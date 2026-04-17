"""Tests for chained Q/K/V projection compile-bridge cache keys, shape validation,
cache layout, and CLI surface.

These tests are pure Python — no NPU hardware needed. Compilation itself is
exercised by end-to-end runs via llama-completion (see CLAUDE.md).
"""

import hashlib
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import (
    qkv_cache_key,
    validate_qkv_shapes,
    get_cached_chained_dir,
    QKV_PROJ_KERNELS,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestQKVCacheKey:
    def test_deterministic(self):
        k1 = qkv_cache_key(1024, 1024, 512, 512, 4)
        k2 = qkv_cache_key(1024, 1024, 512, 512, 4)
        assert k1 == k2

    def test_different_shapes_different_keys(self):
        k1 = qkv_cache_key(1024, 1024, 512, 512, 4)
        k2 = qkv_cache_key(2048, 2048, 512, 512, 4)
        k3 = qkv_cache_key(1024, 1024, 1024, 1024, 4)
        assert len({k1, k2, k3}) == 3

    def test_different_columns_different_keys(self):
        k4 = qkv_cache_key(1024, 1024, 512, 512, 4)
        k8 = qkv_cache_key(1024, 1024, 512, 512, 8)
        assert k4 != k8

    def test_q_vs_kv_dim_distinguished(self):
        """Different per-head Q vs K/V sizing must produce different keys."""
        k_mha = qkv_cache_key(1024, 1024, 1024, 1024, 4)  # MHA: Q=K=V
        k_gqa = qkv_cache_key(1024, 1024, 512, 512, 4)    # GQA: K=V<Q
        assert k_mha != k_gqa

    def test_key_is_hex_string(self):
        k = qkv_cache_key(1024, 1024, 512, 512, 4)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_disjoint_from_gemv_and_gemm_keys(self):
        """QKV keys must never collide with standalone GEMV/GEMM keys."""
        from compile import gemv_cache_key, gemm_cache_key

        keys = {
            qkv_cache_key(1024, 1024, 512, 512, 4),
            gemv_cache_key(1024, 1024, "bf16", "bf16", 4),
            gemm_cache_key(1024, 1024, 1024, "bf16", "bf16", 4),
        }
        assert len(keys) == 3

    def test_disjoint_from_swiglu_keys(self):
        from compile import (
            swiglu_decode_cache_key, swiglu_fused_int8_cache_key,
        )

        keys = {
            qkv_cache_key(1024, 1024, 512, 512, 4),
            swiglu_decode_cache_key(1024, 3584, "bf16", 4),
            swiglu_fused_int8_cache_key(1024, 3584, 4),
        }
        assert len(keys) == 3

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme to guard against silent drift."""
        expected_payload = {
            "op": "qkv",
            "embedding_dim": 1024,
            "q_dim": 1024,
            "k_dim": 512,
            "v_dim": 512,
            "num_aie_columns": 4,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = qkv_cache_key(1024, 1024, 512, 512, 4)
        assert actual == expected, (
            f"qkv cache key scheme drift: got {actual}, expected {expected}."
        )


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestQKVShapeValidation:
    def test_qwen08b_gqa_shape_passes_npu1(self):
        # Qwen3.5-0.8B: embedding_dim=1024, 16 heads × 64 = 1024 Q, 8 KV heads × 64 = 512.
        validate_qkv_shapes(
            embedding_dim=1024, q_dim=1024, k_dim=512, v_dim=512,
            num_aie_columns=4,
        )

    def test_qwen08b_gqa_shape_passes_npu2(self):
        validate_qkv_shapes(
            embedding_dim=1024, q_dim=1024, k_dim=512, v_dim=512,
            num_aie_columns=8,
        )

    def test_invalid_columns_rejected(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            validate_qkv_shapes(1024, 1024, 512, 512, num_aie_columns=2)

    def test_nonpositive_dim_rejected(self):
        with pytest.raises(ValueError, match="q_dim"):
            validate_qkv_shapes(1024, 0, 512, 512, num_aie_columns=4)
        with pytest.raises(ValueError, match="k_dim"):
            validate_qkv_shapes(1024, 1024, -1, 512, num_aie_columns=4)
        with pytest.raises(ValueError, match="v_dim"):
            validate_qkv_shapes(1024, 1024, 512, 0, num_aie_columns=4)

    def test_k_not_multiple_of_vec_size_rejected(self):
        # kernel_vector_size=64 ; K=100 should fail the inner GEMV validator.
        with pytest.raises(ValueError):
            validate_qkv_shapes(
                embedding_dim=100, q_dim=1024, k_dim=512, v_dim=512,
                num_aie_columns=4,
            )


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestQKVCacheLayout:
    """The C++ backend reads combined.xclbin + 3 qkv_*.insts files."""

    def test_kernel_name_constants_pinned(self):
        assert QKV_PROJ_KERNELS == ("gemv_q", "gemv_k", "gemv_v")

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_chained_dir(
            "nonexistent_key", QKV_PROJ_KERNELS, prefix="qkv"
        ) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        for name in QKV_PROJ_KERNELS:
            (d / f"qkv_{name}.insts").write_bytes(b"x")
        # combined.xclbin missing
        assert get_cached_chained_dir(
            key, QKV_PROJ_KERNELS, prefix="qkv"
        ) is None

    def test_cache_miss_when_insts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        # Only first 2 of 3 insts present
        for name in QKV_PROJ_KERNELS[:2]:
            (d / f"qkv_{name}.insts").write_bytes(b"x")
        assert get_cached_chained_dir(
            key, QKV_PROJ_KERNELS, prefix="qkv"
        ) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        for name in QKV_PROJ_KERNELS:
            (d / f"qkv_{name}.insts").write_bytes(b"x")
        result = get_cached_chained_dir(key, QKV_PROJ_KERNELS, prefix="qkv")
        assert result == d

    def test_swiglu_cache_under_same_key_does_not_match_qkv(self, tmp_path, monkeypatch):
        """qkv and swiglu bundles under the same cache key are disjoint by prefix."""
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        # Write *swiglu*-prefixed insts — qkv lookup must still miss.
        for name in QKV_PROJ_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        assert get_cached_chained_dir(
            key, QKV_PROJ_KERNELS, prefix="qkv"
        ) is None


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestQKVCLI:
    def test_qkv_subcommand_registered(self):
        import compile as compile_mod
        import subprocess

        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "qkv", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"qkv --help failed: stderr={result.stderr!r}"
        )
        assert "--embedding-dim" in result.stdout
        assert "--q-dim" in result.stdout
        assert "--k-dim" in result.stdout
        assert "--v-dim" in result.stdout
        assert "--num-aie-columns" in result.stdout
