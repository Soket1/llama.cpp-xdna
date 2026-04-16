"""Tests for fused gate+up+silu+mul INT8 + standalone down GEMV compile-bridge
cache keys, shape validation, cache layout, and CLI surface.

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
    swiglu_fused_int8_cache_key,
    validate_swiglu_decode_int8_shapes,
    get_cached_swiglu_dir,
    SWIGLU_FUSED_CHAINED_KERNELS,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestSwigluFusedInt8CacheKey:
    def test_deterministic(self):
        k1 = swiglu_fused_int8_cache_key(1024, 3584, 4)
        k2 = swiglu_fused_int8_cache_key(1024, 3584, 4)
        assert k1 == k2

    def test_different_shapes_different_keys(self):
        k1 = swiglu_fused_int8_cache_key(1024, 3584, 4)
        k2 = swiglu_fused_int8_cache_key(2048, 3584, 4)
        k3 = swiglu_fused_int8_cache_key(1024, 2048, 4)
        assert len({k1, k2, k3}) == 3

    def test_different_columns_different_keys(self):
        k4 = swiglu_fused_int8_cache_key(1024, 3584, 4)
        k8 = swiglu_fused_int8_cache_key(1024, 3584, 8)
        assert k4 != k8

    def test_different_group_size_different_keys(self):
        k32 = swiglu_fused_int8_cache_key(1024, 3584, 4, group_size=32)
        k64 = swiglu_fused_int8_cache_key(1024, 3584, 4, group_size=64)
        k128 = swiglu_fused_int8_cache_key(1024, 3584, 4, group_size=128)
        assert len({k32, k64, k128}) == 3

    def test_key_is_hex_string(self):
        k = swiglu_fused_int8_cache_key(1024, 3584, 4)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_disjoint_from_chained_int8_keys(self):
        """Fused keys must never collide with chained swiglu_decode_int8 keys."""
        from compile import swiglu_decode_int8_cache_key

        k_chained = swiglu_decode_int8_cache_key(1024, 3584, 4)
        k_fused = swiglu_fused_int8_cache_key(1024, 3584, 4)
        assert k_chained != k_fused

    def test_disjoint_from_bf16_and_gemm_keys(self):
        from compile import (
            swiglu_decode_cache_key, gemm_cache_key, gemv_cache_key,
        )

        keys = {
            swiglu_decode_cache_key(1024, 3584, "bf16", 4),
            gemm_cache_key(1024, 3584, 1024, "i8", "i32", 4),
            gemv_cache_key(1024, 3584, "bf16", "bf16", 4),
            swiglu_fused_int8_cache_key(1024, 3584, 4),
        }
        assert len(keys) == 4

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme to guard against silent drift."""
        expected_payload = {
            "op": "swiglu_fused_int8",
            "embedding_dim": 1024,
            "hidden_dim": 3584,
            "num_aie_columns": 4,
            "group_size": 32,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = swiglu_fused_int8_cache_key(1024, 3584, 4)
        assert actual == expected, (
            f"swiglu_fused_int8 cache key scheme drift: got {actual}, "
            f"expected {expected}."
        )


# ---------------------------------------------------------------------------
# Shape validation (reuses validate_swiglu_decode_int8_shapes — same
# constraints apply since the fused op contains the same inner GEMV stages)
# ---------------------------------------------------------------------------


class TestSwigluFusedInt8ShapeValidation:
    def test_qwen_decode_shape_passes_npu2(self):
        validate_swiglu_decode_int8_shapes(
            embedding_dim=1024, hidden_dim=3584, num_aie_columns=8,
        )

    def test_qwen_decode_shape_passes_npu1(self):
        validate_swiglu_decode_int8_shapes(
            embedding_dim=1024, hidden_dim=3584, num_aie_columns=4,
        )

    def test_invalid_columns_rejected(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            validate_swiglu_decode_int8_shapes(1024, 3584, num_aie_columns=2)

    def test_group_size_not_multiple_of_32_rejected(self):
        with pytest.raises(ValueError, match="group_size"):
            validate_swiglu_decode_int8_shapes(
                1024, 3584, num_aie_columns=4, group_size=48,
            )


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestSwigluFusedInt8CacheLayout:
    """The C++ backend reads combined.xclbin + 2 swiglu_*.insts files."""

    def test_kernel_name_constants_pinned(self):
        assert SWIGLU_FUSED_CHAINED_KERNELS == ("fused", "down_gemv_int8")

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_swiglu_dir(
            "nonexistent_key", SWIGLU_FUSED_CHAINED_KERNELS
        ) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        for name in SWIGLU_FUSED_CHAINED_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        # combined.xclbin missing
        assert get_cached_swiglu_dir(key, SWIGLU_FUSED_CHAINED_KERNELS) is None

    def test_cache_miss_when_insts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        # Only first insts present
        (d / f"swiglu_{SWIGLU_FUSED_CHAINED_KERNELS[0]}.insts").write_bytes(b"x")
        assert get_cached_swiglu_dir(key, SWIGLU_FUSED_CHAINED_KERNELS) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        for name in SWIGLU_FUSED_CHAINED_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        result = get_cached_swiglu_dir(key, SWIGLU_FUSED_CHAINED_KERNELS)
        assert result == d


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestSwigluFusedInt8CLI:
    def test_fused_int8_subcommand_registered(self):
        import compile as compile_mod
        import subprocess

        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "swiglu-fused-int8", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"swiglu-fused-int8 --help failed: stderr={result.stderr!r}"
        )
        assert "--embedding-dim" in result.stdout
        assert "--hidden-dim" in result.stdout
        assert "--num-aie-columns" in result.stdout
        assert "--group-size" in result.stdout
