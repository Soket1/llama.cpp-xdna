"""Tests for W8A8 INT8 SwiGLU prefill compile-bridge cache keys, shape
validation, cache layout, and CLI surface.

These tests are pure Python — no NPU hardware needed.
"""

import hashlib
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import (
    swiglu_prefill_int8_cache_key,
    validate_swiglu_prefill_int8_shapes,
    get_cached_swiglu_dir,
    SWIGLU_PREFILL_INT8_KERNELS,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestSwigluPrefillInt8CacheKey:
    def test_deterministic(self):
        k1 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        k2 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        assert k1 == k2

    def test_different_seq_len_different_keys(self):
        k1 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        k2 = swiglu_prefill_int8_cache_key(128, 1024, 3584, 8)
        assert k1 != k2

    def test_different_shapes_different_keys(self):
        k1 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        k2 = swiglu_prefill_int8_cache_key(64, 2048, 3584, 8)
        k3 = swiglu_prefill_int8_cache_key(64, 1024, 4096, 8)
        assert len({k1, k2, k3}) == 3

    def test_different_columns_different_keys(self):
        k4 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 4)
        k8 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        assert k4 != k8

    def test_different_tile_m_different_keys(self):
        k_default = swiglu_prefill_int8_cache_key(256, 1024, 3584, 8)
        k_16 = swiglu_prefill_int8_cache_key(256, 1024, 3584, 8, tile_m=16)
        k_32 = swiglu_prefill_int8_cache_key(256, 1024, 3584, 8, tile_m=32)
        assert len({k_default, k_16, k_32}) == 3

    def test_key_is_hex_string(self):
        k = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_disjoint_from_bf16_prefill_keys(self):
        from compile import swiglu_prefill_cache_key
        k_bf16 = swiglu_prefill_cache_key(64, 1024, 3584, "bf16", 8, tile_m=16)
        k_i8 = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8, tile_m=16)
        assert k_bf16 != k_i8

    def test_disjoint_from_other_keys(self):
        from compile import (
            swiglu_decode_cache_key, gemm_cache_key, gemv_cache_key,
            swiglu_decode_int8_cache_key,
        )
        keys = {
            swiglu_decode_cache_key(1024, 3584, "bf16", 4),
            gemm_cache_key(1024, 3584, 1024, "i8", "i32", 4),
            gemv_cache_key(1024, 3584, "bf16", "bf16", 4),
            swiglu_decode_int8_cache_key(1024, 3584, 4),
            swiglu_prefill_int8_cache_key(64, 1024, 3584, 4),
        }
        assert len(keys) == 5

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme to guard against silent drift."""
        expected_payload = {
            "op": "swiglu_prefill_int8",
            "seq_len": 64,
            "embedding_dim": 1024,
            "hidden_dim": 3584,
            "num_aie_columns": 8,
            "tile_m": None,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = swiglu_prefill_int8_cache_key(64, 1024, 3584, 8)
        assert actual == expected


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestSwigluPrefillInt8ShapeValidation:
    def test_qwen_shapes_pass(self):
        validate_swiglu_prefill_int8_shapes(
            seq_len=256, embedding_dim=1024, hidden_dim=3584, num_aie_columns=8,
        )

    def test_qwen_shapes_tile_m_16(self):
        validate_swiglu_prefill_int8_shapes(
            seq_len=64, embedding_dim=1024, hidden_dim=3584,
            num_aie_columns=8, tile_m=16,
        )

    def test_invalid_columns_rejected(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            validate_swiglu_prefill_int8_shapes(256, 1024, 3584, num_aie_columns=2)

    def test_tile_m_8_rejected_for_int8(self):
        """INT8 GEMM requires tile_m >= 16, so tile_m=8 is invalid."""
        with pytest.raises(ValueError, match="tile_m"):
            validate_swiglu_prefill_int8_shapes(
                256, 1024, 3584, num_aie_columns=8, tile_m=8,
            )

    def test_seq_len_not_aligned_rejected(self):
        with pytest.raises(ValueError, match="seq_len"):
            validate_swiglu_prefill_int8_shapes(
                48, 1024, 3584, num_aie_columns=8, tile_m=16,
            )


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestSwigluPrefillInt8CacheLayout:
    def test_kernel_name_constants(self):
        assert SWIGLU_PREFILL_INT8_KERNELS == ("gemm_1", "silu", "eltwise_mul", "gemm_2")

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_swiglu_dir(
            "nonexistent_key", SWIGLU_PREFILL_INT8_KERNELS
        ) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        for name in SWIGLU_PREFILL_INT8_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        assert get_cached_swiglu_dir(key, SWIGLU_PREFILL_INT8_KERNELS) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        for name in SWIGLU_PREFILL_INT8_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        result = get_cached_swiglu_dir(key, SWIGLU_PREFILL_INT8_KERNELS)
        assert result == d


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestSwigluPrefillInt8CLI:
    def test_subcommand_registered(self):
        import compile as compile_mod
        import subprocess

        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "swiglu-prefill-int8", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--seq-len" in result.stdout
        assert "--embedding-dim" in result.stdout
        assert "--hidden-dim" in result.stdout
        assert "--num-aie-columns" in result.stdout
        assert "--tile-m" in result.stdout
