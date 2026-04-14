"""Tests for SwiGLU compile-bridge cache keys, shape validation, and CLI surface.

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
    SWIGLU_DECODE_KERNELS,
    SWIGLU_PREFILL_KERNELS,
    swiglu_decode_cache_key,
    swiglu_prefill_cache_key,
    validate_swiglu_decode_shapes,
    validate_swiglu_prefill_shapes,
    get_cached_swiglu_dir,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestSwigluDecodeCacheKey:
    def test_deterministic(self):
        k1 = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        k2 = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        assert k1 == k2

    def test_different_shapes_different_keys(self):
        k1 = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        k2 = swiglu_decode_cache_key(2048, 3584, "bf16", 4)
        k3 = swiglu_decode_cache_key(1024, 2048, "bf16", 4)
        assert len({k1, k2, k3}) == 3

    def test_different_columns_different_keys(self):
        """An NPU1 (4-col) cache entry must not collide with an NPU2 (8-col) entry."""
        k4 = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        k8 = swiglu_decode_cache_key(1024, 3584, "bf16", 8)
        assert k4 != k8

    def test_key_is_hex_string(self):
        k = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_disjoint_from_gemm_and_gemv_keys(self):
        """SwiGLU keys must not collide with GEMM/GEMV keys for matching dims."""
        from compile import gemm_cache_key, gemv_cache_key
        gemm_k = gemm_cache_key(1024, 3584, 1024, "bf16", "bf16", 4)
        gemv_k = gemv_cache_key(1024, 3584, "bf16", "bf16", 4)
        sw_k = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        assert len({gemm_k, gemv_k, sw_k}) == 3

    def test_disjoint_from_swiglu_prefill_key(self):
        """Decode and prefill must produce different keys even for same dims."""
        # Prefill takes seq_len; even if we picked a "natural" seq_len=1, the
        # 'op' field guarantees disjointness.
        k_dec = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        k_pre = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        assert k_dec != k_pre

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme. Bump expected value when changing intentionally."""
        expected_payload = {
            "op": "swiglu_decode",
            "embedding_dim": 1024,
            "hidden_dim": 3584,
            "dtype": "bf16",
            "num_aie_columns": 4,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        assert actual == expected, (
            f"swiglu_decode cache key scheme drift: got {actual}, expected {expected}."
        )


class TestSwigluPrefillCacheKey:
    def test_deterministic(self):
        k1 = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        k2 = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        assert k1 == k2

    def test_different_seq_lens_different_keys(self):
        k1 = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        k2 = swiglu_prefill_cache_key(512, 1024, 3584, "bf16", 4)
        assert k1 != k2

    def test_different_columns_different_keys(self):
        k4 = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        k8 = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 8)
        assert k4 != k8

    def test_key_scheme_pinned(self):
        expected_payload = {
            "op": "swiglu_prefill",
            "seq_len": 256,
            "embedding_dim": 1024,
            "hidden_dim": 3584,
            "dtype": "bf16",
            "num_aie_columns": 4,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        assert actual == expected


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestSwigluDecodeShapeValidation:
    """Mirror of the IRON SwiGLUDecode constraints + GEMV L1 budget."""

    # --- Qwen3.5-0.8B FFN shape (the one we actually need to dispatch) ---

    def test_qwen_decode_shape_passes(self):
        validate_swiglu_decode_shapes(embedding_dim=1024, hidden_dim=3584,
                                      num_aie_columns=4)

    def test_npu2_qwen_decode_shape_passes(self):
        validate_swiglu_decode_shapes(embedding_dim=1024, hidden_dim=3584,
                                      num_aie_columns=8)

    def test_square_2048_passes(self):
        validate_swiglu_decode_shapes(embedding_dim=2048, hidden_dim=2048,
                                      num_aie_columns=4)

    # --- Rejections ---

    def test_invalid_columns_rejected(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            validate_swiglu_decode_shapes(1024, 3584, num_aie_columns=2)

    def test_hidden_not_divisible_by_2cols_rejected(self):
        # hidden_dim=3580, cols=4 -> needs hidden % 8 == 0; 3580 % 8 = 4
        with pytest.raises(ValueError, match="hidden_dim"):
            validate_swiglu_decode_shapes(1024, 3580, num_aie_columns=4)

    def test_embedding_not_divisible_by_cols_rejected(self):
        with pytest.raises(ValueError, match="embedding_dim"):
            validate_swiglu_decode_shapes(1023, 3584, num_aie_columns=4)

    def test_K_not_multiple_of_64_rejected(self):
        # K=1024 ok for gemv_1, but for gemv_2 K=hidden=200 fails GEMV constraint.
        # Use a hidden_dim that's divisible-by-(2*cols) but not by 64.
        # hidden=40, cols=4 -> 40%8=0 ok, but 40%64!=0 fails GEMV K check.
        with pytest.raises(ValueError, match="kernel_vector_size"):
            validate_swiglu_decode_shapes(64, 40, num_aie_columns=4)


class TestSwigluPrefillShapeValidation:
    def test_qwen_prefill_shape_passes(self):
        validate_swiglu_prefill_shapes(seq_len=256, embedding_dim=1024,
                                       hidden_dim=3584, num_aie_columns=4)

    def test_square_2048_passes(self):
        validate_swiglu_prefill_shapes(seq_len=256, embedding_dim=2048,
                                       hidden_dim=2048, num_aie_columns=4)

    def test_invalid_columns_rejected(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            validate_swiglu_prefill_shapes(256, 1024, 3584, num_aie_columns=2)

    def test_seq_len_not_divisible_rejected(self):
        # GEMM requires M % (tile_m * 4) == 0. M=33 has no valid tile_m.
        with pytest.raises(ValueError, match="tile_m"):
            validate_swiglu_prefill_shapes(33, 1024, 3584, num_aie_columns=4)


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestSwigluCacheLayout:
    """The C++ backend reads combined.xclbin + 4 swiglu_<name>.insts files."""

    def test_kernel_name_constants_pinned(self):
        """Pin the kernel-name tuples — the C++ side hard-codes these strings."""
        assert SWIGLU_DECODE_KERNELS == ("gemv_1", "silu", "eltwise_mul", "gemv_2")
        assert SWIGLU_PREFILL_KERNELS == ("gemm_1", "silu", "eltwise_mul", "gemm_2")

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_swiglu_dir("nonexistent_key", SWIGLU_DECODE_KERNELS) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        for name in SWIGLU_DECODE_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        # combined.xclbin missing
        assert get_cached_swiglu_dir(key, SWIGLU_DECODE_KERNELS) is None

    def test_cache_miss_when_any_insts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        # All but the last insts present
        for name in SWIGLU_DECODE_KERNELS[:-1]:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        assert get_cached_swiglu_dir(key, SWIGLU_DECODE_KERNELS) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        for name in SWIGLU_DECODE_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        result = get_cached_swiglu_dir(key, SWIGLU_DECODE_KERNELS)
        assert result == d


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestSwigluCLI:
    """Smoke tests for argparse wiring — actual compilation needs NPU hardware."""

    def test_decode_subcommand_registered(self):
        import compile as compile_mod
        # Sanity: the CLI parser builds without error. Easiest cross-check is
        # to invoke main() with --help on the subcommand.
        import subprocess
        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "swiglu-decode", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--embedding-dim" in result.stdout
        assert "--hidden-dim" in result.stdout
        assert "--num-aie-columns" in result.stdout

    def test_prefill_subcommand_registered(self):
        import compile as compile_mod
        import subprocess
        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "swiglu-prefill", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--seq-len" in result.stdout
        assert "--embedding-dim" in result.stdout
        assert "--hidden-dim" in result.stdout
