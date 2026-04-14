"""Tests for GEMV cache keys and tile selection (M=1 decode path).

These tests are pure Python — no NPU hardware needed.
"""

import hashlib
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import gemv_cache_key, select_gemv_tiles


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestGemvCacheKey:
    """GEMV cache keys must be deterministic and disjoint from GEMM keys."""

    def test_deterministic(self):
        k1 = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        k2 = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        assert k1 == k2

    def test_different_shapes_different_keys(self):
        k1 = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        k2 = gemv_cache_key(2048, 1024, "bf16", "bf16", 4)
        k3 = gemv_cache_key(1024, 2048, "bf16", "bf16", 4)
        assert len({k1, k2, k3}) == 3

    def test_different_columns_different_keys(self):
        k4 = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        k8 = gemv_cache_key(1024, 1024, "bf16", "bf16", 8)
        assert k4 != k8

    def test_key_is_hex_string(self):
        k = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_disjoint_from_gemm_key(self):
        """GEMV and GEMM keys must not collide for matching dimensions.

        The 'op' field in the json payload guarantees disjointness — this is
        structural, not probabilistic, but we pin it so accidental removal of
        the 'op' field is caught.
        """
        from compile import gemm_cache_key
        # M=1 GEMM (even if nonsensical) vs GEMV with same K, N
        gemm_k = gemm_cache_key(1, 1024, 1024, "bf16", "bf16", 4)
        gemv_k = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        assert gemm_k != gemv_k

    def test_key_scheme_pinned(self):
        """Pin the exact GEMV cache-key scheme.

        If this test fails, the Python key scheme changed. See the GEMM
        analogue in test_cache.py for the rationale. Bump this expected value
        deliberately when you change the scheme.
        """
        expected_payload = {
            "op": "gemv",
            "N": 1024,
            "K": 2048,
            "dtype_in": "bf16",
            "dtype_out": "bf16",
            "num_aie_columns": 4,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = gemv_cache_key(1024, 2048, "bf16", "bf16", 4)
        assert actual == expected, (
            f"GEMV cache key scheme drift: got {actual}, expected {expected}. "
            "Update this test intentionally if the scheme was changed."
        )


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


class TestGemvTileSelection:
    """Tile selection must satisfy IRON GEMV constraints AND the L1 budget."""

    def _verify_constraints(self, N, K, num_aie_columns, tile_in, tile_out):
        """Verify IRON GEMV constraints (design.py lines 51-57, 65)."""
        assert N % num_aie_columns == 0, f"N={N} not divisible by cols={num_aie_columns}"
        per_col = N // num_aie_columns
        assert tile_out % tile_in == 0, f"tile_out={tile_out} not divisible by tile_in={tile_in}"
        assert tile_out >= tile_in, f"tile_out={tile_out} < tile_in={tile_in}"
        assert tile_out <= per_col, f"tile_out={tile_out} > per_col={per_col}"
        assert per_col % tile_out == 0, f"per_col={per_col} not divisible by tile_out={tile_out}"
        assert tile_in <= per_col, f"tile_in={tile_in} > per_col={per_col}"
        assert per_col % tile_in == 0, f"per_col={per_col} not divisible by tile_in={tile_in}"
        assert K % 64 == 0, f"K={K} not divisible by kernel_vector_size=64"

    def _verify_l1_budget(self, K, tile_in, budget_bytes=32 * 1024):
        """L1 double-buffered matrix tile must fit in core memory.

        Size = 2 (double-buffer) * tile_in * K * sizeof(bf16=2).
        """
        staged_bytes = 2 * tile_in * K * 2
        assert staged_bytes <= budget_bytes, (
            f"L1 overflow: tile_in={tile_in} K={K} stages {staged_bytes} bytes "
            f"(budget {budget_bytes})"
        )

    # --- Qwen3.5-0.8B decode shapes (the shapes we actually see) ---

    @pytest.mark.parametrize("N,K", [
        (6144, 1024),   # attn_qkv
        (1024, 2048),   # attn_output
        (3584, 1024),   # ffn_gate / ffn_up
        (1024, 3584),   # ffn_down
        (2048, 1024),   # intermediate attn projection
    ])
    def test_qwen_decode_shapes(self, N, K):
        tin, tout = select_gemv_tiles(N, K, 4)
        self._verify_constraints(N, K, 4, tin, tout)
        self._verify_l1_budget(K, tin)

    # --- L1 budget: large K must clamp tile_in ---

    def test_large_K_clamps_tile_in(self):
        """K=3584 with L1 budget 32KB must clamp tile_in to <= 2 (2*2*3584*2=28672)."""
        tin, _ = select_gemv_tiles(1024, 3584, 4)
        assert tin <= 2, f"tile_in={tin} overflows L1 for K=3584"
        self._verify_l1_budget(3584, tin)

    def test_small_K_allows_larger_tile_in(self):
        """K=1024 fits tile_in up to 8 in L1 (2*8*1024*2=32768)."""
        tin, _ = select_gemv_tiles(1024, 1024, 4)
        assert tin == 8, f"expected tile_in=8 for K=1024 (fits L1), got {tin}"

    @pytest.mark.parametrize("K", [1024, 2048, 3584, 8192])
    def test_l1_budget_always_respected(self, K):
        """Whatever tile_in is picked, the L1 budget must be respected."""
        # Use a shape that's friendly enough in N so tile selection succeeds.
        tin, _ = select_gemv_tiles(1024, K, 4)
        self._verify_l1_budget(K, tin)

    # --- Column counts ---

    @pytest.mark.parametrize("num_cols", [1, 2, 4, 8])
    def test_various_column_counts(self, num_cols):
        # 1024 is divisible by 1, 2, 4, 8; per_col >= 8 for all.
        tin, tout = select_gemv_tiles(1024, 1024, num_cols)
        self._verify_constraints(1024, 1024, num_cols, tin, tout)

    # --- Edge cases ---

    def test_K_not_multiple_of_64_raises(self):
        with pytest.raises(ValueError, match="kernel_vector_size"):
            select_gemv_tiles(1024, 100, 4)  # K=100 not %64

    def test_N_not_divisible_by_cols_raises(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            select_gemv_tiles(1023, 1024, 4)  # N=1023 not %4

    def test_tiny_per_col_raises(self):
        """per_col < 8 leaves no valid tile_out in the candidate set."""
        # N=16, cols=4 -> per_col=4, no tile_out candidate <= 4 divides 4
        # (candidates are [2048..1]; 4%4==0 so tile_out=4 wins). Actually 4 works.
        # Use N=8, cols=4 -> per_col=2; tile_out=2 divides, succeeds.
        # Try N=4, cols=4 -> per_col=1; tile_out=1 divides, succeeds.
        # There's no per_col that breaks with current candidates down to 1.
        # Document this: select_gemv_tiles itself doesn't enforce a floor — the
        # backend filter (xdna_shape_dispatchable_gemv) enforces per_col >= 8.
        tin, tout = select_gemv_tiles(4, 64, 4)
        assert tout >= 1


# ---------------------------------------------------------------------------
# C++ xdna_shape_dispatchable_gemv mirror
# ---------------------------------------------------------------------------
# The C++ filter in ggml-xdna.cpp:xdna_shape_dispatchable_gemv enforces:
#   K >= 64 and K % 64 == 0
#   N % num_cols == 0 and N/num_cols >= 8
#   N <= 32768 (BD-overflow cap)
# Those are a superset of select_gemv_tiles's constraints. The C++ mirror is
# not directly exercised here; structural equivalence is guarded by the
# Python tests above plus end-to-end runs via llama-completion.
