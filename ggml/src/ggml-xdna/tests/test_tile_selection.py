"""Tests for GEMM tile size selection.

These tests are pure Python — no NPU hardware needed.
They verify that the tile selection logic satisfies IRON GEMM constraints.
"""

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import select_gemm_tiles


class TestTileSelection:
    """Tile selection must satisfy all IRON GEMM constraints."""

    def _verify_constraints(self, M, K, N, num_aie_columns, tile_m, tile_k, tile_n):
        """Verify that selected tiles satisfy all IRON GEMM constraints."""
        assert M % (tile_m * 4) == 0, f"M={M} not divisible by tile_m*4={tile_m * 4}"
        assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
        assert N % (tile_n * num_aie_columns) == 0, (
            f"N={N} not divisible by tile_n*cols={tile_n * num_aie_columns}"
        )
        assert tile_m >= 8, f"tile_m={tile_m} below minimum 8"
        assert tile_k >= 8, f"tile_k={tile_k} below minimum 8"
        assert tile_n >= 8, f"tile_n={tile_n} below minimum 8"

    # --- Standard shapes (powers of 2) ---

    @pytest.mark.parametrize("M,K,N", [
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 1024, 1024),
        (256, 256, 256),
    ])
    @pytest.mark.parametrize("num_aie_columns", [1, 2, 4, 8])
    def test_power_of_2_shapes(self, M, K, N, num_aie_columns):
        tm, tk, tn = select_gemm_tiles(M, K, N, num_aie_columns)
        self._verify_constraints(M, K, N, num_aie_columns, tm, tk, tn)

    # --- Llama 3.2 1B shapes (the first model we'll target) ---

    @pytest.mark.parametrize("M,K,N", [
        # Attention Q/K/V/O projections (hidden=2048, heads=32, head_dim=64)
        (2048, 2048, 2048),  # Q projection (prefill seq_len=2048)
        (256, 2048, 2048),   # Q projection (prefill seq_len=256)
        # FFN (intermediate_size=5632 or 8192 depending on config)
        (2048, 2048, 8192),  # gate/up projection
        (2048, 8192, 2048),  # down projection
        (256, 2048, 8192),   # gate/up projection (shorter seq)
        (256, 8192, 2048),   # down projection (shorter seq)
    ])
    @pytest.mark.parametrize("num_aie_columns", [4, 8])
    def test_llama_shapes(self, M, K, N, num_aie_columns):
        tm, tk, tn = select_gemm_tiles(M, K, N, num_aie_columns)
        self._verify_constraints(M, K, N, num_aie_columns, tm, tk, tn)

    # --- Non-standard shapes that llama.cpp might produce ---

    @pytest.mark.parametrize("M,K,N,num_aie_columns", [
        (64, 512, 256, 4),    # Small matmul
        (128, 768, 3072, 8),  # BERT-like
        (512, 1024, 4096, 4), # Generic transformer
    ])
    def test_mixed_shapes(self, M, K, N, num_aie_columns):
        tm, tk, tn = select_gemm_tiles(M, K, N, num_aie_columns)
        self._verify_constraints(M, K, N, num_aie_columns, tm, tk, tn)

    # --- Edge cases ---

    def test_minimum_valid_shape(self):
        """Smallest shape that should work: 32x8x8 with 1 column."""
        tm, tk, tn = select_gemm_tiles(32, 8, 8, 1)
        self._verify_constraints(32, 8, 8, 1, tm, tk, tn)

    def test_large_N_with_many_columns(self):
        """Large N with 8 columns should find valid tiles."""
        tm, tk, tn = select_gemm_tiles(2048, 2048, 8192, 8)
        self._verify_constraints(2048, 2048, 8192, 8, tm, tk, tn)

    def test_invalid_M_raises(self):
        """M that can't be tiled should raise ValueError."""
        # M=7 can't be divided by any tile_m * 4
        with pytest.raises(ValueError, match="tile_m"):
            select_gemm_tiles(7, 64, 64, 1)

    def test_invalid_K_raises(self):
        """K that can't be tiled should raise ValueError."""
        with pytest.raises(ValueError, match="tile_k"):
            select_gemm_tiles(256, 3, 64, 1)

    def test_invalid_N_raises(self):
        """N that can't be tiled with given columns should raise ValueError."""
        # N=7, no tile_n * 4 columns divides it
        with pytest.raises(ValueError, match="tile_n"):
            select_gemm_tiles(256, 64, 7, 4)

    # --- int8 constraints ---

    @pytest.mark.parametrize("M,K,N", [
        (2048, 2048, 2048),
        (256, 2048, 2048),
    ])
    def test_int8_shapes(self, M, K, N):
        """INT8 GEMM has different minimum tile sizes."""
        tm, tk, tn = select_gemm_tiles(M, K, N, 4, dtype_in="i8")
        self._verify_constraints(M, K, N, 4, tm, tk, tn)
        # INT8 has higher minimums
        assert tm >= 16, f"INT8 tile_m={tm} below minimum 16"
        assert tn >= 16, f"INT8 tile_n={tn} below minimum 16"
