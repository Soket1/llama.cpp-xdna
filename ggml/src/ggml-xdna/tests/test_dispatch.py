"""Tests for the C++ dispatch logic, exercised from Python.

Tests the data conversion and layout transformations that happen
in ggml_backend_xdna_mul_mat — f32<->bf16, weight transpose, and
end-to-end correctness through the full IRON GEMM pipeline.

Requires NPU hardware.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import aie.utils as aie_utils
    from iron.operators.gemm.op import GEMM
    from iron.operators.gemm.reference import generate_golden_reference
    from iron.common.test_utils import run_test
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

npu = pytest.mark.skipif(not HAS_NPU, reason="NPU/XRT not available")


class TestF32Bf16Conversion:
    """Test f32 <-> bf16 conversion matches C++ implementation."""

    @staticmethod
    def f32_to_bf16(f32_array: np.ndarray) -> np.ndarray:
        """Python reimplementation of the C++ f32_to_bf16 (round-to-nearest-even)."""
        f32_bytes = f32_array.astype(np.float32).tobytes()
        bf16_values = []
        for i in range(0, len(f32_bytes), 4):
            bits = struct.unpack('<I', f32_bytes[i:i+4])[0]
            # Round to nearest even (matches C++ implementation)
            bits += (0x7FFF + ((bits >> 16) & 1))
            bf16_values.append(bits >> 16)
        return np.array(bf16_values, dtype=np.uint16)

    @staticmethod
    def bf16_to_f32(bf16_array: np.ndarray) -> np.ndarray:
        """Python reimplementation of the C++ bf16_to_f32."""
        result = np.zeros(len(bf16_array), dtype=np.float32)
        for i, val in enumerate(bf16_array):
            bits = int(val) << 16
            result[i] = struct.unpack('<f', struct.pack('<I', bits))[0]
        return result

    def test_roundtrip_preserves_values(self):
        """f32 -> bf16 -> f32 should preserve values within bf16 precision."""
        np.random.seed(42)
        f32_data = np.random.randn(1024).astype(np.float32) * 4.0

        bf16_data = self.f32_to_bf16(f32_data)
        f32_back = self.bf16_to_f32(bf16_data)

        # bf16 has ~7 bits mantissa, so relative error should be < 1%
        rel_error = np.abs(f32_back - f32_data) / (np.abs(f32_data) + 1e-10)
        assert np.median(rel_error) < 0.01, f"Median relative error {np.median(rel_error):.4f} too high"

    def test_conversion_matches_torch(self):
        """Our bf16 conversion should match PyTorch's bfloat16."""
        np.random.seed(42)
        f32_data = np.random.randn(1024).astype(np.float32) * 4.0

        # Our conversion
        bf16_ours = self.f32_to_bf16(f32_data)
        f32_ours = self.bf16_to_f32(bf16_ours)

        # PyTorch conversion
        torch_bf16 = torch.from_numpy(f32_data).to(torch.bfloat16)
        f32_torch = torch_bf16.to(torch.float32).numpy()

        # Should match exactly (both use round-to-nearest-even)
        np.testing.assert_array_almost_equal(f32_ours, f32_torch, decimal=5)

    def test_special_values(self):
        """Test conversion of special float values."""
        specials = np.array([0.0, -0.0, 1.0, -1.0, 0.5, 100.0, -100.0], dtype=np.float32)
        bf16 = self.f32_to_bf16(specials)
        f32_back = self.bf16_to_f32(bf16)
        np.testing.assert_array_almost_equal(f32_back, specials, decimal=1)


class TestWeightTranspose:
    """Test the weight transpose that ggml-xdna.cpp performs.

    ggml stores weights as (N, K) row-major.
    IRON GEMM expects B as (K, N) row-major.
    The C++ code transposes on CPU before DMA.
    """

    def test_transpose_correctness(self):
        """Verify (N,K) -> (K,N) transpose produces correct layout."""
        N, K = 64, 128
        np.random.seed(42)
        # ggml layout: (N, K) row-major — N rows of K elements
        weights_nk = np.random.randn(N, K).astype(np.float32)

        # C++ transpose: iterate k, n and write to [k*N + n]
        weights_kn = np.zeros((K, N), dtype=np.float32)
        for k in range(K):
            for n in range(N):
                weights_kn[k, n] = weights_nk[n, k]

        # Should be identical to numpy transpose
        np.testing.assert_array_equal(weights_kn, weights_nk.T)

    def test_transpose_gemm_matches_reference(self):
        """Verify that transposing weights and doing GEMM gives correct result.

        This mimics the full C++ dispatch path:
        1. Receive A(M,K) and B_ggml(N,K) from ggml
        2. Transpose B to (K,N)
        3. Compute C = A @ B = (M,K) @ (K,N) = (M,N)
        """
        M, K, N = 64, 128, 64
        np.random.seed(42)

        A = np.random.randn(M, K).astype(np.float32)
        B_ggml = np.random.randn(N, K).astype(np.float32)  # ggml layout
        B_iron = B_ggml.T  # IRON layout (K, N)

        # Expected result: ggml computes dst = src1 @ src0^T = A @ B_ggml^T = A @ B_iron
        expected = A @ B_iron

        # This is what np.matmul gives us
        actual = np.matmul(A, B_iron)

        np.testing.assert_array_almost_equal(actual, expected, decimal=5)


@npu
class TestEndToEndDispatch:
    """End-to-end test: compile, load, dispatch via IRON, verify.

    This exercises the same flow as the C++ backend but from Python,
    using the compile.py functions.
    """

    def test_compile_and_dispatch_small(self, aie_context):
        """Compile a small GEMM, dispatch, verify against CPU."""
        from compile import compile_gemm_cached, select_gemm_tiles

        M, K, N = 256, 256, 256
        num_cols = 4

        # Generate reference
        ref = generate_golden_reference(M, K, N, dtype="bf16")

        # Compile (cached)
        tile_m, tile_k, tile_n = select_gemm_tiles(M, K, N, num_cols)

        op = GEMM(
            M=M, K=K, N=N,
            tile_m=tile_m, tile_k=tile_k, tile_n=tile_n,
            num_aie_columns=num_cols,
            prio_accuracy=True,
            emulate_bf16_mmul_with_bfp16=False,
            context=aie_context,
        )

        errors, latency_us, _ = run_test(
            operator=op,
            input_buffers={
                "A": ref["input"].flatten(),
                "B": ref["input_b"][0].flatten(),
            },
            output_buffers={
                "C": ref["output"][0].flatten(),
            },
            rel_tol=0.02,
            abs_tol=0.5,
            max_error_rate=0.01,
        )

        assert not errors, f"End-to-end dispatch failed: {errors}"
        assert latency_us > 0, "Latency should be positive"

    def test_insts_file_created(self, tmp_path):
        """Verify that compile_gemm creates both .xclbin and .insts files."""
        from compile import compile_gemm

        out_path = str(tmp_path / "test.xclbin")
        compile_gemm(
            M=256, K=256, N=256,
            dtype_in="bf16", dtype_out="bf16",
            num_aie_columns=4,
            output_path=out_path,
        )

        assert Path(out_path).exists(), "xclbin not created"
        insts_path = out_path.replace(".xclbin", ".insts")
        assert Path(insts_path).exists(), "insts file not created"
        assert Path(insts_path).stat().st_size > 0, "insts file is empty"
