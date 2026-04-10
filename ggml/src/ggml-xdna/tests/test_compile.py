"""Tests for IRON GEMM compilation and correctness.

These tests require NPU hardware and IRON environment:
    source ironenv/bin/activate
    source /opt/xilinx/xrt/setup.sh

Mark tests that need hardware with @pytest.mark.npu so pure tests can run without it.
"""

import os
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import compile_gemm, compile_gemm_cached, gemm_cache_key

# Skip all NPU tests if XRT is not available
try:
    import aie.utils as aie_utils
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

npu = pytest.mark.skipif(not HAS_NPU, reason="NPU/XRT not available")


@npu
class TestCompileGemm:
    """Test that compile_gemm produces valid xclbin files."""

    def test_basic_bf16_gemm(self, tmp_path):
        """Compile a basic bf16 GEMM and verify xclbin is created."""
        out_path = str(tmp_path / "gemm.xclbin")
        result = compile_gemm(
            M=256, K=256, N=256,
            dtype_in="bf16", dtype_out="bf16",
            num_aie_columns=4,
            output_path=out_path,
        )
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_rectangular_bf16_gemm(self, tmp_path):
        """Compile a rectangular (non-square) GEMM."""
        out_path = str(tmp_path / "gemm_rect.xclbin")
        result = compile_gemm(
            M=256, K=2048, N=512,
            dtype_in="bf16", dtype_out="bf16",
            num_aie_columns=4,
            output_path=out_path,
        )
        assert Path(result).exists()

    def test_llama_qkv_shape(self, tmp_path):
        """Compile GEMM with Llama 3.2 1B Q/K/V projection shapes."""
        out_path = str(tmp_path / "gemm_qkv.xclbin")
        result = compile_gemm(
            M=256, K=2048, N=2048,
            dtype_in="bf16", dtype_out="bf16",
            num_aie_columns=4,
            output_path=out_path,
        )
        assert Path(result).exists()

    def test_llama_ffn_gate_shape(self, tmp_path):
        """Compile GEMM with Llama FFN gate/up projection shapes."""
        out_path = str(tmp_path / "gemm_ffn.xclbin")
        result = compile_gemm(
            M=256, K=2048, N=8192,
            dtype_in="bf16", dtype_out="bf16",
            num_aie_columns=4,
            output_path=out_path,
        )
        assert Path(result).exists()


@npu
class TestCompileGemmCached:
    """Test cache hit/miss behavior with real compilation."""

    def test_cache_miss_compiles(self):
        """First call compiles and caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GGML_XDNA_CACHE_DIR"] = tmpdir
            try:
                path = compile_gemm_cached(
                    M=256, K=256, N=256,
                    dtype_in="bf16", dtype_out="bf16",
                    num_aie_columns=4,
                )
                assert path.exists()
                assert path.suffix == ".xclbin"
                assert str(path).startswith(tmpdir)
            finally:
                del os.environ["GGML_XDNA_CACHE_DIR"]

    def test_cache_hit_skips_compile(self):
        """Second call with same params returns cached result without recompiling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GGML_XDNA_CACHE_DIR"] = tmpdir
            try:
                # First call: compiles
                path1 = compile_gemm_cached(
                    M=256, K=256, N=256,
                    dtype_in="bf16", dtype_out="bf16",
                    num_aie_columns=4,
                )
                mtime1 = path1.stat().st_mtime

                # Second call: should return cached, same file
                path2 = compile_gemm_cached(
                    M=256, K=256, N=256,
                    dtype_in="bf16", dtype_out="bf16",
                    num_aie_columns=4,
                )
                mtime2 = path2.stat().st_mtime

                assert path1 == path2
                assert mtime1 == mtime2  # file was not rewritten
            finally:
                del os.environ["GGML_XDNA_CACHE_DIR"]

    def test_different_shapes_different_cache_entries(self):
        """Different shapes produce different cached xclbins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GGML_XDNA_CACHE_DIR"] = tmpdir
            try:
                path1 = compile_gemm_cached(
                    M=256, K=256, N=256,
                    dtype_in="bf16", dtype_out="bf16",
                    num_aie_columns=4,
                )
                path2 = compile_gemm_cached(
                    M=256, K=256, N=512,
                    dtype_in="bf16", dtype_out="bf16",
                    num_aie_columns=4,
                )
                assert path1 != path2
                assert path1.exists()
                assert path2.exists()
            finally:
                del os.environ["GGML_XDNA_CACHE_DIR"]


@npu
class TestCompileAndRun:
    """End-to-end: compile GEMM, load it, run it, verify correctness."""

    def test_bf16_gemm_correctness(self, aie_context):
        """Compile and run a bf16 GEMM, verify results against CPU reference."""
        from iron.operators.gemm.op import GEMM
        from iron.operators.gemm.reference import generate_golden_reference
        from iron.common.test_utils import run_test

        M, K, N = 256, 256, 256
        num_aie_columns = 4

        ref = generate_golden_reference(M, K, N, dtype="bf16")

        op = GEMM(
            M=M, K=K, N=N,
            tile_m=64, tile_k=64, tile_n=64,
            num_aie_columns=num_aie_columns,
            prio_accuracy=True,
            emulate_bf16_mmul_with_bfp16=False,
            context=aie_context,
        )

        errors, latency_us, bandwidth_gbps = run_test(
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

        assert not errors, f"GEMM correctness failed: {errors}"
        assert latency_us > 0

    def test_rectangular_gemm_correctness(self, aie_context):
        """Test non-square GEMM that mimics Llama shapes."""
        from iron.operators.gemm.op import GEMM
        from iron.operators.gemm.reference import generate_golden_reference
        from iron.common.test_utils import run_test

        M, K, N = 256, 2048, 2048
        num_aie_columns = 4

        ref = generate_golden_reference(M, K, N, dtype="bf16")

        op = GEMM(
            M=M, K=K, N=N,
            tile_m=64, tile_k=64, tile_n=64,
            num_aie_columns=num_aie_columns,
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

        assert not errors, f"Rectangular GEMM correctness failed: {errors}"
