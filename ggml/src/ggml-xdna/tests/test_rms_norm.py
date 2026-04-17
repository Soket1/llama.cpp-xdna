"""Tests for the RMSNorm compile-bridge cache keys, shape validation, CLI
surface, and end-to-end numerical correctness.

Pure Python tests (cache-key, validation, CLI, cache layout) run anywhere.
Compile + numerical tests require NPU hardware + an active ironenv and are
skipped when IRON imports fail.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import (
    RMS_NORM_KERNELS,
    rms_norm_cache_key,
    validate_rms_norm_shapes,
    get_cached_chained_dir,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestRmsNormCacheKey:
    def test_deterministic(self):
        k1 = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        k2 = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        assert k1 == k2

    def test_different_sizes_different_keys(self):
        k1 = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        k2 = rms_norm_cache_key(2048, "bf16", 4, 1, 32)
        k3 = rms_norm_cache_key(4096, "bf16", 4, 1, 32)
        assert len({k1, k2, k3}) == 3

    def test_different_tile_sizes_different_keys(self):
        k1 = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        k2 = rms_norm_cache_key(1024, "bf16", 4, 1, 64)
        assert k1 != k2

    def test_different_channel_counts_different_keys(self):
        k1 = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        k2 = rms_norm_cache_key(1024, "bf16", 4, 2, 32)
        assert k1 != k2

    def test_different_columns_different_keys(self):
        k4 = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        k8 = rms_norm_cache_key(1024, "bf16", 8, 1, 32)
        assert k4 != k8

    def test_weighted_disjoint_from_unweighted(self):
        k_u = rms_norm_cache_key(1024, "bf16", 4, 1, 32, weighted=False)
        k_w = rms_norm_cache_key(1024, "bf16", 4, 1, 32, weighted=True)
        assert k_u != k_w

    def test_key_is_hex_string(self):
        k = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_cache_key_stable(self):
        """Pin the exact cache-key scheme. Bump the expected value intentionally
        when the key payload changes."""
        expected_payload = {
            "op": "rms_norm",
            "size": 1024,
            "dtype": "bf16",
            "num_aie_columns": 4,
            "num_channels": 1,
            "tile_size": 32,
            "weighted": False,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        assert actual == expected, (
            f"rms_norm cache-key scheme drift: got {actual}, expected {expected}"
        )

    def test_disjoint_from_gemm_and_gemv_keys(self):
        """RMSNorm key must not collide with GEMM/GEMV keys at matching sizes."""
        from compile import gemm_cache_key, gemv_cache_key
        gemm_k = gemm_cache_key(1024, 1024, 1024, "bf16", "bf16", 4)
        gemv_k = gemv_cache_key(1024, 1024, "bf16", "bf16", 4)
        rms_k = rms_norm_cache_key(1024, "bf16", 4, 1, 32)
        assert len({gemm_k, gemv_k, rms_k}) == 3


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestValidateRmsNormShapes:
    def test_validate_rms_norm_shapes_ok(self):
        ok, reason = validate_rms_norm_shapes(
            size=1024, num_aie_columns=4, num_channels=1, tile_size=32
        )
        assert ok, f"expected OK, got: {reason}"

    def test_validate_rms_norm_shapes_rejects_bad_size(self):
        # 1025 is not divisible by (cols=4 * ch=1 * tile=32) = 128
        ok, reason = validate_rms_norm_shapes(
            size=1025, num_aie_columns=4, num_channels=1, tile_size=32
        )
        assert not ok
        assert "multiple" in (reason or "").lower()

    def test_rejects_channels_zero(self):
        ok, reason = validate_rms_norm_shapes(
            size=1024, num_aie_columns=4, num_channels=0, tile_size=32
        )
        assert not ok

    def test_rejects_cols_zero(self):
        ok, reason = validate_rms_norm_shapes(
            size=1024, num_aie_columns=0, num_channels=1, tile_size=32
        )
        assert not ok

    def test_accepts_two_channels_when_divisible(self):
        # 1024 / (4 * 2 * 32) = 4 — valid
        ok, _ = validate_rms_norm_shapes(
            size=1024, num_aie_columns=4, num_channels=2, tile_size=32
        )
        assert ok

    def test_accepts_8cols_for_npu2(self):
        ok, _ = validate_rms_norm_shapes(
            size=2048, num_aie_columns=8, num_channels=1, tile_size=32
        )
        assert ok


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestRmsNormCacheLayout:
    """The C++ backend reads combined.xclbin + rms_norm_main.insts under
    <cache_dir>/<key>/. Uses the generic get_cached_chained_dir helper with
    prefix="rms_norm"."""

    def test_kernel_name_constant_pinned(self):
        assert RMS_NORM_KERNELS == ("main",)

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_chained_dir(
            "nonexistent_key", RMS_NORM_KERNELS, prefix="rms_norm"
        ) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "rms_norm_main.insts").write_bytes(b"x")
        assert get_cached_chained_dir(
            key, RMS_NORM_KERNELS, prefix="rms_norm"
        ) is None

    def test_cache_miss_when_insts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        assert get_cached_chained_dir(
            key, RMS_NORM_KERNELS, prefix="rms_norm"
        ) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        (d / "rms_norm_main.insts").write_bytes(b"x")
        result = get_cached_chained_dir(
            key, RMS_NORM_KERNELS, prefix="rms_norm"
        )
        assert result == d


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestRmsNormCLI:
    """Smoke tests for argparse wiring."""

    def test_subcommand_registered(self):
        import compile as compile_mod
        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "rms_norm", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--size" in result.stdout
        assert "--num-aie-columns" in result.stdout
        assert "--num-channels" in result.stdout
        assert "--tile-size" in result.stdout
        assert "--weighted" in result.stdout


# ---------------------------------------------------------------------------
# End-to-end compilation (requires NPU + ironenv)
# ---------------------------------------------------------------------------


def _iron_available() -> bool:
    try:
        import aie.utils  # noqa: F401
        import iron.operators.rms_norm.op  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _iron_available(),
                    reason="IRON / aie runtime not importable; needs ironenv + NPU")
class TestRmsNormCompile:
    def test_compile_rms_norm_produces_xclbin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        from compile import compile_rms_norm_cached

        out_dir = compile_rms_norm_cached(
            size=1024, dtype="bf16",
            num_aie_columns=4, num_channels=1, tile_size=32,
        )
        out_dir = Path(out_dir)
        assert (out_dir / "combined.xclbin").exists(), \
            f"combined.xclbin missing under {out_dir}"
        assert (out_dir / "rms_norm_main.insts").exists(), \
            f"rms_norm_main.insts missing under {out_dir}"


@pytest.mark.skipif(not _iron_available(),
                    reason="IRON / aie runtime not importable; needs ironenv + NPU")
class TestRmsNormNumerical:
    """End-to-end numerical correctness: compile, dispatch via IRON's host
    runtime, compare to the reference generator. Tolerance matches the IRON
    suite (rel_tol=0.04)."""

    def test_rms_norm_numerical_correctness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        import aie.utils as aie_utils
        from iron.common import AIEContext
        from iron.operators.rms_norm.op import RMSNorm
        from iron.operators.rms_norm.reference import generate_golden_reference
        from iron.common.test_utils import run_test

        size = 1024
        tile_size = 32
        num_cols = 4
        num_channels = 1
        # size must be divisible by (num_cols * num_channels * tile_size)
        assert size % (num_cols * num_channels * tile_size) == 0

        rows = size // tile_size
        cols = tile_size
        golden = generate_golden_reference(rows=rows, cols=cols, weighted=False)

        ctx = AIEContext()
        try:
            op = RMSNorm(
                size=size,
                num_aie_columns=num_cols,
                num_channels=num_channels,
                tile_size=tile_size,
                weighted=False,
                context=ctx,
            )

            errors, _latency_us, _bw = run_test(
                op,
                {"input1": golden["input"]},
                {"output": golden["output"]},
                rel_tol=0.04, abs_tol=1e-6,
            )
            assert not errors, f"RMSNorm numerical errors: {errors}"
        finally:
            aie_utils.DefaultNPURuntime.cleanup()
