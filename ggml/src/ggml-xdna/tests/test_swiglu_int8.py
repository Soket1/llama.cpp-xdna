"""Tests for W8A16 INT8 SwiGLU decode compile-bridge cache keys, shape
validation, cache layout, and CLI surface.

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
    SWIGLU_DECODE_INT8_KERNELS,
    swiglu_decode_int8_cache_key,
    validate_swiglu_decode_int8_shapes,
    get_cached_swiglu_dir,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestSwigluDecodeInt8CacheKey:
    def test_deterministic(self):
        k1 = swiglu_decode_int8_cache_key(1024, 3584, 4)
        k2 = swiglu_decode_int8_cache_key(1024, 3584, 4)
        assert k1 == k2

    def test_different_shapes_different_keys(self):
        k1 = swiglu_decode_int8_cache_key(1024, 3584, 4)
        k2 = swiglu_decode_int8_cache_key(2048, 3584, 4)
        k3 = swiglu_decode_int8_cache_key(1024, 2048, 4)
        assert len({k1, k2, k3}) == 3

    def test_different_columns_different_keys(self):
        """An NPU1 (4-col) cache entry must not collide with an NPU2 (8-col) entry."""
        k4 = swiglu_decode_int8_cache_key(1024, 3584, 4)
        k8 = swiglu_decode_int8_cache_key(1024, 3584, 8)
        assert k4 != k8

    def test_different_group_size_different_keys(self):
        """group_size is baked into the compiled kernel (-DGROUP_SIZE), so it
        must produce distinct cache entries."""
        k32 = swiglu_decode_int8_cache_key(1024, 3584, 4, group_size=32)
        k64 = swiglu_decode_int8_cache_key(1024, 3584, 4, group_size=64)
        k128 = swiglu_decode_int8_cache_key(1024, 3584, 4, group_size=128)
        assert len({k32, k64, k128}) == 3

    def test_key_is_hex_string(self):
        k = swiglu_decode_int8_cache_key(1024, 3584, 4)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_disjoint_from_bf16_swiglu_keys(self):
        """INT8 keys must never collide with bf16 SwiGLU decode/prefill keys,
        even for identical (embedding_dim, hidden_dim, num_aie_columns)."""
        from compile import swiglu_decode_cache_key, swiglu_prefill_cache_key

        k_bf16_dec = swiglu_decode_cache_key(1024, 3584, "bf16", 4)
        k_bf16_pre = swiglu_prefill_cache_key(256, 1024, 3584, "bf16", 4)
        k_int8 = swiglu_decode_int8_cache_key(1024, 3584, 4)
        assert len({k_bf16_dec, k_bf16_pre, k_int8}) == 3

    def test_disjoint_from_gemm_and_gemv_keys(self):
        """Must not collide with standalone GEMM/GEMV keys either."""
        from compile import gemm_cache_key, gemv_cache_key

        gemm_k = gemm_cache_key(1024, 3584, 1024, "i8", "i32", 4)
        gemv_k = gemv_cache_key(1024, 3584, "bf16", "bf16", 4)
        sw_k = swiglu_decode_int8_cache_key(1024, 3584, 4)
        assert len({gemm_k, gemv_k, sw_k}) == 3

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme. Bump expected value when changing
        intentionally — this guards against silent drift that would invalidate
        all on-disk caches."""
        expected_payload = {
            "op": "swiglu_decode_int8",
            "embedding_dim": 1024,
            "hidden_dim": 3584,
            "num_aie_columns": 4,
            "group_size": 32,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = swiglu_decode_int8_cache_key(1024, 3584, 4)
        assert actual == expected, (
            f"swiglu_decode_int8 cache key scheme drift: got {actual}, "
            f"expected {expected}."
        )


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestSwigluDecodeInt8ShapeValidation:
    """Mirror of the IRON SwiGLUDecodeInt8 constraints + group_size checks."""

    # --- Qwen3.5-0.8B FFN shape (the one we actually need to dispatch) ---

    def test_qwen_decode_shape_passes_npu1(self):
        validate_swiglu_decode_int8_shapes(
            embedding_dim=1024, hidden_dim=3584, num_aie_columns=4,
        )

    def test_qwen_decode_shape_passes_npu2(self):
        validate_swiglu_decode_int8_shapes(
            embedding_dim=1024, hidden_dim=3584, num_aie_columns=8,
        )

    def test_square_2048_passes(self):
        validate_swiglu_decode_int8_shapes(
            embedding_dim=2048, hidden_dim=2048, num_aie_columns=4,
        )

    # --- Rejections ---

    def test_invalid_columns_rejected(self):
        with pytest.raises(ValueError, match="num_aie_columns"):
            validate_swiglu_decode_int8_shapes(1024, 3584, num_aie_columns=2)

    def test_hidden_not_divisible_by_group_size_rejected(self):
        # hidden_dim=3600 divides by 2*4=8 but not by group_size=32.
        # (3600 % 32 = 16). Should trip the group_size check for gemv_int8_2.
        with pytest.raises(ValueError, match="hidden_dim"):
            validate_swiglu_decode_int8_shapes(
                1024, 3600, num_aie_columns=4, group_size=32,
            )

    def test_embedding_not_divisible_by_group_size_rejected(self):
        # embedding_dim=1040 divides by 4 but not by 32 (1040 % 32 = 16).
        with pytest.raises(ValueError, match="embedding_dim"):
            validate_swiglu_decode_int8_shapes(
                1040, 3584, num_aie_columns=4, group_size=32,
            )

    def test_group_size_not_multiple_of_32_rejected(self):
        with pytest.raises(ValueError, match="group_size"):
            validate_swiglu_decode_int8_shapes(
                1024, 3584, num_aie_columns=4, group_size=48,
            )

    def test_group_size_64_accepted_when_shape_compatible(self):
        """The IRON op itself only requires group_size % 32 == 0. If the
        downstream GEMV-int8 op rejects group_size=64 the test documents that;
        otherwise shape validation must let it through."""
        validate_swiglu_decode_int8_shapes(
            1024, 3584, num_aie_columns=4, group_size=64,
        )

    def test_hidden_not_divisible_by_2cols_rejected(self):
        # hidden_dim=3616 is divisible by group_size=32 but not by 2*cols=16
        # (3616 % 16 = 0 — oops). Use 2*cols=16 incompatible and g=32 compatible:
        # hidden=3584+32=3616; 3616 % 16 = 0 too. We need % 8 == 0 and not % 16 == 0
        # AND % 32 == 0 (group_size). 32 is a multiple of 16, so any multiple of
        # 32 automatically clears % 16. Use cols=8 instead: 2*cols=16. hidden=3616
        # — still a multiple of 16.  So at group_size=32 the 2*cols check is
        # redundant with the group_size one for cols in {4,8}. That's fine:
        # it just means the group_size check will be the first to fire. Skip
        # this explicit test case and rely on test_hidden_not_divisible_by_group_size.
        pytest.skip(
            "2*cols divisibility is implied by group_size %==0 for cols in {4,8}"
        )

    def test_embedding_not_divisible_by_cols_rejected(self):
        # embedding=1152 is a multiple of group_size=32 but 1152 % 8 != 0?
        # 1152/8 = 144 — it is. Use embedding=1056 (1056%32=0, 1056%8=0 too).
        # Bottom line: like above, any multiple of group_size=32 is also a
        # multiple of cols in {4,8}. Document that and skip explicit case.
        pytest.skip(
            "cols divisibility is implied by group_size %==0 for cols in {4,8}"
        )

    def test_K_not_multiple_of_64_rejected(self):
        # K constraints from the inner gemv: K % kernel_vector_size (64) == 0.
        # With group_size=32, embedding=32 clears group_size but fails K%64.
        # hidden must also clear 2*cols and group_size, so pick hidden=32 too.
        with pytest.raises(ValueError, match="kernel_vector_size"):
            validate_swiglu_decode_int8_shapes(
                32, 32, num_aie_columns=4, group_size=32,
            )


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestSwigluInt8CacheLayout:
    """The C++ backend reads combined.xclbin + 4 swiglu_<name>.insts files
    (named from SWIGLU_DECODE_INT8_KERNELS)."""

    def test_kernel_name_constants_pinned(self):
        """Pin the kernel-name tuple — the C++ side hard-codes these strings
        when mapping per-slot insts files."""
        assert SWIGLU_DECODE_INT8_KERNELS == (
            "gemv_int8_1", "silu", "eltwise_mul", "gemv_int8_2",
        )

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_swiglu_dir(
            "nonexistent_key", SWIGLU_DECODE_INT8_KERNELS
        ) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        for name in SWIGLU_DECODE_INT8_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        # combined.xclbin missing
        assert get_cached_swiglu_dir(key, SWIGLU_DECODE_INT8_KERNELS) is None

    def test_cache_miss_when_any_insts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        # All but the last insts present
        for name in SWIGLU_DECODE_INT8_KERNELS[:-1]:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        assert get_cached_swiglu_dir(key, SWIGLU_DECODE_INT8_KERNELS) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        for name in SWIGLU_DECODE_INT8_KERNELS:
            (d / f"swiglu_{name}.insts").write_bytes(b"x")
        result = get_cached_swiglu_dir(key, SWIGLU_DECODE_INT8_KERNELS)
        assert result == d


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestSwigluInt8CLI:
    """Smoke test for argparse wiring — actual compilation needs NPU hardware."""

    def test_decode_int8_subcommand_registered(self):
        import compile as compile_mod
        import subprocess

        script = Path(compile_mod.__file__)
        result = subprocess.run(
            ["python3", str(script), "swiglu-decode-int8", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"swiglu-decode-int8 --help failed: stderr={result.stderr!r}"
        )
        assert "--embedding-dim" in result.stdout
        assert "--hidden-dim" in result.stdout
        assert "--num-aie-columns" in result.stdout
        assert "--group-size" in result.stdout
