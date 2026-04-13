"""Tests for xclbin cache key generation and cache management.

These tests are pure Python — no NPU hardware needed.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Add parent dir to path so we can import compile.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import gemm_cache_key, get_cache_dir, get_cached_xclbin


class TestCacheKey:
    """Cache key generation must be deterministic and unique."""

    def test_deterministic(self):
        """Same inputs produce same key."""
        key1 = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        key2 = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        assert key1 == key2

    def test_different_shapes_different_keys(self):
        """Different M/K/N produce different keys."""
        key1 = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        key2 = gemm_cache_key(2048, 2048, 1024, "bf16", "bf16", 4)
        key3 = gemm_cache_key(1024, 2048, 2048, "bf16", "bf16", 4)
        key4 = gemm_cache_key(2048, 1024, 2048, "bf16", "bf16", 4)
        assert len({key1, key2, key3, key4}) == 4

    def test_different_dtypes_different_keys(self):
        """Different dtypes produce different keys."""
        key_bf16 = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        key_i8 = gemm_cache_key(2048, 2048, 2048, "i8", "i8", 4)
        assert key_bf16 != key_i8

    def test_different_columns_different_keys(self):
        """Different num_aie_columns produce different keys."""
        key_4col = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        key_8col = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 8)
        assert key_4col != key_8col

    def test_b_col_maj_different_key(self):
        """b_col_maj flag changes the key."""
        key_row = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4, b_col_maj=False)
        key_col = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4, b_col_maj=True)
        assert key_row != key_col

    def test_key_is_hex_string(self):
        """Key should be a short hex string."""
        key = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme (sha256 of sorted-json, 16-char prefix).

        If this test fails, the Python key scheme changed. That is fine in
        isolation (the C++ side uses its own human-readable key format like
        gemm_MxKxN_bf16_Ncol, intentionally decoupled because C++ passes
        --out to force the output path), but any already-cached xclbins under
        ~/.cache/ggml-xdna will be effectively orphaned. Bump this expected
        value deliberately when you change the scheme.
        """
        import hashlib, json
        expected_payload = {
            "op": "gemm",
            "M": 2048,
            "K": 2048,
            "N": 2048,
            "dtype_in": "bf16",
            "dtype_out": "bf16",
            "num_aie_columns": 4,
            "b_col_maj": False,
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = gemm_cache_key(2048, 2048, 2048, "bf16", "bf16", 4)
        assert actual == expected, (
            f"Cache key scheme drift: got {actual}, expected {expected}. "
            "Update this test intentionally if the scheme was changed."
        )


class TestCacheDirectory:

    def test_default_cache_dir(self):
        """Default cache dir is under ~/.cache/ggml-xdna/xclbin/."""
        # Clear env override if set
        env_backup = os.environ.pop("GGML_XDNA_CACHE_DIR", None)
        try:
            cache_dir = get_cache_dir()
            assert str(cache_dir).endswith(".cache/ggml-xdna/xclbin")
        finally:
            if env_backup:
                os.environ["GGML_XDNA_CACHE_DIR"] = env_backup

    def test_custom_cache_dir_via_env(self):
        """GGML_XDNA_CACHE_DIR env var overrides default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GGML_XDNA_CACHE_DIR"] = tmpdir
            try:
                cache_dir = get_cache_dir()
                assert str(cache_dir) == tmpdir
            finally:
                del os.environ["GGML_XDNA_CACHE_DIR"]

    def test_cache_miss(self):
        """Non-existent key returns None."""
        result = get_cached_xclbin("nonexistent_key_12345")
        assert result is None

    def test_cache_hit(self):
        """Existing xclbin + insts files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GGML_XDNA_CACHE_DIR"] = tmpdir
            try:
                # Create fake cached xclbin and insts
                key = "test_cache_key_1"
                fake_xclbin = Path(tmpdir) / f"{key}.xclbin"
                fake_insts = Path(tmpdir) / f"{key}.insts"
                fake_xclbin.write_bytes(b"fake xclbin data")
                fake_insts.write_bytes(b"fake insts data")

                result = get_cached_xclbin(key)
                assert result is not None
                assert result == fake_xclbin
            finally:
                del os.environ["GGML_XDNA_CACHE_DIR"]

    def test_cache_miss_without_insts(self):
        """xclbin without matching insts file is a cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["GGML_XDNA_CACHE_DIR"] = tmpdir
            try:
                key = "test_cache_key_2"
                fake_xclbin = Path(tmpdir) / f"{key}.xclbin"
                fake_xclbin.write_bytes(b"fake xclbin data")
                # No .insts file

                result = get_cached_xclbin(key)
                assert result is None
            finally:
                del os.environ["GGML_XDNA_CACHE_DIR"]
