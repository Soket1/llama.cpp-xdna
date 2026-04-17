"""Tests for the AttentionBlockPrefill compile-bridge: cache keys, seq-len
bucketing, shape validation, cache layout, CLI surface, and an opt-in
end-to-end compile check.

All tests except the end-to-end compile are pure Python — no NPU hardware
needed.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from compile import (
    attention_prefill_cache_key,
    validate_attention_prefill_shapes,
    select_attention_prefill_seq_bucket,
    get_cached_chained_dir,
    ATTENTION_PREFILL_KERNELS,
    ATTENTION_PREFILL_SEQ_BUCKETS,
)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


class TestAttentionPrefillCacheKey:
    def test_attention_prefill_cache_key_stable(self):
        k1 = attention_prefill_cache_key(
            seq_len=512, embed_dim=1024, num_heads=16, num_kv_heads=2,
            head_dim=64,
        )
        k2 = attention_prefill_cache_key(
            seq_len=512, embed_dim=1024, num_heads=16, num_kv_heads=2,
            head_dim=64,
        )
        assert k1 == k2

    def test_attention_prefill_cache_key_varies_with_shape(self):
        k_base = attention_prefill_cache_key(512, 1024, 16, 2, 64)
        k_embed = attention_prefill_cache_key(512, 2048, 16, 2, 64)
        k_heads = attention_prefill_cache_key(512, 1024, 32, 2, 64)
        k_kv = attention_prefill_cache_key(512, 1024, 16, 4, 64)
        # seq_len moves only if it crosses a bucket
        k_seq = attention_prefill_cache_key(1024, 1024, 16, 2, 64)
        assert len({k_base, k_embed, k_heads, k_kv, k_seq}) == 5

    def test_cache_key_shared_within_bucket(self):
        """seq_len values inside the same bucket must produce the same key —
        that's the whole point of bucketing."""
        k_256 = attention_prefill_cache_key(200, 1024, 16, 2, 64)
        k_bucket = attention_prefill_cache_key(256, 1024, 16, 2, 64)
        assert k_256 == k_bucket

    def test_key_is_hex_string(self):
        k = attention_prefill_cache_key(512, 1024, 16, 2, 64)
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)

    def test_key_scheme_pinned(self):
        """Pin the exact cache-key scheme to guard against silent drift."""
        expected_payload = {
            "op": "attention_prefill",
            "seq_len_padded": 512,
            "embed_dim": 1024,
            "num_heads": 16,
            "num_kv_heads": 2,
            "head_dim": 64,
            "dtype": "bf16",
        }
        expected = hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        actual = attention_prefill_cache_key(512, 1024, 16, 2, 64)
        assert actual == expected

    def test_disjoint_from_other_keys(self):
        from compile import (
            swiglu_decode_cache_key, gemm_cache_key, gemv_cache_key,
            swiglu_prefill_cache_key, qkv_cache_key, rms_norm_cache_key,
            swiglu_prefill_int8_cache_key,
        )
        keys = {
            attention_prefill_cache_key(512, 1024, 16, 2, 64),
            swiglu_decode_cache_key(1024, 3584, "bf16", 4),
            gemm_cache_key(1024, 1024, 1024, "bf16", "bf16", 8),
            gemv_cache_key(1024, 1024, "bf16", "bf16", 4),
            swiglu_prefill_cache_key(64, 1024, 3584, "bf16", 8),
            swiglu_prefill_int8_cache_key(64, 1024, 3584, 8),
            qkv_cache_key(1024, 1024, 128, 128, 4),
            rms_norm_cache_key(1024, "bf16", 4, 1, 32, False),
        }
        assert len(keys) == 8


# ---------------------------------------------------------------------------
# Seq-len bucketing
# ---------------------------------------------------------------------------


class TestAttentionPrefillSeqBucket:
    def test_select_seq_bucket_rounds_up(self):
        assert select_attention_prefill_seq_bucket(1) == 128
        assert select_attention_prefill_seq_bucket(128) == 128
        assert select_attention_prefill_seq_bucket(129) == 256
        assert select_attention_prefill_seq_bucket(200) == 256
        assert select_attention_prefill_seq_bucket(256) == 256
        assert select_attention_prefill_seq_bucket(257) == 512
        assert select_attention_prefill_seq_bucket(512) == 512
        assert select_attention_prefill_seq_bucket(1000) == 1024
        assert select_attention_prefill_seq_bucket(4096) == 4096

    def test_select_seq_bucket_rejects_too_large(self):
        largest = ATTENTION_PREFILL_SEQ_BUCKETS[-1]
        assert select_attention_prefill_seq_bucket(largest + 1) is None
        assert select_attention_prefill_seq_bucket(100_000) is None

    def test_select_seq_bucket_rejects_non_positive(self):
        assert select_attention_prefill_seq_bucket(0) is None
        assert select_attention_prefill_seq_bucket(-5) is None

    def test_buckets_are_sorted_ascending(self):
        b = ATTENTION_PREFILL_SEQ_BUCKETS
        assert list(b) == sorted(b)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


class TestAttentionPrefillShapeValidation:
    def test_validate_shapes_accepts_qwen_3_5_0_8b(self):
        ok, reason = validate_attention_prefill_shapes(
            seq_len=512, embed_dim=1024, num_heads=16, num_kv_heads=2,
            head_dim=64,
        )
        assert ok, reason

    def test_validate_shapes_accepts_qwen_4b_shape(self):
        # Qwen 4B: embed=2560, H=32, KV=8, d=64 -> H*d=2048, KV*d=512
        ok, reason = validate_attention_prefill_shapes(
            seq_len=512, embed_dim=2560, num_heads=32, num_kv_heads=8,
            head_dim=64,
        )
        assert ok, reason

    def test_validate_shapes_rejects_head_dim_not_64(self):
        ok, reason = validate_attention_prefill_shapes(
            seq_len=512, embed_dim=1024, num_heads=16, num_kv_heads=2,
            head_dim=128,
        )
        assert not ok
        assert "head_dim" in reason

    def test_validate_shapes_rejects_non_divisible_gqa_ratio(self):
        ok, reason = validate_attention_prefill_shapes(
            seq_len=512, embed_dim=1024, num_heads=15, num_kv_heads=2,
            head_dim=64,
        )
        assert not ok
        assert "num_heads" in reason

    def test_validate_shapes_rejects_zero_seq_len(self):
        ok, reason = validate_attention_prefill_shapes(
            seq_len=0, embed_dim=1024, num_heads=16, num_kv_heads=2,
            head_dim=64,
        )
        assert not ok
        assert "seq_len" in reason

    def test_validate_shapes_rejects_embed_not_divisible_by_8(self):
        ok, reason = validate_attention_prefill_shapes(
            seq_len=512, embed_dim=100, num_heads=16, num_kv_heads=2,
            head_dim=64,
        )
        assert not ok
        assert "embed_dim" in reason

    def test_validate_shapes_rejects_kv_dim_not_multiple_of_64(self):
        # 3 * 64 = 192 -> not multiple of 64? 192 % 64 == 0 actually. Use KV=1
        # to get KV*d = 64 (passes) vs something that fails. Construct a case:
        # if we allowed head_dim != 64 we could fail this, but that check fires
        # first. Smallest construction that reaches the kv_dim check: head_dim=64
        # is forced, so KV*64 % 64 == 0 always. Skip — the check is still a
        # defensive guard kept for the day head_dim widens.
        pytest.skip("kv_dim % 64 check is unreachable while head_dim is fixed at 64")


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


class TestAttentionPrefillCacheLayout:
    def test_kernel_name_constants(self):
        assert ATTENTION_PREFILL_KERNELS == (
            "rms_norm", "gemm_q", "gemm_kv", "rope_q", "rope_k",
            "perm_q", "perm_kv", "mha", "perm_o", "gemm_o", "add",
        )
        assert len(ATTENTION_PREFILL_KERNELS) == 11

    def test_cache_miss_when_dir_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        assert get_cached_chained_dir(
            "nonexistent_key", ATTENTION_PREFILL_KERNELS, prefix="attn_prefill"
        ) is None

    def test_cache_miss_when_xclbin_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        for name in ATTENTION_PREFILL_KERNELS:
            (d / f"attn_prefill_{name}.insts").write_bytes(b"x")
        assert get_cached_chained_dir(
            key, ATTENTION_PREFILL_KERNELS, prefix="attn_prefill"
        ) is None

    def test_cache_miss_when_one_insts_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        # Drop the last insts file
        for name in ATTENTION_PREFILL_KERNELS[:-1]:
            (d / f"attn_prefill_{name}.insts").write_bytes(b"x")
        assert get_cached_chained_dir(
            key, ATTENTION_PREFILL_KERNELS, prefix="attn_prefill"
        ) is None

    def test_cache_hit_when_complete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        key = "abc123"
        d = tmp_path / key
        d.mkdir()
        (d / "combined.xclbin").write_bytes(b"x")
        for name in ATTENTION_PREFILL_KERNELS:
            (d / f"attn_prefill_{name}.insts").write_bytes(b"x")
        result = get_cached_chained_dir(
            key, ATTENTION_PREFILL_KERNELS, prefix="attn_prefill"
        )
        assert result == d


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestAttentionPrefillCLI:
    def test_subcommand_registered(self):
        import compile as compile_mod

        script = Path(compile_mod.__file__)
        result = subprocess.run(
            [sys.executable, str(script), "attention-prefill", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "--seq-len" in result.stdout
        assert "--embed-dim" in result.stdout
        assert "--num-heads" in result.stdout
        assert "--num-kv-heads" in result.stdout
        assert "--head-dim" in result.stdout
        assert "--dtype" in result.stdout


# ---------------------------------------------------------------------------
# End-to-end compilation (requires NPU + ironenv; ~60s)
# ---------------------------------------------------------------------------


def _iron_available() -> bool:
    try:
        import aie.utils  # noqa: F401
        import iron.operators.attention_block_prefill.op  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _iron_available(),
                    reason="IRON / aie runtime not importable; needs ironenv + NPU")
class TestAttentionPrefillCompile:
    def test_compile_attention_prefill_produces_all_artifacts(self, tmp_path,
                                                              monkeypatch):
        """Full compile of a Qwen3.5-0.8B attention block. Takes ~60s; guarded
        by the IRON-availability skip so CI without the NPU env passes quickly.
        """
        monkeypatch.setenv("GGML_XDNA_CACHE_DIR", str(tmp_path))
        from compile import compile_attention_prefill_cached

        out_dir = compile_attention_prefill_cached(
            seq_len=512, embed_dim=1024,
            num_heads=16, num_kv_heads=2, head_dim=64,
        )
        out_dir = Path(out_dir)

        assert (out_dir / "combined.xclbin").exists(), (
            f"combined.xclbin missing under {out_dir}"
        )
        for name in ATTENTION_PREFILL_KERNELS:
            insts = out_dir / f"attn_prefill_{name}.insts"
            assert insts.exists(), f"{insts.name} missing under {out_dir}"
