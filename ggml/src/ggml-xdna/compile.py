#!/usr/bin/env python3
"""
ggml-xdna compilation bridge.

Called by the C++ backend to compile IRON operators into cached xclbin binaries.
Not a user-facing tool — internal to the ggml-xdna backend.

Usage (from C++ backend via subprocess):
    python compile.py gemm --M 2048 --K 2048 --N 256 --dtype bf16 --out /path/to/cache/hash.xclbin
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import aie.utils as aie_utils
import numpy as np
import os

if os.name == 'nt':
    # Add XRT driver directory to DLL search path for pyxrt
    driver_dir = r'C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810'
    if os.path.exists(driver_dir):
        os.add_dll_directory(driver_dir)

    # Add XRT SDK python folder to DLL path too
    xrt_sdk_dlls = r'C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\bin\condautils'
    if os.path.exists(xrt_sdk_dlls):
        os.add_dll_directory(xrt_sdk_dlls)

    # Patch Peano path for Windows
    import aie.utils.config as aie_config
    peano_dir = r'C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano'
    if os.path.exists(peano_dir):
        aie_config.peano_install_dir = lambda: peano_dir


def get_device_cols(requested_cols: int) -> int:
    """Safely get the number of columns on the current device, with a fallback."""
    if os.name == 'nt':
        return 8 # Strix Point / Ryzen AI 300
    try:
        import aie.utils as aie_utils
        dev = aie_utils.get_current_device()
        if dev is not None:
            print(f"DEBUG: detected hardware cols={dev.cols}")
            return dev.cols
    except Exception as e:
        print(f"DEBUG: hardware detection failed: {e}")
    return requested_cols


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------

def gemm_cache_key(M: int, K: int, N: int, dtype_in: str, dtype_out: str,
                   num_aie_columns: int, b_col_maj: bool = False) -> str:
    """Generate a deterministic cache key for a GEMM configuration."""
    key_data = {
        "op": "gemm",
        "M": M,
        "K": K,
        "N": N,
        "dtype_in": dtype_in,
        "dtype_out": dtype_out,
        "num_aie_columns": num_aie_columns,
        "b_col_maj": b_col_maj,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def gemv_cache_key(N: int, K: int, dtype_in: str, dtype_out: str,
                   num_aie_columns: int) -> str:
    """Generate a deterministic cache key for a GEMV configuration.

    N is the matrix-row dimension (== IRON GEMV's M); K is the reduction dim.
    Tile sizes are derived from (N, K, cols) so they're not part of the key.
    """
    key_data = {
        "op": "gemv",
        "N": N,
        "K": K,
        "dtype_in": dtype_in,
        "dtype_out": dtype_out,
        "num_aie_columns": num_aie_columns,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def swiglu_decode_cache_key(embedding_dim: int, hidden_dim: int,
                            dtype: str, num_aie_columns: int) -> str:
    """Cache key for a SwiGLU decode (M=1) configuration.

    Tile sizes for the inner GEMV/SiLU/EltwiseMul ops are derived from
    (embedding_dim, hidden_dim, num_aie_columns) so they aren't part of the key.
    """
    key_data = {
        "op": "swiglu_decode",
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dtype": dtype,
        "num_aie_columns": num_aie_columns,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def swiglu_prefill_cache_key(seq_len: int, embedding_dim: int, hidden_dim: int,
                             dtype: str, num_aie_columns: int,
                             tile_m: int | None = None,
                             tile_n: int | None = None) -> str:
    """Cache key for a SwiGLU prefill configuration.

    tile_m and tile_n are distinguishing inputs because they change the compiled
    xclbin (different MAC tiling). tile_k is left at IRON's default (64) and
    omitted from the key.
    """
    key_data = {
        "op": "swiglu_prefill",
        "seq_len": seq_len,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "dtype": dtype,
        "num_aie_columns": num_aie_columns,
        "tile_m": tile_m,
    }
    if tile_n is not None and tile_n != 64:
        key_data["tile_n"] = tile_n
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def swiglu_decode_int8_cache_key(embedding_dim: int, hidden_dim: int,
                                 num_aie_columns: int,
                                 group_size: int = 32) -> str:
    """Cache key for a W8A16 INT8 SwiGLU decode (M=1) configuration.

    Disjoint from bf16 swiglu / gemm / gemv keys by virtue of the "op" field.
    group_size participates because it is baked into the inner gemv_int8
    kernel at compile time via -DGROUP_SIZE. dtype is implied (int8 weights +
    bf16 activations) and thus omitted.
    """
    key_data = {
        "op": "swiglu_decode_int8",
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_aie_columns": num_aie_columns,
        "group_size": group_size,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def swiglu_fused_int8_cache_key(embedding_dim: int, hidden_dim: int,
                                num_aie_columns: int,
                                group_size: int = 32) -> str:
    """Cache key for the fused gate+up+silu+mul INT8 operator + standalone down GEMV.

    Distinct from swiglu_decode_int8 (chained) by the "op" field.
    """
    key_data = {
        "op": "swiglu_fused_int8",
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_aie_columns": num_aie_columns,
        "group_size": group_size,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def rms_norm_cache_key(size: int, dtype: str, num_aie_columns: int,
                       num_channels: int, tile_size: int,
                       weighted: bool = False) -> str:
    """Cache key for a standalone RMSNorm configuration.

    size is the norm-axis length; tile_size, num_channels, and num_aie_columns
    together determine the AIE dataflow layout and are baked into the compiled
    xclbin — all participate. ``weighted=False`` is currently the only wired
    path but the field participates so a future weighted variant won't collide.
    """
    key_data = {
        "op": "rms_norm",
        "size": size,
        "dtype": dtype,
        "num_aie_columns": num_aie_columns,
        "num_channels": num_channels,
        "tile_size": tile_size,
        "weighted": weighted,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def flowkv_decode_cache_key(num_heads: int, num_kv_heads: int, head_dim: int,
                            seq_len: int, chunk_size: int,
                            num_cols: int) -> str:
    """Cache key for a FlowKV decode attention configuration.

    Single xclbin with 2-tile pipeline (score + value) per KV head group.
    Disjoint from other ops by the "op" field.
    """
    key_data = {
        "op": "flowkv_decode",
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "num_cols": num_cols,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


FLOWKV_DECODE_KERNELS = ("flowkv_decode",)


def qkv_cache_key(embedding_dim: int, q_dim: int, k_dim: int, v_dim: int,
                  num_aie_columns: int) -> str:
    """Cache key for a chained Q/K/V projection configuration.

    Three bf16 GEMVs with shared input (K=embedding_dim) and distinct output
    dims fused into one xclbin. Disjoint from gemm/gemv/swiglu keys by the
    ``"op"`` field. Tile sizes are derived from shapes by IRON and so are
    not part of the key.
    """
    key_data = {
        "op": "qkv",
        "embedding_dim": embedding_dim,
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "num_aie_columns": num_aie_columns,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def swiglu_prefill_int8_cache_key(seq_len: int, embedding_dim: int, hidden_dim: int,
                                   num_aie_columns: int,
                                   tile_m: int | None = None,
                                   tile_n: int | None = None) -> str:
    """Cache key for a W8A8 INT8 SwiGLU prefill configuration.

    tile_m and tile_n are distinguishing inputs because they change the compiled
    xclbin (different MAC tiling). Disjoint from bf16 swiglu_prefill by the
    "op" field.
    """
    key_data = {
        "op": "swiglu_prefill_int8",
        "seq_len": seq_len,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_aie_columns": num_aie_columns,
        "tile_m": tile_m,
    }
    if tile_n is not None and tile_n != 64:
        key_data["tile_n"] = tile_n
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def select_swiglu_prefill_tile_m(seq_len: int) -> int:
    """Pick the largest valid tile_m for a given prefill seq_len.

    IRON GEMM requires seq_len % (tile_m * 4) == 0 with tile_m in {64,32,16,8}.
    Returns 0 if no candidate works (caller should fall back / reject).
    """
    for tm in (64, 32, 16, 8):
        if seq_len % (tm * 4) == 0:
            return tm
    return 0


def select_swiglu_prefill_tile_n(embedding_dim: int, hidden_dim: int,
                                 num_aie_columns: int) -> int:
    """Pick the largest valid tile_n for SwiGLU prefill.

    Both inner GEMMs must tile cleanly:
      - GEMM1 (gate/up): N = hidden_dim  → hidden_dim  % (tile_n * cols) == 0
      - GEMM2 (down):    N = embedding_dim → embedding_dim % (tile_n * cols) == 0
    Returns 0 if no candidate works (caller should fall back / reject).
    """
    for tn in (64, 32, 16, 8):
        if (hidden_dim % (tn * num_aie_columns) == 0 and
                embedding_dim % (tn * num_aie_columns) == 0):
            return tn
    return 0


# Sub-kernel attribute prefixes inside the chained SwiGLU xclbin. These must
# match the names passed to chain_swiglu_artifacts in iron/operators/swiglu_*/op.py.
# The C++ backend looks up insts files by these names.
# IRON-windows SwiGLU decode: 2 sub-kernels (fused gate+up+silu+mul, down GEMV)
# instead of the old 4-kernel chain. Names match _stage_swiglu_decode_new().
SWIGLU_DECODE_KERNELS = ("fused", "gemv_2")
# IRON-windows SwiGLU prefill: 3 sub-kernels (gemm_1, silu_mul, gemm_2).
# Names match _stage_swiglu_prefill_new().
SWIGLU_PREFILL_KERNELS = ("gemm_1", "silu_mul", "gemm_2")
# W8A16 (INT8 weights + bf16 activations) SwiGLU decode. Sub-op prefixes come
# from iron/operators/swiglu_decode_int8/op.py chain_swiglu_artifacts() call.
# SiLU / eltwise_mul share the same compiled binaries as the bf16 chain
# (they operate on bf16 intermediates), but are keyed under their int8-namespace
# insts files inside this bundle — the C++ backend loads them via distinct tags.
SWIGLU_DECODE_INT8_KERNELS = ("gemv_int8_1", "silu", "eltwise_mul", "gemv_int8_2")
# Fused gate+up+silu+mul + standalone down GEMV, chained into one xclbin.
# Names must match chain_swiglu_artifacts() call in chained.py.
SWIGLU_FUSED_CHAINED_KERNELS = ("fused", "down_gemv_int8")
# W8A8 INT8 SwiGLU prefill. Same 4-stage structure as bf16 prefill but
# GEMMs use dtype_in="i8", dtype_out="bf16s". Kernel names match the
# chain_swiglu_artifacts() call in swiglu_prefill_int8/op.py.
SWIGLU_PREFILL_INT8_KERNELS = ("gemm_1", "silu", "eltwise_mul", "gemm_2")
# Chained Q/K/V projection: IRON-windows uses a single fused GEMV kernel
# (Wq/Wk/Wv concatenated, one xclbin, one dispatch). Names must match
# _stage_qkv_fused_new() output filenames.
QKV_PROJ_KERNELS = ("main",)
# Standalone (non-chained) RMSNorm. Single kernel, staged with prefix "rms_norm"
# so it lives alongside chained bundles under a per-key cache dir.
RMS_NORM_KERNELS = ("main",)
# Chained AttentionBlockPrefill: 11 sub-kernels in one xclbin. Names must match
# the chain_artifacts() call in iron/operators/attention_block_prefill/op.py
# (prefix="attn_prefill").
ATTENTION_PREFILL_KERNELS = (
    "rms_norm",
    "gemm_q",
    "gemm_kv",
    "rope_q",
    "rope_k",
    "perm_q",
    "perm_kv",
    "mha",
    "perm_o",
    "gemm_o",
    "add",
)

# Seq-len buckets for AttentionBlockPrefill. The prefill ubatch size is set at
# llama.cpp run-start, so in the steady state one session maps to a single
# bucket — avoids xclbin sprawl while keeping wasted-pad compute bounded.
# 256 minimum: IRON bf16 GEMM with default tile_m=64 requires M % 256 == 0.
# Smaller seq_lens fall to CPU. Dropping below 256 requires passing smaller
# tile_m through the composite (future optimization for partial-batch prefill).
ATTENTION_PREFILL_SEQ_BUCKETS = (256, 512, 768, 1024, 1536, 2048, 4096)


# Chained TransformerBlockPrefill: 17 sub-kernels in one xclbin (attention +
# SwiGLU FFN + two residuals). Names must match the chain_artifacts() call in
# iron/operators/transformer_block_prefill/op.py (prefix="tblock_prefill").
TRANSFORMER_BLOCK_PREFILL_KERNELS = (
    "rms_norm_attn",
    "gemm_q",
    "gemm_kv",
    "rope_q",
    "rope_k",
    "perm_q",
    "perm_kv",
    "mha",
    "perm_o",
    "gemm_o",
    "add_attn",
    "rms_norm_ffn",
    "gemm_gate_up",
    "silu",
    "eltwise_mul",
    "gemm_down",
    "add_ffn",
)

# Seq-len buckets shared with AttentionBlockPrefill: prefill ubatch is a
# run-start constant, so one session maps to one bucket.
TRANSFORMER_BLOCK_PREFILL_SEQ_BUCKETS = ATTENTION_PREFILL_SEQ_BUCKETS


# ---------------------------------------------------------------------------
# Tile size selection
# ---------------------------------------------------------------------------

def select_gemm_tiles(M: int, K: int, N: int, num_aie_columns: int,
                      dtype_in: str = "bf16") -> tuple[int, int, int]:
    """Select tile sizes that satisfy IRON GEMM constraints.

    Constraints:
        M % (tile_m * 4) == 0   (4 rows of AIE tiles)
        K % tile_k == 0
        N % (tile_n * num_aie_columns) == 0

    Returns:
        (tile_m, tile_k, tile_n)
    """
    # Default tile sizes that work well for most shapes
    candidates_m = [64, 32, 16, 8]
    candidates_k = [64, 32, 16, 8]
    candidates_n = [64, 32, 16, 8]

    if dtype_in == "i8":
        min_m, min_k, min_n = 16, 8, 16
    else:
        # bf16 4x8x8 MAC in IRON's aie2p mm.cc requires both tile_m % (2*r) == 0
        # with r=4 (min tile_m=8) AND tile_n % (2*t) == 0 with t=8 (min tile_n=16).
        # aie2 4x8x4 bf16 kernel has the equivalent tile_n % (4*t) == 0 with t=4
        # (also min tile_n=16), so 16 is the floor on both device generations.
        # Attempting tile_n=8 for bf16 triggers a kernel static_assert.
        min_m, min_k, min_n = 8, 8, 16

    tile_m = None
    for tm in candidates_m:
        if tm >= min_m and M % (tm * 4) == 0:
            tile_m = tm
            break
    if tile_m is None:
        raise ValueError(f"Cannot find valid tile_m for M={M}")

    tile_k = None
    for tk in candidates_k:
        if tk >= min_k and K % tk == 0:
            tile_k = tk
            break
    if tile_k is None:
        raise ValueError(f"Cannot find valid tile_k for K={K}")

    tile_n = None
    for tn in candidates_n:
        if tn >= min_n and N % (tn * num_aie_columns) == 0:
            tile_n = tn
            break
    if tile_n is None:
        raise ValueError(f"Cannot find valid tile_n for N={N}, num_aie_columns={num_aie_columns}")

    return tile_m, tile_k, tile_n


def select_gemv_tiles(N: int, K: int, num_aie_columns: int,
                      kernel_vector_size: int = 64) -> tuple[int, int]:
    """Select (tile_size_input, tile_size_output) for IRON GEMV.

    IRON GEMV constraints (iron/operators/gemv/design.py):
        N % cols == 0
        tile_out % tile_in == 0, tile_out >= tile_in
        tile_out <= N//cols
        (N//cols) % tile_out == 0
        (N//cols) % tile_in == 0
        K % kernel_vector_size == 0

    Note: IRON calls the matrix-row dim "M" — here we use N (matches ggml/GEMM usage).
    """
    if K % kernel_vector_size != 0:
        raise ValueError(f"K={K} must be a multiple of kernel_vector_size={kernel_vector_size}")
    if N % num_aie_columns != 0:
        raise ValueError(f"N={N} must be divisible by num_aie_columns={num_aie_columns}")

    per_col = N // num_aie_columns

    # tile_out: largest divisor of per_col up to a reasonable cap (matches test configs)
    tso_candidates = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    tile_out = None
    for tso in tso_candidates:
        if tso <= per_col and per_col % tso == 0:
            tile_out = tso
            break
    if tile_out is None:
        raise ValueError(f"Cannot find tile_size_output for N={N}, cols={num_aie_columns}")

    # tile_in: L1 double-buffered matrix tile is 2 * tile_in * K * sizeof(bf16) bytes.
    # AIE2p (Strix Point / NPU2) has 64KB L1 data memory per tile.
    # Must also reserve space for B (input vector, depth=1) and C (output, depth=2).
    l1_budget_bytes = 64 * 1024  # AIE2p: 64KB per tile
    bf16 = 2
    # B (K elements, depth=1) + C (tile_out elements, depth=2) must fit.
    bc_bytes = (K * bf16) + (2 * tile_out * bf16)
    if bc_bytes >= l1_budget_bytes:
        raise ValueError(
            f"L1 overflow: B+C alone need {bc_bytes} bytes but budget is "
            f"{l1_budget_bytes}. K={K}, tile_out={tile_out}, cols={num_aie_columns}"
        )
    l1_available_for_a = l1_budget_bytes - bc_bytes
    max_tsi_by_l1 = l1_available_for_a // (2 * K * bf16)  # 2 = double-buffer
    tsi_candidates = [8, 4, 2, 1]
    tile_in = None
    for tsi in tsi_candidates:
        if (tsi <= tile_out and tsi <= max_tsi_by_l1
                and tile_out % tsi == 0 and per_col % tsi == 0):
            tile_in = tsi
            break
    if tile_in is None:
        raise ValueError(
            f"Cannot find tile_size_input for N={N}, K={K}, cols={num_aie_columns}. "
            f"L1 available for A: {l1_available_for_a} bytes, "
            f"B+C: {bc_bytes} bytes, budget: {l1_budget_bytes} bytes."
        )

    return tile_in, tile_out


def validate_swiglu_decode_shapes(embedding_dim: int, hidden_dim: int,
                                  num_aie_columns: int) -> None:
    """Verify a (embedding_dim, hidden_dim, cols) tuple is dispatchable to SwiGLUDecode.

    Mirrors the divisibility constraints encoded in iron/operators/swiglu_decode/op.py
    (silu/eltwise_mul tile_size = hidden_dim // (n_cols * 2) and hidden_dim // n_cols).
    Raises ValueError on any violation so the C++ backend can fall back cleanly.
    """
    if num_aie_columns not in (4, 8):
        raise ValueError(
            f"num_aie_columns must be 4 (NPU1) or 8 (NPU2), got {num_aie_columns}"
        )
    if hidden_dim % (num_aie_columns * 2) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by 2*num_aie_columns="
            f"{num_aie_columns * 2} (SiLU per-column tile)"
        )
    if embedding_dim % num_aie_columns != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} must be divisible by "
            f"num_aie_columns={num_aie_columns}"
        )
    # Static kernel buffers (left_buf/right_buf) are compiled with -DM_OUTPUT_MAX.
    # m_output = hidden_dim / fused_cols (fused kernel uses 4 cols on NPU2).
    # Safety check: hidden_dim must fit in a reasonable buffer. The operator
    # passes -DM_OUTPUT_MAX at compile time, but we validate here to fail early.
    fused_cols = 4  # hardcoded in AIESwiGLUDecode for fused kernel
    m_output = hidden_dim // fused_cols
    max_supported = 4096  # default M_OUTPUT_MAX in kernel source
    if m_output > max_supported:
        raise ValueError(
            f"hidden_dim // {fused_cols} = {m_output} exceeds static kernel "
            f"buffer capacity ({max_supported}). Increase M_OUTPUT_MAX or "
            f"reduce hidden_dim."
        )
    # The two GEMV stages must themselves be dispatchable. Reuse the GEMV tile
    # selector — it raises ValueError on K%64 / per-col / L1 budget violations.
    select_gemv_tiles(N=hidden_dim, K=embedding_dim, num_aie_columns=num_aie_columns)
    select_gemv_tiles(N=embedding_dim, K=hidden_dim, num_aie_columns=num_aie_columns)


def validate_swiglu_decode_int8_shapes(embedding_dim: int, hidden_dim: int,
                                       num_aie_columns: int,
                                       group_size: int = 32) -> None:
    """Verify an (embedding_dim, hidden_dim, cols, group_size) tuple dispatches
    to SwiGLUDecodeInt8.

    Adds the INT8-specific group_size divisibility constraints on top of the
    bf16 SwiGLUDecode checks. The two inner gemv_int8 stages must also satisfy
    the standard bf16 GEMV L1-budget + per-col tiling constraints (K is the
    dim reduced in each stage; activations are still bf16).
    """
    if num_aie_columns not in (4, 8):
        raise ValueError(
            f"num_aie_columns must be 4 (NPU1) or 8 (NPU2), got {num_aie_columns}"
        )
    if group_size % 32 != 0:
        raise ValueError(
            f"group_size={group_size} must be a multiple of 32 (Q8_0 block size)"
        )
    if embedding_dim % group_size != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} must be a multiple of "
            f"group_size={group_size} (gemv_int8_1 K constraint)"
        )
    if hidden_dim % group_size != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be a multiple of "
            f"group_size={group_size} (gemv_int8_2 K constraint)"
        )
    if hidden_dim % (num_aie_columns * 2) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by 2*num_aie_columns="
            f"{num_aie_columns * 2} (SiLU per-column tile)"
        )
    if embedding_dim % num_aie_columns != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} must be divisible by "
            f"num_aie_columns={num_aie_columns}"
        )
    # Reuse the bf16 GEMV tile selector for the K%64 + per-col checks (the
    # gemv_int8 kernel uses the same kernel_vector_size=64 under the hood).
    select_gemv_tiles(N=hidden_dim, K=embedding_dim, num_aie_columns=num_aie_columns)
    select_gemv_tiles(N=embedding_dim, K=hidden_dim, num_aie_columns=num_aie_columns)


def validate_swiglu_prefill_shapes(seq_len: int, embedding_dim: int, hidden_dim: int,
                                   num_aie_columns: int,
                                   tile_m: int | None = None,
                                   tile_n: int | None = None) -> None:
    """Verify a SwiGLU prefill shape is dispatchable.

    If tile_m is None, IRON defaults to 64 → min_M = 256. If overridden, the
    constraint loosens to seq_len % (tile_m * 4) == 0. tile_k remains at
    IRON's default of 64. tile_n defaults to auto-selected via
    select_swiglu_prefill_tile_n when not provided.
    """
    if num_aie_columns not in (4, 8):
        raise ValueError(
            f"num_aie_columns must be 4 (NPU1) or 8 (NPU2), got {num_aie_columns}"
        )
    if hidden_dim % (num_aie_columns * 2) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by 2*num_aie_columns="
            f"{num_aie_columns * 2} (SiLU per-column tile)"
        )
    effective_tile_m = tile_m if tile_m is not None else 64
    if effective_tile_m not in (64, 32, 16, 8):
        raise ValueError(
            f"tile_m={effective_tile_m} must be one of 64, 32, 16, 8"
        )
    min_M = effective_tile_m * 4
    if seq_len % min_M != 0:
        raise ValueError(
            f"seq_len={seq_len} must be a multiple of {min_M} (tile_m={effective_tile_m})"
        )
    if embedding_dim % 64 != 0:
        raise ValueError(f"embedding_dim={embedding_dim} must be a multiple of 64")
    if hidden_dim % 64 != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be a multiple of 64")
    effective_tile_n = tile_n if tile_n is not None else select_swiglu_prefill_tile_n(
        embedding_dim, hidden_dim, num_aie_columns
    )
    if effective_tile_n == 0:
        raise ValueError(
            f"Cannot find valid tile_n for embedding_dim={embedding_dim} "
            f"hidden_dim={hidden_dim} num_aie_columns={num_aie_columns}"
        )
    if effective_tile_n not in (64, 32, 16, 8):
        raise ValueError(
            f"tile_n={effective_tile_n} must be one of 64, 32, 16, 8"
        )
    if hidden_dim % (effective_tile_n * num_aie_columns) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be a multiple of tile_n*num_aie_columns="
            f"{effective_tile_n * num_aie_columns} (gemm_1 tile_n*cols)"
        )
    if embedding_dim % (effective_tile_n * num_aie_columns) != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} must be a multiple of tile_n*num_aie_columns="
            f"{effective_tile_n * num_aie_columns} (gemm_2 tile_n*cols)"
        )


def validate_swiglu_prefill_int8_shapes(seq_len: int, embedding_dim: int,
                                         hidden_dim: int, num_aie_columns: int,
                                         tile_m: int | None = None,
                                         tile_n: int | None = None) -> None:
    """Verify a W8A8 INT8 SwiGLU prefill shape is dispatchable.

    INT8 GEMM requires tile_m >= 16, so min seq_len = 64 (vs 32 for bf16).
    tile_n is auto-selected via select_swiglu_prefill_tile_n when not provided.
    """
    if num_aie_columns not in (4, 8):
        raise ValueError(
            f"num_aie_columns must be 4 (NPU1) or 8 (NPU2), got {num_aie_columns}"
        )
    if hidden_dim % (num_aie_columns * 2) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by 2*num_aie_columns="
            f"{num_aie_columns * 2} (SiLU per-column tile)"
        )
    effective_tile_m = tile_m if tile_m is not None else 64
    # INT8 GEMM requires tile_m >= 16 (vs 8 for bf16)
    if effective_tile_m not in (64, 32, 16):
        raise ValueError(
            f"tile_m={effective_tile_m} must be one of 64, 32, 16 for INT8 GEMM"
        )
    min_M = effective_tile_m * 4
    if seq_len % min_M != 0:
        raise ValueError(
            f"seq_len={seq_len} must be a multiple of {min_M} (tile_m={effective_tile_m})"
        )
    if embedding_dim % 64 != 0:
        raise ValueError(f"embedding_dim={embedding_dim} must be a multiple of 64")
    if hidden_dim % 64 != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be a multiple of 64")
    effective_tile_n = tile_n if tile_n is not None else select_swiglu_prefill_tile_n(
        embedding_dim, hidden_dim, num_aie_columns
    )
    if effective_tile_n == 0:
        raise ValueError(
            f"Cannot find valid tile_n for embedding_dim={embedding_dim} "
            f"hidden_dim={hidden_dim} num_aie_columns={num_aie_columns}"
        )
    # INT8 GEMM min_tile_n is 16 (vs 8 for bf16)
    if effective_tile_n not in (64, 32, 16):
        raise ValueError(
            f"tile_n={effective_tile_n} must be one of 64, 32, 16 for INT8 GEMM"
        )
    if hidden_dim % (effective_tile_n * num_aie_columns) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be a multiple of tile_n*num_aie_columns="
            f"{effective_tile_n * num_aie_columns}"
        )
    if embedding_dim % (effective_tile_n * num_aie_columns) != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} must be a multiple of tile_n*num_aie_columns="
            f"{effective_tile_n * num_aie_columns}"
        )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_gemm(M: int, K: int, N: int, dtype_in: str, dtype_out: str,
                 num_aie_columns: int, output_path: str,
                 b_col_maj: bool = False) -> str:
    """Compile an IRON GEMM operator and save the xclbin.

    Args:
        M, K, N: Matrix dimensions.
        dtype_in: Input dtype ("bf16" or "i8").
        dtype_out: Output dtype ("bf16", "i8", "i16", "i32").
        num_aie_columns: Number of AIE columns to use.
        output_path: Where to write the xclbin.
        b_col_maj: Whether B matrix is column-major.

    Returns:
        Path to the compiled xclbin.
    """
    from iron.operators.gemm.op import GEMM

    tile_m, tile_k, tile_n = select_gemm_tiles(M, K, N, num_aie_columns, dtype_in)

    gemm_kwargs = {
        "M": M,
        "K": K,
        "N": N,
        "tile_m": tile_m,
        "tile_k": tile_k,
        "tile_n": tile_n,
        "num_aie_columns": num_aie_columns,
        "b_col_maj": b_col_maj,
        "dtype_in": dtype_in,
        "dtype_out": dtype_out,
    }

    # Don't set bf16-specific flags for int8
    if dtype_in != "i8":
        # False enables tile_m=4 (4x8x8 MAC) on XDNA2 for more shape coverage
        gemm_kwargs["emulate_bf16_mmul_with_bfp16"] = False
        gemm_kwargs["prio_accuracy"] = True

    op = GEMM(**gemm_kwargs)
    op.compile()

    # The compiled artifacts live in context.build_dir
    build_dir = op.context.build_dir
    compiled_xclbin = build_dir / op.xclbin_artifact.filename
    compiled_insts = build_dir / op.insts_artifact.filename

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(str(compiled_xclbin), output_path)

    # Also copy the insts file alongside the xclbin (same name, .insts extension)
    insts_output = output_path.replace(".xclbin", ".insts")
    shutil.copy2(str(compiled_insts), insts_output)

    return output_path


def compile_gemv(N: int, K: int, dtype_in: str, dtype_out: str,
                 num_aie_columns: int, output_path: str) -> str:
    """Compile an IRON GEMV operator and save the xclbin.

    Args:
        N: Matrix-row dimension (output length). Maps to IRON GEMV's M.
        K: Reduction dimension (vector length).
        dtype_in: Input dtype (currently only "bf16" supported by iron GEMV op).
        dtype_out: Output dtype (currently only "bf16").
        num_aie_columns: Number of AIE columns to use.
        output_path: Where to write the xclbin.
    """
    from iron.operators.gemv.op import GEMV

    if dtype_in != "bf16" or dtype_out != "bf16":
        raise ValueError(f"GEMV currently supports only bf16 in/out, got {dtype_in}/{dtype_out}")

    tile_in, tile_out = select_gemv_tiles(N, K, num_aie_columns)

    op = GEMV(
        M=N,                        # IRON calls it M; it's the matrix-row dim
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size_input=tile_in,
        tile_size_output=tile_out,
    )
    op.compile()

    build_dir = op.context.build_dir
    compiled_xclbin = build_dir / op.xclbin_artifact.filename
    compiled_insts = build_dir / op.insts_artifact.filename

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.copy2(str(compiled_xclbin), output_path)
    insts_output = output_path.replace(".xclbin", ".insts")
    shutil.copy2(str(compiled_insts), insts_output)

    return output_path


def compile_swiglu_decode(embedding_dim: int, hidden_dim: int, dtype: str,
                          num_aie_columns: int, output_dir: str) -> str:
    """Compile an IRON SwiGLU decode operator and stage its artifacts into output_dir.

    SwiGLU is a chained xclbin: one combined xclbin with 4 kernels
    (gemv_1, silu, eltwise_mul, gemv_2), plus 4 separate insts files. We stage
    everything under a single directory so the C++ side has a deterministic
    layout to consume.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/swiglu_gemv_1.insts
        <output_dir>/swiglu_silu.insts
        <output_dir>/swiglu_eltwise_mul.insts
        <output_dir>/swiglu_gemv_2.insts
    """
    if dtype != "bf16":
        raise ValueError(f"SwiGLU currently supports only bf16, got {dtype}")
    validate_swiglu_decode_shapes(embedding_dim, hidden_dim, num_aie_columns)

    # SwiGLUDecode reads num_aie_columns from the live device. The C++ caller
    # always passes the device's actual cols, so we cross-check here to fail
    # loudly if a stale cache key is requested against a different device.
    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.swiglu_decode.op import AIESwiGLUDecode

    op = AIESwiGLUDecode(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_aie_columns=num_aie_columns,
    )
    op.compile()

    _stage_swiglu_decode_new(op, output_dir)
    return output_dir


def compile_swiglu_decode_int8(embedding_dim: int, hidden_dim: int,
                               num_aie_columns: int, output_dir: str,
                               group_size: int = 32) -> str:
    """Compile an IRON SwiGLUDecodeInt8 operator and stage its artifacts.

    Mirrors compile_swiglu_decode layout: one combined xclbin with 4 sub-ops
    (gemv_int8_1, silu, eltwise_mul, gemv_int8_2) plus 4 insts files, staged
    under ``output_dir``. No weights are packed here — weights are runtime data
    that the C++ backend loads from the GGUF and repacks per-tile at dispatch.
    """
    validate_swiglu_decode_int8_shapes(embedding_dim, hidden_dim,
                                       num_aie_columns, group_size=group_size)

    # SwiGLUDecodeInt8 reads num_aie_columns from the live device (mirrors bf16).
    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    raise NotImplementedError(
        "SwiGLUDecodeInt8 is not available in IRON-windows. "
        "No INT8 SwiGLU decode operator exists in this fork."
    )
    return output_dir


def compile_swiglu_fused_int8(embedding_dim: int, hidden_dim: int,
                              num_aie_columns: int, output_dir: str,
                              group_size: int = 32) -> str:
    """Compile the chained fused+down INT8 SwiGLU composite.

    Produces one combined xclbin with 2 kernel entries (fused gate+up+silu+mul
    + standalone down GEMV) plus 2 insts files, staged under output_dir.
    Both kernels share one hw_context and can be batched via xrt::runlist.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/swiglu_fused.insts
        <output_dir>/swiglu_down_gemv_int8.insts
    """
    validate_swiglu_decode_int8_shapes(embedding_dim, hidden_dim,
                                       num_aie_columns, group_size=group_size)

    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    raise NotImplementedError(
        "SwiGLUDecodeInt8FusedChained is not available in IRON-windows. "
        "No fused INT8 SwiGLU decode operator exists in this fork."
    )
    return output_dir


def compile_swiglu_prefill(seq_len: int, embedding_dim: int, hidden_dim: int,
                           dtype: str, num_aie_columns: int, output_dir: str,
                           tile_m: int | None = None,
                           tile_n: int | None = None) -> str:
    """Compile an IRON SwiGLU prefill operator and stage its artifacts.

    Output layout mirrors compile_swiglu_decode but with gemm_1/gemm_2 kernel names.
    When tile_m/tile_n are provided they are forwarded to SwiGLUPrefill,
    overriding the defaults (64) in both inner GEMMs. tile_n is auto-selected
    when not explicitly provided, enabling models whose dims aren't multiples
    of 512 (= 64 * 8 cols).
    """
    if dtype != "bf16":
        raise ValueError(f"SwiGLU currently supports only bf16, got {dtype}")

    # Auto-select tile_n if not explicitly provided.
    if tile_n is None:
        tile_n = select_swiglu_prefill_tile_n(embedding_dim, hidden_dim,
                                              num_aie_columns)
        if tile_n == 0:
            raise ValueError(
                f"Cannot find valid tile_n for embedding_dim={embedding_dim} "
                f"hidden_dim={hidden_dim} num_aie_columns={num_aie_columns}"
            )

    validate_swiglu_prefill_shapes(seq_len, embedding_dim, hidden_dim,
                                   num_aie_columns, tile_m=tile_m,
                                   tile_n=tile_n)

    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.swiglu_prefill.op import AIESwiGLUPrefill

    kwargs = dict(
        seq_len=seq_len, embedding_dim=embedding_dim, hidden_dim=hidden_dim
    )
    if tile_m is not None:
        kwargs["tile_m"] = tile_m
    if tile_n != 64:
        kwargs["tile_n"] = tile_n

    op = AIESwiGLUPrefill(**kwargs)
    op.compile()

    _stage_swiglu_prefill_new(op, output_dir)
    return output_dir


def compile_swiglu_prefill_int8(seq_len: int, embedding_dim: int, hidden_dim: int,
                                 num_aie_columns: int, output_dir: str,
                                 tile_m: int | None = None,
                                 tile_n: int | None = None) -> str:
    """Compile a W8A8 INT8 SwiGLU prefill operator and stage its artifacts.

    Uses GEMM(dtype_in="i8", dtype_out="bf16s") for both matmul stages.
    The C++ backend handles dynamic activation quantization at runtime.
    """
    # Auto-select tile_n if not explicitly provided.
    if tile_n is None:
        tile_n = select_swiglu_prefill_tile_n(embedding_dim, hidden_dim,
                                              num_aie_columns)
        if tile_n == 0:
            raise ValueError(
                f"Cannot find valid tile_n for embedding_dim={embedding_dim} "
                f"hidden_dim={hidden_dim} num_aie_columns={num_aie_columns}"
            )

    validate_swiglu_prefill_int8_shapes(seq_len, embedding_dim, hidden_dim,
                                         num_aie_columns, tile_m=tile_m,
                                         tile_n=tile_n)

    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    raise NotImplementedError(
        "SwiGLUPrefillInt8 is not available in IRON-windows. "
        "No INT8 SwiGLU prefill operator exists in this fork."
    )
    return output_dir


def _stage_chained_artifacts(op, kernel_names: tuple[str, ...], output_dir: str,
                             prefix: str = "swiglu") -> None:
    """Copy a chained composite op's combined xclbin + per-kernel insts to output_dir.

    Insts filenames are ``{prefix}_{name}.insts`` so bundles under different
    prefixes (swiglu, qkv, ...) cannot collide within the same cache dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir

    combined_src = build_dir / op.combined_xclbin.filename
    shutil.copy2(str(combined_src), os.path.join(output_dir, "combined.xclbin"))

    for name in kernel_names:
        insts_artifact = getattr(op, f"{name}_insts")
        src = build_dir / insts_artifact.filename
        dst = os.path.join(output_dir, f"{prefix}_{name}.insts")
        shutil.copy2(str(src), dst)


def _stage_swiglu_artifacts(op, kernel_names: tuple[str, ...], output_dir: str) -> None:
    """Backward-compatible wrapper: stages a SwiGLU bundle with ``prefix="swiglu"``."""
    _stage_chained_artifacts(op, kernel_names, output_dir, prefix="swiglu")


def _stage_swiglu_decode_new(op, output_dir: str) -> None:
    """Stage AIESwiGLUDecode (IRON-windows) artifacts into output_dir.

    IRON-windows AIESwiGLUDecode uses a dual-GEMV fused design with 2 sub-kernels
    (fused gate+up+silu+mul, down GEMV) instead of the old 4-kernel chain.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/swiglu_fused.insts
        <output_dir>/swiglu_gemv_2.insts
    """
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir

    combined_src = build_dir / op.combined_xclbin.filename
    shutil.copy2(str(combined_src), os.path.join(output_dir, "combined.xclbin"))

    fused_insts_src = build_dir / op.fused_insts.filename
    shutil.copy2(str(fused_insts_src), os.path.join(output_dir, "swiglu_fused.insts"))

    gemv_2_insts_src = build_dir / op.gemv_2_insts.filename
    shutil.copy2(str(gemv_2_insts_src), os.path.join(output_dir, "swiglu_gemv_2.insts"))


def _stage_swiglu_prefill_new(op, output_dir: str) -> None:
    """Stage AIESwiGLUPrefill (IRON-windows) artifacts into output_dir.

    IRON-windows AIESwiGLUPrefill uses 3 sub-kernels (gemm_1, silu_mul, gemm_2).

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/swiglu_gemm_1.insts
        <output_dir>/swiglu_silu_mul.insts
        <output_dir>/swiglu_gemm_2.insts
    """
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir

    combined_src = build_dir / op.combined_xclbin.filename
    shutil.copy2(str(combined_src), os.path.join(output_dir, "combined.xclbin"))

    gemm_1_insts_src = build_dir / op.gemm_1_insts.filename
    shutil.copy2(str(gemm_1_insts_src), os.path.join(output_dir, "swiglu_gemm_1.insts"))

    silu_mul_insts_src = build_dir / op.silu_mul_insts.filename
    shutil.copy2(str(silu_mul_insts_src), os.path.join(output_dir, "swiglu_silu_mul.insts"))

    gemm_2_insts_src = build_dir / op.gemm_2_insts.filename
    shutil.copy2(str(gemm_2_insts_src), os.path.join(output_dir, "swiglu_gemm_2.insts"))


def _stage_qkv_fused_new(op, output_dir: str) -> None:
    """Stage AIEFusedQKVProj (IRON-windows) artifacts into output_dir.

    IRON-windows AIEFusedQKVProj is a single-kernel operator that concatenates
    Wq/Wk/Wv into one weight matrix and runs a single GEMV. The host splits
    the output into Q, K, V segments.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/qkv_main.insts
    """
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir

    combined_src = build_dir / op.xclbin_artifact.filename
    shutil.copy2(str(combined_src), os.path.join(output_dir, "combined.xclbin"))

    insts_src = build_dir / op.insts_artifact.filename
    shutil.copy2(str(insts_src), os.path.join(output_dir, "qkv_main.insts"))


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_cache_dir() -> Path:
    """Get the xclbin cache directory."""
    return Path(os.environ.get("GGML_XDNA_CACHE_DIR",
                               Path.home() / ".cache" / "ggml-xdna" / "xclbin"))


def get_cached_xclbin(cache_key: str) -> Path | None:
    """Check if an xclbin is cached. Returns path if found, None otherwise.

    Both .xclbin and .insts files must exist for a cache hit.
    """
    cache_dir = get_cache_dir()
    xclbin_path = cache_dir / f"{cache_key}.xclbin"
    insts_path = cache_dir / f"{cache_key}.insts"
    if xclbin_path.exists() and insts_path.exists():
        return xclbin_path
    return None


def compile_gemm_cached(M: int, K: int, N: int, dtype_in: str = "bf16",
                        dtype_out: str = "bf16", num_aie_columns: int = 4,
                        b_col_maj: bool = False) -> Path:
    """Compile a GEMM operator with caching.

    Returns path to the xclbin (cached or newly compiled).
    """
    cache_key = gemm_cache_key(M, K, N, dtype_in, dtype_out, num_aie_columns, b_col_maj)

    cached = get_cached_xclbin(cache_key)
    if cached is not None:
        return cached

    cache_dir = get_cache_dir()
    output_path = str(cache_dir / f"{cache_key}.xclbin")
    compile_gemm(M, K, N, dtype_in, dtype_out, num_aie_columns, output_path, b_col_maj)
    return Path(output_path)


def compile_gemv_cached(N: int, K: int, dtype_in: str = "bf16",
                        dtype_out: str = "bf16", num_aie_columns: int = 4) -> Path:
    """Compile a GEMV operator with caching.

    Returns path to the xclbin (cached or newly compiled).
    """
    cache_key = gemv_cache_key(N, K, dtype_in, dtype_out, num_aie_columns)

    cached = get_cached_xclbin(cache_key)
    if cached is not None:
        return cached

    cache_dir = get_cache_dir()
    output_path = str(cache_dir / f"{cache_key}.xclbin")
    compile_gemv(N, K, dtype_in, dtype_out, num_aie_columns, output_path)
    return Path(output_path)


def get_cached_chained_dir(cache_key: str, kernel_names: tuple[str, ...],
                           prefix: str = "swiglu") -> Path | None:
    """Check if a chained-composite bundle is cached. Returns directory if found.

    The combined.xclbin and *all* expected per-kernel insts files (named
    ``{prefix}_{name}.insts``) must exist.
    """
    cache_dir = get_cache_dir() / cache_key
    if not (cache_dir / "combined.xclbin").exists():
        return None
    for name in kernel_names:
        if not (cache_dir / f"{prefix}_{name}.insts").exists():
            return None
    return cache_dir


def get_cached_swiglu_dir(cache_key: str, kernel_names: tuple[str, ...]) -> Path | None:
    """Backward-compatible wrapper: looks up a SwiGLU bundle with ``prefix="swiglu"``."""
    return get_cached_chained_dir(cache_key, kernel_names, prefix="swiglu")


def compile_swiglu_decode_cached(embedding_dim: int, hidden_dim: int,
                                 dtype: str = "bf16",
                                 num_aie_columns: int = 4) -> Path:
    """Compile a SwiGLU decode operator with caching.

    Returns path to the cache directory containing combined.xclbin + 4 insts files.
    """
    key = swiglu_decode_cache_key(embedding_dim, hidden_dim, dtype, num_aie_columns)

    cached = get_cached_swiglu_dir(key, SWIGLU_DECODE_KERNELS)
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_swiglu_decode(embedding_dim, hidden_dim, dtype, num_aie_columns, output_dir)
    return Path(output_dir)


def compile_swiglu_decode_int8_cached(embedding_dim: int, hidden_dim: int,
                                      num_aie_columns: int = 4,
                                      group_size: int = 32) -> Path:
    """Compile a SwiGLUDecodeInt8 operator with caching.

    Returns path to the cache directory containing combined.xclbin + 4 insts
    files (SWIGLU_DECODE_INT8_KERNELS).
    """
    key = swiglu_decode_int8_cache_key(
        embedding_dim, hidden_dim, num_aie_columns, group_size=group_size
    )

    cached = get_cached_swiglu_dir(key, SWIGLU_DECODE_INT8_KERNELS)
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_swiglu_decode_int8(
        embedding_dim, hidden_dim, num_aie_columns, output_dir,
        group_size=group_size,
    )
    return Path(output_dir)


def compile_swiglu_fused_int8_cached(embedding_dim: int, hidden_dim: int,
                                      num_aie_columns: int = 4,
                                      group_size: int = 32) -> Path:
    """Compile the chained fused+down INT8 SwiGLU with caching.

    Returns path to the cache directory containing combined.xclbin + 2 insts.
    """
    key = swiglu_fused_int8_cache_key(
        embedding_dim, hidden_dim, num_aie_columns, group_size=group_size
    )

    cached = get_cached_swiglu_dir(key, SWIGLU_FUSED_CHAINED_KERNELS)
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_swiglu_fused_int8(
        embedding_dim, hidden_dim, num_aie_columns, output_dir,
        group_size=group_size,
    )
    return Path(output_dir)


def validate_qkv_shapes(embedding_dim: int, q_dim: int, k_dim: int, v_dim: int,
                        num_aie_columns: int) -> None:
    """Verify a Q/K/V projection tuple is dispatchable to QKVProjChained.

    Each of the 3 inner GEMV stages must satisfy K % kernel_vector_size (64),
    per-column tiling, and L1-budget constraints. Reuses ``select_gemv_tiles``
    for the arithmetic checks so the rules stay in one place.
    """
    if num_aie_columns not in (4, 8):
        raise ValueError(
            f"num_aie_columns must be 4 (NPU1) or 8 (NPU2), got {num_aie_columns}"
        )
    for name, m in (("q", q_dim), ("k", k_dim), ("v", v_dim)):
        if m <= 0:
            raise ValueError(f"{name}_dim must be > 0, got {m}")
        # GEMV calls N the matrix-row dim (== output length); validator uses N=m.
        select_gemv_tiles(N=m, K=embedding_dim, num_aie_columns=num_aie_columns)


def compile_qkv(embedding_dim: int, q_dim: int, k_dim: int, v_dim: int,
                num_aie_columns: int, output_dir: str) -> str:
    """Compile the chained Q/K/V projection composite (3 bf16 GEMVs).

    Produces one combined xclbin with 3 kernel entries (Q, K, V projections)
    plus 3 insts files, staged under output_dir. All three kernels share one
    hw_context and can be batched via xrt::runlist.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/qkv_gemv_q.insts
        <output_dir>/qkv_gemv_k.insts
        <output_dir>/qkv_gemv_v.insts
    """
    validate_qkv_shapes(embedding_dim, q_dim, k_dim, v_dim, num_aie_columns)

    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.fused_qkv_proj.op import AIEFusedQKVProj

    op = AIEFusedQKVProj(
        embedding_dim=embedding_dim,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        num_aie_columns=num_aie_columns,
    )
    op.compile()

    _stage_qkv_fused_new(op, output_dir)
    return output_dir


def compile_qkv_cached(embedding_dim: int, q_dim: int, k_dim: int, v_dim: int,
                       num_aie_columns: int = 4) -> Path:
    """Compile the chained Q/K/V projection with caching.

    Returns path to the cache directory containing combined.xclbin + 3 insts files.
    """
    key = qkv_cache_key(embedding_dim, q_dim, k_dim, v_dim, num_aie_columns)

    cached = get_cached_chained_dir(key, QKV_PROJ_KERNELS, prefix="qkv")
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_qkv(embedding_dim, q_dim, k_dim, v_dim, num_aie_columns, output_dir)
    return Path(output_dir)


# ---------------------------------------------------------------------------
# FlowKV Decode Attention (single xclbin, 2-tile streaming pipeline)
# ---------------------------------------------------------------------------

def _stage_flowkv_decode_artifacts(op, output_dir: str) -> None:
    """Stage a compiled FlowKV decode op's xclbin + insts under output_dir.

    Produces:
        <output_dir>/combined.xclbin
        <output_dir>/flowkv_decode_main.insts
    """
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir
    compiled_xclbin = build_dir / op.xclbin_artifact.filename
    compiled_insts = build_dir / op.insts_artifact.filename
    shutil.copy2(str(compiled_xclbin), os.path.join(output_dir, "combined.xclbin"))
    shutil.copy2(str(compiled_insts), os.path.join(output_dir, "flowkv_decode_main.insts"))


def compile_flowkv_decode(num_heads: int, num_kv_heads: int, head_dim: int,
                          seq_len: int, chunk_size: int = 32,
                          num_cols: int = 4, output_dir: str = "") -> str:
    """Compile a FlowKV decode attention operator.

    Produces one xclbin with a 2-tile streaming pipeline (score + value)
    per KV head group, with fused RoPE on Q. Uses online softmax for
    exact FlashAttention semantics in a single streaming pass.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/flowkv_decode_main.insts
    """
    from iron.operators.flowkv_decode.op import AIEFlowKVDecode

    op = AIEFlowKVDecode(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        chunk_size=chunk_size,
        num_cols=num_cols,
    )
    op.compile()

    _stage_flowkv_decode_artifacts(op, output_dir)
    return output_dir


def compile_flowkv_decode_cached(num_heads: int, num_kv_heads: int,
                                 head_dim: int, seq_len: int,
                                 chunk_size: int = 32,
                                 num_cols: int = 4) -> Path:
    """Compile a FlowKV decode attention operator with caching.

    Returns path to the cache directory containing combined.xclbin and
    flowkv_decode_main.insts.
    """
    key = flowkv_decode_cache_key(
        num_heads, num_kv_heads, head_dim, seq_len, chunk_size, num_cols
    )

    cached = get_cached_chained_dir(
        key, FLOWKV_DECODE_KERNELS, prefix="flowkv_decode"
    )
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_flowkv_decode(
        num_heads, num_kv_heads, head_dim, seq_len, chunk_size, num_cols,
        output_dir,
    )
    return Path(output_dir)


def compile_swiglu_prefill_cached(seq_len: int, embedding_dim: int, hidden_dim: int,
                                  dtype: str = "bf16",
                                  num_aie_columns: int = 4,
                                  tile_m: int | None = None,
                                  tile_n: int | None = None) -> Path:
    """Compile a SwiGLU prefill operator with caching.

    Returns path to the cache directory containing combined.xclbin + 4 insts files.
    """
    key = swiglu_prefill_cache_key(
        seq_len, embedding_dim, hidden_dim, dtype, num_aie_columns,
        tile_m=tile_m, tile_n=tile_n,
    )

    cached = get_cached_swiglu_dir(key, SWIGLU_PREFILL_KERNELS)
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_swiglu_prefill(
        seq_len, embedding_dim, hidden_dim, dtype, num_aie_columns, output_dir,
        tile_m=tile_m, tile_n=tile_n,
    )
    return Path(output_dir)


def compile_swiglu_prefill_int8_cached(seq_len: int, embedding_dim: int, hidden_dim: int,
                                        num_aie_columns: int = 4,
                                        tile_m: int | None = None,
                                        tile_n: int | None = None) -> Path:
    """Compile a W8A8 INT8 SwiGLU prefill operator with caching."""
    key = swiglu_prefill_int8_cache_key(
        seq_len, embedding_dim, hidden_dim, num_aie_columns,
        tile_m=tile_m, tile_n=tile_n,
    )

    cached = get_cached_swiglu_dir(key, SWIGLU_PREFILL_INT8_KERNELS)
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_swiglu_prefill_int8(
        seq_len, embedding_dim, hidden_dim, num_aie_columns, output_dir,
        tile_m=tile_m, tile_n=tile_n,
    )
    return Path(output_dir)


# ---------------------------------------------------------------------------
# RMSNorm (standalone, single-kernel)
# ---------------------------------------------------------------------------

def validate_rms_norm_shapes(size: int, num_aie_columns: int,
                             num_channels: int, tile_size: int,
                             weighted: bool = False) -> tuple[bool, str | None]:
    """Verify an (size, cols, channels, tile_size) tuple is dispatchable to RMSNorm.

    Mirrors the constraints in iron/operators/rms_norm/op.py::RMSNorm.__post_init__
    but returns (ok, reason) instead of raising so the C++ backend can fall back
    cleanly. Shim-DMA budget is checked against the live device.
    """
    if num_aie_columns < 1:
        return False, f"num_aie_columns must be >= 1, got {num_aie_columns}"
    if num_channels < 1:
        return False, f"num_channels must be >= 1, got {num_channels}"
    if tile_size < 1:
        return False, f"tile_size must be >= 1, got {tile_size}"
    max_multiple = num_aie_columns * num_channels * tile_size
    if size % max_multiple != 0:
        return False, (
            f"size ({size}) must be a multiple of "
            f"num_aie_columns * num_channels * tile_size ({max_multiple})"
        )
    try:
        dev = aie_utils.get_current_device()
        if dev is None:
            shim_limit = None
        else:
            from iron.common.utils import get_shim_dma_limit
            shim_limit = get_shim_dma_limit(dev)
    except Exception:
        shim_limit = None
    if shim_limit is not None:
        total = num_aie_columns * num_channels
        if total > shim_limit:
            return False, (
                f"num_aie_columns * num_channels ({total}) exceeds ShimDMA "
                f"limit of {shim_limit} for this device"
            )
        if weighted and num_channels * (num_aie_columns + 1) > shim_limit:
            return False, (
                f"weighted RMSNorm requires "
                f"{num_channels * (num_aie_columns + 1)} ShimDMA output "
                f"channels but device only has {shim_limit}"
            )
    return True, None


def _stage_rms_norm_artifacts(op, output_dir: str) -> None:
    """Stage a compiled RMSNorm op's single xclbin + insts under output_dir.

    Produces:
        <output_dir>/combined.xclbin
        <output_dir>/rms_norm_main.insts
    so the on-disk layout mirrors chained composites (for the bundle-lookup
    helper) even though only one kernel is present.
    """
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir
    compiled_xclbin = build_dir / op.xclbin_artifact.filename
    compiled_insts = build_dir / op.insts_artifact.filename
    shutil.copy2(str(compiled_xclbin), os.path.join(output_dir, "combined.xclbin"))
    shutil.copy2(str(compiled_insts), os.path.join(output_dir, "rms_norm_main.insts"))


def compile_rms_norm(size: int, dtype: str, num_aie_columns: int,
                     num_channels: int, tile_size: int, weighted: bool,
                     output_dir: str) -> str:
    """Compile an IRON RMSNorm operator and stage artifacts under output_dir.

    Epsilon is hardcoded to 1e-5 inside the AIE kernel (not runtime-configurable).
    """
    if dtype != "bf16":
        raise ValueError(f"RMSNorm currently supports only bf16, got {dtype}")
    ok, reason = validate_rms_norm_shapes(
        size, num_aie_columns, num_channels, tile_size, weighted
    )
    if not ok:
        raise ValueError(reason)

    actual_cols = get_device_cols(num_aie_columns)
    if actual_cols < num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.rms_norm.op import RMSNorm

    op = RMSNorm(
        size=size,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        tile_size=tile_size,
        weighted=weighted,
    )
    op.compile()

    _stage_rms_norm_artifacts(op, output_dir)
    return output_dir


def compile_rms_norm_cached(size: int, dtype: str = "bf16",
                            num_aie_columns: int = 4,
                            num_channels: int = 1,
                            tile_size: int = 32,
                            weighted: bool = False) -> Path:
    """Compile an RMSNorm operator with caching.

    Returns path to the cache directory containing combined.xclbin and
    rms_norm_main.insts.
    """
    key = rms_norm_cache_key(
        size, dtype, num_aie_columns, num_channels, tile_size, weighted
    )

    cached = get_cached_chained_dir(key, RMS_NORM_KERNELS, prefix="rms_norm")
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_rms_norm(
        size, dtype, num_aie_columns, num_channels, tile_size, weighted,
        output_dir,
    )
    return Path(output_dir)


# ---------------------------------------------------------------------------
# AttentionBlockPrefill (chained 11-kernel composite)
# ---------------------------------------------------------------------------


def select_attention_prefill_seq_bucket(seq_len: int) -> int | None:
    """Return the smallest bucket >= seq_len, or None if seq_len exceeds the cap.

    Used to pad up the prefill seq_len dim so one session reuses a single
    xclbin. The caller should treat a None return as a hard reject
    (fall back to CPU).
    """
    if seq_len <= 0:
        return None
    for b in ATTENTION_PREFILL_SEQ_BUCKETS:
        if seq_len <= b:
            return b
    return None


def attention_prefill_cache_key(seq_len: int, embed_dim: int, num_heads: int,
                                num_kv_heads: int, head_dim: int,
                                dtype: str = "bf16",
                                rope_method_type: int = 0) -> str:
    """Cache key for an AttentionBlockPrefill configuration.

    seq_len is rounded up to the nearest bucket before hashing so that two
    callers requesting lengths in the same bucket share one xclbin. Disjoint
    from every other op's key by the ``"op"`` field. ``rope_method_type``
    must be 0 (TWO_HALVES / ggml NEOX) or 1 (INTERLEAVED / ggml NORMAL); it
    is included in the key so each rotation method gets its own bundle.
    """
    seq_len_padded = select_attention_prefill_seq_bucket(seq_len)
    key_data = {
        "op": "attention_prefill",
        "seq_len_padded": seq_len_padded,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype,
        "rope_method_type": rope_method_type,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def validate_attention_prefill_shapes(seq_len: int, embed_dim: int,
                                      num_heads: int, num_kv_heads: int,
                                      head_dim: int) -> tuple[bool, str | None]:
    """Verify an (seq_len, embed_dim, num_heads, num_kv_heads, head_dim) tuple
    is dispatchable to AttentionBlockPrefill.

    Returns (ok, reason). Callers (typically the C++ backend) can fall back to
    CPU cleanly on a False return without raising.
    """
    if seq_len <= 0:
        return False, f"seq_len must be > 0, got {seq_len}"
    if head_dim != 64:
        return False, (
            f"head_dim must be 64 (IRON MHA hardcode), got {head_dim}"
        )
    if num_kv_heads <= 0 or num_heads <= 0:
        return False, (
            f"num_heads and num_kv_heads must be > 0, got "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}"
        )
    if num_heads % num_kv_heads != 0:
        return False, (
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads "
            f"({num_kv_heads}) for GQA grouping"
        )
    if embed_dim <= 0:
        return False, f"embed_dim must be > 0, got {embed_dim}"
    # GEMM_Q uses all device columns (we target NPU2 with 8). Q output dim =
    # num_heads * head_dim; that and embed_dim must tile cleanly against 8 cols
    # with tile_n >= 16 (bf16 floor). Most practical models (embed=1024/2048/4096,
    # H*d = multiple-of-128) satisfy this.
    if embed_dim % 8 != 0:
        return False, (
            f"embed_dim ({embed_dim}) must be divisible by 8 (Q/O GEMM column count)"
        )
    kv_dim = num_kv_heads * head_dim
    # GEMM_KV uses num_aie_columns = KV_d // tile_n (tile_n=64). KV_d must be a
    # multiple of 64 and yield at least one column.
    if kv_dim % 64 != 0:
        return False, (
            f"num_kv_heads*head_dim ({kv_dim}) must be a multiple of 64 for GEMM_KV"
        )
    return True, None


def compile_attention_prefill(seq_len: int, embed_dim: int, num_heads: int,
                              num_kv_heads: int, head_dim: int,
                              dtype: str, output_dir: str,
                              rope_method_type: int = 0) -> str:
    """Compile AttentionBlockPrefill and stage its 11-kernel chained bundle.

    seq_len is rounded up to the nearest bucket and forwarded to the composite
    as the padded length. The C++ backend is responsible for pad/unpad.

    Output layout:
        <output_dir>/combined.xclbin
        <output_dir>/attn_prefill_rms_norm.insts
        <output_dir>/attn_prefill_gemm_q.insts
        <output_dir>/attn_prefill_gemm_kv.insts
        <output_dir>/attn_prefill_rope_q.insts
        <output_dir>/attn_prefill_rope_k.insts
        <output_dir>/attn_prefill_perm_q.insts
        <output_dir>/attn_prefill_perm_kv.insts
        <output_dir>/attn_prefill_mha.insts
        <output_dir>/attn_prefill_perm_o.insts
        <output_dir>/attn_prefill_gemm_o.insts
        <output_dir>/attn_prefill_add.insts
    """
    if dtype != "bf16":
        raise ValueError(
            f"AttentionBlockPrefill currently supports only bf16, got {dtype}"
        )
    ok, reason = validate_attention_prefill_shapes(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim
    )
    if not ok:
        raise ValueError(reason)

    seq_len_padded = select_attention_prefill_seq_bucket(seq_len)
    if seq_len_padded is None:
        raise ValueError(
            f"seq_len={seq_len} exceeds the largest attention-prefill bucket "
            f"({ATTENTION_PREFILL_SEQ_BUCKETS[-1]}); caller should fall back to CPU"
        )

    # AttentionBlockPrefill reads num_aie_columns from the live device inside
    # set_up_artifacts (via aie_utils.get_current_device()). We don't take a
    # --num-aie-columns flag for this op, but fail fast if the device isn't up.
    _ = get_device_cols(0)

    raise NotImplementedError(
        "AttentionBlockPrefill is not available in IRON-windows. "
        "No chained attention-block prefill operator exists in this fork."
    )
    return output_dir


def compile_attention_prefill_cached(seq_len: int, embed_dim: int,
                                     num_heads: int, num_kv_heads: int,
                                     head_dim: int,
                                     dtype: str = "bf16",
                                     rope_method_type: int = 0) -> Path:
    """Compile AttentionBlockPrefill with caching.

    Returns the cache directory (contains combined.xclbin + 11 insts).
    """
    key = attention_prefill_cache_key(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, dtype,
        rope_method_type=rope_method_type,
    )

    cached = get_cached_chained_dir(
        key, ATTENTION_PREFILL_KERNELS, prefix="attn_prefill"
    )
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_attention_prefill(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, dtype, output_dir,
        rope_method_type=rope_method_type,
    )
    return Path(output_dir)


# ---------------------------------------------------------------------------
# TransformerBlockPrefill (chained 17-kernel composite: attn + SwiGLU FFN)
# ---------------------------------------------------------------------------


def select_transformer_block_prefill_seq_bucket(seq_len: int) -> int | None:
    if seq_len <= 0:
        return None
    for b in TRANSFORMER_BLOCK_PREFILL_SEQ_BUCKETS:
        if seq_len <= b:
            return b
    return None


def transformer_block_prefill_cache_key(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
    dtype: str = "bf16",
    rope_method_type: int = 0,
) -> str:
    seq_len_padded = select_transformer_block_prefill_seq_bucket(seq_len)
    key_data = {
        "op": "transformer_block_prefill",
        "seq_len_padded": seq_len_padded,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "ffn_hidden_dim": ffn_hidden_dim,
        "dtype": dtype,
        "rope_method_type": rope_method_type,
    }
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def validate_transformer_block_prefill_shapes(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
) -> tuple[bool, str | None]:
    ok, reason = validate_attention_prefill_shapes(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim
    )
    if not ok:
        return False, reason
    if ffn_hidden_dim <= 0:
        return False, f"ffn_hidden_dim must be > 0, got {ffn_hidden_dim}"
    # gemm_gate_up: N=ffn_hidden_dim; gemm_down: K=ffn_hidden_dim. Both use
    # num_aie_columns=n_cols (typically 8 on NPU2) with tile_n=64. Require
    # ffn_hidden_dim divisible by (64 * 8)=512 (aligns with 4-col NPU1 too).
    if ffn_hidden_dim % 64 != 0:
        return False, (
            f"ffn_hidden_dim ({ffn_hidden_dim}) must be a multiple of 64 "
            f"(tile_n) for gemm_gate_up / gemm_down"
        )
    return True, None


def compile_transformer_block_prefill(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
    dtype: str, output_dir: str,
    rope_method_type: int = 0,
) -> str:
    if dtype != "bf16":
        raise ValueError(
            f"TransformerBlockPrefill currently supports only bf16, got {dtype}"
        )
    ok, reason = validate_transformer_block_prefill_shapes(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim
    )
    if not ok:
        raise ValueError(reason)

    seq_len_padded = select_transformer_block_prefill_seq_bucket(seq_len)
    if seq_len_padded is None:
        raise ValueError(
            f"seq_len={seq_len} exceeds the largest transformer-block-prefill "
            f"bucket ({TRANSFORMER_BLOCK_PREFILL_SEQ_BUCKETS[-1]}); caller "
            f"should fall back to CPU"
        )

    _ = get_device_cols(0)

    raise NotImplementedError(
        "TransformerBlockPrefill is not available in IRON-windows. "
        "No chained transformer-block prefill operator exists in this fork."
    )
    return output_dir


def compile_transformer_block_prefill_cached(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
    dtype: str = "bf16",
    rope_method_type: int = 0,
) -> Path:
    key = transformer_block_prefill_cache_key(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim,
        dtype, rope_method_type=rope_method_type,
    )
    cached = get_cached_chained_dir(
        key, TRANSFORMER_BLOCK_PREFILL_KERNELS, prefix="tblock_prefill"
    )
    if cached is not None:
        return cached
    output_dir = str(get_cache_dir() / key)
    compile_transformer_block_prefill(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim,
        dtype, output_dir, rope_method_type=rope_method_type,
    )
    return Path(output_dir)


# ---------------------------------------------------------------------------
# TransformerBlockPrefillFused (Layer 4A Phase 2: monolithic single-ELF)
#
# Unlike the chained TransformerBlockPrefill above (17 separate xclbin kernels
# chained by host-side dispatch), this bundles the entire attn+FFN block into
# ONE ELF with ONE kernel entry ("main:sequence") taking 3 BO args:
# input_buffer, output_buffer, scratch_buffer. The C++ backend allocates
# those 3 BOs once, writes named tensors (x / w_q / w_norm_ffn / ...) at
# offsets from the sidecar layout JSON, and issues a SINGLE xrt::run per
# layer instead of 17. Target: recover the per-dispatch overhead that Layer
# 3A chained dispatch cannot avoid.
# ---------------------------------------------------------------------------


TRANSFORMER_BLOCK_PREFILL_FUSED_SEQ_BUCKETS = ATTENTION_PREFILL_SEQ_BUCKETS


def select_transformer_block_prefill_fused_seq_bucket(seq_len: int) -> int | None:
    if seq_len <= 0:
        return None
    for b in TRANSFORMER_BLOCK_PREFILL_FUSED_SEQ_BUCKETS:
        if seq_len <= b:
            return b
    return None


def transformer_block_prefill_fused_cache_key(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
    dtype: str = "bf16",
    rope_method_type: int = 0,
    num_layers: int = 1,
    w8a16: bool = False,
    w8a16_ffn: bool = False,
) -> str:
    if w8a16_ffn and not w8a16:
        raise ValueError("w8a16_ffn=True requires w8a16=True")
    seq_len_padded = select_transformer_block_prefill_fused_seq_bucket(seq_len)
    key_data = {
        "op": "transformer_block_prefill_fused",
        "seq_len_padded": seq_len_padded,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "ffn_hidden_dim": ffn_hidden_dim,
        "dtype": dtype,
        "rope_method_type": rope_method_type,
    }
    # Only fold num_layers into the key when >1 so N=1 cache entries
    # remain bit-identical to pre-Layer-4B bundles (no thrash).
    if num_layers > 1:
        key_data["num_layers"] = num_layers
    # Only fold w8a16 when True so the bf16 cache entries remain
    # bit-identical to pre-W8A16 bundles (no thrash). Three modes:
    #   - w8a16=False                      : no key entry (pure bf16)
    #   - w8a16=True, w8a16_ffn=False (A.7): key_data["w8a16"] = "attn"
    #     (INT8 attn projections only; FFN GEMMs are bf16). This tag was
    #     introduced alongside the w8a16_ffn split — it forces a fresh
    #     cache miss vs any pre-split w8a16 bundles (which were Phase B).
    #   - w8a16_ffn=True              (B)  : key_data["w8a16"] = "ffn"
    #     (INT8 attn + INT8 FFN; preserves the existing Phase-B cache key).
    if w8a16_ffn:
        key_data["w8a16"] = "ffn"
    elif w8a16:
        key_data["w8a16"] = "attn"
    key_json = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_json.encode()).hexdigest()[:16]


def _tblock_fused_bundle_filenames(
    output_dir: str, w8a16: bool = False, w8a16_ffn: bool = False
) -> dict:
    """Canonical filenames produced by the fused staging helper.

    Kept in one place so both the producer and the C++-side consumer (via
    the bundle_present check) reference the same strings.
    """
    if w8a16_ffn and not w8a16:
        raise ValueError("w8a16_ffn=True requires w8a16=True")
    if w8a16_ffn:
        return {
            "elf": os.path.join(output_dir, "tblock_fused_w8a16ffn.elf"),
            "layout": os.path.join(output_dir, "tblock_fused_w8a16ffn.layout.json"),
        }
    if w8a16:
        return {
            "elf": os.path.join(output_dir, "tblock_fused_w8a16.elf"),
            "layout": os.path.join(output_dir, "tblock_fused_w8a16.layout.json"),
        }
    return {
        "elf": os.path.join(output_dir, "tblock_fused.elf"),
        "layout": os.path.join(output_dir, "tblock_fused.layout.json"),
    }


def _stage_tblock_fused_artifacts(
    op, output_dir: str, w8a16: bool = False, w8a16_ffn: bool = False
) -> None:
    """Copy the FusedMLIROperator ELF and write its buffer layout as JSON.

    FusedMLIROperator produces a single ``{name}.elf`` artifact in
    ``op.context.build_dir`` plus an in-memory ``subbuffer_layout`` dict
    ``{name: (buf_type, offset_bytes, length_bytes, dtype)}`` where buf_type
    is one of ``input``, ``output``, ``scratch`` and dtype is the numpy
    dtype (bfloat16 for activations/bf16 weights/scales, uint8 for the
    packed INT8+scales weight buffers under W8A16). The C++ backend needs
    the ELF to load the kernel and the layout to know where to write each
    named tensor. Serialize both to predictable filenames under
    ``output_dir``.
    """
    if w8a16_ffn and not w8a16:
        raise ValueError("w8a16_ffn=True requires w8a16=True")
    os.makedirs(output_dir, exist_ok=True)
    names = _tblock_fused_bundle_filenames(
        output_dir, w8a16=w8a16, w8a16_ffn=w8a16_ffn
    )

    # ELF
    elf_artifact = op.artifacts[0]
    build_dir = op.context.build_dir
    elf_src = build_dir / elf_artifact.filename
    shutil.copy2(str(elf_src), names["elf"])

    # Layout JSON. buffer_sizes is a 4-tuple when split_dynamic is True
    # (x/angles carved into a separate dynamic-input BO), 3-tuple otherwise.
    if len(op.buffer_sizes) == 4:
        input_sz, dynamic_input_sz, output_sz, scratch_sz = op.buffer_sizes
    else:
        input_sz, output_sz, scratch_sz = op.buffer_sizes
        dynamic_input_sz = 0
    buffers = {}
    for bname, entry in op.subbuffer_layout.items():
        # IRON subbuffer_layout entries are 4-tuples:
        # (buf_type, offset_bytes, length_bytes, dtype). The dtype is a
        # numpy dtype (bfloat16 for fp activations / scales / bf16 weights,
        # uint8 for packed INT8+scales buffers). Normalize to the numpy
        # dtype name so the C++ side can reason about it as a plain string
        # ("bfloat16", "uint8", ...).
        btype, off, length, dtype = entry
        buffers[bname] = {
            "buf_type": btype,
            "offset_bytes": int(off),
            "length_bytes": int(length),
            "dtype": np.dtype(dtype).name,
        }
    # Preserve slice_info too so the C++ side can resolve sliced buffer
    # names if a future runlist uses slice notation (currently none do,
    # but it's cheap to emit).
    slices = {
        sname: {"parent": parent, "start": int(s), "end": int(e)}
        for sname, (parent, s, e) in op.slice_info.items()
    }
    layout = {
        "version": 1,
        "kernel_name": "main:sequence",
        "buffer_sizes": {
            "input_bytes": int(input_sz),
            "dynamic_input_bytes": int(dynamic_input_sz),
            "output_bytes": int(output_sz),
            "scratch_bytes": int(scratch_sz),
        },
        "input_args": list(op.input_args),
        "dynamic_input_args": list(getattr(op, "dynamic_input_args", []) or []),
        "output_args": list(op.output_args),
        "buffers": buffers,
        "slices": slices,
        "meta": {
            "seq_len": op.seq_len,
            "embed_dim": op.embed_dim,
            "num_heads": op.num_heads,
            "num_kv_heads": op.num_kv_heads,
            "head_dim": op.head_dim,
            "ffn_hidden_dim": op.ffn_hidden_dim,
            "rope_method_type": op.rope_method_type,
            "num_layers": getattr(op, "num_layers", 1),
        },
    }
    with open(names["layout"], "w") as f:
        json.dump(layout, f, indent=2, sort_keys=True)


def tblock_fused_bundle_present(
    output_dir: str, w8a16: bool = False, w8a16_ffn: bool = False
) -> bool:
    if w8a16_ffn and not w8a16:
        raise ValueError("w8a16_ffn=True requires w8a16=True")
    names = _tblock_fused_bundle_filenames(
        output_dir, w8a16=w8a16, w8a16_ffn=w8a16_ffn
    )
    return all(os.path.exists(p) for p in names.values())


def validate_transformer_block_prefill_fused_shapes(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
) -> tuple[bool, str | None]:
    # Same guard rails as the chained composite — the fused op wraps the
    # same sub-ops, so the same divisibility constraints apply.
    return validate_transformer_block_prefill_shapes(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim
    )


def compile_transformer_block_prefill_fused(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
    dtype: str, output_dir: str,
    rope_method_type: int = 0,
    num_layers: int = 1,
    w8a16: bool = False,
    w8a16_ffn: bool = False,
) -> str:
    if w8a16_ffn and not w8a16:
        raise ValueError("w8a16_ffn=True requires w8a16=True")
    if dtype != "bf16":
        raise ValueError(
            f"TransformerBlockPrefillFused currently supports only bf16, "
            f"got {dtype}"
        )
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    ok, reason = validate_transformer_block_prefill_fused_shapes(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim
    )
    if not ok:
        raise ValueError(reason)

    seq_len_padded = select_transformer_block_prefill_fused_seq_bucket(seq_len)
    if seq_len_padded is None:
        raise ValueError(
            f"seq_len={seq_len} exceeds the largest transformer-block-prefill-"
            f"fused bucket ({TRANSFORMER_BLOCK_PREFILL_FUSED_SEQ_BUCKETS[-1]}); "
            f"caller should fall back to CPU"
        )

    _ = get_device_cols(0)

    raise NotImplementedError(
        "TransformerBlockPrefillFused (AttentionBlockPrefillFused) is not available "
        "in IRON-windows. No monolithic fused transformer-block prefill operator "
        "exists in this fork."
    )
    return output_dir


def compile_transformer_block_prefill_fused_cached(
    seq_len: int, embed_dim: int, num_heads: int, num_kv_heads: int,
    head_dim: int, ffn_hidden_dim: int,
    dtype: str = "bf16",
    rope_method_type: int = 0,
    num_layers: int = 1,
    w8a16: bool = False,
    w8a16_ffn: bool = False,
) -> Path:
    if w8a16_ffn and not w8a16:
        raise ValueError("w8a16_ffn=True requires w8a16=True")
    key = transformer_block_prefill_fused_cache_key(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim,
        dtype, rope_method_type=rope_method_type, num_layers=num_layers,
        w8a16=w8a16, w8a16_ffn=w8a16_ffn,
    )
    output_dir = str(get_cache_dir() / key)
    if tblock_fused_bundle_present(output_dir, w8a16=w8a16, w8a16_ffn=w8a16_ffn):
        return Path(output_dir)
    compile_transformer_block_prefill_fused(
        seq_len, embed_dim, num_heads, num_kv_heads, head_dim, ffn_hidden_dim,
        dtype, output_dir, rope_method_type=rope_method_type,
        num_layers=num_layers, w8a16=w8a16, w8a16_ffn=w8a16_ffn,
    )
    return Path(output_dir)


# ---------------------------------------------------------------------------
# CLI entry point (called by C++ backend)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ggml-xdna IRON compilation bridge")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stdout output (for use from ggml-xdna backend)")
    subparsers = parser.add_subparsers(dest="op", required=True)

    # GEMM subcommand
    gemm_parser = subparsers.add_parser("gemm", help="Compile GEMM operator")
    gemm_parser.add_argument("--M", type=int, required=True)
    gemm_parser.add_argument("--K", type=int, required=True)
    gemm_parser.add_argument("--N", type=int, required=True)
    gemm_parser.add_argument("--dtype-in", default="bf16", choices=["bf16", "i8"])
    gemm_parser.add_argument("--dtype-out", default="bf16", choices=["bf16", "i8", "i16", "i32"])
    gemm_parser.add_argument("--num-aie-columns", type=int, default=4)
    gemm_parser.add_argument("--b-col-maj", action="store_true")
    gemm_parser.add_argument("--out", type=str, help="Output xclbin path (default: cache)")

    # GEMV subcommand
    gemv_parser = subparsers.add_parser("gemv", help="Compile GEMV operator (M=1 decode)")
    gemv_parser.add_argument("--N", type=int, required=True, help="Matrix-row dim (output length)")
    gemv_parser.add_argument("--K", type=int, required=True, help="Reduction dim (vector length)")
    gemv_parser.add_argument("--dtype-in", default="bf16", choices=["bf16"])
    gemv_parser.add_argument("--dtype-out", default="bf16", choices=["bf16"])
    gemv_parser.add_argument("--num-aie-columns", type=int, default=4)
    gemv_parser.add_argument("--out", type=str, help="Output xclbin path (default: cache)")

    # SwiGLU decode subcommand
    swd_parser = subparsers.add_parser(
        "swiglu-decode", help="Compile fused SwiGLU FFN (M=1 decode path)"
    )
    swd_parser.add_argument("--embedding-dim", type=int, required=True)
    swd_parser.add_argument("--hidden-dim", type=int, required=True)
    swd_parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    swd_parser.add_argument("--num-aie-columns", type=int, default=4)
    swd_parser.add_argument("--out", type=str,
                            help="Output directory (default: cache)")

    # SwiGLU decode INT8 subcommand (W8A16)
    swdi_parser = subparsers.add_parser(
        "swiglu-decode-int8",
        help="Compile fused W8A16 SwiGLU FFN (INT8 weights + bf16 activations, "
             "M=1 decode path)",
    )
    swdi_parser.add_argument("--embedding-dim", type=int, required=True)
    swdi_parser.add_argument("--hidden-dim", type=int, required=True)
    swdi_parser.add_argument("--num-aie-columns", type=int, default=4)
    swdi_parser.add_argument("--group-size", type=int, default=32,
                             help="Quantization group size (must match Q8_0=32 "
                                  "for ggml Q8_0 weights)")
    swdi_parser.add_argument("--out", type=str,
                             help="Output directory (default: cache)")

    # SwiGLU fused INT8 subcommand (gate+up+silu+mul fused + standalone down GEMV)
    swfi_parser = subparsers.add_parser(
        "swiglu-fused-int8",
        help="Compile fused gate+up+silu+mul INT8 + standalone down GEMV "
             "(2-dispatch path, W8A16)",
    )
    swfi_parser.add_argument("--embedding-dim", type=int, required=True)
    swfi_parser.add_argument("--hidden-dim", type=int, required=True)
    swfi_parser.add_argument("--num-aie-columns", type=int, default=4)
    swfi_parser.add_argument("--group-size", type=int, default=32,
                             help="Quantization group size (must match Q8_0=32)")
    swfi_parser.add_argument("--out", type=str,
                             help="Output directory (default: cache)")

    # QKV projection subcommand (3 chained bf16 GEMVs, shared input)
    qkv_parser = subparsers.add_parser(
        "qkv",
        help="Compile chained Q/K/V projection (3 bf16 GEMVs sharing input, "
             "one xclbin, one xrt::runlist dispatch)"
    )
    qkv_parser.add_argument("--embedding-dim", type=int, required=True,
                            help="Input activation size (K for all 3 GEMVs)")
    qkv_parser.add_argument("--q-dim", type=int, required=True,
                            help="Q projection output size")
    qkv_parser.add_argument("--k-dim", type=int, required=True,
                            help="K projection output size")
    qkv_parser.add_argument("--v-dim", type=int, required=True,
                            help="V projection output size")
    qkv_parser.add_argument("--num-aie-columns", type=int, default=4)
    qkv_parser.add_argument("--out", type=str,
                            help="Output directory (default: cache)")

    # SwiGLU prefill subcommand
    swp_parser = subparsers.add_parser(
        "swiglu-prefill", help="Compile fused SwiGLU FFN (M>=32 prefill path)"
    )
    swp_parser.add_argument("--seq-len", type=int, required=True)
    swp_parser.add_argument("--embedding-dim", type=int, required=True)
    swp_parser.add_argument("--hidden-dim", type=int, required=True)
    swp_parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    swp_parser.add_argument("--num-aie-columns", type=int, default=4)
    swp_parser.add_argument("--tile-m", type=int, default=None,
                            help="Override tile_m for both inner GEMMs "
                                 "(default: IRON SwiGLUPrefill default of 64)")
    swp_parser.add_argument("--tile-n", type=int, default=None,
                            help="Override tile_n for both inner GEMMs "
                                 "(default: auto-selected for shape compatibility)")
    swp_parser.add_argument("--out", type=str,
                            help="Output directory (default: cache)")

    # RMSNorm subcommand (standalone, single-kernel)
    rms_parser = subparsers.add_parser(
        "rms_norm",
        help="Compile standalone RMSNorm (bf16, epsilon hardcoded to 1e-5)",
    )
    rms_parser.add_argument("--size", type=int, required=True,
                            help="Norm-axis length")
    rms_parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    rms_parser.add_argument("--num-aie-columns", type=int, default=4)
    rms_parser.add_argument("--num-channels", type=int, default=1)
    rms_parser.add_argument("--tile-size", type=int, default=32)
    rms_parser.add_argument("--weighted", action="store_true",
                            help="Include per-feature weight multiply "
                                 "(not yet wired through the backend)")
    rms_parser.add_argument("--cache-dir", type=str, default=None,
                            help="Override GGML_XDNA_CACHE_DIR for this run")
    rms_parser.add_argument("--out", type=str,
                            help="Output directory (default: cache)")

    # FlowKV Decode Attention subcommand (streaming decode attention with online softmax)
    fkvd_parser = subparsers.add_parser(
        "flowkv-decode",
        help="Compile FlowKV decode attention (streaming online softmax, fused RoPE, "
             "2-tile pipeline per KV head group)",
    )
    fkvd_parser.add_argument("--num-heads", type=int, required=True,
                             help="Number of query heads (e.g. 32 for Llama 3.2 1B)")
    fkvd_parser.add_argument("--num-kv-heads", type=int, required=True,
                             help="Number of KV heads (e.g. 8 for Llama 3.2 1B)")
    fkvd_parser.add_argument("--head-dim", type=int, default=64,
                             help="Per-head dimension (only 64 supported)")
    fkvd_parser.add_argument("--seq-len", type=int, required=True,
                             help="KV cache sequence length")
    fkvd_parser.add_argument("--chunk-size", type=int, default=32,
                             help="K/V chunk size for streaming (default: 32)")
    fkvd_parser.add_argument("--num-cols", type=int, default=4,
                             help="Number of AIE columns (default: 4)")
    fkvd_parser.add_argument("--out", type=str,
                             help="Output directory (default: cache)")

    # SwiGLU prefill INT8 subcommand
    swpi_parser = subparsers.add_parser(
        "swiglu-prefill-int8",
        help="Compile fused SwiGLU FFN (M>=32 prefill, W8A8 INT8)"
    )
    swpi_parser.add_argument("--seq-len", type=int, required=True)
    swpi_parser.add_argument("--embedding-dim", type=int, required=True)
    swpi_parser.add_argument("--hidden-dim", type=int, required=True)
    swpi_parser.add_argument("--num-aie-columns", type=int, default=4)
    swpi_parser.add_argument("--tile-m", type=int, default=None,
                             help="Override tile_m for both inner GEMMs (min 16 for INT8)")
    swpi_parser.add_argument("--tile-n", type=int, default=None,
                             help="Override tile_n for both inner GEMMs "
                                  "(default: auto-selected for shape compatibility)")
    swpi_parser.add_argument("--out", type=str,
                             help="Output directory (default: cache)")

    # AttentionBlockPrefill subcommand (11-kernel chained composite)
    attnp_parser = subparsers.add_parser(
        "attention-prefill",
        help="Compile fused attention block prefill (RMSNorm + QKV + RoPE + "
             "MHA + O + residual, 11 kernels chained into one xclbin)",
    )
    attnp_parser.add_argument("--seq-len", type=int, required=True,
                              help="Prompt seq length (rounded up to bucket)")
    attnp_parser.add_argument("--embed-dim", type=int, required=True,
                              help="Model hidden size")
    attnp_parser.add_argument("--num-heads", type=int, required=True,
                              help="Q head count")
    attnp_parser.add_argument("--num-kv-heads", type=int, required=True,
                              help="K/V head count (GQA group = H / KV)")
    attnp_parser.add_argument("--head-dim", type=int, default=64,
                              help="Per-head dim (only 64 supported by MHA)")
    attnp_parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    attnp_parser.add_argument("--rope-method-type", type=int, default=0,
                              choices=[0, 1],
                              help="RoPE rotation: 0=TWO_HALVES (HF/ggml NEOX), "
                                   "1=INTERLEAVED (ggml NORMAL adjacent-pair)")
    attnp_parser.add_argument("--cache-dir", type=str, default=None,
                              help="Override GGML_XDNA_CACHE_DIR for this run")
    attnp_parser.add_argument("--out", type=str,
                              help="Output directory (default: cache)")

    # TransformerBlockPrefill subcommand (17-kernel chained composite)
    tblockp_parser = subparsers.add_parser(
        "transformer-block-prefill",
        help="Compile fused transformer block prefill (attention + SwiGLU FFN "
             "+ two residuals, 17 kernels chained into one xclbin)",
    )
    tblockp_parser.add_argument("--seq-len", type=int, required=True,
                                help="Prompt seq length (rounded up to bucket)")
    tblockp_parser.add_argument("--embed-dim", type=int, required=True,
                                help="Model hidden size")
    tblockp_parser.add_argument("--num-heads", type=int, required=True,
                                help="Q head count")
    tblockp_parser.add_argument("--num-kv-heads", type=int, required=True,
                                help="K/V head count (GQA group = H / KV)")
    tblockp_parser.add_argument("--head-dim", type=int, default=64,
                                help="Per-head dim (only 64 supported by MHA)")
    tblockp_parser.add_argument("--ffn-hidden", type=int, required=True,
                                help="FFN intermediate dim (gate/up N, down K)")
    tblockp_parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    tblockp_parser.add_argument("--rope-method-type", type=int, default=0,
                                choices=[0, 1],
                                help="RoPE rotation: 0=TWO_HALVES (HF/ggml NEOX), "
                                     "1=INTERLEAVED (ggml NORMAL adjacent-pair)")
    tblockp_parser.add_argument("--cache-dir", type=str, default=None,
                                help="Override GGML_XDNA_CACHE_DIR for this run")
    tblockp_parser.add_argument("--out", type=str,
                                help="Output directory (default: cache)")

    # TransformerBlockPrefillFused subcommand — Layer 4A Phase 2 monolithic
    # ELF variant of the above. Produces a single-ELF bundle (tblock_fused.elf
    # + tblock_fused.layout.json) instead of chained combined.xclbin + N insts.
    tblockf_parser = subparsers.add_parser(
        "transformer-block-prefill-fused",
        help="Compile fully-fused monolithic transformer block prefill "
             "(attn + SwiGLU FFN, single @main runtime_sequence -> single "
             "xrt::run per layer).",
    )
    tblockf_parser.add_argument("--seq-len", type=int, required=True,
                                help="Prompt seq length (rounded up to bucket)")
    tblockf_parser.add_argument("--embed-dim", type=int, required=True,
                                help="Model hidden size")
    tblockf_parser.add_argument("--num-heads", type=int, required=True,
                                help="Q head count")
    tblockf_parser.add_argument("--num-kv-heads", type=int, required=True,
                                help="K/V head count (GQA group = H / KV)")
    tblockf_parser.add_argument("--head-dim", type=int, default=64,
                                help="Per-head dim (only 64 supported by MHA)")
    tblockf_parser.add_argument("--ffn-hidden", type=int, required=True,
                                help="FFN intermediate dim (gate/up N, down K)")
    tblockf_parser.add_argument("--dtype", default="bf16", choices=["bf16"])
    tblockf_parser.add_argument("--rope-method-type", type=int, default=0,
                                choices=[0, 1],
                                help="RoPE rotation: 0=TWO_HALVES (HF/ggml NEOX), "
                                     "1=INTERLEAVED (ggml NORMAL adjacent-pair)")
    tblockf_parser.add_argument("--num-layers", type=int, default=1,
                                help="Number of consecutive transformer blocks "
                                     "packed into one ELF (Layer 4B multi-layer "
                                     "packing). Default 1 preserves the Phase "
                                     "3.7 bundle format.")
    tblockf_parser.add_argument("--w8a16", action="store_true",
                                help="W8A16 mode: INT8 attention projections "
                                     "(Wq/Wk/Wv/Wo), bf16 FFN. Bundle is named "
                                     "tblock_fused_w8a16.{elf,layout.json} "
                                     "and its cache key is disambiguated from "
                                     "the bf16 variant.")
    tblockf_parser.add_argument("--w8a16-ffn", action="store_true",
                                help="Phase B: additionally quantize the FFN "
                                     "GEMMs (gate/up/down) to INT8. Requires "
                                     "--w8a16. Bundle is named "
                                     "tblock_fused_w8a16ffn.{elf,layout.json}.")
    tblockf_parser.add_argument("--cache-dir", type=str, default=None,
                                help="Override GGML_XDNA_CACHE_DIR for this run")
    tblockf_parser.add_argument("--out", type=str,
                                help="Output directory (default: cache)")

    args = parser.parse_args()

    if args.op == "gemm":
        if args.out:
            path = compile_gemm(
                args.M, args.K, args.N,
                args.dtype_in, args.dtype_out,
                args.num_aie_columns, args.out,
                args.b_col_maj,
            )
        else:
            path = compile_gemm_cached(
                args.M, args.K, args.N,
                args.dtype_in, args.dtype_out,
                args.num_aie_columns,
                args.b_col_maj,
            )
        if not args.quiet:
            print(path)
    elif args.op == "gemv":
        if args.out:
            path = compile_gemv(
                args.N, args.K,
                args.dtype_in, args.dtype_out,
                args.num_aie_columns, args.out,
            )
        else:
            path = compile_gemv_cached(
                args.N, args.K,
                args.dtype_in, args.dtype_out,
                args.num_aie_columns,
            )
        if not args.quiet:
            print(path)
    elif args.op == "swiglu-decode":
        if args.out:
            path = compile_swiglu_decode(
                args.embedding_dim, args.hidden_dim,
                args.dtype, args.num_aie_columns, args.out,
            )
        else:
            path = compile_swiglu_decode_cached(
                args.embedding_dim, args.hidden_dim,
                args.dtype, args.num_aie_columns,
            )
        if not args.quiet:
            print(path)
    elif args.op == "swiglu-decode-int8":
        if args.out:
            path = compile_swiglu_decode_int8(
                args.embedding_dim, args.hidden_dim,
                args.num_aie_columns, args.out,
                group_size=args.group_size,
            )
        else:
            path = compile_swiglu_decode_int8_cached(
                args.embedding_dim, args.hidden_dim,
                args.num_aie_columns,
                group_size=args.group_size,
            )
        if not args.quiet:
            print(path)
    elif args.op == "swiglu-fused-int8":
        if args.out:
            path = compile_swiglu_fused_int8(
                args.embedding_dim, args.hidden_dim,
                args.num_aie_columns, args.out,
                group_size=args.group_size,
            )
        else:
            path = compile_swiglu_fused_int8_cached(
                args.embedding_dim, args.hidden_dim,
                args.num_aie_columns,
                group_size=args.group_size,
            )
        if not args.quiet:
            print(path)
    elif args.op == "qkv":
        if args.out:
            path = compile_qkv(
                args.embedding_dim, args.q_dim, args.k_dim, args.v_dim,
                args.num_aie_columns, args.out,
            )
        else:
            path = compile_qkv_cached(
                args.embedding_dim, args.q_dim, args.k_dim, args.v_dim,
                args.num_aie_columns,
            )
        if not args.quiet:
            print(path)
    elif args.op == "swiglu-prefill":
        if args.out:
            path = compile_swiglu_prefill(
                args.seq_len, args.embedding_dim, args.hidden_dim,
                args.dtype, args.num_aie_columns, args.out,
                tile_m=args.tile_m, tile_n=args.tile_n,
            )
        else:
            path = compile_swiglu_prefill_cached(
                args.seq_len, args.embedding_dim, args.hidden_dim,
                args.dtype, args.num_aie_columns,
                tile_m=args.tile_m, tile_n=args.tile_n,
            )
        if not args.quiet:
            print(path)
    elif args.op == "swiglu-prefill-int8":
        if args.out:
            path = compile_swiglu_prefill_int8(
                args.seq_len, args.embedding_dim, args.hidden_dim,
                args.num_aie_columns, args.out,
                tile_m=args.tile_m, tile_n=args.tile_n,
            )
        else:
            path = compile_swiglu_prefill_int8_cached(
                args.seq_len, args.embedding_dim, args.hidden_dim,
                args.num_aie_columns,
                tile_m=args.tile_m, tile_n=args.tile_n,
            )
        if not args.quiet:
            print(path)
    elif args.op == "rms_norm":
        if args.cache_dir:
            os.environ["GGML_XDNA_CACHE_DIR"] = args.cache_dir
        if args.out:
            path = compile_rms_norm(
                args.size, args.dtype, args.num_aie_columns,
                args.num_channels, args.tile_size, args.weighted,
                args.out,
            )
        else:
            path = compile_rms_norm_cached(
                args.size, args.dtype, args.num_aie_columns,
                args.num_channels, args.tile_size, args.weighted,
            )
        if not args.quiet:
            print(path)
    elif args.op == "flowkv-decode":
        if args.out:
            path = compile_flowkv_decode(
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.seq_len, args.chunk_size, args.num_cols,
                args.out,
            )
        else:
            path = compile_flowkv_decode_cached(
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.seq_len, args.chunk_size, args.num_cols,
            )
        if not args.quiet:
            print(path)
    elif args.op == "attention-prefill":
        if args.cache_dir:
            os.environ["GGML_XDNA_CACHE_DIR"] = args.cache_dir
        if args.out:
            path = compile_attention_prefill(
                args.seq_len, args.embed_dim,
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.dtype, args.out,
                rope_method_type=args.rope_method_type,
            )
        else:
            path = compile_attention_prefill_cached(
                args.seq_len, args.embed_dim,
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.dtype,
                rope_method_type=args.rope_method_type,
            )
        if not args.quiet:
            print(path)
    elif args.op == "transformer-block-prefill":
        if args.cache_dir:
            os.environ["GGML_XDNA_CACHE_DIR"] = args.cache_dir
        if args.out:
            path = compile_transformer_block_prefill(
                args.seq_len, args.embed_dim,
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.ffn_hidden,
                args.dtype, args.out,
                rope_method_type=args.rope_method_type,
            )
        else:
            path = compile_transformer_block_prefill_cached(
                args.seq_len, args.embed_dim,
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.ffn_hidden,
                args.dtype,
                rope_method_type=args.rope_method_type,
            )
        if not args.quiet:
            print(path)
    elif args.op == "transformer-block-prefill-fused":
        if args.cache_dir:
            os.environ["GGML_XDNA_CACHE_DIR"] = args.cache_dir
        if args.w8a16_ffn and not args.w8a16:
            parser.error("--w8a16-ffn requires --w8a16")
        if args.out:
            path = compile_transformer_block_prefill_fused(
                args.seq_len, args.embed_dim,
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.ffn_hidden,
                args.dtype, args.out,
                rope_method_type=args.rope_method_type,
                num_layers=args.num_layers,
                w8a16=args.w8a16,
                w8a16_ffn=args.w8a16_ffn,
            )
        else:
            path = compile_transformer_block_prefill_fused_cached(
                args.seq_len, args.embed_dim,
                args.num_heads, args.num_kv_heads, args.head_dim,
                args.ffn_hidden,
                args.dtype,
                rope_method_type=args.rope_method_type,
                num_layers=args.num_layers,
                w8a16=args.w8a16,
                w8a16_ffn=args.w8a16_ffn,
            )
        if not args.quiet:
            print(path)


if __name__ == "__main__":
    import traceback
    from datetime import datetime
    try:
        main()
    except Exception as e:
        with open("compile_error.log", "a") as f:
            f.write(f"\n--- ERROR at {datetime.now()} ---\n")
            traceback.print_exc(file=f)
        sys.exit(1)
