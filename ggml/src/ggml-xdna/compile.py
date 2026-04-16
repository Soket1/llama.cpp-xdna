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
                             tile_m: int | None = None) -> str:
    """Cache key for a SwiGLU prefill configuration.

    tile_m is a distinguishing input because it changes the compiled xclbin
    (different MAC tiling). tile_k/tile_n are left at IRON's defaults and
    omitted from the key — revisit if we ever vary them too.
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


def select_swiglu_prefill_tile_m(seq_len: int) -> int:
    """Pick the largest valid tile_m for a given prefill seq_len.

    IRON GEMM requires seq_len % (tile_m * 4) == 0 with tile_m in {64,32,16,8}.
    Returns 0 if no candidate works (caller should fall back / reject).
    """
    for tm in (64, 32, 16, 8):
        if seq_len % (tm * 4) == 0:
            return tm
    return 0


# Sub-kernel attribute prefixes inside the chained SwiGLU xclbin. These must
# match the names passed to chain_swiglu_artifacts in iron/operators/swiglu_*/op.py.
# The C++ backend looks up insts files by these names.
SWIGLU_DECODE_KERNELS = ("gemv_1", "silu", "eltwise_mul", "gemv_2")
SWIGLU_PREFILL_KERNELS = ("gemm_1", "silu", "eltwise_mul", "gemm_2")
# W8A16 (INT8 weights + bf16 activations) SwiGLU decode. Sub-op prefixes come
# from iron/operators/swiglu_decode_int8/op.py chain_swiglu_artifacts() call.
# SiLU / eltwise_mul share the same compiled binaries as the bf16 chain
# (they operate on bf16 intermediates), but are keyed under their int8-namespace
# insts files inside this bundle — the C++ backend loads them via distinct tags.
SWIGLU_DECODE_INT8_KERNELS = ("gemv_int8_1", "silu", "eltwise_mul", "gemv_int8_2")
# Fused gate+up+silu+mul + standalone down GEMV, chained into one xclbin.
# Names must match chain_swiglu_artifacts() call in chained.py.
SWIGLU_FUSED_CHAINED_KERNELS = ("fused", "down_gemv_int8")


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
        # bf16 4x8x8 MAC in IRON's aie2p mm.cc requires tile_m % (2*r) == 0 with r=4,
        # so minimum tile_m is 8 (not 4). Attempting 4 triggers a static_assert.
        min_m, min_k, min_n = 8, 8, 8

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
    # AIE2p core-tile usable budget is ~32KB — cap tile_in accordingly so we
    # don't blow past L1. Prefer larger tile_in for throughput when it fits.
    l1_budget_bytes = 32 * 1024
    bf16 = 2
    max_tsi_by_l1 = max(1, l1_budget_bytes // (2 * K * bf16))  # 2 = double-buffer
    tsi_candidates = [8, 4, 2, 1]
    tile_in = None
    for tsi in tsi_candidates:
        if (tsi <= tile_out and tsi <= max_tsi_by_l1
                and tile_out % tsi == 0 and per_col % tsi == 0):
            tile_in = tsi
            break
    if tile_in is None:
        raise ValueError(f"Cannot find tile_size_input for N={N}, K={K}, cols={num_aie_columns}")

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
                                   tile_m: int | None = None) -> None:
    """Verify a SwiGLU prefill shape is dispatchable.

    If tile_m is None, IRON defaults to 64 → min_M = 256. If overridden, the
    constraint loosens to seq_len % (tile_m * 4) == 0. tile_k/tile_n remain
    at IRON's default of 64.
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
    if hidden_dim % (64 * num_aie_columns) != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be a multiple of 64*num_aie_columns="
            f"{64 * num_aie_columns} (gemm_1 tile_n*cols)"
        )
    if embedding_dim % (64 * num_aie_columns) != 0:
        raise ValueError(
            f"embedding_dim={embedding_dim} must be a multiple of 64*num_aie_columns="
            f"{64 * num_aie_columns} (gemm_2 tile_n*cols)"
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
    actual_cols = aie_utils.get_current_device().cols
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.swiglu_decode.op import SwiGLUDecode

    op = SwiGLUDecode(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    op.compile()

    _stage_swiglu_artifacts(op, SWIGLU_DECODE_KERNELS, output_dir)
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
    actual_cols = aie_utils.get_current_device().cols
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.swiglu_decode_int8.op import SwiGLUDecodeInt8

    op = SwiGLUDecodeInt8(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        group_size=group_size,
    )
    op.compile()

    _stage_swiglu_artifacts(op, SWIGLU_DECODE_INT8_KERNELS, output_dir)
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

    actual_cols = aie_utils.get_current_device().cols
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.swiglu_decode_int8_fused.chained import (
        SwiGLUDecodeInt8FusedChained,
    )

    op = SwiGLUDecodeInt8FusedChained(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        group_size=group_size,
    )
    op.compile()

    _stage_swiglu_artifacts(op, SWIGLU_FUSED_CHAINED_KERNELS, output_dir)
    return output_dir


def compile_swiglu_prefill(seq_len: int, embedding_dim: int, hidden_dim: int,
                           dtype: str, num_aie_columns: int, output_dir: str,
                           tile_m: int | None = None) -> str:
    """Compile an IRON SwiGLU prefill operator and stage its artifacts.

    Output layout mirrors compile_swiglu_decode but with gemm_1/gemm_2 kernel names.
    When tile_m is provided it is forwarded to SwiGLUPrefill, overriding the
    default (64) in both inner GEMMs. Requires the IRON PR adding tile
    overrides on SwiGLUPrefill to be merged locally.
    """
    if dtype != "bf16":
        raise ValueError(f"SwiGLU currently supports only bf16, got {dtype}")
    validate_swiglu_prefill_shapes(seq_len, embedding_dim, hidden_dim,
                                   num_aie_columns, tile_m=tile_m)

    actual_cols = aie_utils.get_current_device().cols
    if actual_cols != num_aie_columns:
        raise ValueError(
            f"Device column mismatch: requested {num_aie_columns}, current "
            f"device reports {actual_cols}. Compile on the target device."
        )

    from iron.operators.swiglu_prefill.op import SwiGLUPrefill

    kwargs = dict(
        seq_len=seq_len, embedding_dim=embedding_dim, hidden_dim=hidden_dim
    )
    if tile_m is not None:
        kwargs["tile_m"] = tile_m

    op = SwiGLUPrefill(**kwargs)
    op.compile()

    _stage_swiglu_artifacts(op, SWIGLU_PREFILL_KERNELS, output_dir)
    return output_dir


def _stage_swiglu_artifacts(op, kernel_names: tuple[str, ...], output_dir: str) -> None:
    """Copy a compiled SwiGLU op's combined xclbin + per-kernel insts to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    build_dir = op.context.build_dir

    combined_src = build_dir / op.combined_xclbin.filename
    shutil.copy2(str(combined_src), os.path.join(output_dir, "combined.xclbin"))

    for name in kernel_names:
        insts_artifact = getattr(op, f"{name}_insts")
        src = build_dir / insts_artifact.filename
        dst = os.path.join(output_dir, f"swiglu_{name}.insts")
        shutil.copy2(str(src), dst)


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


def get_cached_swiglu_dir(cache_key: str, kernel_names: tuple[str, ...]) -> Path | None:
    """Check if a SwiGLU bundle is cached. Returns the directory if found, else None.

    The combined.xclbin and *all* expected per-kernel insts files must exist.
    """
    cache_dir = get_cache_dir() / cache_key
    if not (cache_dir / "combined.xclbin").exists():
        return None
    for name in kernel_names:
        if not (cache_dir / f"swiglu_{name}.insts").exists():
            return None
    return cache_dir


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


def compile_swiglu_prefill_cached(seq_len: int, embedding_dim: int, hidden_dim: int,
                                  dtype: str = "bf16",
                                  num_aie_columns: int = 4,
                                  tile_m: int | None = None) -> Path:
    """Compile a SwiGLU prefill operator with caching.

    Returns path to the cache directory containing combined.xclbin + 4 insts files.
    """
    key = swiglu_prefill_cache_key(
        seq_len, embedding_dim, hidden_dim, dtype, num_aie_columns, tile_m=tile_m
    )

    cached = get_cached_swiglu_dir(key, SWIGLU_PREFILL_KERNELS)
    if cached is not None:
        return cached

    output_dir = str(get_cache_dir() / key)
    compile_swiglu_prefill(
        seq_len, embedding_dim, hidden_dim, dtype, num_aie_columns, output_dir,
        tile_m=tile_m,
    )
    return Path(output_dir)


# ---------------------------------------------------------------------------
# CLI entry point (called by C++ backend)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ggml-xdna IRON compilation bridge")
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
    swp_parser.add_argument("--out", type=str,
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
        print(path)
    elif args.op == "swiglu-prefill":
        if args.out:
            path = compile_swiglu_prefill(
                args.seq_len, args.embedding_dim, args.hidden_dim,
                args.dtype, args.num_aie_columns, args.out,
                tile_m=args.tile_m,
            )
        else:
            path = compile_swiglu_prefill_cached(
                args.seq_len, args.embedding_dim, args.hidden_dim,
                args.dtype, args.num_aie_columns,
                tile_m=args.tile_m,
            )
        print(path)


if __name__ == "__main__":
    main()
