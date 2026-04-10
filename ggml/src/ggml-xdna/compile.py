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
        gemm_kwargs["emulate_bf16_mmul_with_bfp16"] = True

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


if __name__ == "__main__":
    main()
