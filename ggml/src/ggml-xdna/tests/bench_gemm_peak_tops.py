#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: measure peak achievable INT8 GEMM TOPS on compute-optimal shapes.

Background
----------
project_int8_prefill.md observed 5.1 TOPS on the 8B gate/up shape
(K=4096, N=12288, M=256). The 32-tile aie2p theoretical peak for INT8
is ~4-16 TOPS depending on op counting convention, so 5.1 might already
be near peak — OR the asymmetric N=12288 across 8 columns is wasting
tiles and there's headroom.

This script tests that by running square/balanced shapes where each
column has identical, large, well-aligned N work. If we still see ~5
TOPS, we're at practical peak and the only lever is fewer dispatches
(bigger fused xclbins = option L). If we see 10-15 TOPS, the gate/up
geometry specifically is leaving perf on the table and there's
low-hanging fruit in tile/column selection.

Shapes tested
-------------
  A)  M=256,  K=4096, N=4096   -- square, standard
  B)  M=512,  K=4096, N=4096   -- bigger M, fills pipeline more
  C)  M=1024, K=4096, N=4096   -- even bigger M (may hit BD overflow)
  D)  M=256,  K=4096, N=8192   -- 2x N, still multiple of 64*8
  E)  M=256,  K=4096, N=12288  -- reference point matching the 8B lever bench

All shapes satisfy M % (tile_m * 4) == 0 with tile_m in {16, 32, 64}
and N % (tile_n * 8) == 0 with tile_n=64.

Output
------
Per-shape: npu_us median, GOPs, TOPS, % of 16-TOPS-peak.
"""

import statistics
import sys
import time

import aie.utils as aie_utils
import torch

from iron.common import AIEContext
from iron.operators.gemm.op import GEMM
from iron.operators.gemm.reference import generate_golden_reference

NUM_AIE_COLUMNS = 8
WARMUP_ITERS = 2
TIMED_ITERS = 5

# (label, M, K, N)
SHAPES = [
    ("A_square_256",    256,  4096, 4096),
    ("B_square_512",    512,  4096, 4096),
    ("C_square_1024",  1024,  4096, 4096),
    ("D_M256_N8192",    256,  4096, 8192),
    ("E_8B_gateup",     256,  4096, 12288),  # reference: the lever bench shape
]

# int8 min_tile_m=16, min_tile_n=16, min_tile_k=8
def pick_tile_m(M: int) -> int:
    for tm in (64, 32, 16):
        if M % (tm * 4) == 0:
            return tm
    raise ValueError(f"no tile_m for M={M}")


def pick_tile_n(N: int) -> int:
    for tn in (64, 32, 16):
        if N % (tn * NUM_AIE_COLUMNS) == 0:
            return tn
    raise ValueError(f"no tile_n for N={N} cols={NUM_AIE_COLUMNS}")


def bench_one(M: int, K: int, N: int, ctx: AIEContext):
    tm = pick_tile_m(M)
    tn = pick_tile_n(N)
    tk = 64 if K % 64 == 0 else 32

    op = GEMM(
        M=M, K=K, N=N,
        tile_m=tm, tile_k=tk, tile_n=tn,
        num_aie_columns=NUM_AIE_COLUMNS,
        b_col_maj=False, c_col_maj=False,
        dtype_in="i8", dtype_out="i32",
        context=ctx,
    )
    op.compile()
    op_func = op.get_callable()

    golden = generate_golden_reference(
        M=M, K=K, N=N, dtype="i8", dtype_out="i32",
        b_col_maj=False, c_col_maj=False,
    )
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    arg_spec = op.get_arg_spec()
    A = XRTTensor.from_torch(golden["input"].flatten())
    B = XRTTensor.from_torch(golden["input_b"][0].flatten())
    C = XRTTensor(arg_spec[2].shape, dtype=arg_spec[2].dtype)

    for _ in range(WARMUP_ITERS):
        op_func(A, B, C)

    npu_us = []
    for _ in range(TIMED_ITERS):
        r = op_func(A, B, C)
        npu_us.append(r.npu_time / 1e3)

    return statistics.median(npu_us), tm, tk, tn


def main():
    dev = aie_utils.get_current_device()
    print(f"\nDevice: {dev.resolve().name}  num_aie_columns={NUM_AIE_COLUMNS}")
    print(f"Warmup: {WARMUP_ITERS}  Timed: {TIMED_ITERS}  (median reported)")
    print(f"Per-tile peak (aie2p int8 8x8x8 @ 1GHz): 128 GOP/s")
    print(f"32-tile theoretical peak: 4.1 TOPS (1 op/MAC) or 8.2 TOPS (2 ops/MAC)\n")

    ctx = AIEContext(mlir_verbose=False)
    results = []
    for label, M, K, N in SHAPES:
        try:
            npu_us, tm, tk, tn = bench_one(M, K, N, ctx)
        except Exception as e:
            print(f"[FAIL] {label} M={M} K={K} N={N}: {e}")
            continue
        gops = 2.0 * M * K * N / 1e9
        tops = gops / (npu_us / 1e6) / 1e3
        results.append((label, M, K, N, npu_us, gops, tops, tm, tk, tn))
        print(f"[OK] {label:18s} M={M:4d} K={K:4d} N={N:5d}  "
              f"tile=({tm},{tk},{tn})  "
              f"npu={npu_us:7.1f}us  {gops:6.2f} GOps  {tops:5.2f} TOPS")

    print("\n" + "=" * 78)
    print("SUMMARY  (TOPS = ops/sec measured, 2 ops per MAC convention)")
    print("=" * 78)
    hdr = f"{'shape':<18} {'M':>5} {'K':>5} {'N':>6} {'npu_us':>9} {'TOPS':>7} {'% 8T peak':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, M, K, N, npu_us, gops, tops, *_ in results:
        pct = tops / 8.2 * 100
        print(f"{label:<18} {M:>5} {K:>5} {N:>6} {npu_us:>9.1f} {tops:>7.2f} {pct:>9.1f}%")

    print("\nInterpretation:")
    print("  - If square shapes (A/B/C) hit ~same TOPS as 8B gate/up (E): we're at")
    print("    NPU practical peak. Only lever left is dispatch-count reduction (L).")
    print("  - If square shapes hit 2x+ of gate/up: N=12288 asymmetry across 8 cols")
    print("    is wasting tiles. Tune tile_n/num_cols for skewed shapes; low-hanging.")


if __name__ == "__main__":
    main()
