#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic benchmark: GEMM latency scaling with M at the 8B gate/up shape.

Background
----------
project_int8_prefill.md closed the 8B INT8 prefill investigation at M=64
with the finding that both bf16 and INT8 achieve ~2 TOPS absolute (same
wall-clock latency band), ruling out dispatch overhead as the cause.
The one untested hypothesis from that memo is:

    "Try larger M (M=128, 256) to increase arithmetic intensity — if
     bandwidth-bound, larger M should help INT8 more than bf16."

This script tests that directly. It runs a single IRON GEMM at the 8B
gate/up shape (K=4096, N=12288) across M in {64, 128, 256, 512}, for
both bf16->bf16 and i8->i32 inputs, on 8 AIE columns (NPU2).

Interpretation
--------------
Denote `t_bf16(M)` and `t_i8(M)` as median NPU-side latencies.

  * If t_bf16 grows roughly linearly with M but t_i8 is sub-linear,
    the kernel is DDR-bound: weights dominate the DMA, INT8 halves them,
    so INT8 effective throughput climbs with M. This is a real lever:
    rewire prefill to batch more tokens per dispatch (and eat the
    padding cost — still a net win).

  * If both grow linearly with similar slope, the kernel is
    compute/core-bound at this shape, not bandwidth-bound. INT8 prefill
    at 8B is genuinely closed; nothing at the dispatch layer helps.
    (A kernel-level rewrite of mm_i8.cc would be the only move, which
    is a separate effort from this plan.)

Output is a plain-text table; no side effects beyond the IRON compile
cache and the test's own XCLBINs. The operator is compiled once per
(M, dtype_in) combination, so subsequent runs are fast.
"""

import statistics
import sys
import time
from pathlib import Path

import aie.utils as aie_utils
import torch

from iron.common import AIEContext
from iron.operators.gemm.op import GEMM
from iron.operators.gemm.reference import generate_golden_reference


# 8B gate/up projection shape (Qwen3.5-8B gate/up: K=embedding_dim, N=hidden_dim).
K = 4096
N = 12288

# NPU2 has 8 columns; match what the ggml-xdna backend would pick in prod.
NUM_AIE_COLUMNS = 8

# M values to sweep. Must satisfy tile_m*4 | M for the selected tile_m:
#   M=64  -> tile_m=16 (i8 & bf16)   64  = 16*4
#   M=128 -> tile_m=32               128 = 32*4
#   M=256 -> tile_m=64               256 = 64*4
#   M=512 -> tile_m=64 works too     512 = 64*4*2  (2 tile-rows, safer for GEMM)
# Order ascending so cache hits help subsequent iterations only slightly.
M_VALUES = [64, 128, 256, 512]

WARMUP_ITERS = 2
TIMED_ITERS = 5

# Same tile selection rule the bridge (compile.py:select_gemm_tiles) uses.
def pick_tile_m(M: int, dtype_in: str) -> int:
    min_m = 16 if dtype_in == "i8" else 8
    for tm in (64, 32, 16, 8):
        if tm >= min_m and M % (tm * 4) == 0:
            return tm
    raise ValueError(f"no tile_m for M={M} dtype_in={dtype_in}")


def pick_tile_k(K: int, dtype_in: str) -> int:
    min_k = 8 if dtype_in == "i8" else 8
    for tk in (64, 32, 16, 8):
        if tk >= min_k and K % tk == 0:
            return tk
    raise ValueError(f"no tile_k for K={K}")


def pick_tile_n(N: int, dtype_in: str, num_cols: int) -> int:
    min_n = 16 if dtype_in == "i8" else 8
    for tn in (64, 32, 16, 8):
        if tn >= min_n and N % (tn * num_cols) == 0:
            return tn
    raise ValueError(f"no tile_n for N={N} num_cols={num_cols}")


def bench_one(M: int, dtype_in: str, dtype_out: str, ctx: AIEContext):
    """Compile GEMM(M,K,N) once, warm, then time TIMED_ITERS dispatches.

    Returns (npu_us_median, wall_us_median, tile_m, tile_k, tile_n).
    """
    tm = pick_tile_m(M, dtype_in)
    tk = pick_tile_k(K, dtype_in)
    tn = pick_tile_n(N, dtype_in, NUM_AIE_COLUMNS)

    # GEMM op matching the production-path settings used by swiglu_prefill_int8
    # and the standalone GEMM prefill (ggml-xdna.cpp): no bf16s accum emulation,
    # row-major A/B/C, prio_accuracy only for bf16 (int8 rejects that flag).
    op_kwargs = dict(
        M=M,
        K=K,
        N=N,
        tile_m=tm,
        tile_k=tk,
        tile_n=tn,
        num_aie_columns=NUM_AIE_COLUMNS,
        b_col_maj=False,
        c_col_maj=False,
        dtype_in=dtype_in,
        dtype_out=dtype_out,
        context=ctx,
    )
    if dtype_in == "bf16":
        op_kwargs["prio_accuracy"] = True
        op_kwargs["emulate_bf16_mmul_with_bfp16"] = False

    op = GEMM(**op_kwargs)
    op.compile()
    op_func = op.get_callable()

    # Generate input buffers. We don't verify correctness here — the test.py
    # suite already covers that. We just need plausibly-sized buffers.
    golden = generate_golden_reference(
        M=M, K=K, N=N, dtype=dtype_in, dtype_out=dtype_out,
        b_col_maj=False, c_col_maj=False,
    )

    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    arg_spec = op.get_arg_spec()

    A = XRTTensor.from_torch(golden["input"].flatten())
    B = XRTTensor.from_torch(golden["input_b"][0].flatten())
    C = XRTTensor(arg_spec[2].shape, dtype=arg_spec[2].dtype)

    # Warmup
    for _ in range(WARMUP_ITERS):
        op_func(A, B, C)

    npu_us = []
    wall_us = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        result = op_func(A, B, C)
        t1 = time.perf_counter()
        npu_us.append(result.npu_time / 1e3)
        wall_us.append((t1 - t0) * 1e6)

    return (statistics.median(npu_us),
            statistics.median(wall_us),
            tm, tk, tn)


def gflops(M: int, lat_us: float) -> float:
    """Useful-work throughput. bf16 and int8 use the same 2*M*K*N FLOP count
    (mul+add per MAC); i8 should be called ops/s but we keep the same unit
    for an apples-to-apples ratio."""
    return (2.0 * M * K * N) / (lat_us * 1e-6) / 1e9


def bytes_moved(M: int, dtype_in: str, dtype_out: str) -> int:
    bpe_in = 1 if dtype_in == "i8" else 2
    bpe_out = 4 if dtype_out == "i32" else 2
    return M * K * bpe_in + K * N * bpe_in + M * N * bpe_out


def gbps(M: int, dtype_in: str, dtype_out: str, lat_us: float) -> float:
    return bytes_moved(M, dtype_in, dtype_out) / (lat_us * 1e-6) / 1e9


def main():
    dev = aie_utils.get_current_device()
    cols = dev.cols
    devname = dev.resolve().name
    print(f"\nDevice: {devname} (max cols={cols}). Using num_aie_columns={NUM_AIE_COLUMNS}.")
    if cols < NUM_AIE_COLUMNS:
        print(f"WARNING: requested {NUM_AIE_COLUMNS} cols but device has {cols}. "
              f"Falling back to {cols}.", file=sys.stderr)
        globals()["NUM_AIE_COLUMNS"] = cols

    print(f"Shape:  K={K} N={N}  (Qwen3.5-8B gate/up class)")
    print(f"Warmup: {WARMUP_ITERS} iters  Timed: {TIMED_ITERS} iters  (median reported)\n")

    results = {}  # (dtype_in, M) -> (npu_us, wall_us, tm, tk, tn)
    for dtype_in, dtype_out in (("bf16", "bf16"), ("i8", "i32")):
        ctx = AIEContext(mlir_verbose=False)
        for M in M_VALUES:
            tag = f"{dtype_in}->{dtype_out} M={M}"
            try:
                npu_us, wall_us, tm, tk, tn = bench_one(M, dtype_in, dtype_out, ctx)
            except Exception as e:
                print(f"[FAIL] {tag}: {e}")
                continue
            results[(dtype_in, M)] = (npu_us, wall_us, tm, tk, tn)
            print(f"[OK]   {tag}  tile=({tm},{tk},{tn})  "
                  f"npu={npu_us:8.1f}us  wall={wall_us:8.1f}us")
        aie_utils.DefaultNPURuntime.cleanup()

    # Summary tables.
    print("\n" + "=" * 80)
    print("LATENCY (NPU-side, us, median of {} iters)".format(TIMED_ITERS))
    print("=" * 80)
    hdr = f"{'M':>6}  {'bf16 npu_us':>14}  {'i8 npu_us':>14}  {'i8/bf16 ratio':>16}"
    print(hdr)
    print("-" * len(hdr))
    for M in M_VALUES:
        bf = results.get(("bf16", M))
        i8 = results.get(("i8", M))
        if bf and i8:
            ratio = i8[0] / bf[0]
            print(f"{M:>6}  {bf[0]:>14.1f}  {i8[0]:>14.1f}  {ratio:>16.3f}")

    print("\n" + "=" * 80)
    print("THROUGHPUT (GFLOP/s or GOP/s)  --  MAC throughput")
    print("=" * 80)
    print(f"{'M':>6}  {'bf16 GFLOP/s':>14}  {'i8 GOP/s':>14}  {'i8/bf16':>10}")
    for M in M_VALUES:
        bf = results.get(("bf16", M))
        i8 = results.get(("i8", M))
        if bf and i8:
            g_bf = gflops(M, bf[0])
            g_i8 = gflops(M, i8[0])
            print(f"{M:>6}  {g_bf:>14.2f}  {g_i8:>14.2f}  {g_i8/g_bf:>10.3f}")

    print("\n" + "=" * 80)
    print("EFFECTIVE DMA BANDWIDTH (GB/s total host<->device traffic)")
    print("=" * 80)
    print(f"{'M':>6}  {'bf16 GB/s':>12}  {'i8 GB/s':>12}")
    for M in M_VALUES:
        bf = results.get(("bf16", M))
        i8 = results.get(("i8", M))
        if bf and i8:
            b_bf = gbps(M, "bf16", "bf16", bf[0])
            b_i8 = gbps(M, "i8", "i32", i8[0])
            print(f"{M:>6}  {b_bf:>12.2f}  {b_i8:>12.2f}")

    # Scaling coefficient: if lat = a + b*M (compute-bound) vs lat ~ const
    # (bandwidth-bound), we can eyeball it from the delta lat / delta M.
    print("\n" + "=" * 80)
    print("SCALING: Δlat_us per extra token of M  (lower = better amortization)")
    print("=" * 80)
    print(f"{'M range':>12}  {'bf16':>10}  {'i8':>10}")
    for a, b in zip(M_VALUES, M_VALUES[1:]):
        bf_a, bf_b = results.get(("bf16", a)), results.get(("bf16", b))
        i8_a, i8_b = results.get(("i8", a)), results.get(("i8", b))
        if bf_a and bf_b and i8_a and i8_b:
            dbf = (bf_b[0] - bf_a[0]) / (b - a)
            di8 = (i8_b[0] - i8_a[0]) / (b - a)
            print(f"{a:>5}->{b:<5}  {dbf:>10.2f}  {di8:>10.2f}")

    print("\nInterpretation:")
    print("  - If i8/bf16 latency ratio stays near 1.0 across all M: both are")
    print("    compute-bound on the core. INT8 prefill investigation is closed.")
    print("  - If ratio drifts toward 0.5 at larger M: bandwidth-bound regime,")
    print("    INT8 has untapped headroom. Next step: wire prefill to feed larger M.")


if __name__ == "__main__":
    main()
