// SPDX-License-Identifier: MIT
// Phase 9 Day 1 spike: validate that XRT async dispatch (xrt::run::start()
// without intervening waits) actually pipelines on AMD XDNA NPU, vs the
// baseline sync (start + wait per op) pattern.
//
// What this measures:
//   1. SYNC baseline:  for i in N: rs[i].start(); rs[i].wait();
//   2. ASYNC pipeline: for i in N: rs[i].start();
//                      for i in N: rs[i].wait();
//
// Both use the same xclbin (INT4 GEMV K=2048 N=8192 8col g32), one
// xrt::hw_context shared, fresh per-run BOs (no aliasing) so there's no
// data-hazard reason for the driver to serialize beyond its own choice.
//
// Expected (per XRT.pdf and ypapadop-amd/ggml-hsa code inspection):
//   sync   ~= N * (kernel_us + ~1500 us overhead per round-trip)
//   async  ~= N * kernel_us (dispatches hidden behind NPU compute)
//
// If async ~= sync, XRT/XDNA is serializing under the hood and Phase 9
// needs deeper rework (or runlist instead of independent runs, IRON #83
// notwithstanding).
//
// Build (Visual Studio Developer Cmd or MSYS):
//   see compile_spike.bat alongside this file.

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>   // xrt::runlist

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

long long us_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        clk::now() - t0).count();
}

std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "spike: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf((size_t)n);
    f.read(buf.data(), n);
    return buf;
}

// Build a self-contained dispatch context for one shape so the spike
// doesn't depend on the rest of ggml-xdna. We mirror the load pattern
// from ggml-xdna.cpp lines 1342..1371 exactly.
struct dispatch_ctx {
    xrt::device       device;
    xrt::xclbin       xclbin;
    xrt::hw_context   hw_ctx;
    xrt::kernel       kernel;
    std::vector<char> insts;
    xrt::bo           insts_bo;

    // Cached weight (group_id 3), reused across all runs in the test.
    xrt::bo           weight_bo;

    int K = 0;
    int N = 0;
};

dispatch_ctx make_dispatch_ctx(const std::string & xclbin_path,
                               const std::string & insts_path,
                               int K, int N) {
    dispatch_ctx d;
    d.K = K;
    d.N = N;
    d.device = xrt::device(0);                     // first NPU
    d.xclbin = xrt::xclbin(xclbin_path);
    d.device.register_xclbin(d.xclbin);
    auto uuid = d.xclbin.get_uuid();
    d.hw_ctx = xrt::hw_context(d.device, uuid);
    d.kernel = xrt::kernel(d.hw_ctx, "MLIR_AIE");

    d.insts = read_file(insts_path);
    d.insts_bo = xrt::bo(d.device, d.insts.size(),
                         xrt::bo::flags::cacheable,
                         d.kernel.group_id(1));
    d.insts_bo.write(d.insts.data());
    d.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Packed INT4 weight buffer. Real size = num_cols * tiles_per_col *
    // (m_input * K/2 + m_input * (K/g) * 2). For K=2048 N=8192 cols=8
    // tsi=4 g=32: per_col=1024, tpc=256, ptb = 4*1024 + 4*64*2 = 4608 B.
    // Total = 8 * 256 * 4608 = 9 437 184 B. We don't need correct contents
    // for a perf spike -- random fill is fine since we're not validating
    // output, just timing the dispatch.
    const size_t group_size = 32;
    const int m_input = 4;
    const int cols = 8;
    const int64_t per_col_rows = N / cols;
    const int64_t tiles_per_col = per_col_rows / m_input;
    const int64_t num_groups_per_row = K / group_size;
    const size_t packed_tile_bytes =
        (size_t)m_input * K / 2 + (size_t)m_input * num_groups_per_row * 2;
    const size_t weight_bytes = (size_t)cols * tiles_per_col * packed_tile_bytes;

    d.weight_bo = xrt::bo(d.device, weight_bytes, xrt::bo::flags::host_only,
                          d.kernel.group_id(3));
    std::memset(d.weight_bo.map<void*>(), 0, weight_bytes);
    d.weight_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    fprintf(stderr, "spike: ctx ready K=%d N=%d insts=%zuB weight=%zuB\n",
            K, N, d.insts.size(), weight_bytes);
    fflush(stderr);

    return d;
}

// Build a single fresh dispatch (input BO + output BO + run object).
// We don't share input/output BOs between dispatches so there's no
// inter-run data hazard that could force serialization.
struct dispatch {
    xrt::bo  in_bo;     // group_id 4
    xrt::bo  out_bo;    // group_id 5
    xrt::run run;
};

dispatch make_dispatch(dispatch_ctx & d) {
    dispatch r;
    r.in_bo  = xrt::bo(d.device, (size_t)d.K * 2, xrt::bo::flags::host_only,
                       d.kernel.group_id(4));
    r.out_bo = xrt::bo(d.device, (size_t)d.N * 2, xrt::bo::flags::host_only,
                       d.kernel.group_id(5));
    std::memset(r.in_bo.map<void*>(), 0, (size_t)d.K * 2);
    r.in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    r.run = xrt::run(d.kernel);
    // Same arg order ggml-xdna uses: (opcode, insts, n_insts, mat, vec, out)
    r.run.set_arg(0, (uint32_t)3);
    r.run.set_arg(1, d.insts_bo);
    r.run.set_arg(2, (uint32_t)d.insts.size());
    r.run.set_arg(3, d.weight_bo);
    r.run.set_arg(4, r.in_bo);
    r.run.set_arg(5, r.out_bo);
    return r;
}

void bench(dispatch_ctx & d, int N_iters) {
    fprintf(stderr, "\nspike: building %d dispatches...\n", N_iters);
    std::vector<dispatch> ds;
    ds.reserve(N_iters);
    auto t_build_s = clk::now();
    for (int i = 0; i < N_iters; i++) ds.push_back(make_dispatch(d));
    auto build_us = us_since(t_build_s);
    fprintf(stderr, "spike: built in %lld us (%.0f us/dispatch)\n",
            build_us, (double)build_us / N_iters);

    // Warm-up: 5 sync dispatches so the driver / hw_ctx is "armed".
    for (int i = 0; i < 5; i++) {
        ds[i].run.start();
        ds[i].run.wait();
    }

    // ---- SYNC baseline ----
    auto t_sync_s = clk::now();
    for (int i = 0; i < N_iters; i++) {
        ds[i].run.start();
        ds[i].run.wait();
    }
    auto sync_us = us_since(t_sync_s);

    // ---- ASYNC pipeline ----
    auto t_async_s = clk::now();
    for (int i = 0; i < N_iters; i++) ds[i].run.start();
    auto submit_done_us = us_since(t_async_s);
    for (int i = 0; i < N_iters; i++) ds[i].run.wait();
    auto async_us = us_since(t_async_s);

    // ---- RUNLIST: all N runs in ONE submitted runlist ----
    // Note: XRT.pdf warns IRON #83 can corrupt output with shared kernel
    // handle, but here all 100 runs share one kernel -- so this is a
    // perf-only smoke test, NOT a correctness test.
    long long runlist_us = -1;
    long long runlist_exec_only_us = -1;
    try {
        xrt::runlist rl(d.hw_ctx);
        for (int i = 0; i < N_iters; i++) rl.add(ds[i].run);
        auto t_rl_s = clk::now();
        rl.execute();
        auto rl_exec_us = us_since(t_rl_s);
        rl.wait();
        runlist_us = us_since(t_rl_s);
        runlist_exec_only_us = rl_exec_us;
    } catch (const std::exception & e) {
        fprintf(stderr, "spike: runlist test threw: %s\n", e.what());
    }

    fprintf(stderr, "\n=== spike results (N=%d) ===\n", N_iters);
    fprintf(stderr, "  sync     total %8lld us  (%.0f us/op)\n",
            sync_us, (double)sync_us / N_iters);
    fprintf(stderr, "  async    total %8lld us  (%.0f us/op)\n",
            async_us, (double)async_us / N_iters);
    fprintf(stderr, "  async    submit-only %8lld us  (%.0f us/op)\n",
            submit_done_us, (double)submit_done_us / N_iters);
    if (runlist_us >= 0) {
        fprintf(stderr, "  runlist  total %8lld us  (%.0f us/op)\n",
                runlist_us, (double)runlist_us / N_iters);
        fprintf(stderr, "  runlist  exec-only %8lld us\n", runlist_exec_only_us);
    }
    fprintf(stderr, "  speedup async   / sync = %.2fx\n",
            (double)sync_us / (double)async_us);
    if (runlist_us >= 0) {
        fprintf(stderr, "  speedup runlist / sync = %.2fx\n",
                (double)sync_us / (double)runlist_us);
    }
    fflush(stderr);
}

// ----------------------------------------------------------------------
// Token-sequence mode (Phase 9 v2 re-spike).
//
// Models one Llama 3.2 1B decode token's worth of INT4 GEMV dispatches
// using the v2 xclbins. Per-layer sequence (7 ops/layer × num_layers):
//     2× attn_q/o   (K=2048 N=2048)
//     2× attn_k/v   (K=2048 N=512)
//     1× ffn_gate   (K=2048 N=8192)
//     1× ffn_up     (K=2048 N=8192)
//     1× ffn_down   (K=8192 N=2048)
//
// Three modes compared:
//     SYNC          submit + wait per op   (current ggml-xdna behavior)
//     ASYNC_LAYER   submit 7 ops, wait at end of layer, then busy_us CPU
//                   work (mimics deferred bias compensation at a barrier)
//     ASYNC_TOKEN   submit ALL N×7 ops, wait once on the final op
//                   (upper bound -- ignores data deps + bias barriers)
//
// busy_us optionally injects a CPU busy-loop between dispatches to model
// the host-side bias compensation step that mul_mat_gemv_int4 currently
// does immediately after rl.wait(). Set ~700 us to mimic the worst-case
// FFN-down bias work observed in earlier profiling.

struct LayerOp {
    const char * label;
    const char * xclbin;
    const char * insts;
    int K;
    int N;
    int count_per_layer;
};

void busy_loop_us(long long us) {
    if (us <= 0) return;
    auto t0 = clk::now();
    volatile unsigned x = 1;
    while (us_since(t0) < us) {
        for (int i = 0; i < 1000; i++) x = x * 1103515245u + 12345u;
    }
    (void)x;
}

void bench_token_sequence(const std::string & cache_dir,
                          int num_layers,
                          long long busy_us) {
    static const LayerOp layer_ops[] = {
        {"attn_q",  "gemv_int4_v2_K2048_N2048_8col_g32.xclbin",
                    "gemv_int4_v2_K2048_N2048_8col_g32.insts", 2048, 2048, 1},
        {"attn_o",  "gemv_int4_v2_K2048_N2048_8col_g32.xclbin",
                    "gemv_int4_v2_K2048_N2048_8col_g32.insts", 2048, 2048, 1},
        {"attn_k",  "gemv_int4_v2_K2048_N512_8col_g32.xclbin",
                    "gemv_int4_v2_K2048_N512_8col_g32.insts",  2048,  512, 1},
        {"attn_v",  "gemv_int4_v2_K2048_N512_8col_g32.xclbin",
                    "gemv_int4_v2_K2048_N512_8col_g32.insts",  2048,  512, 1},
        {"ffn_gate","gemv_int4_v2_K2048_N8192_8col_g32.xclbin",
                    "gemv_int4_v2_K2048_N8192_8col_g32.insts", 2048, 8192, 1},
        {"ffn_up",  "gemv_int4_v2_K2048_N8192_8col_g32.xclbin",
                    "gemv_int4_v2_K2048_N8192_8col_g32.insts", 2048, 8192, 1},
        {"ffn_down","gemv_int4_v2_K8192_N2048_8col_g32.xclbin",
                    "gemv_int4_v2_K8192_N2048_8col_g32.insts", 8192, 2048, 1},
    };
    const int ops_per_layer = (int)(sizeof(layer_ops) / sizeof(layer_ops[0]));

    // Deduplicate contexts by (xclbin path) so we only load each xclbin once.
    fprintf(stderr, "\nspike: token-sequence mode num_layers=%d busy_us=%lld\n",
            num_layers, busy_us);
    std::vector<dispatch_ctx> ctxs;
    std::vector<int> op_to_ctx(ops_per_layer);
    for (int i = 0; i < ops_per_layer; i++) {
        int found = -1;
        for (int j = 0; j < (int)ctxs.size(); j++) {
            if (ctxs[j].K == layer_ops[i].K && ctxs[j].N == layer_ops[i].N) {
                found = j;
                break;
            }
        }
        if (found < 0) {
            ctxs.push_back(make_dispatch_ctx(
                cache_dir + "/" + layer_ops[i].xclbin,
                cache_dir + "/" + layer_ops[i].insts,
                layer_ops[i].K, layer_ops[i].N));
            found = (int)ctxs.size() - 1;
        }
        op_to_ctx[i] = found;
    }
    fprintf(stderr, "spike: loaded %zu unique shapes for %d ops/layer\n",
            ctxs.size(), ops_per_layer);

    // Pre-build all dispatches for one token (num_layers * ops_per_layer).
    const int total_ops = num_layers * ops_per_layer;
    fprintf(stderr, "spike: building %d dispatches (one token's worth)...\n",
            total_ops);
    std::vector<dispatch> ds;
    ds.reserve(total_ops);
    std::vector<int> ds_ctx(total_ops);
    auto t_build_s = clk::now();
    for (int L = 0; L < num_layers; L++) {
        for (int op = 0; op < ops_per_layer; op++) {
            int cidx = op_to_ctx[op];
            ds.push_back(make_dispatch(ctxs[cidx]));
            ds_ctx[L * ops_per_layer + op] = cidx;
        }
    }
    auto build_us = us_since(t_build_s);
    fprintf(stderr, "spike: built in %lld us (%.0f us/dispatch)\n",
            build_us, (double)build_us / total_ops);

    // Warm-up: one full token sync to arm driver + hw_ctx for every shape.
    for (int i = 0; i < total_ops; i++) {
        ds[i].run.start();
        ds[i].run.wait();
    }

    // ---- SYNC ----
    long long sync_bias_us = 0;
    auto t_sync_s = clk::now();
    for (int i = 0; i < total_ops; i++) {
        ds[i].run.start();
        ds[i].run.wait();
        if (busy_us > 0) {
            auto t_b = clk::now();
            busy_loop_us(busy_us);
            sync_bias_us += us_since(t_b);
        }
    }
    long long sync_us = us_since(t_sync_s);

    // ---- ASYNC_LAYER: submit 7 ops/layer, wait barrier, busy work ----
    long long async_layer_bias_us = 0;
    auto t_al_s = clk::now();
    for (int L = 0; L < num_layers; L++) {
        const int base = L * ops_per_layer;
        for (int op = 0; op < ops_per_layer; op++) ds[base + op].run.start();
        for (int op = 0; op < ops_per_layer; op++) ds[base + op].run.wait();
        if (busy_us > 0) {
            auto t_b = clk::now();
            // One bias compensation per op-with-bias. Approximation: do
            // ops_per_layer busy steps at the layer barrier (real path
            // would interleave them; this overestimates the cost slightly).
            busy_loop_us(busy_us * ops_per_layer);
            async_layer_bias_us += us_since(t_b);
        }
    }
    long long async_layer_us = us_since(t_al_s);

    // ---- ASYNC_TOKEN: submit everything, wait only at end ----
    auto t_at_s = clk::now();
    for (int i = 0; i < total_ops; i++) ds[i].run.start();
    auto at_submit_done_us = us_since(t_at_s);
    for (int i = 0; i < total_ops; i++) ds[i].run.wait();
    long long async_token_us = us_since(t_at_s);

    // ---- Report ----
    fprintf(stderr, "\n=== token-sequence results (layers=%d, ops/layer=%d, total=%d) ===\n",
            num_layers, ops_per_layer, total_ops);
    fprintf(stderr, "  SYNC         total %8lld us  (%.0f us/op)",
            sync_us, (double)sync_us / total_ops);
    if (busy_us > 0)
        fprintf(stderr, "  [bias %lld us]", sync_bias_us);
    fprintf(stderr, "\n");
    fprintf(stderr, "  ASYNC_LAYER  total %8lld us  (%.0f us/op)",
            async_layer_us, (double)async_layer_us / total_ops);
    if (busy_us > 0)
        fprintf(stderr, "  [bias %lld us]", async_layer_bias_us);
    fprintf(stderr, "\n");
    fprintf(stderr, "  ASYNC_TOKEN  total %8lld us  (%.0f us/op)\n",
            async_token_us, (double)async_token_us / total_ops);
    fprintf(stderr, "                 submit-only %lld us\n",
            at_submit_done_us);
    fprintf(stderr, "  speedup ASYNC_LAYER / SYNC = %.3fx\n",
            (double)sync_us / (double)async_layer_us);
    fprintf(stderr, "  speedup ASYNC_TOKEN / SYNC = %.3fx\n",
            (double)sync_us / (double)async_token_us);

    // Translate to t/s projection assuming the per-token decode budget
    // outside NPU work stays constant. Current measured baseline:
    //   sync v2 INT4 decode = 5.6 t/s = 178 ms/token, of which the NPU
    //   work is the sync_us above. So non-NPU overhead ≈ (178000 -
    //   sync_us/1000) us. Async ROI is reflected in the new total:
    //   new_token_us = async_X_us + non_NPU_overhead.
    const long long non_npu_us = 178000LL - sync_us;
    if (non_npu_us > 0) {
        auto proj_ts = [non_npu_us](long long npu_us) -> double {
            return 1e6 / (double)(npu_us + non_npu_us);
        };
        fprintf(stderr, "  projected end-to-end decode t/s "
                "(non-NPU overhead = %lld us, assumed constant):\n",
                non_npu_us);
        fprintf(stderr, "    SYNC         %.2f t/s\n", proj_ts(sync_us));
        fprintf(stderr, "    ASYNC_LAYER  %.2f t/s\n", proj_ts(async_layer_us));
        fprintf(stderr, "    ASYNC_TOKEN  %.2f t/s\n", proj_ts(async_token_us));
    }
    fflush(stderr);
}

} // namespace

int main(int argc, char ** argv) {
    // Two modes:
    //   1. Single-shape (default): one xclbin, N dispatches, sync/async/runlist.
    //        argv: <xclbin> <insts> <K> <N> <N_iters>
    //   2. Token-sequence (Phase 9 v2 re-spike): pass --token as argv[1].
    //        argv: --token <cache_dir> <num_layers> <busy_us>
    if (argc > 1 && std::string(argv[1]) == "--token") {
        const char * cache_dir = (argc > 2) ? argv[2] : "npu_kernels_win_8col";
        int num_layers = (argc > 3) ? std::atoi(argv[3]) : 16;
        long long busy_us = (argc > 4) ? std::atoll(argv[4]) : 0;
        try {
            bench_token_sequence(cache_dir, num_layers, busy_us);
        } catch (const std::exception & e) {
            fprintf(stderr, "spike: exception: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    const char * xclbin =
        (argc > 1) ? argv[1]
                   : "npu_kernels_win_8col/gemv_int4_K2048_N8192_8col_g32.xclbin";
    const char * insts =
        (argc > 2) ? argv[2]
                   : "npu_kernels_win_8col/gemv_int4_K2048_N8192_8col_g32.insts";
    int K = (argc > 3) ? std::atoi(argv[3]) : 2048;
    int N = (argc > 4) ? std::atoi(argv[4]) : 8192;
    int N_iters = (argc > 5) ? std::atoi(argv[5]) : 100;

    try {
        auto d = make_dispatch_ctx(xclbin, insts, K, N);
        bench(d, N_iters);
    } catch (const std::exception & e) {
        fprintf(stderr, "spike: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
