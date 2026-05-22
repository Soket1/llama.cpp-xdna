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

} // namespace

int main(int argc, char ** argv) {
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
