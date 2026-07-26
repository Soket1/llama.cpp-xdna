// Measure switch cost between two xclbins on the SAME device.
// Build: cmake --build . --target xclbin_switch_cost --config Release
//
// Usage:
//   xclbin_switch_cost <xclbin_A> <insts_A> <xclbin_B> <insts_B>
//                      <bo0_B> <bo1_B> <bo2_B> <bo3_B> <bo4_B>
//                      [iters=3]

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

long long us_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
}

std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "error: cannot open %s\n", path.c_str()); std::exit(2); }
    std::streamsize n = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<char> buf((size_t)n); f.read(buf.data(), n); return buf;
}

int main(int argc, char ** argv) {
    if (argc < 10) {
        fprintf(stderr,
            "usage: %s <xclbin_A> <insts_A> <xclbin_B> <insts_B> "
            "<bo0_B> <bo1_B> <bo2_B> <bo3_B> <bo4_B> [iters=3]\n", argv[0]);
        return 2;
    }

    std::string xclbin_A = argv[1], insts_A = argv[2];
    std::string xclbin_B = argv[3], insts_B = argv[4];
    size_t bo_sizes[5];
    for (int i = 0; i < 5; i++) bo_sizes[i] = (size_t)std::strtoull(argv[5 + i], nullptr, 0);
    int n_iters = (argc > 10) ? atoi(argv[10]) : 3;

    auto device = xrt::device(0);
    fprintf(stderr, "=== Xclbin switch cost measurement ===\n");

    // Load xclbins and get kernels
    auto load_one = [&](const std::string & xp, const std::string & ip) {
        auto x = xrt::xclbin(xp);
        device.register_xclbin(x);
        auto hw_ctx = xrt::hw_context(device, x.get_uuid());
        auto kernel = xrt::kernel(hw_ctx, "MLIR_AIE");
        auto insts = read_file(ip);
        fprintf(stderr, "  loaded %s: insts=%zuB\n", xp.c_str(), insts.size());
        return std::make_tuple(std::move(hw_ctx), std::move(kernel), std::move(insts));
    };

    auto [hA, kA, iA] = load_one(xclbin_A, insts_A);
    auto [hB, kB, iB] = load_one(xclbin_B, insts_B);

    // Instruction BOs (matches xclbin_replay: exact size, cacheable, group_id(1))
    auto make_insts_bo = [&](xrt::kernel & k, std::vector<char> & d) {
        xrt::bo b(device, d.size(), xrt::bo::flags::cacheable, k.group_id(1));
        b.write(d.data()); b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return b;
    };
    xrt::bo iboA = make_insts_bo(kA, iA);
    xrt::bo iboB = make_insts_bo(kB, iB);

    // Data BOs (host_only, group_id(0), caller-specified sizes)
    auto make_data_bo = [&](xrt::kernel & k, size_t sz) {
        if (sz == 0) sz = 65536;
        return xrt::bo(device, sz, xrt::bo::flags::host_only, k.group_id(0));
    };
    std::vector<xrt::bo> bosA, bosB;
    for (int i = 0; i < 5; i++) {
        bosA.push_back(make_data_bo(kA, bo_sizes[i]));
        bosB.push_back(make_data_bo(kB, bo_sizes[i]));
    }

    auto dispatch = [&](xrt::kernel & k, xrt::bo & ibo, std::vector<xrt::bo> & bos, size_t isize) {
        auto run = xrt::run(k);
        run.set_arg(0, (uint32_t)3);
        run.set_arg(1, ibo);
        run.set_arg(2, (uint32_t)isize);
        for (int i = 0; i < 5; i++) run.set_arg(3 + i, bos[i]);
        auto t0 = clk::now();
        run.start();
        run.wait();
        return us_since(t0);
    };

    // Warmup phases
    fprintf(stderr, "\nWarming A: ");
    for (int i = 0; i < n_iters; i++) {
        auto t = dispatch(kA, iboA, bosA, iA.size());
        fprintf(stderr, "%lldus ", t); fflush(stderr);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Warming B: ");
    for (int i = 0; i < n_iters; i++) {
        auto t = dispatch(kB, iboB, bosB, iB.size());
        fprintf(stderr, "%lldus ", t); fflush(stderr);
    }
    fprintf(stderr, "\n");

    // Measure: A warm (baseline)
    auto tA_warm = dispatch(kA, iboA, bosA, iA.size());
    fprintf(stderr, "\nA warm (baseline): %lld us\n", tA_warm);

    // Measure: B warm (baseline)
    auto tB_warm = dispatch(kB, iboB, bosB, iB.size());
    fprintf(stderr, "B warm (baseline): %lld us\n", tB_warm);

    // Cross switch: A -> B
    dispatch(kA, iboA, bosA, iA.size());  // last A
    auto tB_after_A = dispatch(kB, iboB, bosB, iB.size());
    fprintf(stderr, "\nB after A: %lld us (switch cost: %lld us)\n",
            tB_after_A, tB_after_A - tB_warm);

    // Cross switch: B -> A
    dispatch(kB, iboB, bosB, iB.size());  // last B
    auto tA_after_B = dispatch(kA, iboA, bosA, iA.size());
    fprintf(stderr, "A after B: %lld us (switch cost: %lld us)\n",
            tA_after_B, tA_after_B - tA_warm);

    // Same-xclbin back-to-back (A -> A)
    dispatch(kA, iboA, bosA, iA.size());
    auto tA_again = dispatch(kA, iboA, bosA, iA.size());
    fprintf(stderr, "\nA after A (same xclbin): %lld us\n", tA_again);

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
