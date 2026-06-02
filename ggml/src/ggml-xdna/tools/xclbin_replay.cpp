// SPDX-License-Identifier: MIT
//
// Generic MLIR_AIE xclbin replay/dispatch harness.
//
// Loads an arbitrary "MLIR_AIE" (dpu_kernel_id 0x901, 8-arg ABI) xclbin,
// binds a raw instruction (control-code) blob to arg 1, allocates up to 5
// data BOs (bo0..bo4 = kernel args 3..7) of caller-specified sizes, then
// dispatches and reports the ERT completion state + wall time. Data BOs are
// zero-filled — this proves the dispatch PATH (kernel loads, BD topology is
// valid, NPU completes without hanging) and measures per-dispatch wall time.
// Output is not validated here (dummy inputs); a separate step fills real
// data for correctness.
//
// The 8-arg ABI (from the xclbin EMBEDDED_METADATA):
//   arg0 opcode(u64) | arg1 instr(BO) | arg2 ninstr(u32) | arg3..7 bo0..bo4
//
// ninstr: for transaction-format control code (XAIE aie2txn, header carries
// its own size) the kernel ignores ninstr — pass 0. For word-counted formats
// pass the word/byte count. Override via the last positional arg if needed.
//
// Usage:
//   xclbin_replay <xclbin> <insts> <bo0_B> <bo1_B> <bo2_B> <bo3_B> <bo4_B>
//                 [iters=1] [ninstr=0] [opcode=3]
//   A BO size of 0 means "do not allocate / do not bind that arg".
//
// Build: add_executable target in ggml-xdna/CMakeLists.txt (see xrt_async_spike).

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
        fprintf(stderr, "replay: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf((size_t)n);
    f.read(buf.data(), n);
    return buf;
}

size_t parse_sz(const char * s) {
    // Accept plain bytes or hex (0x...) — generous; sizes can be large.
    return (size_t)std::strtoull(s, nullptr, 0);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr,
            "usage: %s <xclbin> <insts> <bo0_B> <bo1_B> <bo2_B> <bo3_B> "
            "<bo4_B> [iters=1] [ninstr=0] [opcode=3]\n", argv[0]);
        return 2;
    }
    const std::string xclbin_path = argv[1];
    const std::string insts_path  = argv[2];
    size_t bo_bytes[5];
    for (int i = 0; i < 5; i++) bo_bytes[i] = parse_sz(argv[3 + i]);
    const int      iters  = (argc > 8)  ? std::atoi(argv[8])  : 1;
    const uint32_t ninstr = (argc > 9)  ? (uint32_t)std::strtoul(argv[9], nullptr, 0) : 0u;
    const uint32_t opcode = (argc > 10) ? (uint32_t)std::strtoul(argv[10], nullptr, 0) : 3u;

    try {
        auto device = xrt::device(0);
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        auto hw_ctx = xrt::hw_context(device, xclbin.get_uuid());
        auto kernel = xrt::kernel(hw_ctx, "MLIR_AIE");

        fprintf(stderr, "replay: loaded %s\n", xclbin_path.c_str());
        fprintf(stderr, "replay: kernel group_ids:");
        for (int a = 0; a <= 7; a++) {
            try { fprintf(stderr, " [%d]=%zu", a, (size_t)kernel.group_id(a)); }
            catch (...) { fprintf(stderr, " [%d]=?", a); }
        }
        fprintf(stderr, "\n");

        // instr BO (arg1). cacheable group_id(1) — mirrors ggml-xdna.cpp.
        auto insts = read_file(insts_path);
        xrt::bo insts_bo(device, insts.size(), xrt::bo::flags::cacheable,
                         kernel.group_id(1));
        insts_bo.write(insts.data());
        insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        fprintf(stderr, "replay: insts=%zuB ninstr=%u opcode=%u\n",
                insts.size(), ninstr, opcode);

        // data BOs bo0..bo4 = args 3..7. Filled with a non-zero pattern so a
        // genuinely-executed kernel produces a non-zero, input-dependent
        // change in some output BO (proves the NPU actually computed, not just
        // walked BDs over zeros where matmul(0)=0 hides everything).
        auto checksum = [](const uint8_t * p, size_t n) -> uint64_t {
            uint64_t h = 1469598103934665603ull;           // FNV-1a
            for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
            return h;
        };
        auto nonzero = [](const uint8_t * p, size_t n) -> size_t {
            size_t c = 0; for (size_t i = 0; i < n; i++) c += (p[i] != 0); return c;
        };
        // Optional per-BO file contents: argv tokens "boN=path" load a file
        // into bo N (BO size = max(declared, file size)); else fill a pattern.
        std::string bo_file[5];
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a.size() > 4 && a[0] == 'b' && a[1] == 'o' && a[3] == '=') {
                int idx = a[2] - '0';
                if (idx >= 0 && idx < 5) bo_file[idx] = a.substr(4);
            }
        }

        std::vector<xrt::bo> data_bos(5);
        size_t act[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < 5; i++) {
            std::vector<char> fd;
            act[i] = bo_bytes[i];
            if (!bo_file[i].empty()) {
                fd = read_file(bo_file[i]);
                if (fd.size() > act[i]) act[i] = fd.size();
            }
            if (act[i] == 0) continue;
            data_bos[i] = xrt::bo(device, act[i], xrt::bo::flags::host_only,
                                  kernel.group_id(3 + i));
            uint8_t * m = data_bos[i].map<uint8_t*>();
            // bf16-safe fill: small positive values (~0.008..0.016), never
            // NaN/inf, so dummy activation/KV inputs don't poison the GEMV.
            // exp field 0x78 (=2^-7) with varying mantissa.
            uint16_t * w = reinterpret_cast<uint16_t*>(m);
            size_t nw = act[i] / 2;
            for (size_t b = 0; b < nw; b++)
                w[b] = (uint16_t)(0x3C00u | ((b * 7 + (size_t)i) & 0x3FFu));
            if (act[i] & 1) m[act[i] - 1] = 0;
            if (!fd.empty()) std::memcpy(m, fd.data(), fd.size());
            data_bos[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
            fprintf(stderr, "replay: bo%d (arg%d) = %zu B %spre-csum=%016llx\n",
                    i, 3 + i, act[i], (bo_file[i].empty() ? "" : "[file] "),
                    (unsigned long long)checksum(m, act[i]));
        }

        auto run = xrt::run(kernel);
        run.set_arg(0, opcode);
        run.set_arg(1, insts_bo);
        run.set_arg(2, ninstr);
        for (int i = 0; i < 5; i++) {
            if (act[i] != 0) run.set_arg(3 + i, data_bos[i]);
        }

        // Warm dispatch (also the go/no-go: does the NPU complete?).
        fprintf(stderr, "replay: dispatching (timeout 10s)...\n");
        fflush(stderr);
        auto t0 = clk::now();
        run.start();
        ert_cmd_state st = run.wait(std::chrono::seconds(10));
        long long first_us = us_since(t0);
        fprintf(stderr, "replay: first dispatch state=%d (%s) in %lld us\n",
                (int)st, (st == ERT_CMD_STATE_COMPLETED ? "COMPLETED" : "NOT-COMPLETED"),
                first_us);
        if (st != ERT_CMD_STATE_COMPLETED) {
            fprintf(stderr, "replay: dispatch did NOT complete — aborting.\n");
            return 1;
        }

        // Read every BO back: which one(s) did the NPU write? A changed csum
        // (vs the pre-dispatch pattern) proves the kernel actually executed.
        for (int i = 0; i < 5; i++) {
            if (act[i] == 0) continue;
            data_bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const uint8_t * m = data_bos[i].map<uint8_t*>();
            fprintf(stderr, "replay: bo%d post-csum=%016llx nonzero=%zu/%zu\n",
                    i, (unsigned long long)checksum(m, act[i]),
                    nonzero(m, act[i]), act[i]);
        }

        // Dump bo0 (the layer output activation) for off-line analysis.
        if (act[0] != 0) {
            const uint8_t * m = data_bos[0].map<uint8_t*>();
            std::ofstream of("replay_bo0_out.bin", std::ios::binary);
            of.write(reinterpret_cast<const char*>(m), (std::streamsize)act[0]);
            fprintf(stderr, "replay: wrote bo0 -> replay_bo0_out.bin (%zu B)\n", act[0]);
        }

        if (iters > 1) {
            // Warm repeat for a steady-state per-dispatch number.
            auto tb = clk::now();
            for (int i = 0; i < iters; i++) {
                run.start();
                run.wait();
            }
            long long tot = us_since(tb);
            fprintf(stderr,
                "replay: %d iters total %lld us = %.1f us/dispatch "
                "(%.2f dispatch/s)\n",
                iters, tot, (double)tot / iters, 1e6 * iters / (double)tot);
        }
        fprintf(stderr, "replay: OK\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "replay: exception: %s\n", e.what());
        return 1;
    }
}
