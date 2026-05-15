// ============================================================================
// xdna_diag_offset.cpp — Diagnostic tests for phantom DMA offset hypothesis
//
// Tests:
//   1. Compare physical addresses: host_only vs normal allocation
//   2. Dump Shim DMA BD registers via xrt-smi
//   3. Export buffer and compare addresses
//
// Build (Windows, MSVC):
//   cl /std:c++17 /EHsc xdna_diag_offset.cpp /I<XRT>/include ^
//      /link /LIBPATH:<XRT>/lib xrt_coreutil.lib
//
// Run on Windows with NPU:
//   xdna_diag_offset.exe
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"

// Helper: print 64-bit address
static void print_addr(const char * label, uint64_t addr) {
    fprintf(stderr, "  %-30s 0x%016llX\n", label, (unsigned long long)addr);
}

// Helper: print hex dump of first N bf16 values
static void print_bf16_hex(const char * label, const void * ptr, int n) {
    const uint16_t * p = (const uint16_t *)ptr;
    fprintf(stderr, "  %s:", label);
    for (int i = 0; i < n; i++) fprintf(stderr, " 0x%04X", p[i]);
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  XDNA DMA Offset Diagnostic Tool\n");
    fprintf(stderr, "========================================\n\n");

    // ------------------------------------------------------------------
    // Open device and load any xclbin to get a kernel reference
    // ------------------------------------------------------------------
    // NOTE: User must provide path to a compiled xclbin. For FlowKV:
    //   --xclbin <path_to_flowkv.xclbin>
    // If no xclbin provided, we still run address tests with dummy kernel.
    const char * xclbin_path = nullptr;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--xclbin") == 0) {
            xclbin_path = argv[i + 1];
            break;
        }
    }

    // Try to open device 0
    xrt::device device(0);
    fprintf(stderr, "[OK] Device opened: %s\n", device.get_info<xrt::info::device::name>().c_str());

    // ------------------------------------------------------------------
    // TEST 1: Address comparison — host_only vs normal
    // ------------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 1: host_only vs normal address comparison ---\n");
    {
        const size_t buf_size = 32768;  // 32KB = 16384 bf16 elements

        // Create two buffers: one host_only, one normal
        // Use group_id(3) as placeholder (matches FlowKV K buffer position)
        xrt::bo bo_host, bo_norm;
        try {
            if (xclbin_path) {
                auto xclbin = xrt::xclbin(xclbin_path);
                auto kernel = xrt::kernel(device, xclbin, "flowkv_decode",
                                          xrt::kernel::cu_access_mode::exclusive);
                bo_host = xrt::bo(device, buf_size, xrt::bo::flags::host_only, kernel.group_id(3));
                bo_norm = xrt::bo(device, buf_size, xrt::bo::flags::normal, kernel.group_id(3));
            } else {
                fprintf(stderr, "  [WARN] No xclbin provided, using device-level allocation\n");
                fprintf(stderr, "  [WARN] group_id unavailable — addresses may differ from real run\n");
                // Fallback: allocate without kernel context
                bo_host = xrt::bo(device, buf_size, xrt::bo::flags::host_only);
                bo_norm = xrt::bo(device, buf_size, xrt::bo::flags::normal);
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "  [ERROR] Allocation failed: %s\n", e.what());
            fprintf(stderr, "  Skipping TEST 1\n");
            goto test2;
        }

        {
            uint64_t addr_host = bo_host.address();
            uint64_t addr_norm = bo_norm.address();
            int64_t delta = (int64_t)addr_host - (int64_t)addr_norm;

            print_addr("host_only address", addr_host);
            print_addr("normal address", addr_norm);
            fprintf(stderr, "  %-30s %lld bytes (%lld KB)\n", "delta", delta, delta / 1024);

            if (delta == 32768 || delta == -32768) {
                fprintf(stderr, "\n  *** HYPOTHESIS 2 CONFIRMED: 32KB offset between host_only and normal ***\n");
                fprintf(stderr, "  *** kipudrv.inf likely adds hidden metadata prefix.              ***\n");
            } else if (delta == 0) {
                fprintf(stderr, "\n  [OK] No offset between allocation types.\n");
                fprintf(stderr, "  Hypothesis 2 NOT confirmed by this test.\n");
            } else {
                fprintf(stderr, "\n  [INFO] Non-zero delta (%lld). Investigate further.\n", delta);
            }
        }
    }

test2:
    // ------------------------------------------------------------------
    // TEST 2: Dump Shim DMA registers via xrt-smi
    // ------------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 2: Shim DMA register dump ---\n");
    fprintf(stderr, "  Run manually on Windows:\n");
    fprintf(stderr, "    xrt-smi examine\n");
    fprintf(stderr, "  Then look for BD (Buffer Descriptor) registers:\n");
    fprintf(stderr, "    - Compare 'base address' field with xrt::bo::address()\n");
    fprintf(stderr, "    - If BD addr = bo.address() + 32KB → phantom offset confirmed\n");
    fprintf(stderr, "\n  For detailed register dump:\n");
    fprintf(stderr, "    xrt-smi examine -d <bdf> --report dma\n");
    fprintf(stderr, "  Or directly read BD registers:\n");
    fprintf(stderr, "    xbutil examine --report electrical\n");

    // ------------------------------------------------------------------
    // TEST 3: Export buffer and compare addresses
    // ------------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 3: Buffer export address comparison ---\n");
    {
        const size_t buf_size = 32768;
        try {
            xrt::bo bo_orig(device, buf_size, xrt::bo::flags::host_only);

            // Export the buffer — creates a new handle to same physical memory
            auto exported = xrt::bo::export_buffer(bo_orig);

            uint64_t addr_orig = bo_orig.address();
            uint64_t addr_exp = exported.address();

            print_addr("original address", addr_orig);
            print_addr("exported address", addr_exp);

            if (addr_orig != addr_exp) {
                fprintf(stderr, "\n  *** EXPORT CHANGED ADDRESS: delta = %lld bytes ***\n",
                        (int64_t)addr_exp - (int64_t)addr_orig);
                fprintf(stderr, "  This suggests driver remaps the buffer on export.\n");
            } else {
                fprintf(stderr, "\n  [OK] Export preserved address.\n");
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "  [ERROR] Export test failed: %s\n", e.what());
        }
    }

    // ------------------------------------------------------------------
    // TEST 4 (bonus): Verify FlowKV K/V address relationship
    // ------------------------------------------------------------------
    fprintf(stderr, "\n--- TEST 4: FlowKV K/V address relationship ---\n");
    if (xclbin_path) {
        try {
            auto xclbin = xrt::xclbin(xclbin_path);
            auto kernel = xrt::kernel(device, xclbin, "flowkv_decode",
                                      xrt::kernel::cu_access_mode::exclusive);

            // Mimic FlowKV buffer creation from ggml-xdna.cpp
            const int head_dim = 64;
            const int seq_len = 128;  // example
            size_t k_size = 1 * seq_len * head_dim * sizeof(uint16_t);
            size_t v_size = 1 * seq_len * head_dim * sizeof(uint16_t);
            size_t q_size = 1 * (8 * head_dim + head_dim) * sizeof(uint16_t);
            size_t out_size = 8 * head_dim * sizeof(uint16_t);

            xrt::bo bo_k(device, k_size, xrt::bo::flags::host_only, kernel.group_id(3));
            xrt::bo bo_v(device, v_size, xrt::bo::flags::host_only, kernel.group_id(4));
            xrt::bo bo_q(device, q_size, xrt::bo::flags::host_only, kernel.group_id(5));
            xrt::bo bo_out(device, out_size, xrt::bo::flags::host_only, kernel.group_id(6));

            uint64_t addr_k = bo_k.address();
            uint64_t addr_v = bo_v.address();
            uint64_t addr_q = bo_q.address();
            uint64_t addr_out = bo_out.address();

            print_addr("bo_k (group_id 3)", addr_k);
            print_addr("bo_v (group_id 4)", addr_v);
            print_addr("bo_q (group_id 5)", addr_q);
            print_addr("bo_out (group_id 6)", addr_out);

            fprintf(stderr, "\n  K→V delta: %lld bytes (%lld KB)\n",
                    (int64_t)(addr_v - addr_k), (int64_t)(addr_v - addr_k) / 1024);
            fprintf(stderr, "  K→Q delta: %lld bytes (%lld KB)\n",
                    (int64_t)(addr_q - addr_k), (int64_t)(addr_q - addr_k) / 1024);

            // Write test pattern to K and verify it reads back correctly
            auto k_ptr = bo_k.map<uint16_t *>();
            for (int i = 0; i < 8; i++) k_ptr[i] = (uint16_t)(0x3E00 + i);
            bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Read back from K
            bo_k.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto k_read = bo_k.map<uint16_t *>();
            print_bf16_hex("K readback", k_read, 8);

            // Check if V contains K's data (would indicate shared memory / aliasing)
            bo_v.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto v_read = bo_v.map<uint16_t *>();
            bool v_has_k_data = true;
            for (int i = 0; i < 8; i++) {
                if (v_read[i] != k_ptr[i]) { v_has_k_data = false; break; }
            }
            if (v_has_k_data) {
                fprintf(stderr, "\n  *** V BUFFER CONTAINS K DATA — buffers may be aliased! ***\n");
            } else {
                fprintf(stderr, "\n  [OK] K and V buffers are independent.\n");
            }

        } catch (const std::exception & e) {
            fprintf(stderr, "  [ERROR] FlowKV test failed: %s\n", e.what());
        }
    } else {
        fprintf(stderr, "  [SKIP] Requires --xclbin <path>\n");
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "  Diagnostic complete.\n");
    fprintf(stderr, "  Copy this output and share with the\n");
    fprintf(stderr, "  development team for analysis.\n");
    fprintf(stderr, "========================================\n\n");

    return 0;
}
