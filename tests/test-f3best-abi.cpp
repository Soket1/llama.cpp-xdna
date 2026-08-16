// SPDX-License-Identifier: MIT
// CPU-only regression for the f3best decode-layer host ABI (#235).
//
// Locks the single shared geometry descriptor (f3best_pack.h: make_f3best_host_abi)
// to the exact XR spans, packed-weight tile counts, per-phase A-stream offsets, and
// byte sizes that pack_bcast()/pack_gemv() and the NPU kernel consume. The legacy
// host-only "PACKED" formula equaled the 4608-byte bcast tile only at E=2048; at
// E=3072 it was 6912 and corrupted the 3B weight feed. This test needs no XRT, NPU,
// or IRON — it runs on the host geometry alone.

#include "testing.h"

#include "f3best_pack.h"

#include <cstdint>
#include <vector>

static void test_bcast_tile_counts(testing & t) {
    t.assert_equal("4608-byte bcast tile", (size_t)4608, f3b::bcast_tile_bytes);
    t.assert_equal("1B Q tiles (rows=per_tile=256, K=E=2048)", (int64_t)64,
                   f3b::bcast_tile_count(256, 2048));
    t.assert_equal("3B Q tiles (rows=per_tile=384, K=E=3072)", (int64_t)144,
                   f3b::bcast_tile_count(384, 3072));
    t.assert_equal("3B gate/up tiles (rows=H8=1024, K=E=3072)", (int64_t)384,
                   f3b::bcast_tile_count(1024, 3072));
    t.assert_equal("3B down tiles (rows=E=3072, K=H8=1024)", (int64_t)384,
                   f3b::bcast_tile_count(3072, 1024));
}

static void test_1b_no_kv_abi(testing & t) {
    auto a = f3b::make_f3best_host_abi(2048, 8192, 64, 4, false);  // Llama-3.2-1B
    t.assert_equal("XB", (int64_t)2320, a.XB);
    t.assert_equal("XR_ELEMS", (int64_t)6416, a.xr_elems);
    t.assert_equal("wt_tiles/head", (int64_t)896, a.wt_tiles);
    t.assert_equal("packed/head", (size_t)4608, a.packed);
    t.assert_equal("wt_bytes/head", (size_t)4128768, a.wt_bytes);
    t.assert_equal("A total (8 heads)", (size_t)33030144, a.a_total);
    t.assert_equal("Q start", (int64_t)0, a.off_q);
    t.assert_equal("O start", (int64_t)64, a.off_o);
    t.assert_equal("gate start", (int64_t)128, a.off_gate);
    t.assert_equal("up start", (int64_t)384, a.off_up);
    t.assert_equal("down start", (int64_t)640, a.off_down);
    t.assert_equal("end", (int64_t)896, a.off_end);
    t.assert_true("end == wt_tiles", a.off_end == a.wt_tiles);
}

static void test_3b_no_kv_abi(testing & t) {
    auto a = f3b::make_f3best_host_abi(3072, 8192, 128, 3, false);  // Llama-3.2-3B
    t.assert_equal("rope_elems", (int64_t)384, a.rope_elems);
    t.assert_equal("XB", (int64_t)3472, a.XB);
    t.assert_equal("XR_ELEMS", (int64_t)9616, a.xr_elems);
    t.assert_equal("wt_tiles/head", (int64_t)1440, a.wt_tiles);
    t.assert_equal("packed/head (M-derived, 3B)", (size_t)6912, a.packed);
    t.assert_equal("wt_bytes/head", (size_t)9953280, a.wt_bytes);
    t.assert_equal("A total (8 heads)", (size_t)79626240, a.a_total);
    t.assert_equal("Q start", (int64_t)0, a.off_q);
    t.assert_equal("O start", (int64_t)144, a.off_o);
    t.assert_equal("gate start", (int64_t)288, a.off_gate);
    t.assert_equal("up start", (int64_t)672, a.off_up);
    t.assert_equal("down start", (int64_t)1056, a.off_down);
    t.assert_equal("end", (int64_t)1440, a.off_end);
    t.assert_true("end == wt_tiles", a.off_end == a.wt_tiles);
    // per-phase byte offsets = tile start * PACKED (6912 for 3B)
    t.assert_equal("O byte offset", (size_t)995328, (size_t)a.off_o * a.packed);
    t.assert_equal("gate byte offset", (size_t)1990656, (size_t)a.off_gate * a.packed);
    t.assert_equal("up byte offset", (size_t)4644864, (size_t)a.off_up * a.packed);
    t.assert_equal("down byte offset", (size_t)7299072, (size_t)a.off_down * a.packed);
    // XR region invariants
    t.assert_true("rope_end <= seq_off", a.rope_elems + a.E <= a.seq_off);
    t.assert_true("seq_off + 16 == XB", a.seq_off + 16 == a.XB);
    t.assert_true("XB + 2*E == xr_elems", a.XB + 2*a.E == a.xr_elems);
}

static void test_3b_with_kv_abi(testing & t) {
    auto a = f3b::make_f3best_host_abi(3072, 8192, 128, 3, true);  // 3B + NPU-KV (deferred #190)
    t.assert_equal("K/V tiles", (int64_t)48, a.kv_t);
    t.assert_equal("wt_tiles/head", (int64_t)1536, a.wt_tiles);
    t.assert_equal("wt_bytes/head", (size_t)10616832, a.wt_bytes);
    t.assert_equal("K start", (int64_t)144, a.off_k);
    t.assert_equal("V start", (int64_t)192, a.off_v);
    t.assert_equal("O start", (int64_t)240, a.off_o);
    t.assert_equal("gate start", (int64_t)384, a.off_gate);
    t.assert_equal("up start", (int64_t)768, a.off_up);
    t.assert_equal("down start", (int64_t)1152, a.off_down);
    t.assert_equal("end", (int64_t)1536, a.off_end);
}

// Sentinel: pack_bcast() writes a 4608-byte payload per slot at the caller's
// pad_to stride. The regression must show the payload is placed at slot*pad_to
// offsets and that nothing outside the described slots is touched.
static void test_pack_bcast_bounds(testing & t) {
    const int64_t E = 3072, H8 = 1024, group = 32;
    const auto a = f3b::make_f3best_host_abi(E, 8192, 128, 3, false);
    const int64_t num_groups = E / group;                  // 96
    const size_t  row_bytes  = (size_t)num_groups * 18;
    std::vector<uint8_t> w((size_t)H8 * row_bytes, 0x3C);  // arbitrary Q4_0 payload
    const int64_t gate_tiles = f3b::bcast_tile_count(H8, E);
    const size_t  payload_per_slot = f3b::bcast_tile_bytes;
    const size_t  stride          = a.packed;              // 6912 for 3B

    std::vector<uint8_t> region(a.wt_bytes, 0xAA);
    f3b::pack_bcast(w.data(), H8, E, E, 0, (int)group, stride,
                    (uint8_t *)region.data());
    // nothing past the last slot's stride is touched
    t.assert_equal("no write past last slot", (uint8_t)0xAA,
                   region[(size_t)gate_tiles * stride]);
    // each slot's payload (first 4608 bytes of each 6912 stride) was written
    bool wrote = false;
    for (int64_t s = 0; s < gate_tiles; s++) {
        size_t base = (size_t)s * stride;
        for (size_t i = 0; i < payload_per_slot; i++) {
            if (region[base + i] != 0xAA) { wrote = true; break; }
        }
        if (wrote) break;
    }
    t.assert_true("payload written", wrote);
}

int main(void) {
    testing t(std::cout);
    t.test("bcast tile counts", test_bcast_tile_counts);
    t.test("1B no-KV host ABI", test_1b_no_kv_abi);
    t.test("3B no-KV host ABI", test_3b_no_kv_abi);
    t.test("3B with-KV host ABI", test_3b_with_kv_abi);
    t.test("pack_bcast respects descriptor bounds", test_pack_bcast_bounds);
    return t.summary();
}