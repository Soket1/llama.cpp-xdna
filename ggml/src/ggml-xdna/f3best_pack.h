// SPDX-License-Identifier: MIT
// Host-side weight packers for the f3best fused decode layer (XDNA_OP_DECODE_LAYER_F3BEST).
// Ported byte-for-byte from the validated Python reference build_ofold8_f3best_real.py
// (pack_gemv / pack_gate_bcast / pack_down_bcast). Header-only so the backend dispatch
// fn AND the standalone packing test share one source of truth.
//
// A-stream per head (NH=8) = [Wq | Wo | gate | up | down], PACKED=4608 bytes/tile:
//   Wq, Wo   -> f3best_pack_gemv   (row-tile: m rows nibbles, then m rows bf16 scales)
//   gate, up -> f3best_pack_bcast  (32x256 block, COLUMN-major nibbles + 8 scale groups)
//   down     -> f3best_pack_bcast  (same layout; K=H8 instead of E)
#pragma once
#include <cstdint>
#include <cstring>

namespace f3b {

// fp16 (IEEE half) bits -> bf16 bits, round-to-nearest-even. Mirrors
// ggml-xdna.cpp::fp16_to_bf16 exactly (must stay in sync).
static inline uint16_t fp16_to_bf16(uint16_t h) {
    const uint32_t sign     = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp_f16  = (h >> 10) & 0x1F;
    const uint32_t mant_f16 = h & 0x3FF;
    uint32_t f32_bits;
    if (exp_f16 == 0) {
        if (mant_f16 == 0) {
            f32_bits = sign;
        } else {
            uint32_t m = mant_f16;
            int e = -1;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3FF;
            const uint32_t exp_f32 = (uint32_t)(127 - 15 + e + 1);
            f32_bits = sign | (exp_f32 << 23) | (m << 13);
        }
    } else if (exp_f16 == 0x1F) {
        f32_bits = sign | (0xFFu << 23) | (mant_f16 << 13);
    } else {
        const uint32_t exp_f32 = (uint32_t)((int)exp_f16 - 15 + 127);
        f32_bits = sign | (exp_f32 << 23) | (mant_f16 << 13);
    }
    f32_bits += (0x7FFF + ((f32_bits >> 16) & 1));
    return (uint16_t)(f32_bits >> 16);
}

// Row-tile GEMV pack (Wq/Wo): per tile of m_input consecutive output rows, emit the
// signed int4 nibbles (2/byte) for all rows, then the bf16 group scales for all rows.
// Identical to ggml-xdna.cpp::xdna_ffn16_pack_tiles(..., sub8=true). q4_0_src is the
// raw Q4_0 weight (n_rows x (K/G) blocks of 18 bytes). pad_to = PACKED (4608).
static inline void pack_gemv(const uint8_t * q4_0_src, int64_t n_rows, int64_t K,
                             int m_input, int group_size, size_t pad_to,
                             uint8_t * tiles_out) {
    const int64_t num_groups = K / group_size;
    const size_t  Q4_0_BLOCK = 18;                       // fp16 scale + 16 nibble bytes
    const size_t  row_stride  = (size_t)num_groups * Q4_0_BLOCK;
    const size_t  wbytes_row  = (size_t)K / 2;
    const size_t  sbytes_row  = (size_t)num_groups * 2;
    const int64_t n_tiles     = n_rows / m_input;
    for (int64_t t = 0; t < n_tiles; t++) {
        uint8_t * tile_dst = tiles_out + (size_t)t * pad_to;
        for (int r = 0; r < m_input; r++) {
            const uint8_t * row_src = q4_0_src + (size_t)(t * m_input + r) * row_stride;
            uint8_t * row_dst = tile_dst + (size_t)r * wbytes_row;
            for (int64_t g = 0; g < num_groups; g++) {
                const uint8_t * qs = row_src + (size_t)g * Q4_0_BLOCK + 2;
                uint8_t e[32];
                for (int j = 0; j < 16; j++) {
                    uint8_t lo = (uint8_t)(((qs[j]      & 0x0F) - 8) & 0x0F);
                    uint8_t hi = (uint8_t)((((qs[j] >> 4) & 0x0F) - 8) & 0x0F);
                    e[j] = lo; e[j + 16] = hi;
                }
                uint8_t * blk = row_dst + (size_t)g * 16;
                for (int k = 0; k < 16; k++) blk[k] = (uint8_t)(e[2*k] | (e[2*k + 1] << 4));
            }
        }
        const size_t scale_region = (size_t)m_input * wbytes_row;
        for (int r = 0; r < m_input; r++) {
            const uint8_t * row_src = q4_0_src + (size_t)(t * m_input + r) * row_stride;
            uint16_t * sdst = (uint16_t *)(tile_dst + scale_region + (size_t)r * sbytes_row);
            for (int64_t g = 0; g < num_groups; g++) {
                uint16_t fp16; memcpy(&fp16, row_src + (size_t)g * Q4_0_BLOCK, 2);
                sdst[g] = fp16_to_bf16(fp16);
            }
        }
    }
}

// Broadcast-GEMV pack (gate/up/down): weight is n_rows (output) x K_full (reduction),
// row-major Q4_0. We pack the K-slice [k_off, k_off+K_pack) (K_pack columns), tiled into
// (n_rows/32) x (K_pack/256) blocks, block-major then chunk-minor. Per tile:
//   * COLUMN-major nibbles: for each of 256 K-columns, the 32 output-row signed int4
//     weights packed 2 rows/byte (16 bytes) -> 4096 bytes.
//   * scales: for each of 8 K-groups in the chunk, the 32 row scales (bf16) -> 512 bytes.
// Total 4608 = PACKED. Mirrors Python pack_gate_bcast/pack_down_bcast.
//   gate/up (per-head ROW slice): src already offset to the head; K_pack=K_full=E, k_off=0.
//   down (per-head COLUMN slice of [E x HH]): src = full down; K_pack=H8, K_full=HH, k_off=h*H8.
static inline void pack_bcast(const uint8_t * q4_0_src, int64_t n_rows,
                              int64_t K_pack, int64_t K_full, int64_t k_off,
                              int group_size, size_t pad_to, uint8_t * tiles_out) {
    const int N = 32, KC = 256;
    const int64_t num_groups_full = K_full / group_size;     // groups per full output row
    const size_t  row_stride  = (size_t)num_groups_full * 18;
    const int64_t g_off = k_off / group_size;                // first group of the slice
    const int64_t blocks = n_rows / N;
    const int64_t chunks = K_pack / KC;
    const int     sg_per_chunk = KC / group_size;            // 8
    auto nib = [&](int64_t row, int64_t kcol) -> uint8_t {   // kcol is slice-local
        int64_t g = (k_off + kcol) / group_size; int within = (int)((k_off + kcol) % group_size);
        const uint8_t * qs = q4_0_src + (size_t)row * row_stride + (size_t)g * 18 + 2;
        uint8_t v = (within < 16) ? (qs[within] & 0x0F) : ((qs[within - 16] >> 4) & 0x0F);
        return (uint8_t)((v - 8) & 0x0F);
    };
    for (int64_t block = 0; block < blocks; block++) {
        for (int64_t chunk = 0; chunk < chunks; chunk++) {
            uint8_t * tile_dst = tiles_out + (size_t)(block * chunks + chunk) * pad_to;
            int64_t r0 = block * N, c0 = chunk * KC;
            uint8_t * wdst = tile_dst;
            for (int c = 0; c < KC; c++) {
                for (int k = 0; k < N / 2; k++) {
                    uint8_t lo = nib(r0 + 2*k,     c0 + c);
                    uint8_t hi = nib(r0 + 2*k + 1, c0 + c);
                    *wdst++ = (uint8_t)(lo | (hi << 4));
                }
            }
            uint16_t * sdst = (uint16_t *)(tile_dst + (size_t)KC * (N / 2));
            for (int kg = 0; kg < sg_per_chunk; kg++) {
                int64_t g = g_off + chunk * sg_per_chunk + kg;
                for (int r = 0; r < N; r++) {
                    uint16_t fp16;
                    memcpy(&fp16, q4_0_src + (size_t)(r0 + r) * row_stride + (size_t)g * 18, 2);
                    *sdst++ = fp16_to_bf16(fp16);
                }
            }
        }
    }
}


// Q-bcast pack wrapper for #86 diagnostics: full K columns, k_off=0.
static inline void pack_qo_bcast(const uint8_t * q4_0_src, int64_t n_rows, int64_t K_full,
                                 int group_size, size_t pad_to, uint8_t * tiles_out) {
    pack_bcast(q4_0_src, n_rows, K_full, K_full, 0, group_size, pad_to, tiles_out);
}

}  // namespace f3b
