// SPDX-License-Identifier: MIT
// Private host-side reduction for the f3best fused decode layer.
#pragma once

#include <cstddef>
#include <cstdint>

struct xdna_f3best_reduce_diagnostics {
    float * post_attention = nullptr;
    float * ffn_contribution = nullptr;
    size_t capacity = 0;
};

// Reduce BF16 f3best partial streams into an F32 final output. Each partial stream
// begins at partials + h * partial_stride; post_attention contains embedding_dim
// BF16 values. Diagnostics, when requested, write the two semantic intermediates
// into caller-owned regions with capacity values each. No tensor metadata or
// allocation ownership is inferred here.
bool xdna_f3best_reduce_f32(
        const uint16_t * partials,
        const uint16_t * post_attention,
        int64_t embedding_dim,
        int64_t num_partials,
        int64_t partial_stride,
        float * final_output,
        const xdna_f3best_reduce_diagnostics * diagnostics = nullptr);
