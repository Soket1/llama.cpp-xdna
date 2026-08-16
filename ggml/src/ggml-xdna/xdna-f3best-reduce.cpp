// SPDX-License-Identifier: MIT
#include "xdna-f3best-reduce.h"

#include <cstring>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

inline float xdna_f3best_bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool xdna_f3best_reduce_dimensions_valid(
        int64_t embedding_dim,
        int64_t num_partials,
        int64_t partial_stride) {
    if (embedding_dim <= 0 || num_partials <= 0 || partial_stride < embedding_dim) {
        return false;
    }

    const size_t stride = static_cast<size_t>(partial_stride);
    const size_t count = static_cast<size_t>(num_partials);
    return stride <= std::numeric_limits<size_t>::max() / count;
}

} // namespace

bool xdna_f3best_reduce_f32(
        const uint16_t * partials,
        const uint16_t * post_attention,
        int64_t embedding_dim,
        int64_t num_partials,
        int64_t partial_stride,
        float * final_output,
        const xdna_f3best_reduce_diagnostics * diagnostics) {
    if (!partials || !post_attention || !final_output ||
        !xdna_f3best_reduce_dimensions_valid(embedding_dim, num_partials, partial_stride)) {
        return false;
    }

    if (diagnostics && (!diagnostics->post_attention || !diagnostics->ffn_contribution ||
                        diagnostics->capacity < static_cast<size_t>(embedding_dim))) {
        return false;
    }

    const bool write_diagnostics = diagnostics != nullptr;

#if defined(__AVX2__)
    int64_t e = 0;
    for (; e + 8 <= embedding_dim; e += 8) {
        __m256 ffn = _mm256_setzero_ps();
        for (int64_t h = 0; h < num_partials; ++h) {
            const uint16_t * stream = partials +
                static_cast<size_t>(h) * static_cast<size_t>(partial_stride) +
                static_cast<size_t>(e);
            const __m128i bf16_values = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(stream));
            const __m256i expanded = _mm256_cvtepu16_epi32(bf16_values);
            ffn = _mm256_add_ps(
                ffn, _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16)));
        }
        const __m128i attention_bf16 = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(post_attention + e));
        const __m256i attention_expanded = _mm256_cvtepu16_epi32(attention_bf16);
        const __m256 attention = _mm256_castsi256_ps(
            _mm256_slli_epi32(attention_expanded, 16));
        _mm256_storeu_ps(final_output + e, _mm256_add_ps(attention, ffn));
        if (write_diagnostics) {
            _mm256_storeu_ps(diagnostics->post_attention + e, attention);
            _mm256_storeu_ps(diagnostics->ffn_contribution + e, ffn);
        }
    }
#else
    int64_t e = 0;
#endif

    for (; e < embedding_dim; ++e) {
        float ffn = 0.0f;
        for (int64_t h = 0; h < num_partials; ++h) {
            ffn += xdna_f3best_bf16_to_f32(partials[
                static_cast<size_t>(h) * static_cast<size_t>(partial_stride) +
                static_cast<size_t>(e)]);
        }
        const float attention = xdna_f3best_bf16_to_f32(post_attention[e]);
        final_output[e] = attention + ffn;
        if (write_diagnostics) {
            diagnostics->post_attention[e] = attention;
            diagnostics->ffn_contribution[e] = ffn;
        }
    }

    return true;
}
