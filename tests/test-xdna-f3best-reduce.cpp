#include "xdna-f3best-reduce.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t kCanaryBits = 0x7fc01234;

float bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

bool same_bits(float lhs, float rhs) {
    uint32_t lhs_bits;
    uint32_t rhs_bits;
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
    return lhs_bits == rhs_bits;
}

bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-xdna-f3best-reduce: %s\n", message);
    }
    return condition;
}

float partial_value(int64_t head, int64_t channel) {
    const int pattern = static_cast<int>((head * 37 + channel * 19) % 73) - 36;
    return static_cast<float>(pattern) * 0.0625f;
}

float attention_value(int64_t channel) {
    const int pattern = static_cast<int>((channel * 29) % 61) - 30;
    return static_cast<float>(pattern) * 0.03125f;
}

bool test_3b_geometry_with_diagnostics() {
    constexpr int64_t kEmbeddingDim = 3072;
    constexpr int64_t kNumPartials = 8;
    constexpr int64_t kPartialStride = kEmbeddingDim;
    constexpr size_t kGuard = 17;

    std::vector<uint16_t> partials(
        static_cast<size_t>(kNumPartials * kPartialStride));
    std::vector<uint16_t> post_attention(static_cast<size_t>(kEmbeddingDim));
    for (int64_t channel = 0; channel < kEmbeddingDim; ++channel) {
        post_attention[static_cast<size_t>(channel)] = f32_to_bf16(attention_value(channel));
        for (int64_t head = 0; head < kNumPartials; ++head) {
            partials[static_cast<size_t>(head * kPartialStride + channel)] =
                f32_to_bf16(partial_value(head, channel));
        }
    }

    std::vector<float> final(kGuard + kEmbeddingDim + kGuard, 0.0f);
    std::vector<float> diagnostic_attention(kGuard + kEmbeddingDim + kGuard, 0.0f);
    std::vector<float> diagnostic_ffn(kGuard + kEmbeddingDim + kGuard, 0.0f);
    const float canary = [] {
        float result;
        std::memcpy(&result, &kCanaryBits, sizeof(result));
        return result;
    }();
    for (size_t i = 0; i < kGuard; ++i) {
        final[i] = final[kGuard + kEmbeddingDim + i] = canary;
        diagnostic_attention[i] = diagnostic_attention[kGuard + kEmbeddingDim + i] = canary;
        diagnostic_ffn[i] = diagnostic_ffn[kGuard + kEmbeddingDim + i] = canary;
    }

    xdna_f3best_reduce_diagnostics diagnostics{
        diagnostic_attention.data() + kGuard,
        diagnostic_ffn.data() + kGuard,
        static_cast<size_t>(kEmbeddingDim),
    };
    if (!check(xdna_f3best_reduce_f32(
            partials.data(), post_attention.data(), kEmbeddingDim, kNumPartials,
            kPartialStride, final.data() + kGuard, &diagnostics),
            "3B reduction unexpectedly failed")) {
        return false;
    }

    for (int64_t channel = 0; channel < kEmbeddingDim; ++channel) {
        float expected_ffn = 0.0f;
        for (int64_t head = 0; head < kNumPartials; ++head) {
            expected_ffn += bf16_to_f32(
                partials[static_cast<size_t>(head * kPartialStride + channel)]);
        }
        const float expected_attention = bf16_to_f32(post_attention[static_cast<size_t>(channel)]);
        const size_t index = kGuard + static_cast<size_t>(channel);
        if (!check(same_bits(final[index], expected_attention + expected_ffn),
                   "final output differs from independent scalar expectation") ||
            !check(same_bits(diagnostic_attention[index], expected_attention),
                   "post-attention diagnostic differs from independent scalar expectation") ||
            !check(same_bits(diagnostic_ffn[index], expected_ffn),
                   "FFN diagnostic differs from independent scalar expectation")) {
            return false;
        }
    }

    for (size_t i = 0; i < kGuard; ++i) {
        if (!check(same_bits(final[i], canary) && same_bits(final[kGuard + kEmbeddingDim + i], canary),
                   "final-output canary changed") ||
            !check(same_bits(diagnostic_attention[i], canary) &&
                   same_bits(diagnostic_attention[kGuard + kEmbeddingDim + i], canary),
                   "post-attention diagnostic canary changed") ||
            !check(same_bits(diagnostic_ffn[i], canary) &&
                   same_bits(diagnostic_ffn[kGuard + kEmbeddingDim + i], canary),
                   "FFN diagnostic canary changed")) {
            return false;
        }
    }

    return true;
}

bool test_null_diagnostics_do_not_write_implicit_scratch() {
    constexpr int64_t kEmbeddingDim = 13;
    constexpr int64_t kNumPartials = 2;
    constexpr int64_t kPartialStride = 16;
    constexpr size_t kGuard = 11;

    std::vector<uint16_t> partials(static_cast<size_t>(kNumPartials * kPartialStride), 0);
    std::vector<uint16_t> post_attention(static_cast<size_t>(kEmbeddingDim), 0);
    for (int64_t channel = 0; channel < kEmbeddingDim; ++channel) {
        post_attention[static_cast<size_t>(channel)] = f32_to_bf16(attention_value(channel));
        for (int64_t head = 0; head < kNumPartials; ++head) {
            partials[static_cast<size_t>(head * kPartialStride + channel)] =
                f32_to_bf16(partial_value(head, channel));
        }
    }

    const float sentinel = -12345.0f;
    std::vector<float> output(kGuard + kEmbeddingDim + kGuard, sentinel);
    float * final = output.data() + kGuard;
    if (!check(xdna_f3best_reduce_f32(
            partials.data(), post_attention.data(), kEmbeddingDim, kNumPartials,
            kPartialStride, final, nullptr),
            "null-diagnostics reduction unexpectedly failed")) {
        return false;
    }

    for (size_t i = 0; i < kGuard; ++i) {
        if (!check(output[i] == sentinel && output[kGuard + kEmbeddingDim + i] == sentinel,
                   "null diagnostics modified memory outside ordinary final output")) {
            return false;
        }
    }

    xdna_f3best_reduce_diagnostics invalid_diagnostics{
        nullptr,
        nullptr,
        static_cast<size_t>(kEmbeddingDim),
    };
    if (!check(!xdna_f3best_reduce_f32(
            partials.data(), post_attention.data(), kEmbeddingDim, kNumPartials,
            kPartialStride, final, &invalid_diagnostics),
            "invalid diagnostics descriptor did not fail closed")) {
        return false;
    }

    return true;
}

} // namespace

int main() {
    return test_3b_geometry_with_diagnostics() &&
            test_null_diagnostics_do_not_write_implicit_scratch() ? 0 : 1;
}
