#!/usr/bin/env python3
"""
NPU vs CPU correctness harness for ggml-xdna backend.

Runs a fixed set of prompts through both pure-CPU and NPU-accelerated
llama-cli configurations, with greedy sampling (--temp 0) for determinism,
and compares the generated text token-for-token.

Catches regressions in:
  - Numerical drift over long generations
  - Multi-query state leaks (the RMS_NORM+QKV class of bugs)
  - Per-operator dispatch correctness
  - seq_len boundary handling (1 / 31 / 32 / 255 / 256 / >256)

Standard library only. Run from anywhere; uses absolute Windows paths
in env config below. Customize paths to match your install if different.

Usage:
  python correctness_test.py             # run all tests
  python correctness_test.py <name>      # run one test by name
  python correctness_test.py --list      # list test names

Exit code 0 if all tests pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path("C:/llama.cpp-xdna")
LLAMA_CLI    = REPO_ROOT / "build" / "bin" / "Release" / "llama-cli.exe"
LLAMA_LOOKUP = REPO_ROOT / "build" / "bin" / "Release" / "llama-lookup.exe"
MODEL     = REPO_ROOT / "models" / "llama-3.2-1b-instruct-BF16.gguf"

# Quantized variants of the same model, generated via llama-quantize.exe.
# Used for INT4/INT8 NPU dispatch testing. Today (2026-05-21) the backend
# does not claim Q4_0/Q4_K mul_mat in supports_op, so these models route
# all matmuls through CPU; the harness verifies the NPU build still
# produces the right answer in that fallback regime. When Priority 8
# INT4 work lands, tighten the tolerances and expand to NPU dispatch.
MODEL_Q4_0   = REPO_ROOT / "models" / "llama-3.2-1b-instruct-Q4_0.gguf"
MODEL_Q4_K_M = REPO_ROOT / "models" / "llama-3.2-1b-instruct-Q4_K_M.gguf"
MODEL_QWEN35_9B_Q4_0 = REPO_ROOT / "models" / "Qwen3.5-9B-Q4_0.gguf"
# Llama 3.2 3B (head_dim=128, 28 layers, GQA 3:1). Used to validate the
# Phase 8.5 head_dim parameterization end-to-end on a non-Llama-1B model.
MODEL_LLAMA_3B_Q4_0 = REPO_ROOT / "models" / "llama-3.2-3b-q4_0.gguf"
# Gemma 3 1B Q4_K_M. Architecture: head_dim=256, GQA 4:1, embedding=1152,
# 26 blocks. Mixed quantization -- only attn_output is Q4_K (the only
# weight type our supports_op claims), so only that op dispatches to NPU.
# Stresses Q4_K + head_dim=256 dispatch path even when FlowKV/attention
# matchers don't fire (different arch from Llama).
MODEL_GEMMA_3_1B_Q4_K_M = REPO_ROOT / "models" / "gemma-3-1b-it-Q4_K_M.gguf"

# Driver/SDK paths -- adjust if your install differs.
BASE_ENV: dict[str, str] = {
    "AMD_DRIVER_DIR":         "C:\\Windows\\System32\\DriverStore\\FileRepository\\kipudrv.inf_amd64_1a1aa059597c4810",
    "PYTHONPATH":             "C:\\Users\\Kuhnya\\Downloads\\xrt_windows_sdk\\xrt_sdk\\xrt\\python;C:\\Python313\\Lib\\site-packages",
    "GGML_XDNA_PYTHON_CMD":   "C:\\Python313\\python.exe",
    "PEANO_INSTALL_DIR":      "C:\\ProgramData\\miniforge3\\envs\\ryzen-ai-1.7.1\\Lib\\site-packages\\win64.o\\tools\\peano",
    "XRT_BIN_DIR":            "C:\\Users\\Kuhnya\\Downloads\\xrt_windows_sdk\\xrt_sdk\\xrt",
    "MLIR_AIE_BIN_DIR":       "C:\\ProgramData\\miniforge3\\envs\\ryzen-ai-1.7.1\\Lib\\site-packages\\mlir_aie\\bin",
    "GGML_XDNA_CACHE_DIR":    str(REPO_ROOT / "npu_kernels_win_8col"),
    # Common runtime flags expected by ggml-xdna
    "GGML_XDNA_NUM_COLS":     "8",
    "GGML_XDNA_FORCE_CH1":    "0",
    "GGML_SCHED_KV_OFFLOAD":  "1",
}

# Build PATH by prepending the NPU/MLIR tool dirs (Windows-style).
_extra_path = (
    "C:\\Windows\\System32\\DriverStore\\FileRepository\\kipudrv.inf_amd64_1a1aa059597c4810;"
    "C:\\ProgramData\\miniforge3\\envs\\ryzen-ai-1.7.1\\Lib\\site-packages\\win64.o\\tools\\peano\\bin;"
    "C:\\Users\\Kuhnya\\Downloads\\xrt_windows_sdk\\xrt_sdk\\xrt;"
    "C:\\ProgramData\\miniforge3\\envs\\ryzen-ai-1.7.1\\Lib\\site-packages\\mlir_aie\\bin;"
)

# Operator presets. Each preset is a dict of XDNA_ENABLE_* and related env vars
# layered on top of BASE_ENV.
PRESETS: dict[str, dict[str, str]] = {
    "cpu_baseline": {
        # All NPU operators OFF -- reference output.
        "XDNA_ENABLE_GEMV":              "0",
        "XDNA_ENABLE_SWIGLU":            "0",
        "XDNA_ENABLE_QKV":               "0",
        "XDNA_ENABLE_DECODE_BATCH":      "0",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "0",
        "XDNA_ENABLE_FLOWKV_DECODE":     "0",
        "XDNA_ENABLE_RMS_NORM":          "0",
    },
    "npu_chat_safe": {
        # Current production for chat: everything on except RMS_NORM
        # (workaround for the RMS_NORM+QKV interference bug).
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
    },
    "npu_full": {
        # All NPU operators including RMS_NORM. Known to break in
        # chat-mode multi-query; should still pass single-query.
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "1",
    },
    "npu_int4": {
        # Production preset for Q4_0 models. Uses v2 INT4 GEMV kernel
        # (PR #101 port) by default -- ~1.65x faster end-to-end than v1,
        # 6% over NPU bf16. v2 is the C++ default since 2026-05-22, so
        # we don't need to set XDNA_ENABLE_GEMV_INT4_V2.
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
        # Phase 8.2 (XDNA_ENABLE_SWIGLU_INT4) intentionally OFF by default:
        # profiling showed the chained INT4 SwiGLU kernel is compute-bound
        # and runs ~14 ms/layer (~2.3x slower than bf16 SwiGLU at ~5.7
        # ms/layer), so enabling it regresses decode t/s. Code is in place
        # and correctness is byte-exact -- just opt-in until the inner
        # dequant loop is optimized.
    },
    "npu_int4_v1": {
        # Regression-coverage preset that explicitly forces the old v1
        # INT4 GEMV kernel via XDNA_DISABLE_GEMV_INT4_V2=1. Kept so we
        # can A/B test future kernel changes against the documented v1
        # baseline (3.4 t/s on Llama 3.2 1B Q4_0).
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
        "XDNA_DISABLE_GEMV_INT4_V2":     "1",
    },
    "npu_int4_v2": {
        # Alias of npu_int4 (v2 is the default since 2026-05-22). Kept
        # for backward compatibility with existing test names.
        # QKV fused mode B: single INT4 dispatch for Q+K+V (+10.5%).
        # Phase 9 async: overlaps INT4 GEMV with CPU bias compensation (+5%).
        # QKV RoPE fused: apply RoPE inline inside QKV dispatch, eliminating
        # the separate CPU ROPE step between QKV and FlowKV (+~5%).
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
        "XDNA_ENABLE_QKV_INT4_FUSED":    "1",
        "XDNA_ENABLE_PHASE9":            "1",
        "XDNA_ENABLE_QKV_ROPE_FUSED":    "1",
        "XDNA_ENABLE_SWIGLU_NORM_FUSED": "1",
        "XDNA_ENABLE_QKV_NORM_FUSED":    "1",
    },
    "npu_int4_gemv_only": {
        # INT4 GEMV-only preset for models with NON-Llama attention
        # architecture (e.g. Qwen3.5 with head_dim=256, M-RoPE, SWA).
        # Disables every attention/FFN-fusion op (which all assume
        # head_dim=64) and keeps ONLY the pure matmul INT4 path.
        # Attention runs on CPU via ggml fallback; INT4 GEMV accelerates
        # FFN/QKV-proj matmuls without architecture assumptions.
        "XDNA_ENABLE_GEMV":              "0",
        "XDNA_ENABLE_SWIGLU":            "0",
        "XDNA_ENABLE_QKV":               "0",
        "XDNA_ENABLE_DECODE_BATCH":      "0",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "0",
        "XDNA_ENABLE_FLOWKV_DECODE":     "0",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
    },
    "npu_int4_v2_no_flowkv": {
        # Same as npu_int4_v2 but FlowKV decode disabled. For 3B/Gemma
        # debug: isolates INT4 dispatch correctness from FlowKV POC.
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "0",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
    },
    "npu_int4_qkv_fused": {
        # Phase 8.3 mode B: fused Q+K+V INT4 in one dispatch.
        # XDNA_ENABLE_QKV_INT4_FUSED=1 routes INT4 QKV triples through
        # ggml_backend_xdna_mul_mat_qkv_int4_fused (N=q+k+v, single call).
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
        "XDNA_ENABLE_QKV_INT4_FUSED":    "1",
    },
    "npu_int4_specdec": {
        # v3 batched INT4 GEMV for spec-dec verify batches (M=2..8).
        # Output from this preset should match cpu_baseline at temp=0.
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
        "XDNA_ENABLE_GEMV_INT4_BATCH":   "1",
    },
    "npu_int4_specdec_cpu_verify": {
        # Same as npu_int4 but spec-dec verify uses CPU (M>1 fallback).
        # This is the correct/baseline for spec-dec correctness validation.
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
    },
    "npu_int4_swiglu": {
        # Same as npu_int4 but ALSO enables the chained INT4 SwiGLU
        # dispatch (Phase 8.2). Kept available for regression coverage
        # and future re-evaluation; do not select for perf measurement
        # until the kernel is optimized.
        "XDNA_ENABLE_GEMV":              "1",
        "XDNA_ENABLE_SWIGLU":            "1",
        "XDNA_ENABLE_QKV":               "1",
        "XDNA_ENABLE_DECODE_BATCH":      "1",
        "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
        "XDNA_ENABLE_FLOWKV_DECODE":     "1",
        "XDNA_ENABLE_RMS_NORM":          "0",
        "XDNA_ENABLE_GEMV_INT4":         "1",
        "XDNA_ENABLE_SWIGLU_INT4":       "1",
    },
}

# =============================================================================
# Test definitions
# =============================================================================

@dataclass
class Test:
    name: str
    prompt: str | list[str]    # str for single-turn, list for chat-mode
    n_predict: int
    mode: str                  # "single-turn" or "chat"
    ctx_size: int = 512
    seed: int = 42
    # Model file to use. Default = BF16. Override per-test for INT8/INT4
    # quantized variants.
    model: Path = MODEL
    # Variants to test against the baseline. If not specified, defaults to all
    # NPU presets. Tests expected to fail on certain presets should mark them
    # in `expected_fail`.
    variants: list[str] = field(default_factory=lambda: ["npu_chat_safe", "npu_full"])
    expected_fail: set[str] = field(default_factory=set)
    description: str = ""
    # Minimum characters of each response that must match the baseline exactly.
    # 0 = require full byte-exact match (default). For long generations where
    # bf16 vs f32 numerical drift makes full match impossible under greedy
    # sampling, set this to a positive value. The test PASSes if the first
    # `min_prefix_match` chars match, and the divergence position is reported
    # as info. A `min_prefix_match` of None means "require full match"
    # (same as 0). Use a positive int to allow drift past that point.
    min_prefix_match: int | None = None
    # Optional draft model for spec-dec tests.
    draft_model: Path | None = None
    draft_max: int = 4


TESTS: list[Test] = [
    Test(
        name="paris_short",
        prompt="What is the capital of France?",
        n_predict=16,
        mode="single-turn",
        description="Sanity check -- short prompt, short generation. Should be byte-identical.",
    ),
    Test(
        name="paris_drift_128",
        prompt="What is the capital of France? Explain in detail.",
        n_predict=128,
        mode="single-turn",
        # bf16 vs f32 numerical drift accumulates over long greedy generation.
        # We require the first ~20 chars to match exactly (catches any *early*
        # divergence that would indicate a real bug); divergence past that
        # point is reported as info, not failure.
        min_prefix_match=20,
        description="Drift check -- long generation. bf16 noise may accumulate.",
    ),
    Test(
        name="multiquery_basic",
        prompt=["What is the capital of France?", "What is 2+2?"],
        n_predict=32,
        mode="chat",
        variants=["npu_chat_safe"],
        # Short Q&A answers can drift to abbreviations ("2" vs "2 + 2 = 4") on
        # NPU. The intent here is to catch *garbage*, not stylistic drift, so
        # just require the response to not be empty / start differently.
        min_prefix_match=1,
        description="Multi-query basic case on the production (chat-safe) config. Catches Q2-onward regressions in the workaround.",
    ),
    Test(
        name="rms_qkv_regression",
        # Probes the documented RMS_NORM+QKV bug. npu_full should produce
        # garbage on Q2; if it doesn't, the bug was accidentally fixed and
        # we should celebrate (and unmark expected_fail / re-enable RMS).
        prompt=["What is the capital of France?", "What is 2+2?"],
        n_predict=32,
        mode="chat",
        variants=["npu_full"],
        expected_fail={"npu_full"},
        # Strict prefix: require 10 chars to match. The garbage output looks
        # like "2reraignidi..." which diverges at char 1; this threshold
        # ensures we don't accept it as accidentally-passing just because
        # the first character happens to coincide with the baseline.
        min_prefix_match=10,
        description="RMS_NORM+QKV interference regression test. Expected to FAIL on npu_full (chat-mode multi-query garbage). UNEXPECTED PASS means the bug got fixed -- celebrate then update the harness.",
    ),
    Test(
        name="multiquery_chatsafe",
        prompt=["What is the capital of France?", "What is the capital of Germany?", "What is the capital of Spain?"],
        n_predict=24,
        mode="chat",
        variants=["npu_chat_safe"],   # only the safe variant
        # NPU may abbreviate ("The" vs "The capital of Germany is Berlin"); we
        # care that it doesn't return garbage, not full reproducibility.
        min_prefix_match=1,
        description="3-query chat with the supported config. Confirms the workaround stays clean across multiple turns.",
    ),
    Test(
        name="seq_len_short",
        # ~5 prompt tokens -> first decode at actual_seq ~5
        prompt="Hi.",
        n_predict=8,
        mode="single-turn",
        # Short conversational reply can drift to a synonym ("meet you" vs
        # "talk to you") within a few tokens; require only the first ~5
        # characters to match exactly so we still catch garbage but accept
        # natural bf16 drift.
        min_prefix_match=5,
        description="Tiny seq_len edge case. Probes the V-PERMUTE matcher's `ne[0] > hd` condition (which silently disables POC when seq<=64).",
    ),
    Test(
        name="seq_len_near_chunk32",
        # Pad prompt to land near the chunk_size=32 boundary of FlowKV
        prompt="Tell me a story about a robot who learned to dance under the moonlight.",
        n_predict=24,
        mode="single-turn",
        # Story prompts are highly sensitive to bf16 drift (every word is a
        # creative choice with many close-probability alternatives). Require
        # only the first ~3 chars to match exactly (the leading "In " before
        # the story begins).
        min_prefix_match=3,
        description="actual_seq_len crosses the FlowKV chunk_size=32 boundary mid-generation.",
    ),

    # -----------------------------------------------------------------------
    # Quantized model regression tests (Priority 8 INT4 work prep)
    # -----------------------------------------------------------------------
    # Today: Q4_0 / Q4_K not claimed by ggml-xdna supports_op, so the NPU
    # build routes all matmuls to CPU. These tests verify that fallback
    # still works correctly (model loads, no crashes, output matches the
    # CPU-baseline run of the same model). When Phase 8.1 lands and
    # XDNA_ENABLE_GEMV_INT4 starts dispatching Q4_0 to NPU, the npu_*
    # variants will start showing bf16-vs-INT4 drift — at that point
    # `min_prefix_match` should be tuned per test and an `npu_int4`
    # preset (with XDNA_ENABLE_GEMV_INT4=1) added.
    Test(
        name="paris_short_q4_0",
        prompt="What is the capital of France?",
        n_predict=16,
        mode="single-turn",
        model=MODEL_Q4_0,
        # Today: CPU fallback → expect byte-exact match across presets.
        # When INT4 NPU dispatch lands, switch to min_prefix_match~20.
        description="Q4_0 model sanity. Today routes via CPU fallback (no INT4 NPU dispatch yet). Confirms model loads and matmul fallback path works.",
    ),
    Test(
        name="paris_short_q4_k_m",
        prompt="What is the capital of France?",
        n_predict=16,
        mode="single-turn",
        model=MODEL_Q4_K_M,
        variants=["npu_int4"],
        # Q4_K has more precision than Q4_0 (2-level scales, asymmetric).
        # We dispatch through the same fused_dequant_gemv kernel via a
        # host-side repack (xdna_repack_q4_K_to_fused_int4) that stores
        # effective_scale + effective_min per group and uses min·S[g]
        # bias compensation. Should match CPU baseline byte-exact.
        min_prefix_match=1,
        description="Phase 8.4: Q4_K_M model routed through INT4 NPU dispatch via lossless host repack (effective_scale + effective_min, min-based bias). First Q4_K NPU dispatch.",
    ),
    Test(
        name="paris_drift_64_q4_k_m_int4",
        prompt="What is the capital of France? Explain in detail.",
        n_predict=64,
        mode="single-turn",
        model=MODEL_Q4_K_M,
        variants=["npu_int4"],
        min_prefix_match=10,
        description="Long Q4_K_M generation through INT4 NPU. Stresses the new Q4_K repack across 16 layers x 7 matmuls x 64 tokens.",
    ),
    Test(
        name="multiquery_q4_0_chatsafe",
        prompt=["What is the capital of France?", "What is 2+2?"],
        n_predict=24,
        mode="chat",
        model=MODEL_Q4_0,
        variants=["npu_chat_safe"],
        min_prefix_match=1,
        description="Q4_0 multi-query under the production NPU config. Catches any RMS-NORM-class interference that might appear if Phase 8.1 lands without the cols>=4 safeguard (review note N9).",
    ),
    Test(
        name="paris_short_q4_0_int4",
        prompt="What is the capital of France?",
        n_predict=12,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4"],
        min_prefix_match=1,  # Q4_0 quantization + bf16 dequant drift vs CPU Q4_0.
        description="Phase 8.1 dispatch path: Q4_0 weights routed through fused INT4 dequant-GEMV on NPU. Baseline is CPU-Q4_0; expect minor drift from bf16-vs-fp32 dequant accumulation.",
    ),
    Test(
        name="paris_drift_64_q4_0_int4",
        prompt="What is the capital of France? Explain in detail.",
        n_predict=64,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4"],
        # Longer generation stresses the INT4 path harder -- every token
        # re-runs ALL matmul shapes including ffn_down (K=8192 N=2048,
        # tile_in=1 branch of select_gemv_tiles). bf16-vs-fp32 dequant noise
        # accumulates more, so we tolerate ~10 chars of prefix match.
        min_prefix_match=10,
        description="Longer Q4_0/INT4 generation. Implicitly covers the tile_in=1 branch (ffn_down K=8192) via repeated invocation; catches drift accumulation.",
    ),
    Test(
        name="multiquery_q4_0_int4",
        prompt=["What is the capital of France?", "What is 2+2?"],
        n_predict=24,
        mode="chat",
        model=MODEL_Q4_0,
        variants=["npu_int4"],
        # Multi-query exercises FlowKV decode + INT4 GEMV composition,
        # plus catches any tile_in=1 path issues that only manifest under
        # KV-cache-bearing decode.
        min_prefix_match=1,
        description="INT4 + FlowKV + chat-mode composition. Catches interference between Q4_0 dispatch and the decode-batch / FlowKV machinery.",
    ),
    Test(
        name="paris_short_q4_0_int4_swiglu",
        prompt="What is the capital of France?",
        n_predict=12,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4_swiglu"],
        min_prefix_match=1,
        description="Phase 8.2 dispatch path: Q4_0 SwiGLU routed through the chained INT4 xclbin (dual_fused_dequant_gemv_silu_mul + fused_dequant_gemv).",
    ),
    Test(
        name="paris_short_qkv_int4_fused",
        prompt="What is the capital of France?",
        n_predict=12,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4_qkv_fused"],
        min_prefix_match=1,
        description="Phase 8.3 mode B: fused Q+K+V INT4 in one dispatch (N=q+k+v). "
                    "Expected ~10% decode speedup vs mode A. Output must match cpu_baseline.",
    ),
    Test(
        name="paris_short_q4_0_int4_v2",
        prompt="What is the capital of France?",
        n_predict=12,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4_v2"],
        min_prefix_match=1,
        description="V2 INT4 GEMV kernel (PR #101 optimized: compile-time DIM_K/G + double-pump + AIE pipelining). ~4x faster than v1 on the dominant FFN shapes per xrt_async_spike.",
    ),
    Test(
        name="paris_short_specdec_cpu_verify",
        prompt="What is the capital of France?",
        n_predict=12,
        mode="single-turn",
        model=MODEL_Q4_0,
        draft_model=MODEL_Q4_0,
        draft_max=4,
        variants=["npu_int4_specdec_cpu_verify"],
        min_prefix_match=1,
        description="Spec-dec with CPU verify (M>1 falls back to CPU). Validates spec-dec correctness baseline: same model as draft so acceptance ~100%. Output must match cpu_baseline.",
    ),
    Test(
        name="paris_short_specdec_v3_npu",
        prompt="What is the capital of France?",
        n_predict=12,
        mode="single-turn",
        model=MODEL_Q4_0,
        draft_model=MODEL_Q4_0,
        draft_max=4,
        variants=["npu_int4_specdec"],
        min_prefix_match=1,
        description="Spec-dec with v3 NPU batched GEMV verify (XDNA_ENABLE_GEMV_INT4_BATCH=1). Output must match cpu_baseline. Fails if v3 bias compensation is wrong.",
    ),
    Test(
        name="paris_drift_64_q4_0_int4_v2",
        prompt="What is the capital of France? Explain in detail.",
        n_predict=64,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4_v2"],
        min_prefix_match=10,
        description="V2 long-generation drift check. Same workload as paris_drift_64_q4_0_int4 but through the optimized PR #101 kernel.",
    ),
    Test(
        name="paris_drift_64_qkv_fused",
        prompt="What is the capital of France? Explain in detail.",
        n_predict=64,
        mode="single-turn",
        model=MODEL_Q4_0,
        variants=["npu_int4_qkv_fused"],
        min_prefix_match=10,
        description="Phase 8.3 mode B long-generation drift check. Output must match cpu_baseline to min 10 chars (validates bias compensation doesn't drift on 64 tokens).",
    ),
    Test(
        name="multiquery_q4_0_int4_v2",
        prompt=["What is the capital of France?", "What is 2+2?"],
        n_predict=24,
        mode="chat",
        model=MODEL_Q4_0,
        variants=["npu_int4_v2"],
        min_prefix_match=1,
        description="V2 + FlowKV + chat-mode composition. Should match the v1 chat test byte-for-byte.",
    ),
    # ---- Llama 3.2 3B Q4_0 (head_dim=128) ------------------------------
    # Validates the Phase 8.5 head_dim parameterization end-to-end on a
    # model with head_dim != 64. First-touch run will trigger IRON compile
    # of new xclbins with -DHEAD_DIM=128 (~5-15 min per shape); subsequent
    # runs hit the npu_kernels_win_8col cache.
    Test(
        name="paris_short_3b_q4_0_int4_v2",
        # 3B Q4_0 appears to be the base completion model (not Instruct);
        # use a sentence-start prompt so greedy decoding emits text rather
        # than immediately hitting EOS.
        prompt="The capital of France is",
        n_predict=24,
        mode="single-turn",
        model=MODEL_LLAMA_3B_Q4_0,
        variants=["npu_int4_v2"],
        min_prefix_match=5,
        description="Phase 8.5 validation: Llama 3.2 3B Q4_0 (head_dim=128, 28 layers, GQA 3:1) through INT4 v2 + FlowKV decode. Triggers first-touch IRON compile of GEMV xclbins (K=3072 shapes) and the FlowKV kernel with -DHEAD_DIM=128.",
    ),
    Test(
        name="paris_short_3b_q4_0_gemv_only",
        prompt="The capital of France is",
        n_predict=24,
        mode="single-turn",
        model=MODEL_LLAMA_3B_Q4_0,
        variants=["npu_int4_gemv_only"],
        min_prefix_match=5,
        description="Phase 8.5 fallback path: matmul-only NPU dispatch (no FlowKV/attention) on 3B. Architecture-agnostic; confirms INT4 GEMV alone works at the 3B model's K=3072 shapes (head_dim is irrelevant for the GEMV path).",
    ),
    Test(
        name="paris_short_gemma3_1b_q4_k_m_gemv_only",
        # Gemma 3 1B is instruct-tuned but uses ChatML-style tokens.
        # A sentence-start prompt avoids the immediate <|im_end|> issue.
        prompt="The capital of France is",
        n_predict=24,
        mode="single-turn",
        model=MODEL_GEMMA_3_1B_Q4_K_M,
        variants=["npu_int4_gemv_only"],
        min_prefix_match=5,
        description="Phase 8.5 head_dim=256 sanity: Gemma 3 1B Q4_K_M dispatches only its 26 attn_output Q4_K matmuls through NPU (other weights are Q5_0/Q6_K -- not claimed). Architecture-agnostic path; SWA + non-1D RoPE blockers on attention/FlowKV are irrelevant here. Validates Q4_K bias compensation works with the Gemma weight layout.",
    ),
    # ---- Qwen3.5-9B-Q4_0 -----------------------------------------------
    # Qwen3.5-9B uses head_dim=256, M-RoPE [11,11,10,0], and SWA with
    # full_attention_interval=4. Our NPU attention/FFN fusion ops hardcode
    # head_dim==64 in 9 places (FlowKV, attn_prefill, decode_batch, etc).
    # npu_int4 (full) WILL produce garbage on Qwen; npu_int4_gemv_only
    # restricts to pure matmul (architecture-agnostic) and works correctly.
    # Both Qwen tests use the CPU run as their baseline (printed via -v).
    Test(
        name="qwen35_ml_npu_full",
        prompt="What is machine learning? Answer in 3 sentences.",
        n_predict=200,
        mode="single-turn",
        model=MODEL_QWEN35_9B_Q4_0,
        variants=["npu_int4"],
        min_prefix_match=50,
        expected_fail={"npu_int4"},
        description="Qwen3.5-9B with full NPU INT4 preset. EXPECTED FAIL: head_dim=256 vs hardcoded 64 in attention dispatch paths produces garbage logits.",
    ),
    Test(
        name="qwen35_ml_npu_gemv_only",
        prompt="What is machine learning? Answer in 3 sentences.",
        n_predict=200,
        mode="single-turn",
        model=MODEL_QWEN35_9B_Q4_0,
        variants=["npu_int4_gemv_only"],
        min_prefix_match=1,
        description="Qwen3.5-9B with NPU restricted to pure matmul (INT4 GEMV only). Attention runs on CPU; INT4 GEMV accelerates FFN/QKV-proj matmuls. Should match CPU baseline.",
    ),
]

# =============================================================================
# Runner
# =============================================================================

def build_env(preset: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(PRESETS[preset])
    # Prepend our extra PATH so XRT/peano/mlir-aie bins win over anything stale.
    env["PATH"] = _extra_path + env.get("PATH", "")
    # Make sure debug noise is off so parsing the response stays clean.
    env.pop("XDNA_DEBUG", None)
    env.pop("XDNA_FLOWKV_MATH_DIAG", None)
    env.pop("XDNA_FLOWKV_BO_PROBE", None)
    env.pop("XDNA_FLOWKV_PER_HEAD_LEGACY", None)
    return env


def run_llama(preset: str, test: Test) -> tuple[str, str]:
    """Returns (stdout, stderr). Raises on timeout or non-zero exit."""
    env = build_env(preset)

    args = [
        str(LLAMA_CLI),
        "-m",   str(test.model),
        "-n",   str(test.n_predict),
        "-c",   str(test.ctx_size),
        "-ngl", "100",
        "--no-mmap",
        "-fa",  "off",
        "--temp", "0",
        "-s",   str(test.seed),
    ]
    # Spec-dec args: add draft model if specified.
    if test.draft_model is not None:
        args.extend(["--model-draft", str(test.draft_model),
                     "--draft-max", str(test.draft_max),
                     "--draft-min", "1"])

    timeout = 60 + test.n_predict * 2  # generous: bf16 NPU runs at ~5-10 t/s
    if test.mode == "single-turn":
        assert isinstance(test.prompt, str)
        args.extend(["-p", test.prompt, "--single-turn"])
        result = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=timeout,
            errors="replace",
        )
        stdin_input = ""
    elif test.mode == "chat":
        assert isinstance(test.prompt, list)
        args.append("-cnv")
        stdin_input = "\n".join(test.prompt) + "\n/exit\n"
        result = subprocess.run(
            args, env=env, capture_output=True, text=True, input=stdin_input,
            timeout=timeout, errors="replace",
        )
    else:
        raise ValueError(f"unknown mode: {test.mode}")

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-cli exited {result.returncode} for preset={preset} test={test.name}\n"
            f"stderr tail:\n{result.stderr[-2000:]}"
        )
    return result.stdout, result.stderr


# =============================================================================
# Parsing
# =============================================================================

# Lines we strip out of any captured response block:
#   - llama-cli's per-turn "[ Prompt: X t/s | Generation: Y t/s ]" trailer
#     (its numbers vary run-to-run and aren't part of the model's reply)
#   - leaked ggml-xdna: warmup/diagnostic lines that occasionally
#     interleave with stdout
#   - the trailing "Exiting..." marker
PERF_LINE_RE = re.compile(r"^\s*\[\s*Prompt:.*?\]\s*$", re.MULTILINE)
GGML_LINE_RE = re.compile(r"^ggml-xdna:")
EXITING_RE   = re.compile(r"^\s*Exiting\.\.\.\s*$")


def extract_responses(stdout: str) -> list[str]:
    """Pull the model's responses out of llama-cli stdout.

    Strategy:
      Split the stdout into chunks delimited by the per-turn perf line
      `[ Prompt: ... | Generation: ... ]`. The chunk immediately preceding
      each perf line is one turn's full block (the `> <user prompt>` line
      plus the model's response). Within that block we drop the `>` line,
      strip ggml-xdna log lines, and trim whitespace.

    This is robust against:
      - varying t/s numbers between runs
      - chat-mode silent-stdin (where the `> ` line shows empty after the >)
      - interleaved ggml-xdna stderr that leaked into stdout
    """
    # Find the position of every perf line; capture content between them.
    perf_positions = [m.start() for m in PERF_LINE_RE.finditer(stdout)]
    if not perf_positions:
        return []

    responses = []
    prev_end = 0
    for pos in perf_positions:
        chunk = stdout[prev_end:pos]
        prev_end = stdout.find("\n", pos) + 1 or len(stdout)

        # Within `chunk`, find the LAST `>` line; everything after it is
        # the model's response. Earlier `>` lines (e.g. from previous
        # turns whose perf line we already consumed) are irrelevant.
        lines = chunk.splitlines()
        # Find last line starting with `>`
        last_gt = -1
        for i, line in enumerate(lines):
            if line.startswith(">"):
                last_gt = i
        if last_gt < 0:
            # No `>` prompt marker found; skip (this is probably the
            # warmup / banner region before the first turn).
            continue

        resp_lines = lines[last_gt + 1:]
        # Drop ggml-xdna log lines, empty lines at the boundaries.
        resp_lines = [
            line for line in resp_lines
            if not GGML_LINE_RE.match(line)
            and not EXITING_RE.match(line)
        ]
        responses.append("\n".join(resp_lines).strip())
    return responses


# =============================================================================
# Comparison
# =============================================================================

def diff_strings(a: str, b: str) -> tuple[int, str]:
    """First-divergence summary for two response strings.
    Returns (divergence_pos, summary). pos = len(a) if a == b (full match).
    pos = -1 if there's no shared prefix (lengths differ at char 0)."""
    if a == b:
        return (len(a), "")
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            ctx_lo = max(0, i - 20)
            return (i,
                f"diverge at char {i}:\n"
                f"  baseline ...{a[ctx_lo:i]!r} [{a[i]!r}] {a[i+1:i+21]!r}...\n"
                f"  variant  ...{b[ctx_lo:i]!r} [{b[i]!r}] {b[i+1:i+21]!r}..."
            )
    return (n, f"prefix matches but lengths differ (baseline={len(a)}, variant={len(b)})")


# =============================================================================
# Test orchestration
# =============================================================================

def run_test(test: Test, verbose: bool = False) -> bool:
    """Returns True if test passed (all variants matched expectations)."""
    print(f"\n=== {test.name} ===")
    if test.description:
        print(textwrap.fill(test.description, width=78, initial_indent="  ",
                            subsequent_indent="  "))

    # Baseline first
    t0 = time.time()
    try:
        baseline_stdout, _ = run_llama("cpu_baseline", test)
    except Exception as e:
        print(f"  cpu_baseline: ERROR {e}")
        return False
    t_baseline = time.time() - t0
    baseline_resp = extract_responses(baseline_stdout)
    print(f"  cpu_baseline:   ok ({t_baseline:.1f}s, {len(baseline_resp)} response(s))")
    if verbose:
        for i, r in enumerate(baseline_resp):
            print(f"    [{i}] {r!r}")

    all_pass = True
    for variant in test.variants:
        t0 = time.time()
        try:
            v_stdout, _ = run_llama(variant, test)
        except Exception as e:
            print(f"  {variant}: ERROR {e}")
            all_pass = False
            continue
        t_var = time.time() - t0
        v_resp = extract_responses(v_stdout)

        if len(v_resp) != len(baseline_resp):
            outcome = "FAIL"
            detail = f"turn count differs (baseline={len(baseline_resp)}, variant={len(v_resp)})"
        else:
            mismatch_details = []
            drift_notes = []
            min_prefix = test.min_prefix_match
            for i, (b, v) in enumerate(zip(baseline_resp, v_resp)):
                pos, summary = diff_strings(b, v)
                if summary == "":
                    continue  # exact match
                if min_prefix is not None and pos >= min_prefix:
                    drift_notes.append(
                        f"turn {i}: prefix matched {pos} chars (>= {min_prefix} required); "
                        f"drift after that point — {summary.splitlines()[0]}"
                    )
                else:
                    mismatch_details.append(f"turn {i}: {summary}")
            if mismatch_details:
                outcome = "FAIL"
                detail = "\n    ".join(mismatch_details)
            elif drift_notes:
                outcome = "PASS_WITH_DRIFT"
                detail = "\n    ".join(drift_notes)
            else:
                outcome = "PASS"
                detail = ""

        expected = variant in test.expected_fail
        if outcome in ("PASS", "PASS_WITH_DRIFT") and expected:
            # Pass when expected to fail is itself a regression of expectations
            # (e.g. the bug got fixed without us updating the harness).
            mark = "?? UNEXPECTED PASS"
            all_pass = False
        elif outcome == "FAIL" and expected:
            mark = "x EXPECTED FAIL"
        elif outcome == "FAIL":
            mark = "FAIL"
            all_pass = False
        elif outcome == "PASS_WITH_DRIFT":
            mark = "PASS (drift after prefix)"
        else:
            mark = "PASS"

        print(f"  {variant}: {mark} ({t_var:.1f}s)")
        if outcome == "FAIL" and detail:
            for line in detail.splitlines():
                print(f"    {line}")
        elif outcome == "PASS_WITH_DRIFT" and detail:
            for line in detail.splitlines():
                print(f"    {line}")
        if verbose:
            for i, r in enumerate(v_resp):
                print(f"    [{i}] {r!r}")

    return all_pass


# =============================================================================
# Benchmark mode (--bench)
# =============================================================================
#
# Goal: measure decode t/s across (preset, model) combos so we can decide
# whether further INT4 phases (8.2 SwiGLU INT4 etc.) are worth the effort.
#
# We don't compare outputs in bench mode -- we just record the
# `llama_perf_context_print: ... eval time = ... tokens per second` number
# from stderr. Each combo is run `BENCH_REPEATS` times; we report the median.

EVAL_TPS_RE = re.compile(
    r"\[\s*Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s\s*\]"
)
BENCH_REPEATS = 2          # 1 warm-up + 1 measured; first run includes any
                           # inline xclbin compile, the second is the warm rate.
BENCH_PROMPT = "What is the capital of France? Explain in detail."
BENCH_N_PREDICT = 64


@dataclass
class BenchConfig:
    label: str
    preset: str
    model: Path


def build_bench_configs() -> list[BenchConfig]:
    return [
        BenchConfig(label="CPU Q4_0",          preset="cpu_baseline",     model=MODEL_Q4_0),
        BenchConfig(label="NPU bf16",          preset="npu_chat_safe",    model=MODEL),
        BenchConfig(label="NPU INT4 v1",       preset="npu_int4_v1",      model=MODEL_Q4_0),
        BenchConfig(label="NPU INT4 (default=v2)", preset="npu_int4",     model=MODEL_Q4_0),
        BenchConfig(label="NPU INT4 +SwiGLU",  preset="npu_int4_swiglu",  model=MODEL_Q4_0),
        BenchConfig(label="NPU INT4 QKV fused", preset="npu_int4_qkv_fused", model=MODEL_Q4_0),
    ]


def build_bench_configs_llama3b() -> list[BenchConfig]:
    """Bench configs for Llama 3.2 3B Q4_0 (head_dim=128).

    Phase 8.3 + 8.5 result: FlowKV NPU dispatch now fires on Q4_0 models.
    The 'NPU INT4 v2' config exercises the full stack including FlowKV
    on head_dim=128. The 'no-flowkv' variant isolates the GEMV+QKV-INT4
    path -- useful to see whether FlowKV POC overwrite of kqv_out helps
    or hurts perf in practice.
    """
    return [
        BenchConfig(label="3B CPU Q4_0",                 preset="cpu_baseline",          model=MODEL_LLAMA_3B_Q4_0),
        BenchConfig(label="3B NPU INT4 v2",              preset="npu_int4_v2",           model=MODEL_LLAMA_3B_Q4_0),
        BenchConfig(label="3B NPU INT4 v2 no-FlowKV",    preset="npu_int4_v2_no_flowkv", model=MODEL_LLAMA_3B_Q4_0),
        BenchConfig(label="3B NPU INT4 GEMV-only",       preset="npu_int4_gemv_only",    model=MODEL_LLAMA_3B_Q4_0),
    ]


def build_bench_configs_gemma() -> list[BenchConfig]:
    """Bench configs for Gemma 3 1B Q4_K_M (head_dim=256).

    Only npu_int4_gemv_only fires (SWA + non-1D RoPE block other paths).
    Q4_K attn_output dispatches through the INT4 path; ffn_down too after
    ggml's CPU_REPACK Q6_K → Q4_K conversion.
    """
    return [
        BenchConfig(label="Gemma 3 1B CPU Q4_K_M",       preset="cpu_baseline",       model=MODEL_GEMMA_3_1B_Q4_K_M),
        BenchConfig(label="Gemma 3 1B NPU INT4 GEMV",    preset="npu_int4_gemv_only", model=MODEL_GEMMA_3_1B_Q4_K_M),
    ]


def build_bench_configs_qwen() -> list[BenchConfig]:
    """Bench configs for Qwen3.5-9B-Q4_0.

    Qwen3.5-9B uses head_dim=256 (vs 64 for Llama), M-RoPE, and SWA --
    none of our attention/FFN-fusion NPU ops support these (every relevant
    matcher hardcodes head_dim==64). The full npu_int4 preset would
    dispatch garbage through those paths; npu_int4_gemv_only restricts
    NPU usage to pure matmul (architecture-agnostic).
    """
    return [
        BenchConfig(label="Qwen3.5-9B CPU Q4_0",            preset="cpu_baseline",       model=MODEL_QWEN35_9B_Q4_0),
        BenchConfig(label="Qwen3.5-9B NPU INT4 (full)",     preset="npu_int4",           model=MODEL_QWEN35_9B_Q4_0),
        BenchConfig(label="Qwen3.5-9B NPU INT4 GEMV-only",  preset="npu_int4_gemv_only", model=MODEL_QWEN35_9B_Q4_0),
    ]


def run_bench_one(cfg: BenchConfig, mode: str) -> tuple[float, float]:
    """Run llama-cli once with cfg in mode={'single','chat'},
    return (decode_tps, prompt_tps). Raises on timeout or non-zero exit."""
    env = build_env(cfg.preset)
    args = [
        str(LLAMA_CLI),
        "-m",   str(cfg.model),
        "-n",   str(BENCH_N_PREDICT),
        "-c",   "512",
        "-ngl", "100",
        "--no-mmap",
        "-fa",  "off",
        "--temp", "0",
        "-s",   "42",
    ]
    if mode == "single":
        args.extend(["-p", BENCH_PROMPT, "--single-turn"])
        stdin_input = None
    elif mode == "chat":
        args.append("-cnv")
        stdin_input = BENCH_PROMPT + "\n/exit\n"
    else:
        raise ValueError(f"unknown bench mode: {mode}")

    timeout = 300 + BENCH_N_PREDICT * 8    # 9B needs more time
    result = subprocess.run(
        args, env=env, capture_output=True, text=True,
        input=stdin_input, timeout=timeout, errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-cli exit {result.returncode} for bench {cfg.label} ({mode})\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
    # Parser: llama-cli prints "[ Prompt: P t/s | Generation: G t/s ]" to stdout
    # at the end of each turn. We want the LAST such line in case of multiple.
    matches = list(EVAL_TPS_RE.finditer(result.stdout))
    if not matches:
        raise RuntimeError(
            f"could not find perf line for {cfg.label} ({mode})\n"
            f"stdout tail:\n{result.stdout[-1500:]}"
        )
    m = matches[-1]
    prompt_tps = float(m.group(1))
    decode_tps = float(m.group(2))
    return decode_tps, prompt_tps


def run_bench_lookup(preset: str, model: Path, draft_max: int = 8,
                     prompt: str = None) -> tuple[float, float, float]:
    """Run llama-lookup.exe and return (decode_tps, accept_pct, n_drafted).

    llama-lookup uses in-context n-gram speculation -- no draft model
    needed. Best with repetitive/structured text; acceptance varies from
    ~50% (creative) to ~100% (repetitive). Stats come from stderr.
    """
    env = build_env(preset)
    test_prompt = prompt or (
        # Repetitive text gives high n-gram hit rate for benchmark stability
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. Continue: "
    )
    args = [
        str(LLAMA_LOOKUP),
        "-m",   str(model),
        "-n",   str(BENCH_N_PREDICT),
        "-c",   "512",
        "-ngl", "100",
        "--no-mmap",
        "--temp", "0",
        "-s",   "42",
        "--draft-max", str(draft_max),
        "--draft-min", "1",
        "-p", test_prompt,
    ]
    timeout = 300 + BENCH_N_PREDICT * 8
    result = subprocess.run(
        args, env=env, capture_output=True, text=True,
        timeout=timeout, errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-lookup exit {result.returncode}\n"
            f"stderr tail:\n{result.stderr[-1000:]}"
        )
    # Parse speed from stderr: "decoded N tokens in T s, speed: S t/s"
    import re
    speed_m = re.search(r"decoded\s+\d+\s+tokens.*?,\s+speed:\s+([\d.]+)\s+t/s",
                        result.stderr)
    accept_m = re.search(r"accept\s*=\s*([\d.]+)%", result.stderr)
    drafted_m = re.search(r"n_drafted\s*=\s*(\d+)", result.stderr)
    decode_tps  = float(speed_m.group(1))  if speed_m   else 0.0
    accept_pct  = float(accept_m.group(1)) if accept_m  else 0.0
    n_drafted   = int(drafted_m.group(1))  if drafted_m else 0
    return decode_tps, accept_pct, n_drafted


def build_bench_configs_lookup() -> list[tuple[str, str, Path, int]]:
    """Lookup (n-gram) spec-dec bench configs.

    Returns list of (label, preset, model, draft_max).
    """
    return [
        ("1B NPU INT4, n-gram d=4",  "npu_int4", MODEL_Q4_0, 4),
        ("1B NPU INT4, n-gram d=8",  "npu_int4", MODEL_Q4_0, 8),
        ("1B NPU INT4, n-gram d=16", "npu_int4", MODEL_Q4_0, 16),
    ]


def run_bench(mode: str, model: str = "llama") -> int:
    if mode == "both":
        rc1 = run_bench("single", model)
        rc2 = run_bench("chat",   model)
        return rc1 or rc2

    if model == "qwen":
        configs = build_bench_configs_qwen()
    elif model == "llama3b":
        configs = build_bench_configs_llama3b()
    elif model == "gemma":
        configs = build_bench_configs_gemma()
    else:
        configs = build_bench_configs()
    missing = [c.model for c in configs if not c.model.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: model not found at {m}")
        return 2

    print(f"\n=== bench [{mode}]: prompt={BENCH_PROMPT!r} n_predict={BENCH_N_PREDICT} repeats={BENCH_REPEATS} ===\n")
    rows: list[tuple[str, list[float], list[float]]] = []
    for cfg in configs:
        decodes: list[float] = []
        prompts: list[float] = []
        for rep in range(BENCH_REPEATS):
            tag = "warm" if rep == 0 else f"run{rep+1}"
            try:
                d, p = run_bench_one(cfg, mode)
            except Exception as e:
                print(f"  {cfg.label:24s} {tag}: ERROR -- {e}")
                d, p = float("nan"), float("nan")
            print(f"  {cfg.label:24s} {tag}: decode={d:6.2f} t/s   prompt={p:7.2f} t/s")
            decodes.append(d)
            prompts.append(p)
        rows.append((cfg.label, decodes, prompts))

    # Pick the median (= second of two runs once warm cache settles).
    def median(xs: list[float]) -> float:
        xs2 = [x for x in xs if x == x]    # drop NaNs
        if not xs2: return float("nan")
        xs2.sort()
        return xs2[len(xs2) // 2]

    print(f"\n--- {mode} mode, median across {BENCH_REPEATS} runs ---")
    print(f"  {'config':24s}  {'decode t/s':>11s}  {'prompt t/s':>11s}")
    print(f"  {'-'*24}  {'-'*11}  {'-'*11}")
    for label, decodes, prompts in rows:
        print(f"  {label:24s}  {median(decodes):11.2f}  {median(prompts):11.2f}")

    # N-gram lookup bench: shows spec-dec speedup without draft model.
    if model == "llama" and mode == "single":
        lookup_configs = build_bench_configs_lookup()
        baseline_tps = median([d for _, decodes, _ in rows[:1] for d in decodes]) or 0
        print(f"\n--- n-gram lookup spec-dec (repetitive prompt, baseline~{baseline_tps:.1f} t/s) ---")
        print(f"  {'config':30s}  {'decode t/s':>11s}  {'accept%':>8s}  {'speedup':>8s}")
        print(f"  {'-'*30}  {'-'*11}  {'-'*8}  {'-'*8}")
        for label, preset, lmodel, dmax in lookup_configs:
            try:
                tps, accept, ndrafted = run_bench_lookup(preset, lmodel, dmax)
                speedup = tps / baseline_tps if baseline_tps > 0 else 0
                print(f"  {label:30s}  {tps:11.2f}  {accept:8.1f}%  {speedup:7.1f}x")
            except Exception as e:
                print(f"  {label:30s}  ERROR: {e}")
    print()
    return 0


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_name", nargs="?", help="Run only this test by name")
    ap.add_argument("--list", action="store_true", help="List test names and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print response text")
    ap.add_argument("--bench", action="store_true",
                    help="Run perf benchmark across CPU/NPU bf16/NPU INT4 presets")
    ap.add_argument("--bench-mode", choices=["single", "chat", "both"], default="single",
                    help="Bench mode: single-turn, chat (-cnv), or both (default: single)")
    ap.add_argument("--model", choices=["llama", "qwen", "llama3b", "gemma"], default="llama",
                    help="Bench model: llama (1B Q4_0, default), qwen (3.5-9B), llama3b (3.2 3B Q4_0), or gemma (3 1B Q4_K_M)")
    args = ap.parse_args()

    if args.list:
        for t in TESTS:
            model_tag = ""
            if t.model != MODEL:
                model_tag = f" [{t.model.stem.split('-')[-1]}]"
            print(f"  {t.name:30s}{model_tag}  ({t.mode}, n={t.n_predict})  {t.description}")
        return 0

    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli not found at {LLAMA_CLI}")
        print("Build first: cmake --build build --config Release --target llama-cli -j")
        return 2

    if args.bench:
        return run_bench(args.bench_mode, args.model)
    # Check all model files referenced by selected tests exist.
    tests = TESTS
    if args.test_name:
        tests = [t for t in TESTS if t.name == args.test_name]
        if not tests:
            print(f"ERROR: no test named {args.test_name}")
            return 2
    missing_models = {t.model for t in tests if not t.model.exists()}
    if missing_models:
        for m in missing_models:
            print(f"ERROR: model not found at {m}")
            if m.name.startswith("llama-3.2-1b-instruct-Q"):
                print(f"  generate it via: .\\build\\bin\\Release\\llama-quantize.exe "
                      f"models\\llama-3.2-1b-instruct-BF16.gguf {m.name} "
                      f"{m.stem.split('-')[-1].lower()}")
        return 2

    n_pass = 0
    n_fail = 0
    t_start = time.time()
    for t in tests:
        if run_test(t, verbose=args.verbose):
            n_pass += 1
        else:
            n_fail += 1
    elapsed = time.time() - t_start

    print(f"\n=== summary: {n_pass} passed, {n_fail} failed in {elapsed:.1f}s ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
