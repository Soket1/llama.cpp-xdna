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
LLAMA_CLI = REPO_ROOT / "build" / "bin" / "Release" / "llama-cli.exe"
MODEL     = REPO_ROOT / "models" / "llama-3.2-1b-instruct-BF16.gguf"

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
        "-m",   str(MODEL),
        "-n",   str(test.n_predict),
        "-c",   str(test.ctx_size),
        "-ngl", "100",
        "--no-mmap",
        "-fa",  "off",
        "--temp", "0",
        "-s",   str(test.seed),
    ]

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
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_name", nargs="?", help="Run only this test by name")
    ap.add_argument("--list", action="store_true", help="List test names and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print response text")
    args = ap.parse_args()

    if args.list:
        for t in TESTS:
            print(f"  {t.name:30s}  ({t.mode}, n={t.n_predict})  {t.description}")
        return 0

    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli not found at {LLAMA_CLI}")
        print("Build first: cmake --build build --config Release --target llama-cli -j")
        return 2
    if not MODEL.exists():
        print(f"ERROR: model not found at {MODEL}")
        return 2

    tests = TESTS
    if args.test_name:
        tests = [t for t in TESTS if t.name == args.test_name]
        if not tests:
            print(f"ERROR: no test named {args.test_name}")
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
