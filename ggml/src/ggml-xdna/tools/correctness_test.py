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
        description="Drift check -- long generation. bf16 noise may accumulate.",
    ),
    Test(
        name="multiquery_basic",
        prompt=["What is the capital of France?", "What is 2+2?"],
        n_predict=32,
        mode="chat",
        expected_fail={"npu_full"},  # known RMS+QKV regression
        description="Multi-query regression test. npu_full should diverge on Q2 (this is the documented RMS_NORM+QKV bug).",
    ),
    Test(
        name="multiquery_chatsafe",
        prompt=["What is the capital of France?", "What is the capital of Germany?", "What is the capital of Spain?"],
        n_predict=24,
        mode="chat",
        variants=["npu_chat_safe"],   # only the safe variant
        description="3-query chat with the supported config. Confirms the workaround stays clean across multiple turns.",
    ),
    Test(
        name="seq_len_short",
        # ~5 prompt tokens -> first decode at actual_seq ~5
        prompt="Hi.",
        n_predict=8,
        mode="single-turn",
        description="Tiny seq_len edge case. Probes the V-PERMUTE matcher's `ne[0] > hd` condition (which silently disables POC when seq<=64).",
    ),
    Test(
        name="seq_len_near_chunk32",
        # Pad prompt to land near the chunk_size=32 boundary of FlowKV
        prompt="Tell me a story about a robot who learned to dance under the moonlight.",
        n_predict=24,
        mode="single-turn",
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

# Matches the per-turn block llama-cli emits in interactive/single-turn mode:
#   > <user prompt>
#   <model response>
#   [ Prompt: ... | Generation: ... ]
RESPONSE_BLOCK_RE = re.compile(
    r"^>\s*(?P<prompt>.*?)\n\n(?P<response>.*?)\n\n\[\s*Prompt:",
    re.MULTILINE | re.DOTALL,
)


def extract_responses(stdout: str) -> list[str]:
    """Pull the model's responses out of llama-cli stdout, ignoring prompts,
    diagnostic noise, and the "Exiting..." trailer."""
    blocks = []
    for m in RESPONSE_BLOCK_RE.finditer(stdout):
        # Strip any in-text ggml-xdna lines that leaked into stdout (warmup
        # logs print to stderr but a few early ones can interleave).
        resp = m.group("response")
        resp_clean = "\n".join(
            line for line in resp.splitlines()
            if not line.startswith("ggml-xdna:")
        ).strip()
        blocks.append(resp_clean)
    return blocks


# =============================================================================
# Comparison
# =============================================================================

def diff_strings(a: str, b: str) -> str:
    """First-divergence summary for two response strings."""
    if a == b:
        return ""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            ctx_lo = max(0, i - 20)
            return (
                f"diverge at char {i}:\n"
                f"  baseline ...{a[ctx_lo:i]!r} [{a[i]!r}] {a[i+1:i+21]!r}...\n"
                f"  variant  ...{b[ctx_lo:i]!r} [{b[i]!r}] {b[i+1:i+21]!r}..."
            )
    return f"prefix matches but lengths differ (baseline={len(a)}, variant={len(b)})"


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
            for i, (b, v) in enumerate(zip(baseline_resp, v_resp)):
                if b != v:
                    mismatch_details.append(f"turn {i}: {diff_strings(b, v)}")
            if mismatch_details:
                outcome = "FAIL"
                detail = "\n    ".join(mismatch_details)
            else:
                outcome = "PASS"
                detail = ""

        expected = variant in test.expected_fail
        if outcome == "PASS" and expected:
            # Pass when expected to fail is itself a regression of expectations
            # (e.g. the bug got fixed without us updating the harness).
            mark = "?? UNEXPECTED PASS"
            all_pass = False
        elif outcome == "FAIL" and expected:
            mark = "x EXPECTED FAIL"
        elif outcome == "FAIL":
            mark = "FAIL"
            all_pass = False
        else:
            mark = "PASS"

        print(f"  {variant}: {mark} ({t_var:.1f}s)")
        if outcome == "FAIL" and detail:
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
