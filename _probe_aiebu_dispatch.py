#!/usr/bin/env python3
"""Mimic correctness_test.py subprocess call with --single-turn so llama-cli
doesn't hang waiting for stdin. Capture stderr to disk."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
EXE = ROOT / "build" / "bin" / "Release" / "llama-cli.exe"
MODEL = ROOT / "models" / "llama-3.2-1b-instruct-Q4_0.gguf"

env = os.environ.copy()
env.update({
    "XDNA_AIEBU_DISPATCH": "1",
    "GGML_XDNA_CACHE_DIR": "npu_kernels_win_8col",
    "XDNA_ENABLE_GEMV": "1",
    "XDNA_ENABLE_SWIGLU": "1",
    "XDNA_ENABLE_QKV": "1",
    "XDNA_ENABLE_DECODE_BATCH": "1",
    "XDNA_ENABLE_TRANSFORMER_BLOCK": "1",
    "XDNA_ENABLE_FLOWKV_DECODE": "1",
    "XDNA_ENABLE_GEMV_INT4": "1",
    "XDNA_ENABLE_SWIGLU_INT4": "1",
    "XDNA_ENABLE_FUSED_LAYER": "1",
})

args = [str(EXE), "-m", str(MODEL), "-p", "hi", "-n", "4", "--temp", "0",
        "--single-turn"]
r = subprocess.run(args, env=env, capture_output=True, text=True,
                   timeout=180, errors="replace")
print("--- exit:", r.returncode)
print("=== stderr (filtered) ===")
for line in r.stderr.splitlines():
    if any(k in line for k in (
        "aiebu", "fullelf", "arg[", "max_arg", "dispatch failed",
        "post_attn_fused", "MISMATCH", "kernel name", "rejected",
        "did not complete", "disabling aiebu",
    )):
        print(line)
print("=== last 1000 chars of stderr ===")
print(r.stderr[-1000:])
