#!/usr/bin/env python3
"""Longer profile: 16-token decode to get stable per-token averages."""
import os, subprocess
from pathlib import Path

ROOT = Path(__file__).parent
EXE = ROOT / "build" / "bin" / "Release" / "llama-cli.exe"
MODEL = ROOT / "models" / "llama-3.2-1b-instruct-Q4_0.gguf"

env = os.environ.copy()
env.update({
    "XDNA_PROFILE_DISPATCHES": "1",
    "XDNA_PROFILE_MAX_TOKENS": "40",
    "XDNA_PROFILE_DISPATCHES_PATH": "xdna_profile_long.csv",
    "GGML_XDNA_CACHE_DIR": "npu_kernels_win_8col",
    "XDNA_ENABLE_GEMV": "1", "XDNA_ENABLE_SWIGLU": "1",
    "XDNA_ENABLE_QKV": "1", "XDNA_ENABLE_DECODE_BATCH": "1",
    "XDNA_ENABLE_TRANSFORMER_BLOCK": "1", "XDNA_ENABLE_FLOWKV_DECODE": "1",
    "XDNA_ENABLE_GEMV_INT4": "1", "XDNA_ENABLE_SWIGLU_INT4": "1",
    "XDNA_ENABLE_FUSED_LAYER": "1",
})
args = [str(EXE), "-m", str(MODEL), "-p", "hi", "-n", "32",
        "--temp", "0", "--single-turn"]
r = subprocess.run(args, env=env, capture_output=True, text=True,
                   timeout=300, errors="replace")
print("exit:", r.returncode)
for line in r.stderr.splitlines():
    if "PROFILE" in line or "tokens per second" in line or "t/s" in line:
        print(line)
