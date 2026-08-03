import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(r"C:/llama.cpp-xdna")
OUT = ROOT / ".rr_stress_20260802"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "correctness_test", ROOT / "ggml/src/ggml-xdna/tools/correctness_test.py")
ct = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ct
spec.loader.exec_module(ct)

# The harness preset supplies DECOUPLE+TRIPLE_B. RR must be injected into every
# independently spawned child so its _rr xclbin is selected.
os.environ["F3BEST_HANDASM_RR"] = "1"
os.environ.pop("XDNA_F3BEST_NPU_KV", None)
os.environ.pop("XDNA_F3BEST_GLUETIME", None)
os.environ.pop("XDNA_FW_PROF", None)
os.environ.pop("XDNA_FW_PROF_VERBOSE", None)

cfg = ct.BenchConfig(
    label="RR+DECOUPLE+TRIPLE_B",
    preset="npu_f3best_loop",
    model=ct.MODEL_Q4_0_VOCABQ4,
)

key_re = re.compile(r"loaded kernel for (decode_layer_f3best_[^\s]+)")
rate_re = re.compile(r"Generation:\s+([\d.]+)\s+t/s")
results = []
for i in range(1, 21):
    try:
        decode, prompt = ct.run_bench_one(cfg, "single")
        result = {"run": i, "status": "OK", "decode": decode, "prompt": prompt}
    except Exception as exc:
        text = str(exc)
        result = {"run": i, "status": "FAIL", "error": text[-1500:]}
    results.append(result)
    print(f"{i:02d}/20 {result['status']}" +
          (f" decode={result['decode']:.2f} prompt={result['prompt']:.2f}" if result['status'] == 'OK' else ""),
          flush=True)
    (OUT / f"{i:02d}.txt").write_text(str(result), encoding="utf-8")

ok = [r["decode"] for r in results if r["status"] == "OK"]
failed = [r for r in results if r["status"] != "OK"]
print("=== RR STRESS SUMMARY ===")
print(f"success={len(ok)}/20 failure={len(failed)}/20")
if ok:
    print(f"decode: min={min(ok):.2f} median={statistics.median(ok):.2f} mean={statistics.mean(ok):.2f} max={max(ok):.2f}")
if failed:
    print("failed runs:", ", ".join(str(r["run"]) for r in failed))
    for r in failed:
        print(f"--- run {r['run']} failure tail ---\n{r['error']}")
raise SystemExit(0 if not failed else 1)
