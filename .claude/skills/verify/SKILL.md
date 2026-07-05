# Verify ggml-xdna f3best loop changes

Use this skill to verify runtime behavior of `ggml/src/ggml-xdna/ggml-xdna.cpp` changes that affect the f3best unified-loop path.

## Surface

The runtime surface is `build/bin/Release/llama-cli.exe` using the `npu_f3best_loop` environment preset from `ggml/src/ggml-xdna/tools/correctness_test.py`.

## Build

```bash
cmake --build /c/llama.cpp-xdna/build --config Release -j 8
```

## Runtime harness pattern

Use a short Python runner that imports `correctness_test.py`, calls `build_env('npu_f3best_loop')`, and launches `llama-cli.exe` with:

```text
-m C:/llama.cpp-xdna/models/llama-3.2-1b-instruct-Q4_0.gguf
-n <N> -c 512 -ngl 100 --no-mmap -fa off --temp 0 -s <seed>
-p "What is the capital of France? Explain in detail." --single-turn
```

## f3best-loop checks

For host-glue timing:

```text
XDNA_F3BEST_GLUETIME=1
```

Expect stderr lines like:

```text
ggml-xdna: [f3best-gluetime] token: layers=16 f3wall=... host=... kvcopy=... npu=... us
```

For opt-in packed-KV append verification:

```text
XDNA_F3BEST_KV_APPEND=1
XDNA_F3BEST_KV_VERIFY=1
```

Expect `f3best-kvverify` lines with `mism=0`. `KV_VERIFY` intentionally rebuilds a full packed-KV reference on CPU, so it makes `kvcopy` timing slow; use a second run without `KV_VERIFY` for performance.

For stable output equality, compare default full-fill vs `XDNA_F3BEST_KV_APPEND=1` at `n=24` on a few prompts/seeds. Longer generations can be nondeterministic even full-fill-vs-full-fill, so do not use long text equality as the primary correctness signal.
