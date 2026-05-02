# ggml-xdna Optimization Plan

> Based on analysis of [albiol2004/llama.cpp@ggml-xdna](https://github.com/albiol2004/llama.cpp/tree/ggml-xdna)
> Issue: [ggml-org/llama.cpp#21725](https://github.com/ggml-org/llama.cpp/issues/21725)
> Last commit: 2026-04-21 — "9tk/s first decode impl"

## Current State

| Metric | Value | Notes |
|--------|-------|-------|
| Prefill (Q8 INT8) | 152 t/s | Good |
| Decode | 9 t/s | Bottleneck: host-side dispatch overhead |
| Attention on NPU | 0.5x CPU speed | Overhead > compute gain |
| Codebase | ~12k lines (9.9k C++ + 2.4k Python) | |
| Supported ops | 10 op kinds + TransformerBlock fusion | |

## Priority 1: Decode Latency (9 → target 30+ t/s)

### 1.1 xrt::runlist Batching

**Problem:** Each `xrt::run` has ~1-5ms host-side overhead. At M=1 decode, NPU compute is ~0.1ms — overhead is 10-50x the actual work. Currently 7 separate dispatches per layer (Q, K, V, O, gate, up, down).

**Solution:** Collect all MUL_MAT ops for one decode step into a single `xrt::runlist`. One submit instead of 7.

- Already partially done for QKV (3 GEMVs in one runlist)
- Extend to full layer: Q + K + V + O + gate + up + down = 7 ops in one batch
- Expected: ~70% overhead reduction on decode path

**Files:** `ggml-xdna.cpp` — `ggml_backend_xdna_mul_mat_qkv()` and `graph_compute()`

### 1.2 Persistent Command Buffers

**Problem:** Per-call kernel setup (BO args, run configuration) adds latency on every token.

**Solution:** Prepare `xrt::run` objects once at model load, then only swap BO payloads and submit.

- Reuse run handles across tokens for same-shape dispatches
- Only memcpy new data into pre-allocated BOs, skip arg setup
- Expected: ~20% additional overhead reduction

**Files:** `ggml-xdna.cpp` — kernel entry structs and dispatch functions

### 1.3 Async NPU Dispatch

**Problem:** NPU dispatch blocks CPU synchronously. CPU idles while waiting for NPU.

**Solution:** Submit NPU run, immediately continue with CPU ops (residual add, RMSNorm). Wait for NPU only when result is needed.

- Use `xrt::run::wait()` only at data dependency points
- Overlap NPU compute with CPU-side tensor operations
- Expected: 10-15% improvement from pipelining

**Files:** `ggml-xdna.cpp` — `graph_compute()` loop

## Priority 2: Attention Performance

### 2.1 Fused Attention Decode Kernel

**Problem:** Attention on NPU at 0.5x CPU speed — dispatch overhead exceeds compute gain for M=1.

**Solution:** Create a single fused `AttentionDecode` kernel: Q·K^T → softmax → V, one dispatch.

- Currently: 5+ separate ops (matmul, softmax, matmul, permute, residual)
- Target: 1 fused xclbin with chained sub-kernels, single xrt::runlist submit
- Similar pattern to existing `AttentionBlockPrefill` but for M=1

**Files:** `compile.py` (new `compile_attention_decode`), `ggml-xdna.cpp` (new `XDNA_OP_ATTENTION_DECODE`)

### 2.2 KV Cache on NPU

**Problem:** Every decode token re-uploads full K/V cache from host to NPU.

**Solution:** Allocate persistent BOs for K/V cache on NPU memory. Update in-place (append new K/V row).

- Pre-allocate BO at model load for max seq_len
- Only DMA the new K/V row each token, not the full cache
- NPU reads cache directly from device memory

**Files:** `ggml-xdna.cpp` — new KV cache management in context struct

### 2.3 FlashAttention-style Tiling for Decode

**Problem:** MHA reads full K/V sequence from DRAM each token — bandwidth-bound.

**Solution:** Tile K/V in NPU SRAM, compute attention in chunks. Keep working set on-chip.

- Leverage AIE core's local memory (~32KB per tile)
- Process K/V in tiles that fit L1, accumulate softmax incrementally
- Reduces DRAM reads from O(seq_len) to O(tile_size) per dispatch

**Files:** `compile.py` — IRON MHA operator modifications

## Priority 3: Prefill Improvements

### 3.1 TransformerBlockPrefillFused Benchmarking

**Status:** Code exists (17 kernels → 1 ELF), needs benchmarking.

**Action:** Test on real models, measure per-layer overhead reduction vs chained variant.

- Compare chained (17 xrt::runs) vs fused (1 xrt::run) per layer
- Test with seq_len buckets: 256, 512, 1024, 2048, 4096

### 3.2 Multi-Layer Packing (Layer 4B)

**Status:** Code exists (`num_layers > 1`), needs validation.

**Action:** Pack N consecutive transformer blocks into one ELF. Reduces host dispatch by Nx.

- Test with N=2, 4, 8 on typical models (Llama 3 8B, Qwen3 8B)
- Memory budget: each layer's weights must fit in BO allocation

### 3.3 W8A16 / W8A8 Quantization

**Status:** Code exists (`w8a16`, `w8a16_ffn` flags).

**Action:** Enable by default for supported models. INT8 GEMM is faster on AIE.

- W8A16: INT8 attention projections (Wq/Wk/Wv/Wo), bf16 FFN
- W8A8: INT8 both attention and FFN (needs activation quantization)
- Expected: 1.5-2x prefill speedup on AIE hardware

## Priority 4: Architecture

### 4.1 Weight Pre-Repacking

**Problem:** Weights are transposed on CPU and DMA'd at first dispatch — cold start penalty.

**Solution:** Repack weights at model load time into AIE-compatible layout (row-major [K,N]).

- Do transpose + pack once at load, not at first inference
- Cache repacked weights to disk (like xclbin cache)
- Expected: 10x faster cold start

### 4.2 Compile-Time Tile Profiling

**Problem:** Tile sizes are selected by shape constraints, not by performance.

**Solution:** Profile multiple tile configurations, pick the fastest for each shape.

- Benchmark tile_m/tile_k/tile_n combinations on target hardware
- Store performance data in a lookup table
- Use at compile time to select optimal tiles

### 4.3 Remove Python Runtime Dependency

**Problem:** `compile.py` called as subprocess, requires Python + IRON at runtime.

**Solution:** Pre-compile all xclbin for a model offline. Ship with pre-built cache.

- Build tool: `ggml-xdna-compile --model model.gguf --output cache/`
- Runtime: pure C++ with XRT, no Python needed
- Users without IRON can use pre-compiled caches

## Priority 5: Quick Wins

### 5.1 Weight BO Lookup Optimization

**Problem:** `unordered_map<void*, xrt::bo>` lookup on every dispatch.

**Solution:** For immutable weights, use direct array indexed by layer position.

- Weights are loaded in order, position is deterministic
- Replace hash map with `std::vector<xrt::bo>` indexed by layer+slot
- O(1) lookup instead of hash + collision handling

### 5.2 Prefetch Next Layer

**Problem:** While NPU processes layer N, CPU idles.

**Solution:** CPU prepares BO args for layer N+1 during NPU compute.

- Double-buffering: submit layer N, immediately prepare N+1 args
- When N completes, N+1 is ready to submit with zero setup latency

### 5.3 Remove Mutex on Hot Path

**Problem:** `weights_mutex` lock on every dispatch — unnecessary for single-threaded decode.

**Solution:** Use atomics or thread-local state for single-writer pattern.

- Decode is single-threaded by nature — no contention
- Only lock during parallel bulk_prewarm phase

## Strategic Notes

### Positioning

- **Don't compete with GPU on raw throughput** — NPU in a laptop targets energy efficiency
- 9 t/s at 5W (NPU) vs 30 t/s at 50W (GPU) = **6x better perf/watt**
- Target: "battery-friendly local inference" — all-day AI assistant on laptop battery

### Target Models

- 1-3B parameter models: NPU handles all compute, CPU minimal
- 7-8B models: NPU handles matmuls (70-80% of compute), CPU handles rest
- Larger models: hybrid NPU+CPU+GPU (if available)

### Collaboration

- IRON (github.com/amd/IRON) is open-source — PR improvements upstream
- XRT is open-source — file issues for dispatch overhead
- AMD is building XDNA ecosystem — potential hardware access / engineering support

## Implementation Order

| Phase | Task | Effort | Impact |
|-------|------|--------|--------|
| 1 | Runlist batching for decode | 1-2 weeks | ⭐⭐⭐⭐⭐ |
| 2 | Persistent command buffers | 1 week | ⭐⭐⭐⭐ |
| 3 | Fused attention decode | 2-3 weeks | ⭐⭐⭐⭐ |
| 4 | KV cache on NPU | 1-2 weeks | ⭐⭐⭐ |
| 5 | Weight pre-repack | 1 week | ⭐⭐⭐ |
| 6 | Async dispatch | 1-2 weeks | ⭐⭐⭐ |
| 7 | Remove Python runtime | 1 week | ⭐⭐ |
| 8 | Quick wins (mutex, prefetch, lookup) | 2-3 days | ⭐⭐ |

---

*Plan created: 2026-05-02*
*Source: `/root/.openclaw/workspace/llama.cpp-xdna/`*
