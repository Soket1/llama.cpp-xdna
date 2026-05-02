# ggml-xdna — Tracker

> План оптимизации: [XDNA_OPTIMIZATION_PLAN.md](./XDNA_OPTIMIZATION_PLAN.md)
> Issue: [ggml-org/llama.cpp#21725](https://github.com/ggml-org/llama.cpp/issues/21725)
> Форк: [albiol2004/llama.cpp@ggml-xdna](https://github.com/albiol2004/llama.cpp/tree/ggml-xdna)
> Последний коммит: 2026-04-21 — "9tk/s first decode impl"

---

## P1: Decode Latency (9 → 30+ t/s)

### 1.1 xrt::runlist Batching
- [x] Аудит текущего dispatch path — найти все точки где `xrt::run` вызывается поодиночке
- [x] Реализовать `xdna_decode_batcher` — собирает несколько kernel dispatches в один `xrt::runlist`
- [ ] Batch для decode layer: Q + K + V + O + gate + up + down (7 ops → 1 submit) — требует cross-layer restructuring
- [x] Batch для SwiGLU decode: gate GEMV + SiLU + eltwise + down GEMV (4 ops → 1 submit)
- [x] Batch для QKV: уже сделано ✅ (проверить корректность)
- [ ] Benchmark: измерить overhead до/после на StrixHalo
- [x] Файлы: `ggml-xdna.cpp` → `ggml_backend_xdna_mul_mat_*`, `graph_compute()`

### 1.2 Persistent Command Buffers
- [ ] Выделить `xrt::run` объекты при загрузке модели (один раз на shape)
- [ ] Перемещать только BO payloads между токенами, пропуская arg setup
- [ ] Кешировать run handles в `xdna_kernel_entry` (добавить поле `xrt::run persistent_run`)
- [ ] Benchmark: измерить per-token overhead до/после
- [ ] Файлы: `ggml-xdna.cpp` → структуры `xdna_kernel_entry`, dispatch функции

### 1.3 Async NPU Dispatch
- [ ] Добавить async submit в dispatch path (`xrt::run` без немедленного wait)
- [ ] Определить точки data dependency — где реально нужен wait
- [ ] Overlap: NPU считает MUL_MAT, CPU делает residual add / RMSNorm
- [ ] Добавить fence/sync перед чтением NPU результатов
- [ ] Benchmark: измерить pipeline throughput
- [ ] Файлы: `ggml-xdna.cpp` → `graph_compute()` loop

---

## P2: Attention Performance

### 2.1 Fused Attention Decode Kernel
- [ ] Дизайн: `AttentionDecode` — Q·K^T → softmax → V, 1 fused xclbin
- [ ] Реализовать в `compile.py`: `compile_attention_decode()`
- [ ] Добавить `XDNA_OP_ATTENTION_DECODE` в `xdna_op_kind`
- [ ] Реализовать dispatch в `ggml-xdna.cpp`
- [ ] Паттерн-матчер в `graph_compute()` для decode attention
- [ ] Benchmark: сравнить с текущими 0.5x CPU speed
- [ ] Файлы: `compile.py`, `ggml-xdna.cpp`

### 2.2 KV Cache on NPU
- [ ] Выделить persistent BOs для K/V cache на NPU memory
- [ ] Pre-allocate при загрузке модели на max seq_len
- [ ] Implement in-place append (DMA только новую строку K/V)
- [ ] Убрать per-token re-upload полного K/V cache
- [ ] Benchmark: измерить bandwidth savings
- [ ] Файлы: `ggml-xdna.cpp` → контекст + dispatch

### 2.3 FlashAttention-style Tiling для Decode
- [ ] Исследовать AIE L1 SRAM budget (32KB per tile)
- [ ] Реализовать tiled attention: K/V processed in chunks that fit L1
- [ ] Incremental softmax accumulation across tiles
- [ ] Интеграция в fused attention decode kernel (2.1)
- [ ] Benchmark: bandwidth-bound vs compute-bound на разных seq_len
- [ ] Файлы: `compile.py` (IRON MHA operator)

---

## P3: Prefill Improvements

### 3.1 TransformerBlockPrefillFused Benchmarking
- [ ] Запустить на реальных моделях (Llama 3 8B, Qwen3 8B)
- [ ] Сравнить chained (17 runs) vs fused (1 run) per layer
- [ ] Протестировать seq_len buckets: 256, 512, 1024, 2048, 4096
- [ ] Определить оптимальный threshold для выбора fused vs chained
- [ ] Файлы: benchmark scripts, `ggml-xdna.cpp`

### 3.2 Multi-Layer Packing (Layer 4B)
- [ ] Протестировать N=2, 4, 8 transformer blocks per ELF
- [ ] Измерить memory overhead per packed ELF
- [ ] Определить max N для типичных моделей (8B, 3B)
- [ ] Validate on Llama 3 8B + Qwen3 8B
- [ ] Файлы: `compile.py`, `ggml-xdna.cpp`

### 3.3 W8A16 / W8A8 Quantization
- [ ] Включить W8A16 по умолчанию для supported моделей
- [ ] Протестировать W8A8 (INT8 attention + INT8 FFN)
- [ ] Измерить quality loss от квантизации на типичных задачах
- [ ] Определить какие модели benefit most от INT8
- [ ] Файлы: `compile.py`, `ggml-xdna.cpp`

---

## P4: Architecture

### 4.1 Weight Pre-Repacking
- [ ] Реализовать репак весов при загрузке модели (row-major [K,N])
- [ ] Кешировать repacked weights на диск (как xclbin cache)
- [ ] Убрать transpose на first dispatch path
- [ ] Benchmark: cold start time до/после
- [ ] Файлы: `ggml-xdna.cpp` → buffer type + weight loading

### 4.2 Compile-Time Tile Profiling
- [ ] Написать benchmark: перебор tile_m/tile_k/tile_n combinations
- [ ] Запустить на целевом hardware (StrixHalo, Strix, Krackan)
- [ ] Построить lookup table: shape → optimal tiles
- [ ] Интегрировать в `compile.py` tile selection
- [ ] Файлы: `compile.py`, new benchmark script

### 4.3 Remove Python Runtime Dependency
- [ ] Написать offline compiler: `ggml-xdna-compile --model model.gguf --output cache/`
- [ ] Pre-compile все xclbin для target модели
- [ ] Проверить что runtime (C++) не вызывает Python
- [ ] Документация: how to pre-compile, cache format
- [ ] Файлы: new CLI tool, `compile.py`, `CMakeLists.txt`

---

## P5: Quick Wins

### 5.1 Weight BO Lookup Optimization
- [ ] Заменить `unordered_map<void*, xrt::bo>` на `vector<xrt::bo>` indexed by layer+slot
- [ ] Определить deterministic order при загрузке весов
- [ ] Benchmark: lookup latency hash map vs array
- [ ] Файлы: `ggml-xdna.cpp` → `xdna_kernel_entry`, `xdna_swiglu_entry`, etc.

### 5.2 Prefetch Next Layer
- [ ] Реализовать double-buffering: submit layer N, prepare N+1 args
- [ ] Добавить `next_layer_args` буфер в контекст
- [ ] Benchmark: per-step latency с/без prefetch
- [ ] Файлы: `ggml-xdna.cpp` → `graph_compute()`

### 5.3 Remove Mutex on Hot Path
- [ ] Аудит: где `weights_mutex` реально нужен (только bulk_prewarm)
- [ ] Убрать lock из per-token dispatch path
- [ ] Использовать атомики или thread-local state
- [ ] Verify: no race conditions under stress test
- [ ] Файлы: `ggml-xdna.cpp` → dispatch functions

---

## Upstream PR Preparation

- [ ] Squash коммиты в clean history
- [ ] Написать PR description с benchmark results
- [ ] Добавить CMake option: `GGML_XDNA` (автоматический detection XRT)
- [ ] Добавить CI workflow для XDNA backend (если есть hardware runner)
- [ ] Документация: build instructions, supported hardware, performance numbers
- [ ] Тесты: unit tests для compile.py + integration tests для dispatch

---

## Status Legend

| Символ | Значение |
|--------|----------|
| `[ ]` | Не начато |
| `[~]` | В процессе |
| `[x]` | Завершено |
| `[-]` | Отменено / не актуально |

## Notes

- 2026-05-02: Анализ завершён, план составлен
