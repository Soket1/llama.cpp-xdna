Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: **FlowKV decode работает, batch num_cols=4, 5.4 t/s.** Baseline 6.0 t/s.

## Ключевые находки

### FlowKV batch num_cols=4 (2026-05-20)

Root cause batch garbage: `num_heads` param к `get_or_load_flowkv_kernel()` был `q_heads_per_kv` (4) вместо `q_heads_per_kv * num_cols` (16). Kernel скомпилирован с `group_size=1`, host готовил данные для `group_size=4`.

Фикс: передавать `q_heads_per_kv * num_cols` как `num_heads`. Cache key: `flowkv_H16_KV4_d64_S256_C32_4col`.

### FlowKV dispatch profiling

| Фаза | ms/dispatch | ms/token (×24) |
|------|-------------|----------------|
| KV+Q memcpy+sync | 0.07 | 1.7 |
| NPU exec (avg) | 0.56 | 13.4 |
| **Итого** | | **~15 ms** |

**83% overhead = NPU execution time.** Host optimization не поможет.

### Что не помогло

- persistent xrt::run (num_cols=1): 3.9 vs 4.8 t/s
- persistent xrt::run (num_cols=4): 5.0 vs 5.3 t/s
- CONT skip + bf16 passthrough: ~5.0 t/s (в пределах noise)
- num_cols=8: XRT runtime выделяет только 4 columns (остальные — Windows Studio Effects)

### Fused FlowKV + O_proj (будущее)

Новый IRON kernel: attention + O_proj в одном xclbin. Экономия ~5 ms/token. ROI низкий (сложность vs выигрыж). Сохранено в NPU_PLAN.md как Priority 5.

### Probe удалён из kernel (2026-05-20)

`g_probe_enabled`, `FLOWKV_PROBE_MAGIC`, diag arrays удалены из `flowkv.cc`.

### Bug 1 (v10): K DMA routing — mirror K into bo_k

IRON compiler читает K FIFO из `bo_k` (arg0), не из `bo_v` (arg1) как предполагал workaround. Host писал K только в `bo_v[0]`, tile видел нули.

Фикс: mirror K data в `bo_k` через memcpy после записи в `bo_v`.

### Bug 2 (v11): CONT node skip

После FlowKV dispatch, `continue` пропускал CONT node. MUL_MAT для `attn_output.weight` читал из CONT output buffer (пустой), а не из `kqv_out->data` (CONT input).

Фикс: убрал `continue`, дал CONT выполниться в CPU range.

### Корневая причина

Два независимых бага маскировали друг друга: первый давал нули для K, второй давал garbage для output. После обоих фиксов модель выдаёт осмысленный текст.

## Архитектура

### FlowKV Decode Attention

2-tile streaming pipeline per KV head group:
- **Score tile (CT0)**: Q*K^T/sqrt(d) + online softmax → packed [F_c | C_c | l]
- **Value tile (CT1)**: weighted V accumulation + normalize → output

### DDR buffer layout

- **KV cache**: `[K_all (num_kv*seq*hd) | V_all (num_kv*seq*hd)]` combined в bo_v
- K data mirror: дополнительно пишется в bo_k (arg0) для DMA routing
- **Q** (arg1): `[Q_group0 (gs*hd) | angles (hd) | actual_seq_len (1)]` per KV group
- **Output** (arg2): `(num_heads * head_dim)` bf16

### Graph structure

ggml scheduler разбивает attention на 3 graph_compute:
1. **QKV** (21 nodes): Q/K/V projection + RoPE → permuted tensors
2. **SOFT_MAX** (1 node): пропускается когда FlowKV включён
3. **CONT+attn_out** (3 nodes): FlowKV POC dispatch → CONT → output projection

### Key env vars

- `XDNA_ENABLE_FLOWKV_DECODE=1` — включить FlowKV
- `XDNA_DEBUG=1` — diagnostic вывод
- `XNA_SCHED_DEBUG=1` — scheduler logging
- `XDNA_FLOWKV_MATH_DIAG=1` — v11 CPU reference (attention math)
- `XDNA_FLOWKV_PROFILE=1` — dispatch profiling (KV+Q memcpy, run_create, exec, sync, scatter)

### Кэш ключ

`flowkv_H16_KV4_d64_S256_C32_4col` — H=q_heads_per_kv*num_cols, KV=num_cols, d=head_dim, S=seq_len, C=chunk_size

## debug_flowkv.bat результат

## debug_flowkv.bat результат

| Test | Output | Status |
|------|--------|--------|
| STEP 2 (без FlowKV) | "The capital of France is Paris." | PASS |
| STEP 3 (с FlowKV) | "The capital of France is Paris." | PASS |

## echo_custom diagnostic tests

| Test | Description | Status |
|------|-------------|--------|
| v1 | Single ObjectFifo copy | PASS |
| v2 | Dual ObjectFifo concat | PASS |
| v3 | FlowKV-like two-tile inter FIFO | PASS |
| v4 | FlowKV-like Q/K/V arg order + TAP | PASS |
| v5 | Multi-chunk sequencing | PASS |
| v6 | Packed inter layout [F_c \| C_c \| l] | PASS |
| v7 | Score-side online softmax math | PASS |
| v8 | Value-side accumulation/normalization | PASS |
| v9 | Production-layout ABI/TAP/Q metadata | PASS |

## Diagnostic tools

- `XDNA_FLOWKV_REAL_PROBE=1` — v10: copies Q/K/V/META through inter FIFO, validates data transport
- `XDNA_FLOWKV_MATH_DIAG=1` — v11: compares NPU attention output with CPU f32 reference
- `flowkv_cpu_reference()` — static helper in ggml-xdna.cpp for CPU attention computation

## IRON compiler bugs

1. **Both K_fifos and V_fifos DMA read from arg1** (not K→arg0, V→arg1). Workaround: combine K+V in single buffer (bo_v). Additional fix: mirror K into bo_k.
2. **K DMA routing reads arg0** (confirmed by v10 probe). The "both read arg1" comment was wrong — K actually reads arg0.

## Файлы

- `IRON-windows/aie_kernels/aie2p/flowkv.cc` — kernel: score softmax, value accum/normalize, v10 probe
- `IRON-windows/iron/operators/flowkv_decode/design.py` — IRON design (TAP offsets, runtime sequence)
- `ggml/src/ggml-xdna/ggml-xdna.cpp` — host dispatch, bo_k mirror, CONT fix, v10/v11 diagnostics
