Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: **FlowKV decode работает.** "The capital of France is Paris." через `debug_flowkv.bat`.

## Ключевые находки

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
- `XDNA_FLOWKV_REAL_PROBE=1` — v10 inline probe (data transport)
- `XDNA_FLOWKV_MATH_DIAG=1` — v11 CPU reference (attention math)

### Кэш ключ

`flowkv_H4_KV1_d64_S256_C32_1col` — H=q_heads_per_kv, KV=1, d=head_dim, S=seq_len, C=chunk_size

## v10 FlowKV inline real-data probe

Env-gated inline probe: score tile forwards Q/K/metadata through inter FIFO, value tile captures and writes to output heads.

```
Qmatch=64  Kmatch=64  Vmatch=64  Mmatch=1
```

Подтвердил: K DMA routing сломан (K FIFO читает arg0, не arg1). Фикс: mirror K в bo_k.

## v11 FlowKV attention math diagnostic

Host-side CPU reference comparison: вычисляет attention на CPU в float32 и сравнивает с NPU output.

```
max_abs=0.000479  max_rel=0.5468  rms=0.000178
```

Подтвердил: **attention math правильный**. Проблема была в graph flow (CONT skip).

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
