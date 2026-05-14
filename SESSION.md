Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: decode 1.6 t/s (was 1.0), но output мусор → исправлено (KV layout bug)

---

## Текущая сессия (2026-05-15)

### Сделали:

- Проанализировали логи step2 (без FlowKV) и step3 (с FlowKV) + Dfff.txt + debug_flowkv.bat
- Нашли **корневой баг**: K/V buffer layout mismatch между host и kernel

#### Проблема: interleaved vs contiguous KV layout

Host укладывал K/V **interleaved** в bo_kv:
```
[K_pos0 | V_pos0 | K_pos1 | V_pos1 | ...]
  byte 0   byte 128  byte 256  byte 384
```

MLIR DMA читал K с stride=128 (пропуская V) → корректно извлекал K[0], K[1], K[2]...
Но **kernel** читал K с stride=64 (contiguous):
```c
const bfloat16 *k_pos = k_chunk + pos * head_dim;  // stride = 64
```

Результат: kernel читал V[0] как K[1], V[1] как K[3], etc.
```
Kernel читает:     K[0]=OK, K[1]=V[0], K[2]=K[1], K[3]=V[1], ...
NPU vs CPU: [0] NPU=0.004333 CPU=-0.879113 diff=0.883446
```

#### Фикс: contiguous layout [K_all | V_all]

**IRON-windows** (commit a9ec332):
- `design.py`: K DMA stride `2*head_dim` → `head_dim`, V DMA offset `+head_dim` → `+kv_region_size`
- `op.py`: `interleave_kv_cache()` → `contiguous_kv_cache()`
- `reference.py`: добавлена `contiguous_kv_cache()` функция
- `test.py`: `KV_interleaved` → `KV_contiguous`

**llama.cpp-xdna** (commit 15d51cf2b):
- `ggml-xdna.cpp`: оба dispatch пути (per_head + POC):
  - `dst_k = pos * 2 * row_bytes` → `pos * row_bytes`
  - `dst_v = (pos * 2 + 1) * row_bytes` → `kv_region + pos * row_bytes`
- `ggml-xdna.cpp`: Q diagnostic fix — f32→bf16 конвертация вместо чтения сырых 2 байт

#### Дополнительные находки:

- **bf16_to_int()** в kernel — формула `(mant << (exp-7))` корректна для integer bf16 значений
- **Q diagnostic** ("Q src raw") обманчиво показывал мусор — читал нижние 2 байта f32 данных (q_nb0=4) как bf16. Реальные bf16 значения в BO check корректны
- **Q tensor**: type=0 (f32), ne=[64,1,32], nb=[4,8192,256] — stride=4 потому что это f32, не bf16
- **actual_seq_len=42** читается из angles region в Q buffer — host корректно кодирует как bf16

### Осталось:

- **Перекомпилировать flowkv.o** на Windows (NPU peano toolchain)
- Удалить старый кэш: `rmdir /s /q npu_kernels_win_8col\flowkv_H4_KV1_d64_S256_C32_1col`
- Протестировать — output должен быть "The capital of France is Paris."
- Если работает → убрать diagnostic код
- Дальше: merge QKV+batch, RMSNorm on NPU

---

## Предыдущие сессии

### Сессия 1: FlowKV POC + double RoPE

- FlowKV POC dispatch работал, но CPU attention перезаписывал результат (нет continue)
- SOFT_MAX выполнялся на CPU вхолостую
- Добавили 3 фикса: continue после POC, skip SOFT_MAX (flag), early dispatch check
- Закоммичено: 4cfd1daf2, запушено в ggml-xdna
- Нашли корневой баг: двойной RoPE (kernel применял RoPE к уже post-RoPE Q)
- Фикс: заменили RoPE в kernel на memcpy
- Ключевой инсайт: identity RoPE (cos=1, sin=0) = identity function → double RoPE не причина мусора
- Реальная причина мусора была в KV layout (найдено в следующей сессии)

---

## Архитектура

### FlowKV Decode Attention

2-tile streaming pipeline per KV head group:
- **Score tile (CT0)**: Q*K^T/sqrt(d) + online softmax → packed [F_c | C_c | l]
- **Value tile (CT1)**: weighted V accumulation + normalize → output

### DDR buffer layout (после фикса):

- **KV cache** (arg0): `[K_all (num_kv*seq*hd) | V_all (num_kv*seq*hd)]` contiguous
- **Q** (arg1): `[Q_group0 (gs*hd) | angles (hd) | actual_seq_len (1)]` per KV group
- **Output** (arg2): `(num_heads * head_dim)` bf16

### Graph structure:

ggml scheduler разбивает attention на 3 graph_compute:
1. **QKV** (21 nodes): Q/K/V projection + RoPE → permuted tensors
2. **SOFT_MAX** (1 node): пропускается когда FlowKV включён
3. **CONT+attn_out** (5 nodes): FlowKV POC dispatch → output projection

### Key env vars:

- `XDNA_ENABLE_FLOWKV_DECODE=1` — включить FlowKV
- `XDNA_DEBUG=1` — diagnostic вывод (Q/K/V data, NPU vs CPU comparison)
- `XNA_SCHED_DEBUG=1` — scheduler logging

### Кэш ключ:

`flowkv_H4_KV1_d64_S256_C32_1col` — H=q_heads_per_kv, KV=1, d=head_dim, S=seq_len, C=chunk_size
