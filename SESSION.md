Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: **ARG SWAP ПОДТВЕРЖДЁН MARKER TEST. Data-swap фикс (855fa12b1). Ожидает тестирования.**

## Ключевые находки

1. **Arg swap** (`a7f0bef` не откачен) — исправлен (`7999a4f`), **не помог**
2. **Phantom offset** — host K[0] в BO правильный (0x3E39...), DMA читает другие данные (0x3BE7...), 0/8 match
3. **BO адреса корректны**: bo_k=0xC2A000, bo_v=0xC32000, K→V delta=32KB
4. **`xrt::bo::flags::normal`** не поддерживается XDNA драйвером (`unsupported buffer type`)
5. **`cacheable`** в прошлых тестах давал K_DIAG = all zeros (ухудшил)
6. **`svm` и `p2p` не поддерживаются** XDNA драйвером. `cacheable` из другого memory pool (delta -18844 KB)
7. **Marker test** (`5d1feb002`): K[0:8]=0xDEAD, V[0:8]=0xBEEF. Если K_DIAG=0xDEAD → DMA читает K. Если 0xBEEF → V. Если random → phantom offset confirmed.
7. **Диагностические инструменты**: `xdna_diag_offset.cpp`, `patch_flowkv_diag.patch` — standalone тесты, не использовались

## 🧪 Тест после arg swap фикса (2026-05-16 04:55)

### Результаты (build b8883-20566573b):

| Тест | Output |
|------|--------|
| STEP 1: FlowKV + swap fix | `"TheDED欠ificio whificio whdio zd wh exclude欠 Bret欠 wh.sub"` — мусор |
| STEP 2: Phantom diag + FlowKV | `"The sometimesieiagma Ler flight scaperorbitsachteiritわり欠 Bretntenioso"` — мусор |
| STEP 3: Baseline (no FlowKV) | `"The capital of France is Paris."` — правильно ✓ |

### Ключевое наблюдение из POC diagnostic:

```
BO check kv_h=0: K[0] first 8 bf16: 0x3E39 0x3D9F 0x3E28 0x3E02 0xBB90 0x3E08 0xBDF3 0xBD73  ← host K[0] ✓
K_DIAG    [0:8]: 0x3BE7 0x3D0F 0x3CD3 0x3C79 0x3C8C 0xBBC6 0x3C69 0x3C00  ← DMA читает ✗
```

**Host K[0] в bo_k правильный** (0x3E39..., совпадает с предыдущими сессиями).
**K_DIAG — данные, которых нет ни в K, ни в V.** DMA читает из неверного адреса.

### Вывод:

- **Arg swap** (коммит `a7f0bef`) — был реальным багом, но его фикс **не решает** проблему.
- **Phantom offset** — основная причина. `host_only` буфер маппится в AIE address space со смещением.
- Diagnostic был вставлен в **per_head path** (не используется), нужен в **POC path** (используется) → исправлено в `d1d3dc125`.

### Результат diagnostic (`6b5a5cf83`):

- BO адреса корректны: bo_k=0xC2A000, bo_v=0xC32000, K→V delta=32KB (размер bo_k)
- `xrt::bo::flags::normal` **не поддерживается** XDNA драйвером: `unsupported buffer type: invalid argument`
- K[0] в BO правильный (0x3E39...), K_DIAG другой (0x3BE7...) — DMA читает не оттуда

### Следующий шаг

Запустить `debug_flowkv_diag.bat` с `6b5a5cf83`. STEP 2 попробует `svm`, `p2p`, `cacheable` флаги и сравнит адреса с `host_only`. Если `svm` даёт другой адрес → phantom offset confirmed.

---

## 🔧 ФИКС: Argument Swap (2026-05-16 04:39)

### Корневая причина

Коммит `a7f0bef` в IRON-windows **поменял порядок аргументов** runtime sequence как diagnostic test:
```diff
-    with rt.sequence(L3_K_ty, L3_V_ty, L3_Q_ty, L3_O_ty) as (K, V, Q, O):
+    with rt.sequence(L3_V_ty, L3_K_ty, L3_Q_ty, L3_O_ty) as (V, K, Q, O):
```
Коммит **не был откачен** после тестирования.

### Цепочка маппинга (до фикса)

```
design.py:  rt.sequence(L3_V_ty, L3_K_ty, ...) → V=arg0, K=arg1
MLIR:       K_0 DMA reads %arg1, V_0 DMA reads %arg0
XRT:        arg0 → group_id(3), arg1 → group_id(4)
host:       bo_k → group_id(3) = arg0 ← design ждёт V!
            bo_v → group_id(4) = arg1 ← design ждёт K!
```

**Итого:** K DMA читает V данные, V DMA читает K данные.

### Фикс

IRON-windows commit `7999a4f`:
1. **Revert arg order**: `L3_V_ty, L3_K_ty` → `L3_K_ty, L3_V_ty`
2. **Remove DIAG offset +64** в `make_k_tap()` — diagnostic из `f208050`, больше не нужен

### Почему все предыдущие тесты выглядели как "phantom +32KB offset"

- Shared buffer: K DMA читает arg1=V → V data начинается со смещения 16384 bf16 = **32KB** от начала → выглядит как phantom offset ✓
- Separate buffer: K DMA читает из V-буфера (чужой буфер) → random garbage ✓
- Swap args: двойной swap → обратно к V data ✓
- Все тесты стабильны (не random) → потому что swap детерминирован ✓

**Phantom offset гипотеза (driver header alignment) — артефакт argument swap'а.** Не нуждается в отдельном тестировании.

### Статус diagnostic инструментов

- `xdna_diag_offset.cpp` — не использовался, можно удалить
- `patch_flowkv_diag.patch` — не наложен на код, можно удалить
- `debug_flowkv_diag.bat` — diagnostic шаг 1.5 не сработал (код не в бинарнике)

### Следующий шаг

Пересобрать IRON MLIR с фиксом → пересобрать llama.cpp-xdna → тест. Ожидание: FlowKV output = корректный текст.

---

## История тестов (2026-05-16, до фикса)

| Тест | K BD offset | K DMA читает из | Ожидание | Результат |
|------|------------|-----------------|----------|-----------|
| Shared buffer | 0 | 16384 (V data) | 0 (K data) | swap: читает V ✓ |
| Separate buffers | 0 | V buffer (чужой) | 0 (K data) | swap: читает V ✓ |
| Swap args (%arg1) | 0 | V data | V data | double swap ✓ |
| K TAP offset +64 | 64 | V data + 64 | K data + 64 | swap ✓ |

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

**IRON-windows** (commits a9ec332, b0446c2, 6f73ceb):
- `design.py`: K DMA stride `2*head_dim` → `head_dim`, V DMA offset `+head_dim` → `+kv_region_size`
- `op.py`: `interleave_kv_cache()` → `contiguous_kv_cache()`
- `reference.py`: добавлена `contiguous_kv_cache()` функция
- `test.py`: `KV_interleaved` → `KV_contiguous`
- **ВАЖНО**: op.py компилирует `aie2p/flowkv.cc`, НЕ `aie2/flowkv.cc`!
- K diagnostic: score_chunk сохраняет K[0] в static buffer, value_normalize пишет в last head output slot

**llama.cpp-xdna** (commits 15d51cf2b, 358024d41, 30dee284b):
- `ggml-xdna.cpp`: оба dispatch пути (per_head + POC):
  - `dst_k = pos * 2 * row_bytes` → `pos * row_bytes`
  - `dst_v = (pos * 2 + 1) * row_bytes` → `kv_region + pos * row_bytes`
- `ggml-xdna.cpp`: Q diagnostic fix — f32→bf16 конвертация
- `ggml-xdna.cpp`: bo_out K_DIAG вывод для last head

#### Результаты тестов:

1. **KV layout фикс работает**: host пишет contiguous (K[0]@0, K[1]@64, V[0]@16384), MLIR DMA stride=64 ✓
2. **Kernel пересобирается**: bo_out изменился (0xBB55→0xBB51) после фикса ✓
3. **Но output всё ещё мусор**: NPU=0.004 vs CPU=-0.894 → проблема не в K/V layout
4. **K diagnostic: K_DIAG ≠ host K[0]** — DMA читает совершенно другие данные

#### K diagnostic результат (2026-05-15):

```
Host K[0]:    0x3E39 0x3D9F 0x3E28 0x3E02 0xBB90 0x3E08 0xBDF3 0xBD73  ← bo_kv после sync ✓
K_DIAG NPU:   0xBB1A 0xBAC8 0x3B0D 0xBBF1 0xB9CC 0x3A11 0xBB64 0x3B8A  ← kernel видит ✗
```

**Ни одно из 8 значений не совпадает.** K_DIAG содержит данные, которых нет ни в:
- исходном K тензоре (raw f16: 0x31C6 0x2CF7...)
- bo_kv после f16→bf16 конверсии (0x3E39 0x3D9F...)
- bo_kv после read-back verif (тот же 0x3E39...)

#### Анализ: DMA config vs host layout

DMA config для K:
```
aie.dma_bd(%arg0, 0, 16384, [<size=1,stride=0>, <size=256,stride=64>, <size=1,stride=0>, <size=64,stride=1>])
```

- offset=0 (K region start) ✓
- 256 итераций × stride=64 bf16 = contiguous K[0], K[1], ... K[255] ✓
- Host пишет K[pos] на `pos * row_bytes` (row_bytes=128=64 bf16) → layout совпадает ✓
- bo_kv size = 65536 bytes = 32768 bf16 (K: 0..16383, V: 16384..32767) ✓

**DMA config корректна.** Данные в bo_kv корректны (read-back подтверждает). Но DMA → tile memory delivers wrong data.

#### Вероятные причины:

1. **NoC cache coherency** (наиболее вероятно): `XCL_BO_SYNC_BO_TO_DEVICE` синхронизирует host→DDR, но AIE NoC DMA controller читает из кэша (stale). Нужен explicit cache invalidation на стороне DMA/NOC.
2. **bo_kv buffer aliasing**: XRT `host_only` buffer может маппиться в host VA space и AIE DDR space по-разному.
3. **Shim tile contention**: Q и K оба через `shim_noc_tile_0_0`, возможна NoC интерференция.

#### Диагностический патч (commits d05385a6a, 001050108):

Добавлен в `ggml-xdna.cpp` (при `XDNA_DEBUG=1`):
- **ВАЖНО**: diagnostic добавлен в **оба** dispatch пути: `per_head` и `POC` (POC — основной, используется в production)
- Первый патч (d05385a6a) стоял только в `per_head` → не работал, т.к. реальный dispatch идёт через POC path
- Второй патч (001050108) добавил проверки в POC path

1. **Pre-dispatch sync verify**: после bo_kv sync TO_DEVICE, делаем FROM_DEVICE read-back и сравниваем с записанными данными. Если `match=NO` → host cache не сбрасывается в DDR.
2. **Post-dispatch bo_kv integrity**: после выполнения kernel, проверяем что DMA не испортил source buffer.
3. **K_DIAG vs host K[0] comparison**: сравниваем что kernel прочитал (K_DIAG из last head output slot) с тем что host записал в bo_kv.

Ожидаемый вывод:
```
[DIAG-SYNC] bo_kv verify after FROM_DEVICE:
  K[0][0:8]: 0x3E39 ...   K[0] match=YES/NO!!
[DIAG-COMPARE] kv_h=0:
  Host  K[0][0:8]: 0x3E39 ...
  K_DIAG    [0:8]: 0xBB1A ...
  Match count (first 8): 0/8 → ZERO MATCH — DMA reads wrong data!
```

#### Что проверить дальше:

- Запустить `debug_flowkv.bat` с патчем d05385a6a
- Если `[DIAG-SYNC] match=NO` → проблема в XRT sync (host→DDR)
- Если `[DIAG-SYNC] match=YES` но `[DIAG-COMPARE] 0/8` → проблема в NoC/DMA cache (DDR→tile)
- Возможные фиксы:
  - Использовать `xrt::bo` с другими флагами (не `host_only`)
  - Добавить explicit cache flush через XRT API
  - Проверить, не нужен ли `XCL_BO_CACHEABLE` флаг
  - Тест: alloc bo_kv как `xrt::bo::flags::cacheable` вместо `host_only`

### Осталось:

- **Запустить debug_flowkv.bat** с патчем 001050108 на Windows (git pull + пересборка)
- Скинуть лог с `[DIAG-SYNC]`, `[DIAG-POST]`, `[DIAG-COMPARE]` выводом
- По результатам: фиксить cache coherency (XRT bo flags / explicit flush)
- После фикса: удалить diagnostic код, протестировать output
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

---

## Сессия 2026-05-16 (продолжение): cacheable не помогает

### Результат теста `cacheable`:
- `host_only`: K_DIAG = random non-zero data (0x3BD5, 0x3D0F, ...)
- `cacheable`: K_DIAG = all zeros (0x0000)
- `cacheable` **ухудшил** ситуацию — DMA вообще не видит данные

### Вывод:
Проблема НЕ в NoC cache coherency. Проблема в адресном пространстве:
DMA controller читает из другого места, не из bo_kv.

### Следующий шаг:
Добавлен diagnostic marker test — заполняем bo_kv[0:8] = 0xDEAD перед dispatch.
Если K_DIAG == 0xDEAD → DMA читает из bo_kv (проблема в данных).
Если K_DIAG ≠ 0xDEAD → DMA читает из WRONG buffer (проблема в адресе).

---

## Сессия 2026-05-16 (продолжение 2): Marker test — DMA reads WRONG buffer

### Marker test result:
```
bo_kv[0:8] = 0xDEAD 0xDEAD ... (marker, synced to device)
K_DIAG[0:8] = 0xB928 0x3B4A ... (this is V[0] data!)
→ K DMA reads from V region (offset 16384 bf16) despite DMA config offset=0
```

### Root cause hypothesis:
Both K and V DMAs read from the same buffer (arg0) with different offsets.
When started concurrently, the V DMA's offset (16384) overwrites K DMA's
offset (0) — shared DMA register or buffer mapping bug.

### Test: separate K/V into different task_groups (sequential DMA)
IRON-windows commit 1603071: K and V fills are now in separate task_groups.
If K reads correct data → concurrent DMA offset corruption confirmed.
If K still reads V data → DMA offset is fundamentally broken for arg0.

---

## Сессия 2026-05-16 (продолжение 3): Phantom Offset Diagnostic

### Анализ от внешней нейросети
Получен анализ 3 гипотез корневой причины:
1. **Argument swap** — уже найден и подтверждён (K→V slot, V→K slot)
2. **Driver header alignment** — kipudrv.inf может добавлять 32KB метаданных к host_only аллокациям
3. **Compiler bitmask defect** — спекулятивная, менее вероятная

### Созданы диагностические инструменты
- `xdna_diag_offset.cpp` — автономный диагност (4 теста: host_only vs normal, DMA registers, export, K/V aliasing)
- `patch_flowkv_diag.patch` — встроенная диагностика в FlowKV путь (XDNA_DIAG_OFFSET=1)
- `debug_flowkv_diag.bat` — обновлённый bat-файл с STEP 1.5 diagnostic

### Что тестирует STEP 1.5
1. Запускает llama-cli с XDNA_DIAG_OFFSET=1
2. Логирует адреса всех BO (bo_k, bo_v, bo_q, bo_out)
3. Создаёт второй K-буфер с флагом `normal` вместо `host_only`
4. Считает delta между host_only и normal
5. **Если delta = 32768 (32KB) → гипотеза 2 ПОДТВЕРЖДЕНА**

### Следующий шаг
Запустить `debug_flowkv_diag.bat` на Windows-машине с NPU.
Результат: `step1.5_diag_offset.log` — искать "XDNA_DIAG_OFFSET" и "PHANTOM OFFSET".

### Commits
- `d84d5bba9` (llama.cpp-xdna ggml-xdna): diag: phantom DMA offset diagnostic tools
- IRON-windows: K TAP offset +64 test (f208050) — результат: мусор, DMA добавляет +16384 к любому offset

---

## Сессия 2026-05-16: Анализ diagnostic logs

### Получены файлы
- `step1_fix_output.log` + `step1_fix_xdna.log` — FlowKV + arg-swap fix (без DIAG)
- `step2_diag_output.log` + `step2_diag_offset.log` — FlowKV + XDNA_DIAG_OFFSET=1
- `step3_baseline_output.log` + `step3_baseline_xdna.log` — Baseline (FlowKV OFF)
- `debug_flowkv_diag.bat` — bat-файл для запуска

### Результаты
| Тест | Output | Статус |
|------|--------|--------|
| STEP 1 (FlowKV fix) | "Theicc dio CallableNormalize tend.raw336udio..." | ❌ мусор |
| STEP 2 (FlowKV + DIAG) | "The Glassuding Forgot lã tendakisachte Bret..." | ❌ мусор |
| STEP 3 (Baseline) | "The capital of France is Paris." | ✅ правильно |

### Анализ step2_diag_offset.log (240 dispatch'ей)

**BO адреса (стабильные):**
- bo_k (group_id 3): `0x0000000002DA9000` size=32768
- bo_v (group_id 4): `0x0000000002DB1000` size=32768
- K→V delta: 32768 bytes (32KB) — корректно
- bo_k_cacheable: `0x0000000004010000` (delta -18844 KB — другой memory pool)
- svm/p2p: FAILED (unsupported)

**K_DIAG vs Host K[0]:**
- 0/8 match на ВСЕХ 240 dispatch'ах
- Host K[0] стабильный: `0x3E39 0x3D9F 0x3E28 0x3E02 0xBB90 0x3E08 0xBDF3 0xBD73`
- K_DIAG **меняется каждую итерацию** — 240 уникальных паттернов из 240
- K_DIAG[0]: 211 уникальных значений из 240
- K_DIAG значения: bf16 floats в диапазоне ~[-0.2, 0.4] (похоже на веса/attention scores)
- bo_k после dispatch (DIAG-POST) не испорчен — данные intact

**Критическая находка: MARKER TEST НЕ ЗАПУСТИЛСЯ!**
- Build: `b8888-62b8c20c0` (коммит ДО marker test `5d1feb002`)
- В логе НЕТ строки `[MARKER] K[0:8]=0xDEAD V[0:8]=0xBEEF`
- K_DIAG содержит обычные bf16 значения, не 0xDEAD/0xBEEF
- **Нужен пересбор с текущим HEAD для запуска marker test**

**DIAG-SYNC отсутствует:**
- Pre-dispatch sync verify (bo_kv read-back после TO_DEVICE) не реализован в коде
- Только DIAG-POST (bo_k после dispatch) и DIAG-COMPARE (K_DIAG vs host K[0])

### Выводы
1. DMA читает не из bo_k — данные не совпадают ни с K, ни с V
2. K_DIAG меняется каждую итерацию → DMA читает из области памяти, которая перезаписывается
3. Это **не** NoC cache coherency (host_only данные корректны после sync)
4. Это **не** arg swap (K и V в отдельных буферах, правильный порядок)
5. **Наиболее вероятная причина**: XRT `host_only` buffer → DMA address mapping сломан
   - `address()` возвращает host VA, но DMA controller видит другой физический адрес
   - Или kipudrv.inf добавляет metadata prefix к host_only allocations

### Следующий шаг
1. **Пересобрать llama.cpp-xdna** с текущим HEAD (`d3afe055e`) — включить marker test
2. **Запустить debug_flowkv_diag.bat** с новым бинарником
3. Ожидаемый результат marker test:
   - K_DIAG == 0xDEAD → DMA читает K buffer (correct!) → проблема в данных
   - K_DIAG == 0xBEEF → DMA читает V buffer (arg swap regression)
   - K_DIAG == random → DMA reads from completely wrong address
4. После marker test: фиксить בהתאם к результату

### Commits сессии
- `7999a4f` (IRON-windows devel): fix(flowkv): revert K/V arg swap, remove DIAG offset +64
- `d3afe055e` (llama.cpp-xdna ggml-xdna): docs: SESSION.md — svm/p2p failed, marker test next
- `5d1feb002` (llama.cpp-xdna ggml-xdna): diag: marker test (НЕ включён в протестированный бинарник)
- `1920fd641` (llama.cpp-xdna ggml-xdna): fix(flowkv): swap bo_k/bo_v in set_arg — **не помог** (K_DIAG=0xDC04)
- `bfc33c7ac` (llama.cpp-xdna ggml-xdna): fix: correct diagnostics after arg swap — **не помог**
- `bd1f27af9` (llama.cpp-xdna ggml-xdna): **fix(flowkv): swap K/V data between bo_k and bo_v** ← DATA-SWAP
- `855fa12b1` (llama.cpp-xdna ggml-xdna): fix: update marker test for data swap

---

## Сессия 2026-05-16 (продолжение): Тестирование фиксов

### Тест 1: arg-swap (set_arg) — `bfc33c7ac`
- Build: `b8894-bfc33c7ac`
- Marker test: K_DIAG = `0xDC04 × 8` — ELSEWHERE (не 0xDEAD, не 0xBEEF)
- Swap set_arg НЕ работает — DMA читает из неизвестного места
- Возможная причина: xclbin кэширует mapping DMA→buffer, set_arg swap не компенсирует

### Тест 2: data-swap — `855fa12b1` (текущий)
- Подход: K data → bo_v (arg1), V data → bo_k (arg0), set_arg порядок оригинальный
- K DMA → arg1 = bo_v = K data
- V DMA → arg0 = bo_k = V data
- Marker test обновлён: 0xBEEF = CORRECT (K DMA читает arg1)
- **Ожидает тестирования**

### Почему arg-swap не работает
Простой swap bo_k↔bo_v в set_arg не компенсирует проблему:
- До swap: K DMA → arg1 = bo_v = V data (0xBEEF)
- После swap: K DMA → arg1 = bo_k = ???  (0xDC04, ELSEWHERE)
- 0xDC04 — не K data, не V data, не маркеры → DMA читает из "phantom" адреса
- Вывод: DMA address mapping зависит не только от set_arg позиции, но и от физического буфера

### Почему data-swap должен работать
- Не меняем mapping буферов (bo_k@arg0, bo_v@arg1)
- Меняем только КАКИЕ ДАННЫЕ кладём в каждый буфер
- K DMA всегда читает arg1 = bo_v → кладём K данные в bo_v
- V DMA всегда читает arg0 = bo_k → кладём V данные в bo_k
