Контекст: форкаю AMD IRON и llama.cpp для XDNA NPU на Windows.

Клонируй:
- https://github.com/Soket1/IRON-windows (ветка devel)
- https://github.com/Soket1/llama.cpp-xdna (ветка ggml-xdna)

Hardware: STX NPU2, 8 columns, model: llama-3.2-1b-BF16

Статус: **aiecc deadlock починен. axpy тест: compilation OK, NPU kernel timeout — fix в процессе (insts_bo sync).**

## Ключевые находки

1. **IRON compiler bug: BOTH K_fifos и V_fifos DMA читают из arg1** (а не K→arg0, V→arg1)
   - Marker test подтвердил: K DMA → arg1 (0xBEEF marker из bo_v)
   - Data-swap подтвердил: V DMA тоже → arg1 (NPU output = 0xBEEF constant = K marker из bo_v)
2. **Фикс: K+V объединены в одном буфере (bo_v = arg1)**
   - K data: offset 0 в bo_v
   - V data: offset kv_region_size в bo_v
   - V TAP: tensor_dims=(2*kv_region), offset=kv_region_size
3. **Arg swap** (`a7f0bef` не откачен) — исправлен (`7999a4f`), **не помог**
4. **BO адреса корректны**: bo_k=0xC2A000, bo_v=0xC32000, K→V delta=32KB
5. **`xrt::bo::flags::normal`** не поддерживается XDNA драйвером
6. **`svm` и `p2p` не поддерживаются** XDNA драйвером
7. **`cacheable`** из другого memory pool (delta -52888 KB), K_DIAG = all zeros
8. **Диагностические инструменты**: marker test (`5d1feb002`), phantom offset diagnostic
9. **`pyxrt.bo.cacheable` crash на XDNA** — access violation при создании insts buffer через CachedXRTRuntime. `host_only` работает. Фикс: monkey-patch в conftest.py
10. **pyxrt.pyd = cp313** — скомпилирован под Python 3.13, несовместим с conda Python 3.12 (ABI mismatch). Весь тестовый стек должен работать на Python 3.13
11. **Environment isolation** — PYTHONPATH не должен включать conda site-packages целиком (конфликт numpy/pytest). Только XRT SDK python + llvm-aie
12. **aiecc.py → aiecc.exe deadlock (ROOT CAUSE FOUND & FIXED)** — `aiecc.py` вызывает `subprocess.run([aiecc.exe, ...])` **без** `stdin=DEVNULL`. Фикс: на Windows `_resolve_aiecc` предпочитает `aiecc.exe` напрямую. Коммит `1f5f7c5` (IRON-windows devel)
13. **insts_bo sync для host_only** — `pyxrt.bo.cacheable` крашит на XDNA → monkey-patch на `host_only`. Но `host_only` буферы не видны NPU без явного `sync_bo(TO_DEVICE)`. Hostruntime НЕ вызывает sync (предполагает cacheable=coherent). Фикс: monkey-patch `XRTHostRuntime.run` для sync insts_bo. Коммит `5048d27` (IRON-windows devel)
14. **echo_custom v1 PASS (2026-05-17)** — standalone raw DMA echo через custom XRT dispatch работает: `run_echo_custom.bat 1` собрал xclbin/insts, скомпилил `echo_test.exe`, `run.start(); run.wait()` завершился `state=4`, output `256/256 match`.
15. **echo_custom v2 PASS (2026-05-17)** — после фикса test design (`out_fifo` должен быть `memref<2*N>`, `N=128` для v2) dual input ObjectFIFO concat работает: `A=128/128 B=128/128`, `PASS: echo v2`.
16. **Вывод из echo_custom v1/v2** — `host_only` BO, custom XRT dispatch, single ObjectFIFO и dual input ObjectFIFO сами по себе исправны. FlowKV баг теперь сужен до FlowKV-specific path: inter-tile FIFO (`inter_0`), K/V/Q/O task ordering, TAP stride/offset, или generated NPU instruction descriptor binding.
17. **Важный runtime факт** — в custom XRT dispatch нужен явный `run.start()` перед `run.wait()`. Без `run.start()` `echo_test.exe` падал с `0xC0000409` после `set_arg`.

## echo_custom raw DMA tests (2026-05-17)

### Что проверяли

`IRON-windows\iron\operators\dma_echo\run_echo_custom.bat 1` — минимальный independent test без FlowKV:

```text
host bo_in -> Shim DMA -> ObjectFIFO -> AIE echo_copy_bf16 -> ObjectFIFO -> Shim DMA -> host bo_out
```

Цель: отделить общий XRT/host_only/raw DMA сбой от FlowKV-specific descriptor/TAP проблемы.

### Что пришлось починить в IRON-windows

Commit `14ca4a0 fix(dma_echo): repair custom Windows echo runner` в `Soket1/IRON-windows:devel`:

- `run_echo_custom.bat`:
  - MSVC env через `vcvars64.bat`;
  - Python313 `mlir_aie\bin\aiecc.exe` напрямую, не через Python wrapper;
  - `llvm-aie\bin\clang.exe` + правильные `mlir_aie`/`llvm-aie` include paths;
  - strip `.tctmemtab`, `.tctmemtabl`, `.tctmemstrtab` из `echo.o`;
  - локальная копия `aiecc_orphan_place.exe`, где `--orphan-handling=error` заменён на `--orphan-handling=place`;
  - `/Zc:__cplusplus` для MSVC, иначе XRT headers требуют `boost/any.hpp`.
- `echo_test.cpp`:
  - include paths под этот XRT SDK: `xrt/experimental/xrt_xclbin.h`, `xrt/experimental/xrt_ext.h`;
  - явный `run.start()` перед `run.wait()`.

### Результат

```text
xclbin + insts compiled
echo_test.exe compiled
[echo_test] after start
[echo_test] after wait state=4
Kernel completed
Output:    0x3F00 0x3F80 0x3FC0 0x4000 0x4020 0x4040 0x4060 0x4080 ...
Result: 256/256 match
PASS: echo v1
```

### v2 dual input FIFO test

Первый запуск `run_echo_custom.bat 2` дошёл до NPU, но вернул `state=8`. Это не было доказательством dual-FIFO bug: test design был некорректен.

Проблемы v2 до фикса:

- MLIR генерировался с `N=128`, а `echo_test.cpp` использовал `N=256`;
- `echo_concat_bf16` пишет `out[0..2*N-1]`, но `out_fifo` был `memref<N>`;
- drain пытался слить `2*N` из FIFO элемента размера `N`.

Фикс в IRON-windows working tree:

- `echo_test.cpp`: `N = (version == 1) ? 256 : 128`;
- `echo_test.cpp`: v2 тоже вызывает `run.start()` перед `run.wait()`;
- `design.py`: `L1_full = np.ndarray[(2 * N,), dtype_in]`, `out_fifo = ObjectFifo(L1_full, ...)`, `Kernel(... [L1_ty, L1_ty, L1_full, np.int32])`.

Результат после фикса:

```text
xclbin + insts compiled
echo_test.exe compiled
echo_test v2 ... N=128
insts: 420 bytes
[echo_test] after v2 start
[echo_test] after v2 wait state=4
Kernel completed
Result: A=128/128 B=128/128
PASS: echo v2
```

Вывод:

- dual input ObjectFIFO + output FIFO работает;
- два `host_only` input BO + output BO работают;
- concat `A+B -> out[0:N], out[N:2N]` работает;
- текущий FlowKV bug не объясняется общим dual-FIFO/XRT/host_only failure.

Следующий минимальный тест должен быть `echo_v3`, ближе к FlowKV:

```text
bo_k -> K_fifo -> tile0
bo_v -> V_fifo -> tile1
tile0 -> inter_fifo -> tile1
tile1 -> out_fifo -> bo_out
```

Цель v3: проверить inter-tile FIFO + раздельные K/V DMA + task ordering без attention математики.

### v3 FlowKV-like inter FIFO test

`run_echo_custom.bat 3` добавлен и проходит. Топология:

```text
bo_k -> K_fifo -> tile0
bo_v -> V_fifo -> tile1
tile0 -> inter_fifo -> tile1
tile1 -> out_fifo -> bo_out
```

Реализация:

- `echo_score_bf16(k, inter, N)` на score tile копирует K в inter FIFO;
- `echo_value_bf16(inter, v, out, N)` на value tile пишет `out[0:N]=K`, `out[N:2N]=V`;
- host test version 3 проверяет все `K=128/128` и `V=128/128` элементы;
- `run_echo_custom.bat all` теперь последовательно прогоняет v1/v2/v3.

Результат:

```text
run_echo_custom.bat 3
[echo_test] after v3 wait state=4
Kernel completed
Result: K=128/128 V=128/128
PASS: echo v3

run_echo_custom.bat all
PASS: echo v1
PASS: echo v2
PASS: echo v3
```

Вывод из v3:

- two-worker / two-tile topology работает;
- inter-tile ObjectFIFO (`tile0 -> tile1`) работает;
- separate `bo_k` + `bo_v` DMA в FlowKV-like схеме работает;
- output FIFO `tile1 -> bo_out` работает;
- FlowKV garbage теперь не объясняется общим inter FIFO, split K/V BO, host_only BO или custom XRT dispatch failure.

Следующий минимальный тест должен быть `echo_v4`, ещё ближе к FlowKV arg/TAP layout: добавить `bo_q`, порядок args как FlowKV, head/seq-like TAP offsets, но оставить простую marker/copy математику.

### v4 FlowKV-like Q/K/V arg + TAP test

`run_echo_custom.bat 4` добавлен и проходит. Топология ближе к FlowKV, чем v3:

```text
rt.sequence(K, V, Q, O)
arg3=K, arg4=V, arg5=Q, arg6=O
Q_fifo + K_fifo -> score tile
score tile -> inter_fifo -> value tile
V_fifo -> value tile
value tile -> O_fifo -> O
```

Реализация:

- `echo_score_qk_bf16(q, k, inter, N)` пишет в inter FIFO `[K, Q]`;
- `echo_value_qkv_bf16(inter, v, out, N)` пишет в output `[K, Q, V]`;
- `echo_v4()` использует `rt.sequence(L3_K_ty, L3_V_ty, L3_Q_ty, L3_O_ty) as (K, V, Q, O)` как FlowKV;
- host test version 4 ставит `run.set_arg(3, bo_k)`, `set_arg(4, bo_v)`, `set_arg(5, bo_q)`, `set_arg(6, bo_out)`;
- V TAP читает с offset `N` внутри `bo_v`, чтобы проверить offset-region read, не просто offset 0;
- host проверяет все `K=64/64`, `Q=64/64`, `V=64/64` элементы.

Результат:

```text
run_echo_custom.bat 4
[echo_test] after v4 wait state=4
Kernel completed
Result: K=64/64 Q=64/64 V=64/64
PASS: echo v4
```

Вывод из v4:

- FlowKV-like arg order `K,V,Q,O` работает в custom XRT dispatch;
- `arg3=K,arg4=V,arg5=Q,arg6=O` не ломает descriptor binding в минимальном графе;
- Q+K на score tile + inter FIFO + V на value tile работает;
- простой V TAP offset работает;
- FlowKV garbage теперь не объясняется общей ошибкой arg order, Q/K/V BO binding или простым V-region TAP offset.

Остаётся искать в настоящем FlowKV-specific деталях: реальные head/group размеры и stride, multi-chunk sequencing, packed inter buffer layout, softmax/RoPE kernels, O layout и descriptor binding в большом графе.

### v5 FlowKV-like multi-chunk sequencing test

`run_echo_custom.bat 5` добавлен и проходит. Он проверяет следующий слой после v4: multi-chunk sequencing без attention math.

Топология:

```text
rt.sequence(K, V, Q, O)
Q_fifo -> score tile, Q удерживается через оба chunk
for chunk in [0, 1]:
    K_chunk -> score tile
    score tile -> inter_fifo -> value tile
    V_chunk -> value tile
value tile -> O_fifo -> O
```

Реализация:

- `echo_score_qk_bf16(q, k, inter, N)` используется на каждом K chunk и пишет `[K_chunk, Q]` в inter FIFO;
- `echo_value_chunk0_bf16` и `echo_value_chunk1_bf16` явно разворачивают value-side chunk calls, чтобы не зависеть от `if chunk == 0` внутри `range_()`;
- host version 5 проверяет отдельные output sections: `K0`, `K1`, `Q`, `V0`, `V1`;
- во время разработки v5 был найден важный IRON/MLIR nuance: переменная `chunk` из `range_()` имеет тип `index`, поэтому её нельзя напрямую передавать в kernel argument `i32`; также branch `if chunk == 0` внутри `range_()` не дал ожидаемого Python-unrolled поведения. Финальная версия использует fixed chunk0/chunk1 kernels.

Результат:

```text
run_echo_custom.bat 5
[echo_test] after v5 wait state=4
Kernel completed
Result: K0=32/32 K1=32/32 Q=32/32 V0=32/32 V1=32/32
PASS: echo v5
```

Вывод из v5:

- multi-chunk Q-held score path работает;
- repeated K acquire/release работает;
- repeated inter FIFO acquire/release между score/value tile работает;
- repeated V acquire/release работает;
- value-side final O write после двух chunks работает;
- FlowKV garbage теперь не объясняется простым multi-chunk sequencing failure.

Остаётся искать в настоящем FlowKV-specific деталях: score/value math kernels, RoPE/softmax state, реальные head/group/chunk strides, O layout и descriptor binding в большом графе.

### v6 FlowKV packed inter layout test

`run_echo_custom.bat 6` добавлен и проходит. Он проверяет следующий слой после v5: настоящий packed inter contract между score/value tile без attention math.

Топология:

```text
rt.sequence(K, V, Q, O)
Q_fifo -> score tile, Q удерживается через оба chunk
for chunk in [0, 1]:
    K_chunk -> score tile
    score tile writes packed inter [F_c | C_c | l]
    packed inter -> value tile
    V_chunk -> value tile
value tile writes [F_c | C_c | l | V_chunk] blocks -> O_fifo -> O
```

Параметры diagnostic:

```text
chunk_size = 16
group_size = 4
head_dim = 64
num_chunks = 2
scores_size = chunk_size * group_size = 64
packed_inter_size = scores_size + 2 * group_size = 72
v_chunk_size = chunk_size * head_dim = 1024
out_block_size = packed_inter_size + v_chunk_size = 1096
```

Реализация:

- `echo_pack_inter_v6_bf16(k, packed_out, chunk_size, group_size, head_dim)` пишет packed layout `[F_c scores | C_c correction | l denom]`;
- `F_c` — marker copy из первых `chunk_size * group_size` элементов K chunk;
- `C_c` и `l` — bf16 `1.0` по `group_size` элементов;
- `echo_value_v6_chunk0_bf16` и `echo_value_v6_chunk1_bf16` явно разворачивают value-side chunk calls;
- `echo_value_v6_bf16` копирует packed inter и затем append'ит V chunk в output block;
- host version 6 проверяет отдельные sections: `F0/C0/L0/V0/F1/C1/L1/V1`.

Важная деталь v6:

- `packed_inter_size = 72` не кратен vector width 16;
- первая версия value copy с `aie::vector<bfloat16, 16>` давала mismatch в `C0/L0/V0`;
- фикс: копировать packed inter и V append через `aie::vector<bfloat16, 8>`, т.к. `72` и `1024` кратны 8.

Результат:

```text
run_echo_custom.bat 6
[echo_test] after v6 wait state=4
Kernel completed
Result: F0=64/64 C0=4/4 L0=4/4 V0=1024/1024 F1=64/64 C1=4/4 L1=4/4 V1=1024/1024
PASS: echo v6
```

Вывод из v6:

- packed inter FIFO layout `[F_c | C_c | l]` работает в минимальном FlowKV-like графе;
- score tile -> packed inter FIFO -> value tile работает для packed size 72 bf16;
- value tile output block `[F_c | C_c | l | V_chunk]` работает;
- FlowKV garbage теперь не объясняется самим packed inter layout или его FIFO/DMA sequencing.

Остаётся искать в настоящем FlowKV-specific деталях: value accumulation/normalization math, реальные production Q/K/V данные, реальные head/group/chunk strides, O layout и descriptor binding в большом графе.

### v7 FlowKV score math packed inter test

`run_echo_custom.bat 7` добавлен и проходит. Он проверяет следующий слой после v6: реальную score-side online-softmax математику, которая создаёт packed inter `[F_c | C_c | l]`, но без value accumulation/normalization.

Топология:

```text
rt.sequence(K, Q, O)
Q_fifo -> score tile, Q удерживается через оба chunk
score_init()
score_rope_q(Q)
for chunk in [0, 1]:
    K_chunk -> score tile
    score_chunk(Q, K_chunk) writes packed inter [F_c | C_c | l]
    packed inter -> drain worker
 drain worker writes packed chunk blocks -> O_fifo -> O
```

Параметры diagnostic:

```text
chunk_size = 16
group_size = 4
head_dim = 64
num_chunks = 2
scores_size = chunk_size * group_size = 64
packed_inter_size = scores_size + 2 * group_size = 72
q_stride = group_size * head_dim + head_dim + 2 = 322
output = num_chunks * packed_inter_size = 144 bf16
```

Реализация:

- `echo_v7_score_init_bf16` сбрасывает running max/sum как FlowKV score tile;
- `echo_v7_score_rope_q_bf16` копирует Q в static `rotated_q` и читает `actual_seq_len` из Q metadata slot;
- `echo_v7_score_chunk_bf16` повторяет FlowKV score-side dot product + online softmax + packed `[F_c | C_c | l]`;
- `echo_v7_copy_pack_chunk0_bf16` и `echo_v7_copy_pack_chunk1_bf16` сливают packed inter chunks в host-visible output;
- host version 7 использует симметричный Q/K pattern: Q one-hot по dim0, каждый K row one-hot по dim0, `actual_seq_len=32`.

Ожидаемая математика для такого pattern:

```text
raw dot = 1.0
scaled score = 1/sqrt(64) = 0.125
chunk 0: F=1, C=0, l=16
chunk 1: F=1, C=1, l=32
```

Результат:

```text
run_echo_custom.bat 7
[echo_test] after v7 wait state=4
Kernel completed
Result: F0=64/64 C0=4/4 L0=4/4 F1=64/64 C1=4/4 L1=4/4
Samples: F0=1.000000 C0=0.000000 L0=16.000000 F1=1.000000 C1=1.000000 L1=32.000000
PASS: echo v7
```

Вывод из v7:

- FlowKV score-side Q copy / actual_seq_len read работает в минимальном графе;
- dot product `Q*K/sqrt(64)` работает для контролируемого Q/K;
- online softmax state across chunks работает (`l: 16 -> 32`, `C: 0 -> 1`);
- packed score output `[F_c | C_c | l]` численно корректен для симметричного pattern;
- FlowKV garbage теперь не объясняется самой score-side math или packed score output на простом controlled input.

Остаётся искать в настоящем FlowKV-specific деталях: value accumulation/normalization math, реальные production Q/K/V данные, реальные head/group/chunk strides, O layout и descriptor binding в большом графе.



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
- `bd1f27af9` (llama.cpp-xdna ggml-xdna): fix(flowkv): swap K/V data — **не помог** (V DMA тоже читает arg1)
- `855fa12b1` (llama.cpp-xdna ggml-xdna): fix: update marker test for data swap
- `0b8e622` (IRON-windows devel): **fix(flowkv): combine K+V in single buffer (arg1)** ← V TAP offset
- `cbfbc9009` (llama.cpp-xdna ggml-xdna): **fix(flowkv): combine K+V in bo_v (arg1)** ← host-side

---

## Сессия 2026-05-16 (продолжение 2): Root cause — BOTH DMAs read arg1

### Открытие
При data-swap тесте: NPU output head0 = `-0.466797` для ВСЕХ 8 значений = `0xBEEF` = V marker.
Это значит **V DMA тоже читает из arg1 (bo_v)**, а не из arg0 (bo_k).

**IRON compiler bug: и K_fifos, и V_fifos DMA маппятся на один и тот же DDR arg (arg1).**

### Marker test подтверждения
1. **Оригинал** (K in bo_k@arg0, V in bo_v@arg1): K_DIAG = 0xBEEF (V marker) → K DMA читает arg1
2. **Data-swap** (K in bo_v@arg1, V in bo_k@arg0): K_DIAG = 0xBEEF (K marker в bo_v) → K DMA читает arg1 ✓
3. NPU output = 0xBEEF constant → V DMA тоже читает arg1 = bo_v = K data ✗

### Фикс: объединить K+V в одном буфере
Раз оба DMA читают arg1, кладём K и V в один буфер (bo_v):
- K data: offset 0 в bo_v
- V data: offset kv_region_size в bo_v
- V TAP: tensor_dims=(2*kv_region), offset=kv_region_size
- bo_v size: 2 × num_kv_heads × seq_len × head_dim × sizeof(bf16)

### Commits фикса
- IRON-windows `0b8e622`: V TAP offset + kv_region_size, tensor_dims doubled
- llama.cpp-xdna `cbfbc9009`: K+V data written to bo_v, bo_v size doubled

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

### Анализ TAP offset → DMA BD (2026-05-16)

**Вопрос:** Генерирует ли IRON compiler DMA BD offset из TAP offset для ObjectFifo fill?

**Ответ: Да, путь полностью трассирован.**

```
fill(in_fifo, source, tap)
  → DMATask.resolve()
    → shim_dma_single_bd_task(alloc, mem, tap=tap)
      → offset = int(tap.offset)         ← напрямую из TAP
      → shim_dma_bd(mem, offset=offset, sizes, strides)
        → dma_bd(mem, offset=offset, ...)
          → DMABDOp в MLIR
```

**Единицы:** offset, len, sizes, strides — всё в element width (не байтах). Подтверждено документацией AIE MLIR:
> "offset, len, sizes and strides are all denominated in element width"

**Итого для K+V combined подхода:**
- `make_k_tap(0)`: offset=0 → DMA BD offset=0 на arg1 (bo_v) → K[0] data ✓
- `make_v_tap(0)`: offset=kv_region_size → DMA BD offset=131072 на arg1 (bo_v) → V[0] data ✓
- `tensor_dims` в TAP НЕ используется при генерации DMA BD (только offset/sizes/strides)

**Статус:** Теоретически подтверждено. Ожидает тестирования на hardware.

### Анализ логов (загруженные файлы)

- `step3_baseline_output.log`: "The capital of France is Paris." — baseline OK (build b8900-795aa56e4)
- `step1_fix_output.log`: мусор — FlowKV + arg-swap fix (тот же build)
- `step2_diag_output.log`: мусор — FlowKV + XDNA_DIAG_OFFSET=1
- `step2_diag_offset.log`: K_DIAG = all zeros (cacheable flag), marker test показал 0xDEAD/0xBEEF markers записаны, но DMA читает 0x0000 → cacheable ухудшает ситуацию

Все три теста — ДО коммитов K+V combined (cbfbc9009, 0b8e622). Combined подход ещё не тестировался.

### Критический баг: L3_V_ty out-of-bounds (2026-05-16)

**Найден при генерации MLIR из design.py.**

До фикса:
```
aie.runtime_sequence(%arg0: memref<131072xbf16>, %arg1: memref<131072xbf16>, ...)
aie.dma_bd(%arg1 : memref<131072xbf16>, 131072, 16384, ...)  ← offset=131072 OUT OF BOUNDS
```

`L3_V_ty` была `num_kv_heads * seq_len * head_dim = 131072`, но V DMA BD offset = `kv_region_size = 131072` — за пределами буфера. На hardware это undefined behavior.

После фикса (`5a86030` IRON-windows devel):
```
aie.runtime_sequence(%arg0: memref<131072xbf16>, %arg1: memref<262144xbf16>, ...)
aie.dma_bd(%arg1 : memref<262144xbf16>, 131072, 16384, ...)  ← OK, within bounds
```

**Это могла быть причина мусора в предыдущих тестах** — DMA читала за пределами буфера.

### DMA echo тест-дизайн (2026-05-16)

Цель: локализовать проблему — DMA path или multi-ObjectFifo sequence.

- **v1**: одна ObjectFifo, один fill/drain. Если не работает → DMA path сломан.
- **v2**: две ObjectFifos, два fill в task_group. Если v1 работает, а v2 нет → multi-fill sequence проблема.

Файлы: `IRON-windows/iron/operators/dma_echo/design.py`, `aie_kernels/aie2p/echo.cc`
Коммит: `faa6c8a` (IRON-windows devel)

### DMA echo инфраструктура (2026-05-17)

Сборка и тест echo через IRON фреймворк. aiecc.exe зависает (PyInstaller), используем aiecc.py.

**Проблемы при создании:**
1. `aiecc.exe --help` зависает → используем `aiecc.py` (IRON ищет .py первым)
2. `import iron` тянет `pyxrt` через цепочку → нужен XRT SDK на PYTHONPATH
3. `iron_repo` конфликт с `IRON-windows` → фильтрация из sys.path
4. `opt.exe`/`llc.exe` не в win64.o Peano → лежат в `C:\Python313\Lib\site-packages\llvm-aie\bin`
5. XDNA NPU не поддерживает `device.load_xclbin()` → используем ELF path (insts.bin → pyxrt.elf → hw_context)

**Файлы:**
- `compile_echo.py` — сборка через IRON compilation framework (aiecc.py)
- `test_echo.py` — запуск на NPU, проверка DMA (ELF path)
- `run_echo.bat` — обёртка с правильным окружением (как debug_flowkv.bat)
- `__init__.py` — package marker

**Ключевые решения:**
- Peano (clang): `win64.o/tools/peano` (AIE-targeted)
- LLVM-AIE (opt/llc): `C:\Python313\Lib\site-packages\llvm-aie` (для aiecc.py)
- Python: `C:\Python313\python.exe` + PYTHONPATH с XRT SDK (не conda Python)
- NPU runtime: ELF path через `pyxrt.elf` + `pyxrt.hw_context` + `pyxrt.ext.kernel`

**Коммиты (IRON-windows devel):**
- `8804c5c` feat: build/test infrastructure
- `3ae2568..d8813f8` fix: environment, paths, unicodeescape
- `6694b23` fix: llvm-aie for aiecc opt/llc
- `7bb31fb` fix: ELF path for NPU

**Следующий шаг:**
Запустить `run_echo.bat 1` на Windows. Если v1 проходит → DMA path работает, тестировать v2.

**Обновление (02:07):** pyxrt ELF/xclbin ручной loading не работает на XDNA NPU.
Перешли на IRON test framework: op.py (AIEEchoV1/V2) + test.py (pytest + run_test).
run_echo.bat теперь вызывает pytest.

---

## Echo Test Debug Session (2026-05-17 03:38–04:11 GMT+8)

### Проблема: Environment hell при запуске run_echo.bat

**Цепочка ошибок и фиксов:**

1. **numpy cp313/cp312 conflict** — PYTHONPATH содержал `C:\Python313\Lib\site-packages` целиком, conda Python 3.12 загружал numpy скомпилированный под 3.13.
   - Фикс: `PYTHONPATH` → только `C:\Python313\Lib\site-packages\llvm-aie`

2. **pyxrt cp313** — `pyxrt.pyd` скомпилирован под Python 3.13, conda env на 3.12. `ctypes.WinDLL` загружает, `import pyxrt` — нет (ABI mismatch).
   - Фикс: `CONDA_PYTHON` → `C:\Python313\python.exe`

3. **pytest from conda** — Python 3.13 загружал conda-прежний pytest (3.12), `allow_abbrev` удалён в 3.13.
   - Фикс: убрать conda site-packages из PYTHONPATH. pytest уже в Python 3.13.

4. **mlir_aie/_mlir C extension cp312** — conda mlir_aie содержит C extensions для 3.13, но Python 3.13 не может их загрузить.
   - Фикс: `mlir_aie` уже установлен в Python 3.13 (`pip install mlir_aie`). Conda-путь не нужен.

5. **conftest nodeid regex** — regex `r"^(.+?)::(.+?)\[(iter\d+-)?(.+?)\]$"` не матчит `test_echo_v1` (без `[iter0-...]` при `--iterations 1`).
   - Фикс: `c41dee0` — regex.optional `[...]` + fallback на group(2).

6. **AIEEchoV1 get_callable() not implemented** — `AIEEchoV1/V2` наследуют `AIEOperatorBase`, а `run_test()` требует `get_callable()` и `get_arg_spec()`.
   - Фикс: `06b0b8b` — реализованы `get_callable()` (через NPUKernel + DefaultNPURuntime) и `get_arg_spec()`.

### Итоговая конфигурация run_echo.bat

```
CONDA_PYTHON=C:\Python313\python.exe
PYTHONPATH=XRT_SDK/python;llvm-aie   # НЕ conda site-packages!
```

Python 3.13 site-packages содержит: mlir_aie, numpy, pytest, torch — всё cp313.

### Коммиты (IRON-windows devel)

- `46c0285` fix(dma_echo): restrict PYTHONPATH to llvm-aie only
- `413e829` fix(dma_echo): use Python 3.13 for pyxrt cp313
- `92f1bdf` fix(dma_echo): add only mlir_aie/python to PYTHONPATH
- `7099548` fix(dma_echo): remove conda mlir_aie from PYTHONPATH
- `c41dee0` fix: conftest nodeid regex optional for --iterations 1
- `06b0b8b` feat(dma_echo): implement get_callable() and get_arg_spec()

7. **pyxrt.bo.cacheable crash on XDNA** — `CachedXRTRuntime` (DefaultNPURuntime) создаёт insts buffer с `pyxrt.bo.cacheable`, что вызывает access violation на XDNA driver. `pyxrt.bo.host_only` работает.
   - Доказано: `pyxrt.bo(d, 300, pyxrt.bo.cacheable, 4)` → crash. `pyxrt.bo(d, 300, pyxrt.bo.host_only, 0)` → OK.
   - Фикс: `91d692e` — monkey-patch `pyxrt.bo.cacheable = pyxrt.bo.host_only` в conftest.py

- `91d692e` fix: monkey-patch pyxrt.bo.cacheable to host_only for XDNA compatibility

### Статус

Echo test v1 проходит environment hell и crash fix. Ожидает тестирования на hardware (NPU execution + DMA verify).

---

## Сессия 2026-05-17: Debug pytest + aiecc.exe deadlock

### Проблема

Pytest (dequant, axpy тесты) зависал бесконечно без вывода. При прерывании (Ctrl+C) создавались сотни зомби-процессов python/aiecc (до 2248 штук).

### Корневая причина

`aiecc.exe` — PyInstaller-бинарник. При запуске через `subprocess.run()` из Python на Windows без `stdin=subprocess.DEVNULL` PyInstaller deadlock'ится (зависает на чтении stdin).

Параллельные экземпляры aiecc.exe (PyInstaller) конфликтуют — каждый пытается извлечь временные файлы в один temp-каталог.

### Фиксы (IRON-windows devel)

1. **`9f242b0`** — убран `capture_output=True` из `ShellCompilationCommand.run()` для видимости вывода aiecc
2. **`162642a`** — добавлены `stdin=subprocess.DEVNULL` + `creationflags=CREATE_NO_WINDOW` (Windows) для обхода PyInstaller deadlock

### Дополнительные находки

- `aiecc.py` — тонкая обёртка, вызывает `aiecc.exe` через subprocess. Переименование exe не помогает.
- `aiecc.exe --help` работает нормально (PyInstaller deadlock только при запуске из Python subprocess).
- `capture_output=True` стоял в 3 местах base.py: строка ~567 (ShellCompilationCommand), строка ~787 (objcopy strip), строка ~1147 (nm symbol map).
- Флаги `--generate-full-elf` vs `--aie-generate-xclbin` + `--aie-generate-npu-insts` — два разных пути компиляции в framework.
- `PYTHONPATH` должен содержать путь с `xrt_sdk` (не `xrt`): `C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python`

### Как избежать зомби-процессов

После прерывания pytest всегда убивать остатки:
```powershell
taskkill /F /IM aiecc.exe 2>$null; taskkill /F /IM python.exe 2>$null
```

### Ожидает тестирования

- Запустить axpy/dequant тесты с фиксом `162642a` на Windows
- Подтвердить что aiecc не deadlock'ится и тесты проходят

---

## Сессия 2026-05-17: Настоящая причина aiecc deadlock

### Диагностика

Добавлены `[AIECC-RESOLVE]` и `[AIECC-RUN]` prints в `_resolve_aiecc` и `ShellCompilationCommand.run()` (коммит `5550f1b`).

### Результат

```
[AIECC-RESOLVE] Using: C:\Python313\Lib\site-packages\mlir_aie\bin\aiecc.py (size=347)
[AIECC-RUN] command=['C:\\Python313\\python.exe', 'C:\\Python313\\Lib\\site-packages\\mlir_aie\\bin\\aiecc.py', '-v', '-j1', ...]
```

`_resolve_aiecc` correctly returns `aiecc.py`. Фикс `162642a` correctly adds `stdin=DEVNULL` к этому вызову. **Но тест всё равно deadlock'ится** — 736 зомби aiecc.exe.

### Настоящая причина

`aiecc.py` (347 байт) — **НЕ** in-process wrapper:

```python
import subprocess, shutil
def _find_aiecc_binary():
    return shutil.which("aiecc")  # → aiecc.exe

def main():
    aiecc_bin = _find_aiecc_binary()
    result = subprocess.run([aiecc_bin, *sys.argv[1:]])  # БЕЗ stdin=DEVNULL!
```

Цепочка deadlock'а:
```
base.py → subprocess.run(python.exe, aiecc.py, stdin=DEVNULL ✓)
              → aiecc.py → subprocess.run(aiecc.exe, stdin=PIPE ✗ ← DEADLOCK)
```

Фикс `162642a` добавил `stdin=DEVNULL` к внешнему вызову, но `aiecc.py` **внутри** запускает `aiecc.exe` через `subprocess.run()` **без** `stdin=DEVNULL`. PyInstaller binary deadlock'ится на чтении stdin.

### Фикс

На Windows вызывать `aiecc.exe` напрямую, минуя `aiecc.py`. Изменён `_resolve_aiecc()`:
- **Windows**: `aiecc.exe` → `aiecc` → `aiecc.py` (exe first)
- **Linux/macOS**: `aiecc.py` → `aiecc` (unchanged)

`ShellCompilationCommand.run()` уже применяет `stdin=DEVNULL` → `aiecc.exe` получает DEVNULL напрямую.

### Коммиты

- `5550f1b` (IRON-windows devel): diag: add logging for aiecc resolve and subprocess invocation
- `1f5f7c5` (IRON-windows devel): fix(aiecc): call aiecc.exe directly on Windows, skip aiecc.py wrapper

### Обновление finding #12 в SESSION.md

`aiecc.py` НЕ in-process wrapper. Он вызывает `shutil.which("aiecc")` → `aiecc.exe` через `subprocess.run()` без `stdin=DEVNULL`. SESSION.md содержал ошибку: ~~"aiecc.py — тонкая обёртка, вызывает aiecc.exe через subprocess"~~ — верно, но это **именно и есть** причина deadlock'а, а не что stdin наследуется.

### Следующий шаг

- `git pull` на Windows, запустить axpy тест
- Подтвердить отсутствие зомби-процессов и успешное выполнение
- **Отозвать GitHub PAT** `github_pat_11ASJ3NRA...` — был использован для push

---

## Сессия 2026-05-17: axpy тест — aiecc deadlock fix + NPU timeout debug

### aiecc deadlock — подтверждён и починен

Фикс `1f5f7c5` работает: `aiecc.exe` вызывается напрямую, zero zombie processes, компиляция завершена за ~30 секунд.

### NPU kernel timeout — root cause: insts_bo не синхронизирован

После фикса aiecc, axpy тест падает с `ERT_CMD_STATE_TIMEOUT` (NPU kernel hang).

**Root cause chain:**
1. `pyxrt.bo.cacheable` крашит на XDNA → monkey-patch на `host_only` (conftest.py)
2. Hostruntime создаёт `insts_bo` с `host_only` (через monkey-patch)
3. `host_only` буфер = только host memory, NPU **не видит** без explicit sync
4. Hostruntime **не вызывает** `sync_bo(TO_DEVICE)` — предполагает `cacheable` = coherent
5. NPU читает stale/zero data → kernel hang → `ERT_CMD_STATE_TIMEOUT`

**Фикс:** monkey-patch `XRTHostRuntime.run` — добавить `insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE)` после создания/получения insts_bo.

### Отладка monkey-patch

1. `_hostrt.XRTRuntime.run` — **не существует** (`AttributeError`). Правильное имя: `_hostrt.XRTHostRuntime.run`
2. Sync стоял только в `else` ветке (fresh creation), но `kernel_handle.insts_bo` был закэширован → sync пропускался. Фикс: sync **всегда** после if/else.

### Коммиты (IRON-windows devel)

- `1f5f7c5` fix(aiecc): call aiecc.exe directly on Windows
- `5048d27` fix: sync insts_bo always, not only on first creation
- `8f56621` fix: use XRTHostRuntime instead of XRTRuntime
- `74fe1c4` diag: add prints to verify conftest monkey-patch

### Следующий шаг

- Запустить axpy тест с фиксом `5048d27`
- Ожидание: `[CONFTEST] insts_bo synced to device (420 bytes)` → `PASSED`
- Если всё ещё timeout — проблема глубже (xclbin format, DMA config, NPU driver)

---

## Сессия 2026-05-17: insts_bo sync — работает, но timeout остаётся

### Результат теста (sync3)

```
[CONFTEST] pyxrt.bo.cacheable patched to host_only
[CONFTEST] XRTRuntime.run patched with insts_bo sync
[CONFTEST] insts_bo synced to device (420 bytes)  ← sync работает!
FAILED - Kernel returned ert_cmd_state.ERT_CMD_STATE_TIMEOUT
```

Sync **применяется** (420 bytes → device), но kernel всё ещё timeout. Проблема **не в sync**.

### Анализ

- Data buffers (x, y, z) sync'ятся через `XRTTensor.to("npu")` → `_sync_to_device()` ✓
- insts_bo sync'ится через наш monkey-patch ✓
- `aiecc.exe` генерирует xclbin (9239 bytes) + insts.bin (420 bytes) — размеры выглядят нормально
- `xclbinutil --info` показывает: UUID, Sections (MEM_TOPOLOGY, AIE_PARTITION, etc.) — структура OK

### Возможные причины timeout

1. **xclbin format mismatch** — `aiecc.exe` (native C++) vs `aiecc.py` (Python wrapper) могут генерировать xclbin по-разному
2. **NPU driver state** — NPU может быть в bad state после FlowKV экспериментов
3. **DMA configuration** — ObjectFifo DMA config в xclbin может быть неправильной для данного NPU
4. **MLIR-AIE / XRT version mismatch** — mlir_aie версия может быть несовместима с XRT 2.21.0

### Следующий шаг

- Проверить, работал ли **какой-нибудь** AIE/NPU тест на этой машине
- Попробовать mlir_aie example напрямую (без IRON)
- Проверить mlir_aie версию: `python -c "import aie; print(aie.__version__)"`
- Если mlir_aie examples тоже timeout → проблема в NPU driver/xclbin, не в IRON

---

## 2026-05-17: FlowKV мусор — корневой баг найден и исправлен

### Тест debug_flowkv.bat (STEP 2 vs STEP 3)

| Test | Output | Status |
|------|--------|--------|
| STEP 2 (без FlowKV) | "The capital of France is Paris." | ✅ |
| STEP 3 (с FlowKV) | "The meansें whbed Bretdio欠欠-steachoeditionagma dio bet exclude" | ❌ |

### Корневой баг: V DMA out-of-bounds

**Проблема в design.py (IRON-windows):**
```python
# L3_V_ty = 16384 элементов, но V TAP читает со смещения 16384 → OOB!
L3_V_ty = np.ndarray[(num_kv_heads * seq_len * head_dim,), dtype_in]  # = 16384

def make_v_tap(kv_head_idx):
    kv_region_size = num_kv_heads * seq_len * head_dim  # = 16384
    base = kv_region_size + ...  # offset = 16384 в буфере 16384 → OOB!
    tensor_dims=(2 * num_kv_heads * seq_len * head_dim,)  # ожидает 32768
```

MLIR output:
```mlir
aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: memref<16384xbf16>, ...)
aie.dma_bd(%arg1 : memref<16384xbf16>, 16384, 16384, ...) ← OOB!
```

**Проблема в ggml-xdna.cpp (per-head dispatch):**
Код делал SWAP (K→bo_v, V→bo_k), но IRON compiler bug: оба DMA читают из arg1 (bo_v).
V DMA читал из bo_v → получал K data вместо V data.

### Фиксы

1. **IRON-windows design.py**: `L3_V_ty = np.ndarray[(2 * num_kv_heads * seq_len * head_dim,)]`
   → MLIR: `%arg1: memref<32768xbf16>`, V DMA offset 16384 within bounds ✓

2. **llama.cpp-xdna ggml-xdna.cpp per-head path**:
   - `bo_v` size: `2 * seq_len * head_dim * sizeof(uint16_t)` (combined K+V)
   - V data → `bo_v` at offset `kv_region_size` (не bo_k)

3. **POC path** уже имел правильный layout (K+V combined в bo_v) — не тронут.

### Статус: фиксы закоммичены, ожидают тестирования на hardware


---

## 2026-05-17: Анализ логов после фиксов + Echo Test через custom dispatch

### Результат теста (build b8914-b38e4a68d)

| Test | Output | Status |
|------|--------|--------|
| STEP 2 (без FlowKV) | "The capital of France is Paris." | ✅ |
| STEP 3 (с FlowKV) | "The sometimes Turingagmaaclesagmaagma..." | ❌ другой мусор |

### Анализ xdna лога (generation phase)

**FlowKV-early dispatch'ится** (H=4 KV=8 d=64 S=256), но **POC path produce all zeros:**
```
bo_out raw: first 8 bf16: 0x0000 0x0000 ... all_zero=1
```

**K data в bo_v = нули**, хотя source K data не нулевой:
```
BO check kv_h=0: K[0] first 8 bf16: 0x0000 ... ← DMA видит нули
K src raw [pos=0,kv_h=0] first 8 bf16: 0x1CF7 0xA33A ... ← данные есть
```

**BO address diagnostic:**
```
bo_k group_id: 1114112
bo_v group_id: 1114112   ← одинаковые!
V-K delta: 36093952 bytes (35248 KB) вместо ожидаемых 32768 (32 KB)
*** UNEXPECTED DELTA — driver may be adding metadata! ***
```

**Marker test не запущен** (XDNA_DIAG_OFFSET не задан в bat-файле).

### Ключевой вывод

DMA читает из ELSEWHERE — не из bo_k, не из bo_v. Все предыдущие фиксы (L3_V_ty,
combined bo_v, POC offset) не решают фундаментальную проблему: **XRT host_only buffer
address не совпадает с тем, что видит DMA controller.**

### Что уже пробовали (и не помогло)

- ❌ Arg swap (a7f0bef → 7999a4f)
- ❌ swap bo_k/bo_v в set_arg (1920fd641)
- ❌ swap K/V data (bd1f27af9)
- ❌ combine K+V в bo_v (cbfbc9009 + 0b8e622)
- ❌ L3_V_ty doubling (b1c9911)
- ❌ cacheable flag → DMA видит нули
- ❌ svm/p2p → unsupported

### Что НЕ работает: IRON framework tests

- axpy тест → ERT_CMD_STATE_TIMEOUT (insts_bo sync не помогает)
- echo test через IRON framework → тоже будет timeout (тот же CachedXRTRuntime path)
- IRON framework + XDNA = broken path

### Что РАБОТАЕТ: custom XRT dispatch

- STEP 2 (QKV, GEMV, SwiGLU через custom dispatch в ggml-xdna.cpp) → "The capital of France is Paris." ✅
- Custom dispatch: xrt::xclbin + xrt::kernel + xrt::run → работает на XDNA

### Следующий шаг: Echo Test через custom dispatch

Создан `echo_test.cpp` — standalone XRT dispatch, bypass IRON framework.
Загружает echo xclbin через xrt::xclbin, dispatch'ит через xrt::run.

**Если echo v1 PASS** → DMA path работает, проблема специфична для FlowKV
**Если echo v1 FAIL** → DMA path сломан фундаментально

Файлы: `iron/operators/dma_echo/echo_test.cpp`, `run_echo_custom.bat`
Коммит: `085cb68` (IRON-windows devel)

### Также: фикс POC path V offset

POC path писал V data на offset `num_kv_heads * seq_len * row_bytes` (262144),
но kernel (compiled for 1 KV head) читает V с offset `seq_len * row_bytes` (32768).
Фикс: `kv_region = seq_len * row_bytes`.
Коммит: `95cc4da4e` (llama.cpp-xdna ggml-xdna)

