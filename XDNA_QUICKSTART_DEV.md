# ggml-xdna — Техническое руководство (для разработчиков)

Подробная документация по сборке, настройке и отладке ggml-xdna бэкенда для AMD XDNA NPU.

> Для обычных пользователей см. [XDNA_QUICKSTART.md](./XDNA_QUICKSTART.md)

## Архитектура

```
┌─────────────────────────────────────────────────┐
│                   llama.cpp                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ ggml-cpu    │  │ ggml-xdna   │  │ ggml-blas│ │
│  │ (fallback)  │  │ (NPU)       │  │ (опц.)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┘ │
│         │                │                       │
│         └────────┬───────┘                       │
│                  │                               │
│         ┌───────▼────────┐                       │
│         │ graph_compute   │                       │
│         │ (scheduler)     │                       │
│         └───────┬────────┘                       │
└─────────────────┼───────────────────────────────┘
                  │
         ┌────────▼────────┐
         │  compile.py     │  (только при первом запуске)
         │  IRON → MLIR    │
         │  → xclbin       │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  XRT Runtime    │
         │  (xrt::kernel,  │
         │   xrt::runlist) │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  AMD XDNA NPU   │
         │  (AIE array)    │
         └─────────────────┘
```

## NPU-операции (xdna_op_kind)

| # | Операция | Когда | Описание |
|---|---------|-------|----------|
| 0 | GEMM | M≥32 | Матричное умножение для prefill |
| 1 | GEMV | M=1 | Матрица-вектор для decode |
| 2 | SWIGLU_DECODE | M=1 | Fused FFN (gate/up/down + SiLU) bf16 |
| 3 | SWIGLU_PREFILL | M≥32 | Fused FFN для prefill bf16 |
| 4 | SWIGLU_DECODE_INT8 | M=1 | Fused FFN W8A16 (int8 веса, bf16 активации) |
| 5 | SWIGLU_FUSED_INT8 | M=1 | Fused gate+up+silu+mul INT8 + down GEMV |
| 6 | SWIGLU_PREFILL_INT8 | M≥32 | W8A8 INT8 prefill |
| 7 | QKV | M=1 | Chained Q/K/V проекции (3 GEMV → 1 runlist) |
| 8 | RMS_NORM | any | Standalone RMSNorm bf16 |
| 9 | ATTENTION_PREFILL | M≥256 | 11 sub-kernels: RMSNorm+QKV+RoPE+MHA+O+residual |

Дополнительно:
- **TransformerBlockPrefill** — 17 sub-kernels (attention + FFN)
- **TransformerBlockPrefillFused** — монолитный single-ELF (1 xrt::run на слой)

## Версии, на которых это собирается и работает

Снято с рабочей машины 2026-07-26. Это не «минимальные требования» — это единственная
проверенная связка; диапазоны совместимости не исследовались.

| Компонент | Версия | Откуда |
|---|---|---|
| Ветка | `ggml-xdna` = upstream `b8746` + 379 коммитов | своих тегов/релизов нет |
| Windows | 11, 10.0.26200 | |
| NPU-драйвер | 32.0.20102.3930 (05.07.2026) | Windows Update / AMD |
| AMD XRT SDK | 2.21.0, hash `4eb1f439` (03.02.2026) | пакет AMD для разработки под NPU |
| MSVC | 19.44.35222 (BuildTools 14.44.35207) | VS 2022 Build Tools |
| CMake | 4.3.2 | pip |
| Python (сборка кернелов) | 3.13, отдельный `C:\Python313` | не conda-окружение |
| `llvm-aie` (**clang 21.0.0**, `7bc5ade6`) | 21.0.0.2026050701+7bc5ade6 | pip — **им и собираются кернелы** |
| `mlir-aie` (aiecc) | 0.0.1.2026033105+e4f35d6 | pip |
| Ryzen AI | 1.7.1, conda `ryzen-ai-1.7.1` | ⚠️ установлен, но в сборке кернелов **не участвует** (см. ниже) |
| `iron` | 0.1.0, editable → `IRON-windows` | наш форк, `pip install -e` |
| `numpy` / `ml_dtypes` | 2.4.4 / 0.5.4 | pip |

Готовые `.xclbin` в `npu_kernels_win_8col/` собраны этой связкой и проверены на этом
драйвере. Загружает их XRT + драйвер, так что заметно более старый драйвер может их
не принять — это не проверялось.

## Предварительные требования

### Ядро Linux

Минимальная версия: **6.14** (модуль `amdxdna` в mainline).

```bash
# Проверить
uname -r
lsmod | grep amdxdna

# Поддерживаемые дистрибутивы:
# - Fedora 41+ (ядро 6.14+ в репозиториях)
# - Ubuntu 25.04+
# - Для старых — сборка ядра из source с патчем amdxdna
```

### AMD XRT (Xilinx Runtime)

```bash
# Установка: https://github.com/amd/xdna-driver

# Проверка
ls /opt/xilinx/xrt/include/xrt/xrt_device.h
ls /opt/xilinx/xrt/lib/libxrt_core.so

# Настройка (добавить в ~/.bashrc)
source /opt/xilinx/xrt/setup.sh

# Проверить NPU
xrt-smi examine
```

### IRON (библиотека NPU-операторов)

Нужен **только для компиляции кернелов** (xclbin). Для запуска с готовым кешем не нужен —
собранные кернелы лежат в `npu_kernels_win_8col/` прямо в репозитории.

⚠️ Нужен **наш форк**, а не upstream. В `amd/iron` нет ~70 операторов, которые здесь
используются (`decode_layer_f3best`, `decode_ffn16_2mm`, `flowkv_decode` и остальные), —
с upstream первый же NPU-диспатч упадёт с ImportError.

```bash
git clone --branch devel https://github.com/Soket1/IRON-windows.git
cd IRON-windows
pip install -e .
pip install numpy
```

IRON-windows намеренно лежит вне этого репозитория (он в `.gitignore`, не submodule):
это отдельная история изменений поверх `amd/iron`. Клонируйте его рядом.

### Windows

```powershell
# 1. NPU Driver — через Windows Update или AMD Support
# 2. AMD XRT Windows SDK:
#    https://github.com/Xilinx/XRT/releases/download/2.21.75/xrt_windows_sdk.zip
#    Нужен ВНУТРЕННИЙ каталог ...\xrt_sdk\xrt (include\ + lib\ + xclbinutil.exe).
#    Установщик Ryzen AI 1.7.1 этого НЕ даёт (RyzenAI\xrt\ пустой);
#    в github.com/amd/xdna-driver релизов нет вообще.
# 3. Visual Studio 2022 Build Tools (C++ Desktop + CMake tools)
# 4. Python 3.10+ (для compile.py)
```

### Какой clang реально компилирует кернелы

На машине лежат два разных Peano, и это сбивает с толку:

| | clang | opt/llc |
|---|---|---|
| conda `ryzen-ai-1.7.1` → `win64.o/tools/peano` | 20.0.0git (`gitenterprise.xilinx.com`, `4dbd91a8`) | **нет** |
| pip `llvm-aie` в `C:\Python313` | **21.0.0** (`github.com/Xilinx/llvm-aie`, `7bc5ade6`) | есть |

`compile.py` патчит `aie_config.peano_install_dir` на conda-путь, а харнесс дополнительно
ставит туда же `PEANO_INSTALL_DIR` — **и то и другое не имеет эффекта.**
`IRON-windows/iron/common/context.py:_resolve_peano_dir()` принимает каталог, только если в
нём есть `opt`; conda-вариант отбрасывается на обоих шагах, и возвращается pip-каталог.
Проверено прямым вызовом: при обоих механизмах, указывающих на conda, функция возвращает
`C:\Python313\Lib\site-packages\llvm-aie`.

Дальше этот один каталог используется целиком: `clang` берётся как
`peano_dir/bin/clang.exe` (`compilation/base.py:1120`), он же передаётся aiecc как
`--peano` (`base.py:906`). ⇒ **кернелы собраны clang 21.0.0, conda-peano не участвует.**

Практический вывод: для сборки кернелов достаточно pip-половины (`mlir-aie` + `llvm-aie` +
наш `iron`). Ryzen AI на этой машине установлен, но в пересборке кернелов не задействован —
полностью удалять его и перепроверять мы не пробовали.

## Сборка

### Linux

```bash
git clone --branch ggml-xdna https://github.com/Soket1/llama.cpp-xdna.git
cd llama.cpp-xdna

source /opt/xilinx/xrt/setup.sh

cmake -B build -DGGML_XDNA=ON
cmake --build build --config Release -j$(nproc)
```

### Windows

```powershell
git clone --branch ggml-xdna https://github.com/Soket1/llama.cpp-xdna.git
cd llama.cpp-xdna

$env:XILINX_XRT = "C:\path\to\xrt_sdk\xrt"

# Developer PowerShell for VS 2022:
cmake -B build -DGGML_XDNA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

⚠️ `XILINX_XRT` — переменная **этапа конфигурации**. Оставленная в окружении при запуске,
она заставляет XRT искать по ней `xrt_core.dll` (который лежит в пакете драйвера, а не в SDK),
и компиляция кернелов падает с кодом 1 и полностью пустым выводом. Поэтому `compile.py`
для собственного поиска использует отдельную `XDNA_XRT_SDK_DIR`.

### Кастомные пути к XRT

```bash
cmake -B build -DGGML_XDNA=ON \
  -DXRT_INCLUDE_DIR=/custom/path/include \
  -DXRT_COREUTIL_LIB=/custom/path/lib/libxrt_coreutil.so
```

Линкуется только `xrt_coreutil`. Переменной `XRT_CORE_LIB` больше нет: в Windows SDK
импортной библиотеки `xrt_core` нет вообще, так что она была NOTFOUND даже на исправной
установке и никем не читалась — чистая приманка для того, кто разбирает неудачную сборку.

## Запуск

```bash
# Базовый
./build/bin/llama-cli -m model.gguf -p "Hello" -n 50

# С Hugging Face
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 50

# Сервер
./build/bin/llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Environment Variables

### Оптимизации (включают NPU-операции)

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `XDNA_ENABLE_SWIGLU` | off | Fused SwiGLU FFN на NPU (gate/up/down + SiLU + mul) |
| `XDNA_ENABLE_QKV` | off | Chained Q/K/V проекции (3 GEMV → 1 runlist) |
| `XDNA_ENABLE_SWIGLU_PREFILL` | off | SwiGLU для prefill (M≥32) |
| `XDNA_ENABLE_SWIGLU_INT8` | off | W8A16 INT8 SwiGLU (int8 веса, bf16 активации) |
| `XDNA_ENABLE_SWIGLU_FUSED` | off | Fused gate+up+silu+mul INT8 + down GEMV |
| `XDNA_ENABLE_RMS_NORM` | off | Standalone RMSNorm на NPU |
| `XDNA_ENABLE_GEMV` | off | GEMV на NPU (по умолчанию CPU) |
| `XDNA_ENABLE_ATTENTION_PREFILL` | off | Attention block prefill (11 sub-kernels) |
| `XDNA_ENABLE_TRANSFORMER_BLOCK` | off | Transformer block prefill (17 sub-kernels) |
| `XDNA_ENABLE_TBLOCK_FUSED` | off | Monolithic single-ELF transformer block |
| `XDNA_ENABLE_TBLOCK_FUSED_N` | 1 | Multi-layer fusion (2 или 4 блока в одном ELF) |
| `XDNA_ENABLE_TBLOCK_FUSED_W8A16` | off | W8A16 для transformer block (INT8 attention) |

### Диагностика

| Переменная | Описание |
|-----------|----------|
| `XDNA_DEBUG` | Отладочный вывод (dispatch stats, match counts, overhead) |
| `XDNA_FORCE_CPU` | Принудительно всё на CPU (для сравнения) |

### Кеш и компиляция

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `GGML_XDNA_CACHE_DIR` | `~/.cache/ggml-xdna/xclbin/` | Директория кеша xclbin |
| `GGML_XDNA_COMPILE_SCRIPT` | `compile.py` in PATH | Путь к compile.py |

## Диагностика

### Отладочный вывод

```bash
export XDNA_DEBUG=1
./build/bin/llama-cli -m model.gguf -p "Hello" -n 10 2>&1 | head -20
```

Пример вывода:
```
ggml-xdna: graph_compute n_nodes=1856 mul_mat=448 npu_dispatchable=448
           glu=128 swiglu=128 swiglu_window=32 swiglu_match=32
           attn_window=0 attn_match=0 tblock_window=0 tblock_match=0
ggml-xdna: QKV plan: 32 triples (64 skip nodes)
ggml-xdna: warm gemv matrix K=2048 N=2048 weight=w_q (1 cached)
ggml-xdna: qkv_prof K=2048 Nq=2048 Nk=512 Nv=512
           in=15us rl_build=8us rl_exec=120us rl_wait=85us out=12us total=240us
```

### Профилирование XRT

```bash
# Собрать trace
XRT_TRACE=1 ./build/bin/llama-cli -m model.gguf -p "Hello" -n 10

# Конвертировать в Chrome trace format
python ggml/src/ggml-xdna/tools/xrt_trace_to_chrome.py \
  --input xrt_trace.log --output trace.json

# Открыть в chrome://tracing
```

### Сравнение NPU vs CPU

```bash
# Baseline на CPU
XDNA_FORCE_CPU=1 ./build/bin/llama-cli -m model.gguf -p "Hello" -n 50

# С NPU оптимизациями
XDNA_ENABLE_SWIGLU=1 XDNA_ENABLE_QKV=1 \
  ./build/bin/llama-cli -m model.gguf -p "Hello" -n 50
```

## Кеш кернелов

xclbin файлы кешируются по ключу `(op, shape, dtype, num_cols)`. По умолчанию кеш в:
- Linux: `~/.cache/ggml-xdna/xclbin/`
- Windows: `%LOCALAPPDATA%\ggml-xdna\xclbin\`

**Готовый набор для XDNA 2 лежит в репозитории** — `npu_kernels_win_8col/` (~1 МБ, 40 файлов,
включая слитый `decode_layer_f3best`). Чтобы бэкенд взял именно его, а не пустой кеш
по умолчанию:

```powershell
$env:GGML_XDNA_CACHE_DIR = "$PWD\npu_kernels_win_8col"
```

Каталоги `*_build/` внутри — промежуточные файлы aiecc, они в `.gitignore` и пересоздаются
при каждой компиляции.

```bash
# Очистить (перекомпилирует при следующем запуске — нужен полный тулчейн)
rm -rf ~/.cache/ggml-xdna/xclbin/
```

## Compile.py

Внутренний мост между C++ бэкендом и IRON. Вызывается автоматически при первом запуске новой shape.

```bash
# Ручная компиляция GEMV
python compile.py gemv --N 4096 --K 2048 --num-aie-columns 8

# Ручная компиляция SwiGLU decode
python compile.py swiglu-decode --embedding-dim 2048 --hidden-dim 5632 --num-aie-columns 8

# Ручная компиляция QKV
python compile.py qkv --embedding-dim 2048 --q-dim 2048 --k-dim 512 --v-dim 512 --num-aie-columns 8

# Слитый decode-слой (тот самый быстрый путь), геометрия Llama-3.2-1B
python compile.py decode-layer-f3best --embed-dim 2048 --hidden-dim 8192 \
  --head-dim 64 --num-kv-heads 8 --attn-group 4 --seq-len 256 --group-size 32
```

`compile.py --help` печатает полный список подкоманд (их около тридцати).

## Квантизация

| Формат | Описание | Поддержка |
|--------|----------|-----------|
| BF16 | Стандартный | ✅ Все модели |
| Q8_0 | 8-bit (group_size=32) | ✅ XDNA1 + XDNA2 |
| W8A16 | INT8 веса, bf16 активации | ✅ Только XDNA2 |
| W8A8 | INT8 веса + INT8 активации | ✅ Только XDNA2 |
| W4ABF16 | 4-bit веса, bf16 активации | ✅ XDNA1 + XDNA2 |

## Известные ограничения

| Ограничение | Описание | Обходной путь |
|-------------|----------|---------------|
| Быстрый путь только 1B/Q4_0 | `decode_layer_f3best` написан под геометрию Llama-3.2-1B (E=2048, H=8192, head_dim=64, GQA 4:1) | другая модель → пооперационный путь |
| XDNA 1 на Windows не поддержан | `compile.py: get_device_cols()` безусловно возвращает 8 колонок без детекции | нужна детекция под 4-колоночный Phoenix/Hawk Point |
| Prefill медленнее CPU | 100 vs 192 т/с на 1B Q4_0 | выигрыш только в decode |
| Контекст ≤256 | окно StreamingLLM в диспатче; при NCHUNK≥4 ломается последняя q-голова | известный баг codegen value-тайла, отложен |
| lm_head только на CPU | перенос на NPU перемерен: 28.5 → 23.0 т/с (140 МБ словаря холодными каждый токен) | путь закрыт |
| Python в рантайме | compile.py при первой встрече новой shape | использовать кеш из репозитория |

## Структура кода

```
ggml/src/ggml-xdna/
├── ggml-xdna.cpp       # C++ бэкенд (~19 400 строк)
│   ├── xdna_kernel_entry    # Кеш кернелов
│   ├── graph_compute()      # Главный диспетчер + f3best LOOP
│   └── ggml_backend_xdna_*  # ggml backend API
├── compile.py          # Python мост к IRON (~3 300 строк)
├── CMakeLists.txt      # Сборка (ищет XRT)
├── tests/
└── tools/
    ├── correctness_test.py    # ГЛАВНЫЙ харнесс: пресеты, token-match, --bench
    ├── xclbin_replay.cpp      # автономный прогон xclbin
    ├── xclbin_switch_cost.cpp # замер стоимости переключения xclbin
    ├── xrt_async_spike.cpp    # замер async-диспатча XRT
    └── xrt_trace_to_chrome.py # профилирование
```

### correctness_test.py

Точка входа для любой проверки — раньше ручного запуска `llama-cli`.

```powershell
python ggml\src\ggml-xdna\tools\correctness_test.py --list          # тесты и пресеты
python ggml\src\ggml-xdna\tools\correctness_test.py paris_short_q4_0_f3best_loop
python ggml\src\ggml-xdna\tools\correctness_test.py --bench --bench-only "f3best LOOP"
```

Пути к тулчейну заданы в `BASE_ENV` в начале файла — под свою машину правьте там.

---

*Последнее обновление: 2026-07-26*
