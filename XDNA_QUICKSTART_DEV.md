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

### IRON (AMD NPU Operator Library)

Нужен **только для компиляции кернелов** (xclbin). Не нужен для запуска с кешем.

```bash
git clone https://github.com/amd/IRON.git
cd IRON
pip install -e .

# Зависимости
pip install numpy
```

### Windows

```powershell
# 1. NPU Driver — через Windows Update или AMD Support
# 2. XRT: https://github.com/amd/xdna-driver/releases
# 3. Visual Studio 2022 Build Tools (C++ Desktop + CMake tools)
# 4. Python 3.10+ (для compile.py)
```

## Сборка

### Linux

```bash
git clone --branch ggml-xdna https://github.com/albiol2004/llama.cpp.git
cd llama.cpp

source /opt/xilinx/xrt/setup.sh

cmake -B build -DGGML_XDNA=ON
cmake --build build --config Release -j$(nproc)
```

### Windows

```powershell
git clone --branch ggml-xdna https://github.com/albiol2004/llama.cpp.git
cd llama.cpp

# Developer PowerShell for VS 2022:
cmake -B build -DGGML_XDNA=ON
cmake --build build --config Release
```

### Кастомные пути к XRT

```bash
cmake -B build -DGGML_XDNA=ON \
  -DXRT_INCLUDE_DIR=/custom/path/include \
  -DXRT_CORE_LIB=/custom/path/lib/libxrt_core.so
```

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

xclbin файлы кешируются по SHA256 ключу (op, shape, dtype). Кеш хранится в:
- Linux: `~/.cache/ggml-xdna/xclbin/`
- Windows: `%LOCALAPPDATA%\ggml-xdna\xclbin\`

```bash
# Посмотреть размер кеша
du -sh ~/.cache/ggml-xdna/xclbin/

# Очистить (перекомпилирует при следующем запуске)
rm -rf ~/.cache/ggml-xdna/xclbin/

# Перенести кеш на другой диск
export GGML_XDNA_CACHE_DIR=/mnt/fast/ssd/ggml-xdna-cache
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
```

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
| Decode ~9 t/s | Host-side dispatch overhead | Ожидается улучшение с runlist batching |
| Attention на NPU 0.5x CPU | Dispatch overhead > compute gain | Использовать CPU для attention |
| head_dim=64 only | MHA kernel hardcoded | Использовать модели с head_dim=64 |
| Min seq_len 256 для prefill | IRON GEMM tile constraint | Меньше → CPU fallback |
| Python runtime | compile.py при первом запуске | Предкомпилировать кеш |

## Структура кода

```
ggml/src/ggml-xdna/
├── ggml-xdna.cpp       # C++ бэкенд (9901 строк)
│   ├── xdna_kernel_entry    # Кеш кернелов
│   ├── xdna_swiglu_entry    # Fused SwiGLU
│   ├── xdna_qkv_entry       # QKV projection
│   ├── xdna_attention_prefill_entry  # Attention
│   ├── graph_compute()      # Главный scheduler
│   └── ggml_backend_xdna_*  # ggml backend API
├── compile.py          # Python мост к IRON (2386 строк)
├── CMakeLists.txt      # Сборка (ищет XRT)
├── tests/              # 15 тестов
│   ├── test_compile.py
│   ├── test_dispatch.py
│   ├── test_swiglu.py
│   ├── test_qkv.py
│   └── ...
└── tools/
    └── xrt_trace_to_chrome.py  # Профилирование
```

---

*Последнее обновление: 2026-05-02*
