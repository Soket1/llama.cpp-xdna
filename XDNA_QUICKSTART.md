# ggml-xdna — Запуск на AMD XDNA NPU

Руководство по запуску llama.cpp с бэкендом ggml-xdna на AMD Ryzen AI процессорах с NPU (XDNA/XDNA2).

## Поддерживаемое оборудование

| Поколение | NPU | AIE Columns | Чипы |
|-----------|-----|-------------|------|
| XDNA 1 | AIE | 4 | Ryzen 7040 (Phoenix), Ryzen 8040 (Hawk Point) |
| XDNA 2 | AIE2 | 8 | Ryzen AI 300 (Strix), Ryzen AI Max (Strix Halo), Krackan Point |

> **Примечание:** Текущий код валидирован преимущественно на **XDNA2 (8 columns)**. XDNA1 (4 columns) имеет ограниченную поддержку.

## Предварительные требования

### 1. AMD XRT (Xilinx Runtime)

XRT — это рантайм для доступа к NPU. Должен быть установлен и настроен.

```bash
# Установка XRT (если ещё не установлен)
# Инструкции: https://github.com/amd/xdna-driver

# Проверка установки
ls /opt/xilinx/xrt/include/xrt/xrt_device.h
ls /opt/xilinx/xrt/lib/libxrt_core.so

# Настройка окружения (добавить в ~/.bashrc)
source /opt/xilinx/xrt/setup.sh
```

### 2. IRON (AMD NPU Operator Library)

IRON нужен **только на этапе компиляции кернелов** (xclbin). Не нужен для запуска с предскомпилированным кешем.

```bash
# Клонирование IRON
git clone https://github.com/amd/IRON.git
cd IRON
pip install -e .
```

### 3. Python зависимости (для compile.py)

```bash
pip install numpy
# IRON устанавливает остальные зависимости (aie, mlir, etc.)
```

## Сборка

### Клонирование форка

```bash
git clone --branch ggml-xdna https://github.com/albiol2004/llama.cpp.git
cd llama.cpp
```

### Сборка с CMake

```bash
# Убедитесь что XRT настроен
source /opt/xilinx/xrt/setup.sh

# Сборка
cmake -B build -DGGML_XDNA=ON
cmake --build build --config Release -j$(nproc)
```

> **Примечание:** CMake автоматически ищет XRT в `/opt/xilinx/xrt/`. Если XRT установлен в другое место, укажите:
> ```bash
> cmake -B build -DGGML_XDNA=ON -DXRT_INCLUDE_DIR=/path/to/xrt/include -DXRT_CORE_LIB=/path/to/libxrt_core.so
> ```

### Проверка сборки

```bash
# Должен показать ggml-xdna в списке бэкендов
./build/bin/llama-cli --list-devices
```

## Запуск

### Базовый запуск

```bash
# С локальным GGUF файлом
./build/bin/llama-cli -m model.gguf -p "Hello, world" -n 50

# С Hugging Face
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 50

# Сервер
./build/bin/llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

### С оптимизациями (экспериментальные)

NPU-бэкенд имеет ряд флагов окружения для включения экспериментальных оптимизаций:

```bash
# SwiGLU FFN на NPU (fused gate/up/down + SiLU + mul)
export XDNA_ENABLE_SWIGLU=1

# Chained Q/K/V projection (3 GEMV → 1 runlist)
export XDNA_ENABLE_QKV=1

# SwiGLU для prefill (M>=32)
export XDNA_ENABLE_SWIGLU_PREFILL=1

# INT8 SwiGLU (W8A16 — int8 веса, bf16 активации)
export XDNA_ENABLE_SWIGLU_INT8=1

# RMSNorm на NPU
export XDNA_ENABLE_RMS_NORM=1

# Attention block prefill (11 sub-kernels в одном xclbin)
export XDNA_ENABLE_ATTENTION_PREFILL=1

# Transformer block prefill (17 sub-kernels — attention + FFN)
export XDNA_ENABLE_TRANSFORMER_BLOCK=1

# Transformer block prefill FUSED (монолитный single-ELF)
export XDNA_ENABLE_TBLOCK_FUSED=1

# Multi-layer fusion (2 или 4 transformer блока в одном ELF)
export XDNA_ENABLE_TBLOCK_FUSED_N=2   # или 4

# W8A16 для transformer block (INT8 attention проекции)
export XDNA_ENABLE_TBLOCK_FUSED_W8A16=1

# Запуск со всеми оптимизациями
XDNA_ENABLE_SWIGLU=1 \
XDNA_ENABLE_QKV=1 \
XDNA_ENABLE_SWIGLU_PREFILL=1 \
XDNA_ENABLE_ATTENTION_PREFILL=1 \
XDNA_ENABLE_TRANSFORMER_BLOCK=1 \
./build/bin/llama-cli -m model.gguf -p "Hello" -n 50
```

## Диагностика

### Отладочный вывод

```bash
# Общий debug — статистика dispatch, match counts, overhead
export XDNA_DEBUG=1
./build/bin/llama-cli -m model.gguf -p "Hello" -n 10

# Вывод покажет:
# ggml-xdna: graph_compute n_nodes=1234 mul_mat=456 npu_dispatchable=389
# ggml-xdna: swiglu_window=32 swiglu_match=32
# ggml-xdna: QKV plan: 32 triples (64 skip nodes)
```

### Пример вывода с XDNA_DEBUG

```
ggml-xdna: graph_compute n_nodes=1856 mul_mat=448 npu_dispatchable=448
           glu=128 swiglu=128 swiglu_window=32 swiglu_match=32
           attn_window=0 attn_match=0 tblock_window=0 tblock_match=0
ggml-xdna: QKV plan: 32 triples (64 skip nodes)
ggml-xdna: warm gemv matrix K=2048 N=2048 weight=w_q (1 cached)
ggml-xdna: qkv_prof K=2048 Nq=2048 Nk=512 Nv=512 in=15us rl_build=8us
           rl_exec=120us rl_wait=85us out=12us total=240us
```

### Профилирование

```bash
# XRT trace → Chrome trace viewer
python ggml/src/ggml-xdna/tools/xrt_trace_to_chrome.py --input xrt_trace.log --output trace.json
# Открыть в chrome://tracing
```

### Кеш кернелов

```bash
# Кеш xclbin хранится в:
ls ~/.cache/ggml-xdna/xclbin/

# Переопределить директорию кеша:
export GGML_XDNA_CACHE_DIR=/path/to/cache

# Очистить кеш (перекомпилирует все кернелы):
rm -rf ~/.cache/ggml-xdna/xclbin/
```

## Рекомендуемые модели

### Для NPU-only (малые модели, 1-3B)

```bash
# Gemma 3 1B — хорошо работает на NPU
./build/bin/llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 100

# Qwen3 0.6B
./build/bin/llama-cli -hf Qwen/Qwen3-0.6B-GGUF -p "Hello" -n 100
```

### Для NPU+CPU hybrid (средние модели, 7-8B)

```bash
# Llama 3 8B — matmuls на NPU, остальное на CPU
./build/bin/llama-cli -hf bartowski/Meta-Llama-3-8B-Instruct-GGUF -p "Hello" -n 100

# Qwen3 8B
./build/bin/llama-cli -hf Qwen/Qwen3-8B-GGUF -p "Hello" -n 100
```

## Известные ограничения

| Ограничение | Описание |
|-------------|----------|
| **Decode speed** | ~9 t/s — bottleneck в host-side dispatch overhead |
| **Attention на NPU** | 0.5x от CPU speed — пока медленнее чем CPU |
| **head_dim** | Только head_dim=64 поддерживается MHA kernel'ом |
| **seq_len prefill** | Минимум 256 для NPU prefill (меньше → CPU) |
| **INT8** | Только Q8_0 квантизация (group_size=32) |
| **Python runtime** | compile.py вызывается при первом запуске новой shape — нужен Python + IRON |

## Типичные проблемы

### "XRT not found"

```bash
# Решение: установить XRT и source setup.sh
source /opt/xilinx/xrt/setup.sh
# Или указать пути явно:
export XILINX_XRT=/opt/xilinx/xrt
```

### "No device found" / "XRT device invalid"

```bash
# Проверить что NPU виден:
xrt-smi examine
# Должен показать NPU device

# Если не виден — проверить драйвер:
lsmod | grep amdxdna
# Если не загружен:
sudo modprobe amdxdna
```

### "compile.py failed" / "IRON not found"

```bash
# IRON нужен только для первой компиляции кернелов
pip install -e /path/to/IRON

# Или использовать предкомпилированный кеш (если есть)
export GGML_XDNA_CACHE_DIR=/path/to/precompiled/cache
```

### Медленный первый запуск

Первый запуск компилирует xclbin кернелы для конкретных shape модели. Это занимает 1-5 минут. Последние запуски используют кеш (~секунды).

---

*Последнее обновление: 2026-05-02*
*Источник: [albiol2004/llama.cpp@ggml-xdna](https://github.com/albiol2004/llama.cpp/tree/ggml-xdna)*
