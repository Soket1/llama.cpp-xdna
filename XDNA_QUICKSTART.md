# ggml-xdna — Запуск на AMD XDNA NPU

Руководство по запуску llama.cpp с бэкендом ggml-xdna на AMD Ryzen AI процессорах с NPU (XDNA/XDNA2).

> **Платформы:** [Linux](#linux) | [Windows](#windows)

## Поддерживаемое оборудование

| Поколение | NPU | AIE Columns | Чипы |
|-----------|-----|-------------|------|
| XDNA 1 | AIE | 4 | Ryzen 7040 (Phoenix), Ryzen 8040 (Hawk Point) |
| XDNA 2 | AIE2 | 8 | Ryzen AI 300 (Strix), Ryzen AI Max (Strix Halo), Krackan Point |

> **Примечание:** Текущий код валидирован преимущественно на **XDNA2 (8 columns)**. XDNA1 (4 columns) имеет ограниченную поддержку.

### Квантизация по поколениям

| Поколение | W8A16 | W4ABF16 | W8A8 |
|-----------|-------|---------|------|
| XDNA 1 (Phoenix, Hawk Point) | ❌ | ✅ | ❌ |
| XDNA 2 (Strix, Strix Halo) | ✅ | ✅ | ✅ |

---

## Linux

### Предварительные требования

#### 1. Ядро Linux

NPU требует модуль ядра `amdxdna`. Минимальная версия: **Linux 6.14** (встроен в mainline).

```bash
# Проверить версию ядра
uname -r  # должно быть >= 6.14

# Проверить что модуль загрушен
lsmod | grep amdxdna

# Если не загружен:
sudo modprobe amdxdna

# Проверить что NPU виден:
xrt-smi examine
```

> **Fedora 41+, Ubuntu 25.04+** — ядро 6.14+ уже в репозиториях.
> Для более старых дистрибутивов — сборка ядра из source с патчем amdxdna.

#### 2. AMD XRT (Xilinx Runtime)

XRT — рантайм для доступа к NPU.

```bash
# Установка XRT
# Инструкции: https://github.com/amd/xdna-driver

# Проверка
ls /opt/xilinx/xrt/include/xrt/xrt_device.h
ls /opt/xilinx/xrt/lib/libxrt_core.so

# Настройка (добавить в ~/.bashrc)
source /opt/xilinx/xrt/setup.sh
```

#### 3. IRON (AMD NPU Operator Library)

IRON нужен **только на этапе компиляции кернелов** (xclbin). Не нужен для запуска с предскомпилированным кешем.

```bash
git clone https://github.com/amd/IRON.git
cd IRON
pip install -e .
```

#### 4. Python зависимости

```bash
pip install numpy
# IRON устанавливает остальные зависимости (aie, mlir, etc.)
```

### Сборка (Linux)

```bash
git clone --branch ggml-xdna https://github.com/albiol2004/llama.cpp.git
cd llama.cpp

source /opt/xilinx/xrt/setup.sh

cmake -B build -DGGML_XDNA=ON
cmake --build build --config Release -j$(nproc)
```

Если XRT установлен не в `/opt/xilinx/xrt/`:
```bash
cmake -B build -DGGML_XDNA=ON \
  -DXRT_INCLUDE_DIR=/path/to/xrt/include \
  -DXRT_CORE_LIB=/path/to/libxrt_core.so
```

---

## Windows

### Предварительные требования

#### 1. NPU Driver

NPU драйвер для Windows поставляется через **Windows Update**. На Ryzen AI ноутбуках обычно уже установлен.

```powershell
# Проверить в Device Manager:
# → System devices → AMD NPU / XDNA Device

# Или через PowerShell:
Get-PnpDevice | Where-Object { $_.FriendlyName -match "NPU|XDNA" }
```

Если драйвер не установлен — обновите Windows или скачайте с [AMD Support](https://www.amd.com/en/support).

#### 2. AMD XRT для Windows

```powershell
# Скачать XRT installer с:
# https://github.com/amd/xdna-driver/releases

# Или через Ryzen AI SDK:
# https://ryzenai.docs.amd.com/

# Проверить:
& "C:\Program Files\AMD\XRT\bin\xrt-smi.exe" examine
```

#### 3. Visual Studio Build Tools

Нужен MSVC компилятор:

```powershell
# Установить Visual Studio 2022 Build Tools
# или Visual Studio 2022 Community с компонентами:
#   - Desktop development with C++
#   - C++ CMake tools for Windows
```

#### 4. Python (для compile.py)

```powershell
# Python 3.10+ с pip
python --version
pip install numpy
# IRON: pip install -e path\to\IRON
```

### Сборка (Windows)

```powershell
git clone --branch ggml-xdna https://github.com/albiol2004/llama.cpp.git
cd llama.cpp

# В Developer PowerShell for VS 2022:
cmake -B build -DGGML_XDNA=ON
cmake --build build --config Release

# Или с явными путями к XRT:
cmake -B build -DGGML_XDNA=ON `
  -DXRT_INCLUDE_DIR="C:\Program Files\AMD\XRT\include" `
  -DXRT_CORE_LIB="C:\Program Files\AMD\XRT\lib\xrt_core.lib"
```

### Запуск (Windows)

```powershell
.\build\bin\Release\llama-cli.exe -m model.gguf -p "Hello" -n 50

# С HF моделью
.\build\bin\Release\llama-cli.exe -hf ggml-org/gemma-3-1b-it-GGUF -p "Hello" -n 50

# Сервер
.\build\bin\Release\llama-server.exe -hf ggml-org/gemma-3-1b-it-GGUF
```

---

## Запуск (общий)

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

На Windows:
```powershell
$env:XDNA_ENABLE_SWIGLU=1
$env:XDNA_ENABLE_QKV=1
$env:XDNA_ENABLE_SWIGLU_PREFILL=1
$env:XDNA_ENABLE_ATTENTION_PREFILL=1
$env:XDNA_ENABLE_TRANSFORMER_BLOCK=1
.\build\bin\Release\llama-cli.exe -m model.gguf -p "Hello" -n 50
```

## Диагностика

### Отладочный вывод

```bash
# Linux
export XDNA_DEBUG=1
./build/bin/llama-cli -m model.gguf -p "Hello" -n 10

# Windows PowerShell
$env:XDNA_DEBUG=1
.\build\bin\Release\llama-cli.exe -m model.gguf -p "Hello" -n 10
```

Вывод покажет:
```
ggml-xdna: graph_compute n_nodes=1234 mul_mat=456 npu_dispatchable=389
ggml-xdna: swiglu_window=32 swiglu_match=32
ggml-xdna: QKV plan: 32 triples (64 skip nodes)
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
# Linux:   ~/.cache/ggml-xdna/xclbin/
# Windows: %LOCALAPPDATA%\ggml-xdna\xclbin\

# Переопределить:
export GGML_XDNA_CACHE_DIR=/path/to/cache          # Linux
$env:GGML_XDNA_CACHE_DIR="C:\path\to\cache"        # Windows

# Очистить (перекомпилирует все кернелы):
rm -rf ~/.cache/ggml-xdna/xclbin/                   # Linux
Remove-Item -Recurse "$env:LOCALAPPDATA\ggml-xdna"  # Windows
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
# Linux
source /opt/xilinx/xrt/setup.sh
# Или:
export XILINX_XRT=/opt/xilinx/xrt

# Windows — проверить что XRT установлен:
& "C:\Program Files\AMD\XRT\bin\xrt-smi.exe" examine
```

### "No device found" / "XRT device invalid"

```bash
# Linux:
xrt-smi examine
lsmod | grep amdxdna
sudo modprobe amdxdna

# Windows:
# Device Manager → System devices → AMD NPU / XDNA Device
# Если нет — обновите Windows Update или установите драйвер с AMD Support
& "C:\Program Files\AMD\XRT\bin\xrt-smi.exe" examine
```

### "compile.py failed" / "IRON not found"

```bash
# IRON нужен только для первой компиляции кернелов
pip install -e /path/to/IRON

# Или использовать предкомпилированный кеш (если есть)
export GGML_XDNA_CACHE_DIR=/path/to/precompiled/cache          # Linux
$env:GGML_XDNA_CACHE_DIR="C:\precompiled\cache"                # Windows
```

### Медленный первый запуск

Первый запуск компилирует xclbin кернелы для конкретных shape модели. Это занимает 1-5 минут. Последние запуски используют кеш (~секунды).

### Windows: "VCRUNTIME140.dll not found"

Установите [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### Windows: "xrt-smi не найден"

```powershell
# Добавить XRT в PATH:
$env:PATH += ";C:\Program Files\AMD\XRT\bin"
# Или полный путь:
& "C:\Program Files\AMD\XRT\bin\xrt-smi.exe" examine
```

---

*Последнее обновление: 2026-05-02*
*Источник: [albiol2004/llama.cpp@ggml-xdna](https://github.com/albiol2004/llama.cpp/tree/ggml-xdna)*
