<#
.SYNOPSIS
    Запускает llama-cli через NPU-путь (слитый decode-слой), выставив всё нужное окружение.

.DESCRIPTION
    Быстрый путь требует полутора десятков переменных окружения, флага -fa off и указания
    на каталог с готовыми кернелами. Копировать это руками -- источник молчаливых ошибок:
    забытая переменная не ломает запуск, а просто уводит на медленный путь.

    Скрипт проверяет предпосылки, настраивает окружение и запускает llama-cli.
    Всё, что передано после -- , уходит в llama-cli как есть.

.EXAMPLE
    .\run-npu.ps1 -Model models\llama-3.2-1b-instruct-Q4_0.gguf -Prompt "What is the capital of France?"

.EXAMPLE
    .\run-npu.ps1 -Model models\llama-3.2-1b-instruct-Q4_0.gguf -- -p "Hi" -n 128 -c 256
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [string] $Prompt = "What is the capital of France?",

    [int] $NPredict = 64,

    # Контекст. Слитый слой собран под окно 256; больше -- см. раздел про ограничения.
    [int] $CtxSize = 256,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ExtraArgs
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$cli = Join-Path $root "build\bin\Release\llama-cli.exe"
if (-not (Test-Path $cli)) {
    throw "Не найден $cli. Сначала соберите проект (см. XDNA_QUICKSTART.md)."
}
if (-not (Test-Path $Model)) {
    throw "Не найдена модель: $Model"
}

$cache = Join-Path $root "npu_kernels_win_8col"
if (-not (Test-Path (Join-Path $cache "decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_rr.xclbin"))) {
    throw "В $cache нет слитого decode-кернела. Каталог должен приехать вместе с репозиторием."
}

# XILINX_XRT нужна только на этапе cmake. Оставленная в окружении, она заставляет XRT
# искать по ней xrt_core.dll и роняет компиляцию кернелов без единой строки вывода.
if ($env:XILINX_XRT) {
    Write-Warning "XILINX_XRT задана ($env:XILINX_XRT). Для запуска она не нужна и мешает -- убираю на время этого процесса."
    $env:XILINX_XRT = $null
}

$env:GGML_XDNA_CACHE_DIR = $cache

# Бэкенд по умолчанию просит 4 колонки (наследие первого XRT-диспатча), а все
# опубликованные замеры сняты на 8. Разница только в том, какой вариант per-op
# GEMV берётся (..._4col_g32 против ..._8col_g32) -- слитый слой одинаков в обоих
# случаях. Ставим 8, чтобы вы получили ровно ту конфигурацию, что в таблице.
$env:GGML_XDNA_NUM_COLS = "8"

# Пресет npu_f3best_loop -- тот же набор, что в ggml/src/ggml-xdna/tools/correctness_test.py
$flags = @(
    "XDNA_ENABLE_GEMV", "XDNA_ENABLE_SWIGLU", "XDNA_ENABLE_QKV",
    "XDNA_ENABLE_DECODE_BATCH", "XDNA_ENABLE_TRANSFORMER_BLOCK",
    "XDNA_ENABLE_FLOWKV_DECODE", "XDNA_ENABLE_RMS_NORM",
    "XDNA_ENABLE_GEMV_INT4", "XDNA_ENABLE_SWIGLU_INT4",
    "XDNA_ENABLE_FUSED_LAYER", "XDNA_LAYER_FUSED",
    "XDNA_ENABLE_LAYER_F3BEST", "XDNA_ATTN_SUPPORTS",
    "XDNA_LAYER_F3BEST_LIVE", "XDNA_F3BEST_LOOP"
)
foreach ($f in $flags) { Set-Item -Path "Env:$f" -Value "1" }

# -fa off обязателен: с включённым flash-attention слитый слой не диспатчится вовсе.
$args = @(
    "-m", $Model,
    "-n", $NPredict,
    "-c", $CtxSize,
    "-ngl", "100",
    "--no-mmap",
    "-fa", "off",
    "--single-turn"
)
if ($ExtraArgs) {
    $args += ($ExtraArgs | Where-Object { $_ -ne "--" })
} else {
    $args += @("-p", $Prompt)
}

Write-Host "NPU-путь включён; кернелы: $cache" -ForegroundColor DarkGray
Write-Host "Признак успеха в stderr: [f3best-LOOP] dispatched 16 layers in ONE call" -ForegroundColor DarkGray

& $cli @args
exit $LASTEXITCODE
