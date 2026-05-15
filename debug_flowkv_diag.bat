@echo off
setlocal

set "AMD_DRIVER_DIR=C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_1a1aa059597c4810"
set "PATH=%AMD_DRIVER_DIR%;%CD%;%PATH%"

set "PYTHONPATH=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt\python;C:\Python313\Lib\site-packages;%PYTHONPATH%"
set "GGML_XDNA_PYTHON_CMD=C:\Python313\python.exe"
set "PEANO_INSTALL_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\win64.o\tools\peano"
set "XRT_BIN_DIR=C:\Users\Kuhnya\Downloads\xrt_windows_sdk\xrt_sdk\xrt"
set "MLIR_AIE_BIN_DIR=C:\ProgramData\miniforge3\envs\ryzen-ai-1.7.1\Lib\site-packages\mlir_aie\bin"
set "PATH=%PEANO_INSTALL_DIR%\bin;%XRT_BIN_DIR%;%MLIR_AIE_BIN_DIR%;%PATH%"

set "GGML_XDNA_CACHE_DIR=C:\llama.cpp-xdna\npu_kernels_win_8col"
set "MODEL_PATH=models\llama-3.2-1b-instruct-BF16.gguf"

REM ===== STEP 0: Clean cache =====
echo === STEP 0: Clean cache ===
powershell -Command "Remove-Item -Path 'C:\llama.cpp-xdna\build\*' -Include *.mlir, *.o, *.bin, *.insts, *.xclbin -Recurse -Force -ErrorAction SilentlyContinue"
powershell -Command "Get-ChildItem -Path 'C:\llama.cpp-xdna\build\*' -Directory -Filter '*.prj' -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
if exist npu_kernels_win_8col rmdir /s /q npu_kernels_win_8col
echo.

REM ===== STEP 1: Test arg-swap fix (FlowKV, no DIAG) =====
echo === STEP 1: Test arg-swap fix ===
set "XDNA_ENABLE_FLOWKV_DECODE=1"
set "XDNA_ENABLE_GEMV=1"
set "XDNA_ENABLE_SWIGLU=1"
set "XDNA_ENABLE_QKV=1"
set "XDNA_ENABLE_DECODE_BATCH=1"
set "XDNA_ENABLE_TRANSFORMER_BLOCK=1"
set GGML_XDNA_NUM_COLS=8
set GGML_XDNA_FORCE_CH1=0
set GGML_XDNA_SCHED_KV_OFFLOAD=1

echo /exit | build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 16 -c 512 -ngl 100 --no-mmap -fa off >step1_fix_output.log 2>step1_fix_xdna.log
echo --- STEP 1 RESULT ---
findstr /C:"Prompt:" step1_fix_output.log
findstr /C:">" step1_fix_output.log | findstr /V "Prompt: Exiting"
echo.

REM ===== STEP 2: Phantom offset diagnostic (XDNA_DIAG_OFFSET=1) =====
echo === STEP 2: Phantom Offset Diagnostic ===
set "XDNA_DIAG_OFFSET=1"
set "XDNA_DEBUG=1"
set "XNA_SCHED_DEBUG=1"

echo /exit | build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 16 -c 512 -ngl 100 --no-mmap -fa off >step2_diag_output.log 2>step2_diag_offset.log
echo --- STEP 2 RESULT ---
echo Look for "=== XDNA_DIAG_OFFSET ===" in step2_diag_offset.log
findstr /C:"XDNA_DIAG_OFFSET" step2_diag_offset.log
findstr /C:"PHANTOM OFFSET" step2_diag_offset.log
findstr /C:"host_only - normal delta" step2_diag_offset.log
echo.
set "XDNA_DIAG_OFFSET="
set "XDNA_DEBUG="
set "XNA_SCHED_DEBUG="

REM ===== STEP 3: Baseline (no FlowKV) =====
echo === STEP 3: Baseline without FlowKV ===
set "XDNA_ENABLE_FLOWKV_DECODE="
set "XDNA_ENABLE_GEMV=1"
set "XDNA_ENABLE_SWIGLU=1"
set "XDNA_ENABLE_QKV=1"
set "XDNA_ENABLE_DECODE_BATCH=1"
set "XDNA_ENABLE_TRANSFORMER_BLOCK=1"
set GGML_XDNA_NUM_COLS=8
set GGML_XDNA_FORCE_CH1=0
set GGML_XDNA_SCHED_KV_OFFLOAD=1

echo /exit | build\bin\Release\llama-cli.exe -m "%MODEL_PATH%" -p "What is the capital of France" -n 16 -c 512 -ngl 100 --no-mmap -fa off >step3_baseline_output.log 2>step3_baseline_xdna.log
echo --- STEP 3 RESULT ---
findstr /C:"Prompt:" step3_baseline_output.log
findstr /C:">" step3_baseline_output.log | findstr /V "Prompt: Exiting"
echo.

echo === DONE ===
echo Files to send:
echo   step1_fix_output.log + step1_fix_xdna.log   *** FIX TEST ***
echo   step2_diag_output.log + step2_diag_offset.log  *** PHANTOM DIAG ***
echo   step3_baseline_output.log + step3_baseline_xdna.log
pause
