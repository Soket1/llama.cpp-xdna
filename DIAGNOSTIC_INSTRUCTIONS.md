# Phantom DMA Offset — Diagnostic Instructions

## What We're Testing

**Hypothesis 2**: Windows driver `kipudrv.inf` adds a hidden 32KB metadata prefix
to `host_only` buffer allocations, causing DMA to read from wrong physical address.

## Files Created

1. **`xdna_diag_offset.cpp`** — Standalone diagnostic tool (4 tests)
2. **`patch_flowkv_diag.patch`** — Inline patch for ggml-xdna.cpp (address dump on FlowKV path)

## How to Run

### Option A: Standalone diagnostic (recommended first)

```powershell
# On Windows machine with NPU and XRT installed
cd <path-to-llama.cpp-xdna>

# Compile
cl /std:c++17 /EHsc xdna_diag_offset.cpp /I"<XRT_SDK>/include" ^
   /link /LIBPATH:<XRT_SDK>/lib xrt_coreutil.lib

# Run with FlowKV xclbin
xdna_diag_offset.exe --xclbin <path-to-flowkv.xclbin>

# Run without xclbin (address-only tests)
xdna_diag_offset.exe
```

### Option B: Inline diagnostic in ggml-xdna

```powershell
# Apply patch
cd <path-to-llama.cpp-xdna>
git apply patch_flowkv_diag.patch

# Build normally
# ...

# Run with diagnostic enabled
set XDNA_DIAG_OFFSET=1
set XDNA_ENABLE_FLOWKV_DECODE=1
llama-cli -m <model> -p "What is the capital of France?" -n 16

# Look for "=== XDNA_DIAG_OFFSET ===" in stderr output
```

## What to Look For

### Test 1: host_only vs normal
- If `delta == 32768` (32 KB) → **Hypothesis 2 CONFIRMED**
- If `delta == 0` → Hypothesis 2 not supported by this test

### Test 2: Shim DMA registers
- Compare `xrt::bo::address()` with BD register base address
- If BD addr = bo.address() + 32KB → phantom offset in hardware

### Test 3: Export buffer
- If export changes address → driver remaps on export

### Test 4: K/V aliasing
- If V contains K's test pattern → buffers share physical memory

## Expected Outcomes

| Result | Meaning | Next Step |
|--------|---------|-----------|
| delta=32KB, host_only vs normal | Driver adds header | Use `normal` flag as workaround |
| delta=32KB, in BD registers | Hardware phantom offset | Use write32 direct register access |
| V contains K data | Buffer aliasing | Fix arg order in host code |
| All clear | Hypothesis 2 wrong | Focus on hypothesis 1 (arg swap) |

## Key Code References

- **Buffer creation**: `ggml-xdna.cpp:4515-4528` (bo_k, bo_v with `host_only`)
- **Argument order**: `ggml-xdna.cpp:11457-11460` (set_arg 3-6)
- **MLIR sequence**: `design.py:277` → `rt.sequence(L3_V_ty, L3_K_ty, ...)`
- **Expected mapping**: arg3=bo_k→%arg0=V_slot, arg4=bo_v→%arg1=K_slot (SWAPPED!)
