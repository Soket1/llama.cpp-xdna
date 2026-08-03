import os, re, subprocess, sys
from pathlib import Path

OUT = Path(r'C:/llama.cpp-xdna/.budget-profiles-20260802')
MODEL = Path(r'C:/llama.cpp-xdna/models/llama-3.2-1b-instruct-Q4_0-vocabQ4.gguf')
PROMPT = 'What is the capital of France? Explain in detail.'
N = 192
CONFIGS = {
    'current': {
        'root': Path(r'C:/llama.cpp-xdna'),
        'exe': Path(r'C:/llama.cpp-xdna/build/bin/Release/llama-cli.exe'),
        'cache': Path(r'C:/llama.cpp-xdna/npu_kernels_win_8col'),
        'key': 'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_mxp_ub_nokv_decouple_tb',
    },
    'prekv': {
        'root': Path(r'C:/llama.cpp-xdna/.bisect-prekv'),
        'exe': Path(r'C:/llama.cpp-xdna/.bisect-prekv/build/bin/Release/llama-cli.exe'),
        'cache': Path(r'C:/llama.cpp-xdna/.bisect-prekv/npu_kernels_win_8col'),
        'key': 'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_mxp_ub_decouple_tb',
    },
}
for name, cfg in CONFIGS.items():
    for ext in ('.xclbin', '.insts'):
        p = cfg['cache'] / f'{cfg["key"]}{ext}'
        if not p.is_file(): raise RuntimeError(f'{name}: missing {p}')
if not MODEL.is_file(): raise RuntimeError(f'missing model: {MODEL}')

# Delete every possible ambient toggle and explicitly restore the matched ABI/options.
CLEAR = [
    'XDNA_F3BEST_NPU_KV', 'XDNA_F3BEST_GLUETIME', 'XDNA_F3BEST_TIME',
    'XDNA_FW_PROF', 'XDNA_FW_PROF_VERBOSE', 'XDNA_FW_PROF_SYNC',
    'F3BEST_MT_DECOUPLE', 'F3BEST_TRIPLE_B', 'F3BEST_HANDASM_RR',
    'F3BEST_FFN_DIV', 'F3BEST_MT_RELAY', 'F3BEST_MT_DEPTH',
    'XDNA_F3BEST_MT_DECOUPLE', 'XDNA_F3BEST_TRIPLE_B',
]
perf_re = re.compile(r'\[\s*Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s\s*\]')
key_re = re.compile(r'loaded kernel for (decode_layer_f3best\S+)')
glue_re = re.compile(r'\[f3best-gluetime\] token: (.*)')
fw_re = re.compile(r'\[fw-prof(?::sched)?\] (.*)')

# Alternation stops progressive clock/thermal state from mapping onto a source.
order = ['current', 'prekv', 'current', 'prekv', 'current', 'prekv']
summary = []
for ordinal, name in enumerate(order, 1):
    cfg = CONFIGS[name]
    env = os.environ.copy()
    for k in CLEAR: env.pop(k, None)
    env.update({
        'XDNA_ENABLE_GEMV': '1',
        'XDNA_ENABLE_SWIGLU': '1',
        'XDNA_ENABLE_QKV': '1',
        'XDNA_ENABLE_DECODE_BATCH': '1',
        'XDNA_ENABLE_TRANSFORMER_BLOCK': '1',
        'XDNA_ENABLE_FLOWKV_DECODE': '1',
        'XDNA_ENABLE_RMS_NORM': '1',
        'XDNA_ENABLE_GEMV_INT4': '1',
        'XDNA_ENABLE_SWIGLU_INT4': '1',
        'XDNA_ENABLE_FUSED_LAYER': '1',
        'XDNA_LAYER_FUSED': '1',
        'XDNA_ENABLE_LAYER_F3BEST': '1',
        'XDNA_ATTN_SUPPORTS': '1',
        'XDNA_LAYER_F3BEST_LIVE': '1',
        'XDNA_F3BEST_LOOP': '1',
        'F3BEST_MT_DECOUPLE': '1',
        'F3BEST_TRIPLE_B': '1',
        'XDNA_F3BEST_GLUETIME': '1',
        # FW_PROF is deliberately disabled for the performance distribution: its
        # verbose per-token printing perturbs the host-side timing we are testing.
        'GGML_XDNA_CACHE_DIR': str(cfg['cache']),
    })
    args = [str(cfg['exe']), '-m', str(MODEL), '-n', str(N), '-c', '512',
            '-ngl', '100', '--no-mmap', '-fa', 'off', '--temp', '0', '-s', '42',
            '-p', PROMPT, '--single-turn']
    print(f'=== {ordinal}/4 {name}: {cfg["exe"]} ===', flush=True)
    r = subprocess.run(args, cwd=cfg['root'], env=env, capture_output=True, text=True,
                       errors='replace', timeout=2000)
    prefix = OUT / f'{ordinal:02d}_{name}_nofw'
    prefix.with_suffix('.stdout.txt').write_text(r.stdout, encoding='utf-8')
    prefix.with_suffix('.stderr.txt').write_text(r.stderr, encoding='utf-8')
    if r.returncode:
        print(f'{name}: FAILED rc={r.returncode}; see {prefix}.stderr.txt', flush=True)
        sys.exit(r.returncode)
    perf = perf_re.findall(r.stdout)
    keys = key_re.findall(r.stderr)
    glue = glue_re.findall(r.stderr)
    fw = fw_re.findall(r.stderr)
    if not perf: raise RuntimeError(f'{name}: no perf line')
    if cfg['key'] not in keys:
        raise RuntimeError(f'{name}: wrong/missing f3best kernel; expected {cfg["key"]}, got {keys[-4:]}')
    # GLUETIME emits only the first several decode tokens by design; retain all.
    if len(glue) < 4:
        raise RuntimeError(f'{name}: insufficient GLUETIME samples ({len(glue)})')
    p, g = perf[-1]
    print(f'{name}: prompt={p} decode={g} t/s gluetimes={len(glue)} fw_lines={len(fw)}', flush=True)
    print(f'{name}: loaded={keys[-1]}', flush=True)
    summary.append((ordinal, name, float(p), float(g), len(glue), len(fw)))

(OUT/'manifest.txt').write_text('\n'.join(
    f'{o}\t{name}\tprompt={p:.2f}\tdecode={g:.2f}\tglue={ng}\tfw={nf}'
    for o,name,p,g,ng,nf in summary) + '\n', encoding='utf-8')
print('=== manifest ===')
print((OUT/'manifest.txt').read_text(encoding='utf-8'), end='')
