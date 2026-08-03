#!/usr/bin/env python3
"""
Fresh budget decomposition for #175: decompose the 253 µs exposed compute on
production RR+DECOUPLE+TRIPLE_B no-KV config.

Two independent measurement axes, cross-validated at the baseline:

  AXIS 1 — FFN_DIV sweep (llama-cli, production emitter)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Varies only the FFN portion of the weight stream. Slope = raw bandwidth,
  intercept = everything that is NOT proportional to weight bytes.
  Runs: F3BEST_FFN_DIV=1 (baseline), 2, 4.  3 runs each, alternating.

  AXIS 2 — Compute stubs (standalone replay, recompiled .o files)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Each stub zeros one component's compute while keeping DMA/locks/BD-chain
  identical. Delta vs baseline = that component's exposed compute cost.
  Stubs: STUB_QKV, STUB_OPROJ, STUB_FFN, STUB_NM, FLOWKV_VALUE_STUB.

  The standalone replay uses emit_ofold8_f3best.py (NOT the production
  f3best_emit_nokv.py).  Absolute times differ, but component deltas are
  structurally comparable because both paths use the same kernel sources.

Combined, the two axes answer:
  (a) Is the weight stream still at 48.5 GB/s?  (FFN_DIV slope)
  (b) What is the current intercept?            (FFN_DIV intercept)
  (c) How does the intercept decompose?          (stub deltas)
  (d) What is the largest remaining component?   (for #175 attack)

Output: .budget-profiles-20260802/budget_YYYYMMDD_HHMMSS/
  - manifest.txt      — one-line-per-run summary
  - */stderr.txt      — full stderr (GLUETIME / replay output)
  - */stdout.txt      — full stdout (perf line)
  - analysis.txt      — fitted slope, intercept, decomposition table
"""

import os, re, subprocess, sys, shutil, json, time
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(r'C:/llama.cpp-xdna')
MODEL     = Path(r'C:/llama.cpp-xdna/models/llama-3.2-1b-instruct-Q4_0-vocabQ4.gguf')
PROMPT    = 'What is the capital of France? Explain in detail.'
N         = 192
CTX       = 512
SEED      = 42

LLAMA_CLI = ROOT / 'build/bin/Release/llama-cli.exe'
STUB_BUILD = ROOT / 'dev_notes/track_a_build/build_ofold8_f3best_real.py'

OUT       = ROOT / f'.budget-profiles-20260802/budget_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# Production f3best cache key (must match what the emitter produces for
# RR+DECOUPLE+TRIPLE_B no-KV).  If this key is stale, update it.
PROD_CACHE = ROOT / 'npu_kernels_win_8col'
# C++ backend cache-key format (ggml-xdna.cpp:1477-1481):
# decode_layer_f3best_K{K}_H{H}_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_mxp_ub{kv_abi}[_d{div}]{dc}{tb}{rr}
# Production = K2048_H8192, _nokv, _decouple, _tb, _rr.
PROD_KEY   = ('decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32'
              '_mc_preq_vexp_vreg_dq8_qp_mxp_ub_nokv_decouple_tb_rr')
# Prefix for key validation — any DIV variant (_d2, _d4) is valid.
KEY_PREFIX = 'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_mxp_ub_nokv'

# ── Validation ────────────────────────────────────────────────────────────────
for p, desc in [(LLAMA_CLI, 'llama-cli'), (MODEL, 'model'), (STUB_BUILD, 'stub-build')]:
    if not p.exists():
        sys.exit(f'MISSING {desc}: {p}')
for ext in ('.xclbin', '.insts'):
    p = PROD_CACHE / (PROD_KEY + ext)
    if not p.is_file():
        sys.exit(f'MISSING production xclbin/insts for key:\n  {PROD_KEY}\n  checked: {p}')

os.makedirs(OUT, exist_ok=True)

# ── Env helpers ───────────────────────────────────────────────────────────────
BASE_ENV = {
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
    'F3BEST_HANDASM_RR': '1',
    'XDNA_F3BEST_GLUETIME': '1',
    'GGML_XDNA_CACHE_DIR': str(PROD_CACHE),
}

# Wipe ambient toggles that could perturb the measurement.
CLEAR = [
    'XDNA_F3BEST_NPU_KV', 'XDNA_F3BEST_TIME',
    'XDNA_FW_PROF', 'XDNA_FW_PROF_VERBOSE', 'XDNA_FW_PROF_SYNC',
    'F3BEST_FFN_DIV', 'F3BEST_MT_RELAY', 'F3BEST_MT_DEPTH',
    'XDNA_F3BEST_MT_DECOUPLE', 'XDNA_F3BEST_TRIPLE_B',
    'F3BEST_RL_FIX', 'F3BEST_HANDASM_RR', 'F3BEST_MT_DECOUPLE',
    'F3BEST_TRIPLE_B',
]


def make_env(**extra):
    e = os.environ.copy()
    for k in CLEAR:
        e.pop(k, None)
    e.update(BASE_ENV)
    e.update({k: str(v) for k, v in extra.items()})
    return e


# ── Parsers ───────────────────────────────────────────────────────────────────
PERF_RE = re.compile(r'\[\s*Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s\s*\]')
GLUE_RE = re.compile(r'\[f3best-gluetime\] token:\s+(.*)')
REPLAY_RE = re.compile(r'(\d+\.?\d*)\s*us/dispatch')
KEY_RE = re.compile(r'loaded kernel for (decode_layer_f3best\S+)')


def parse_gluetimes(stderr_text):
    """Return list of per-token gluetimes, each a dict of component->us."""
    tokens = []
    for m in GLUE_RE.finditer(stderr_text):
        parts = m.group(1).split()
        d = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        if d:
            tokens.append(d)
    return tokens


def gluetimes_summary(tokens, skip_first=2):
    """Median across steady-state tokens (skip first N cold tokens)."""
    if len(tokens) <= skip_first:
        return {}
    keys = tokens[0].keys()
    out = {}
    for k in keys:
        vals = sorted(t[k] for t in tokens[skip_first:] if k in t and isinstance(t[k], (int, float)))
        if vals:
            n = len(vals)
            out[k] = vals[n // 2]  # median
    return out


def median_us_from_replay(stderr_text):
    """Median µs/dispatch from standalone replay output."""
    vals = [float(m) for m in REPLAY_RE.findall(stderr_text)]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


# ── AXIS 1: FFN_DIV sweep (llama-cli, production emitter) ─────────────────────
def run_llama_cli(name, extra_env, ordinal, max_retries=3):
    """Run llama-cli, capture stdout+stderr, return (decode_t_s, gluetimes_median).
    Retries on RR crash (STATUS_STACK_BUFFER_OVERRUN = 0xC0000409)."""
    for attempt in range(1, max_retries + 1):
        env = make_env(**extra_env)
        args = [str(LLAMA_CLI), '-m', str(MODEL), '-n', str(N), '-c', str(CTX),
                '-ngl', '100', '--no-mmap', '-fa', 'off', '--temp', '0', '-s', str(SEED),
                '-p', PROMPT, '--single-turn']
        label = f'{name}' if attempt == 1 else f'{name} (retry{attempt})'
        prefix = OUT / f'{ordinal:02d}_{name}_r{attempt}'
        print(f'  [{ordinal:02d}] {label} ...', flush=True, end=' ')
        t0 = time.time()
        r = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True,
                           errors='replace', timeout=2000)
        elapsed = time.time() - t0
        prefix.with_suffix('.stdout.txt').write_text(r.stdout, encoding='utf-8')
        prefix.with_suffix('.stderr.txt').write_text(r.stderr, encoding='utf-8')
        if r.returncode == 3221226505:  # STATUS_STACK_BUFFER_OVERRUN — RR crash
            print(f'CRASH ({elapsed:.0f}s)', end='')
            if attempt < max_retries:
                print(f', retrying...')
                time.sleep(3)  # let XRT driver settle
                continue
            else:
                print(f', exhausted retries')
                return None, None, f'crash after {max_retries} attempts'
        if r.returncode != 0:
            print(f'FAILED rc={r.returncode} ({elapsed:.0f}s)')
            return None, None, f'rc={r.returncode}'
        perf = PERF_RE.findall(r.stdout)
        if not perf:
            print(f'NO PERF LINE ({elapsed:.0f}s)')
            return None, None, 'no perf line'
        prompt, decode = float(perf[-1][0]), float(perf[-1][1])
        keys = KEY_RE.findall(r.stderr)
        loaded_key = keys[-1] if keys else ''
        if not loaded_key.startswith(KEY_PREFIX):
            print(f'WRONG KEY ({elapsed:.0f}s): {loaded_key[-80:]}')
            return decode, None, f'wrong key; got {loaded_key}'
        gl = parse_gluetimes(r.stderr)
        gs = gluetimes_summary(gl)
        print(f'{decode:.2f} t/s, {len(gl)} gluetimes ({elapsed:.0f}s)')
        return decode, gs, None
    return None, None, 'unreachable'


# ── AXIS 2: Compute stubs (standalone replay) ─────────────────────────────────
def run_standalone_replay(name, extra_env, ordinal):
    """Run standalone replay, capture output, return (median_us, error)."""
    env = make_env(**extra_env)
    # standalone replay needs its own env (no llama-cli flags)
    for k in BASE_ENV:
        env.pop(k, None)
    env['PYTHONPATH'] = str(ROOT / 'dev_notes/track_a_build')
    args = [sys.executable, str(STUB_BUILD)]
    prefix = OUT / f'{ordinal:02d}_{name}'
    print(f'  [{ordinal:02d}] {name} (build+replay ~8min) ...', flush=True, end=' ')
    t0 = time.time()
    r = subprocess.run(args, cwd=ROOT / 'dev_notes/track_a_build', env=env,
                       capture_output=True, text=True, errors='replace', timeout=1200)
    elapsed = time.time() - t0
    prefix.with_suffix('.stdout.txt').write_text(r.stdout, encoding='utf-8')
    prefix.with_suffix('.stderr.txt').write_text(r.stderr, encoding='utf-8')
    # Stubs produce wrong answers (NaN outputs) — the build script may exit non-zero
    # on validation failure but the replay timing is still valid.
    us = median_us_from_replay(r.stderr)
    if us is None:
        us = median_us_from_replay(r.stdout)
    if us is None:
        # Try harder — look for the summary line directly
        for line in (r.stdout + '\n' + r.stderr).split('\n'):
            m = REPLAY_RE.search(line)
            if m:
                us = float(m.group(1))
                break
    if us is None:
        print(f'NO REPLAY TIME ({elapsed:.0f}s) rc={r.returncode}')
        return None, f'no replay time, rc={r.returncode}'
    status = 'OK' if r.returncode == 0 else f'rc={r.returncode}'
    print(f'{us:.1f} us ({elapsed:.0f}s, {status})')
    return us, None


# ── Orchestration ─────────────────────────────────────────────────────────────
print('=== #175 Budget Decomposition ===')
print(f'Output: {OUT}')
print(f'Production key: {PROD_KEY}')
print()

manifest = []

# ═══ AXIS 1: FFN_DIV sweep ════════════════════════════════════════════════════
# 3 alternations × 3 DIV values = 9 runs
print('--- Axis 1: FFN_DIV sweep (llama-cli) ---')
ffn_div_results = {}
for div in [1, 2, 4]:
    for rep in range(3):
        ordinal = len(manifest) + 1
        name = f'ffndiv{div}_r{rep+1}'
        decode, gs, err = run_llama_cli(name, {'F3BEST_FFN_DIV': str(div)}, ordinal)
        manifest.append((ordinal, name, 'llama-cli', f'FFN_DIV={div}', decode, gs, err))
        if decode and gs:
            ffn_div_results.setdefault(div, []).append((decode, gs))
        elif err:
            print(f'    ERROR: {err}')

# ═══ AXIS 2: Compute stubs (standalone replay) ════════════════════════════════
# Baseline first, then each stub.  1 rep each (replay is expensive ~8 min/run).
print('\n--- Axis 2: Compute stubs (standalone replay) ---')
STUB_MATRIX = [
    ('baseline',  {}),
    ('stub_qkv',  {'F3BEST_STUB_QKV': '1'}),
    ('stub_oproj',{'F3BEST_STUB_OPROJ': '1'}),
    ('stub_ffn',  {'F3BEST_STUB_FFN': '1'}),
    ('stub_nm',   {'F3BEST_STUB_NM': '1'}),
    ('stub_value',{'F3BEST_FKV_DEFS': 'FLOWKV_VALUE_STUB'}),
]
# The standalone emitter doesn't set DECOUPLE/TRIPLE_B/RR by default;
# set them so the MLIR matches production structure.
STUB_COMMON = {
    'F3BEST_MT_DECOUPLE': '1',
    'F3BEST_TRIPLE_B': '1',
    'F3BEST_HANDASM_RR': '1',
}

stub_results = {}
for name, extra in STUB_MATRIX:
    ordinal = len(manifest) + 1
    full_env = {**STUB_COMMON, **extra}
    us, err = run_standalone_replay(name, full_env, ordinal)
    manifest.append((ordinal, name, 'replay', str(extra), us, None, err))
    if us:
        stub_results[name] = us
    elif err:
        print(f'    ERROR: {err}')

# ═══ Analysis ═════════════════════════════════════════════════════════════════
print('\n=== Analysis ===')
analysis_lines = []

# FFN_DIV linear fit
if len(ffn_div_results) >= 3:
    analysis_lines.append('--- FFN_DIV slope/intercept ---')
    # Use median decode t/s for each DIV
    div_tokens = {}
    for div, samples in ffn_div_results.items():
        decodes = [s[0] for s in samples if s[0]]
        if decodes:
            decodes.sort()
            div_tokens[div] = decodes[len(decodes)//2]

    if len(div_tokens) == 3:
        # Convert t/s -> µs/token
        div_us = {d: 1e6/t for d, t in div_tokens.items()}
        # WT_BYTES at each DIV (from emitter constants)
        # WT_TILES = GEMV_T + OPROJ_T + 2*GU_T + DN_T
        # DIV=1: 64+64+512+256=896, DIV=2: 64+64+256+128=512, DIV=4: 64+64+128+64=320
        # Weight bytes: WT_TILES * PACKED; PACKED = M*E/2 + M*(E/G)*2 = 4*1024+4*64*2=4096+512=4608
        PACKED = 4608
        E, G, M = 2048, 32, 4
        GROUPS = E // G  # 64
        GEMV_T = 256 // M  # 64 (Q-proj tiles)
        OPROJ_T = 256 // M  # 64 (O-proj tiles)
        # Q and O are NOT divided by FFN_DIV — only FFN is
        for div in sorted(div_tokens.keys()):
            gu_t = (1024 // M) // div  # gate/up tiles per phase
            dn_t = ((E // M) // 2) // div  # down tiles
            wt_tiles = GEMV_T + OPROJ_T + 2*gu_t + dn_t
            wt_bytes = wt_tiles * PACKED * 8  # 8 center tiles
            analysis_lines.append(
                f'  FFN_DIV={div}: {div_tokens[div]:.2f} t/s = {div_us[div]:.0f} us/tok, '
                f'WT_TILES={wt_tiles}, WT_MB={wt_bytes/1e6:.2f}')

        # Linear regression: us = slope * MB + intercept
        divs = sorted(div_tokens.keys())
        xs = []
        for div in divs:
            gu_t = (1024 // M) // div
            dn_t = ((E // M) // 2) // div
            wt_tiles = GEMV_T + OPROJ_T + 2*gu_t + dn_t
            wt_mb = wt_tiles * PACKED * 8 / 1e6
            xs.append(wt_mb)
        ys = [div_us[d] for d in divs]

        n = len(xs)
        mean_x = sum(xs)/n; mean_y = sum(ys)/n
        num = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))
        den = sum((x-mean_x)**2 for x in xs)
        slope = num/den if den else 0  # us/MB
        intercept = mean_y - slope*mean_x  # us

        # Bandwidth = 1/slope GB/s (slope is us/MB, so 1/slope * 1000 = GB/s)
        bw = 1000/slope if slope else 0

        analysis_lines.append(f'  Slope: {slope:.2f} us/MB  ->  Bandwidth: {bw:.2f} GB/s')
        analysis_lines.append(f'  Intercept: {intercept:.0f} us/token  ({intercept/16:.0f} us/layer)')
        analysis_lines.append(f'  Intercept @44.3 t/s (22573 us/tok): {intercept/22573*100:.1f}% of token')

    # Also report NPU time from GLUETIME
    for div in sorted(div_tokens.keys()):
        samples = ffn_div_results.get(div, [])
        for _, gs in samples:
            if gs and 'npu' in gs:
                analysis_lines.append(f'  FFN_DIV={div} GLUETIME npu={gs["npu"]:.0f} us')
                break

# Stub decomposition
if stub_results:
    analysis_lines.append('\n--- Stub decomposition (standalone replay) ---')
    baseline = stub_results.get('baseline')
    if baseline:
        analysis_lines.append(f'  baseline: {baseline:.1f} us/dispatch')
        for name in ['stub_qkv', 'stub_oproj', 'stub_ffn', 'stub_nm', 'stub_value']:
            us = stub_results.get(name)
            if us:
                delta = baseline - us
                analysis_lines.append(f'  {name}: {us:.1f} us  (delta={delta:.1f} us = {delta/baseline*100:.1f}%)')

# ═══ Manifest ══════════════════════════════════════════════════════════════════
manifest_path = OUT / 'manifest.txt'
with open(manifest_path, 'w', encoding='utf-8') as f:
    for o, name, kind, cfg, val, gs, err in manifest:
        if err:
            f.write(f'{o:02d}\t{name}\t{kind}\t{cfg}\tERROR: {err}\n')
        elif kind == 'llama-cli':
            gs_str = ' '.join(f'{k}={v:.0f}' for k,v in (gs or {}).items() if isinstance(v, (int,float)))
            f.write(f'{o:02d}\t{name}\t{kind}\t{cfg}\tdecode={val:.2f} t/s\t{gs_str}\n')
        else:
            f.write(f'{o:02d}\t{name}\t{kind}\t{cfg}\t{val:.1f} us\n')

# ═══ Analysis file ═════════════════════════════════════════════════════════════
analysis_path = OUT / 'analysis.txt'
with open(analysis_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(analysis_lines) + '\n')

print('\n'.join(analysis_lines))
print(f'\nDone.  Manifest: {manifest_path}')
print(f'Analysis: {analysis_path}')
