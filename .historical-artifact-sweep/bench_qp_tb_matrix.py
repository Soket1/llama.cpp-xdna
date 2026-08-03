import os, re, subprocess, sys, time, statistics, hashlib, json
from pathlib import Path

ROOT=Path(r'C:/llama.cpp-xdna/.bisect-pre160')
EXE=ROOT/'build/bin/Release/llama-cli.exe'
CACHE=ROOT/'npu_kernels_win_8col'
MODEL=Path(r'C:/llama.cpp-xdna/models/llama-3.2-1b-instruct-Q4_0-vocabQ4.gguf')
OUT=Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/bench_qp_tb_matrix')
OUT.mkdir(parents=True,exist_ok=True)
PROMPT='What is the capital of France? Explain in detail.'
N=192
KEY='decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_decouple_tb'
XCL=CACHE/f'{KEY}.xclbin'; INST=CACHE/f'{KEY}.insts'
for p in (EXE,MODEL,XCL,INST):
    if not p.is_file(): raise RuntimeError(f'missing {p}')
CLEAR=['XDNA_F3BEST_NPU_KV','XDNA_F3BEST_GLUETIME','XDNA_F3BEST_TIME','XDNA_FW_PROF','XDNA_FW_PROF_VERBOSE','XDNA_FW_PROF_SYNC','F3BEST_MT_DECOUPLE','F3BEST_TRIPLE_B','F3BEST_HANDASM_RR','F3BEST_FFN_DIV','F3BEST_MT_RELAY','F3BEST_MT_DEPTH','F3BEST_RL_FIX','XDNA_F3BEST_MT_DECOUPLE','XDNA_F3BEST_TRIPLE_B']
FLAGS=['XDNA_ENABLE_GEMV','XDNA_ENABLE_SWIGLU','XDNA_ENABLE_QKV','XDNA_ENABLE_DECODE_BATCH','XDNA_ENABLE_TRANSFORMER_BLOCK','XDNA_ENABLE_FLOWKV_DECODE','XDNA_ENABLE_RMS_NORM','XDNA_ENABLE_GEMV_INT4','XDNA_ENABLE_SWIGLU_INT4','XDNA_ENABLE_FUSED_LAYER','XDNA_LAYER_FUSED','XDNA_ENABLE_LAYER_F3BEST','XDNA_ATTN_SUPPORTS','XDNA_LAYER_F3BEST_LIVE','XDNA_F3BEST_LOOP']
perf_re=re.compile(r'\[\s*Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s\s*\]')
key_re=re.compile(r'loaded kernel for (decode_layer_f3best\S+)')
args=[str(EXE),'-m',str(MODEL),'-n',str(N),'-c','512','-ngl','100','--no-mmap','-fa','off','--temp','0','-s','42','-p',PROMPT,'--single-turn']
def mk_env():
    e=os.environ.copy()
    for k in CLEAR: e.pop(k,None)
    e.update({k:'1' for k in FLAGS})
    e.update({'F3BEST_MT_DECOUPLE':'1','F3BEST_TRIPLE_B':'1','GGML_XDNA_CACHE_DIR':str(CACHE)})
    return e
meta={'root':str(ROOT),'exe':str(EXE),'model':str(MODEL),'key':KEY,'xclbin_sha256':hashlib.sha256(XCL.read_bytes()).hexdigest(),'insts_sha256':hashlib.sha256(INST.read_bytes()).hexdigest(),'xclbin_bytes':XCL.stat().st_size,'insts_bytes':INST.stat().st_size,'n_predict':N,'prompt':PROMPT,'env':{k:mk_env().get(k) for k in FLAGS+['F3BEST_MT_DECOUPLE','F3BEST_TRIPLE_B','GGML_XDNA_CACHE_DIR']}}
(OUT/'meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
rates=[]
for i in range(6):
    print(f'=== qp_tb {i+1}/6 ===',flush=True)
    r=subprocess.run(args,cwd=ROOT,env=mk_env(),text=True,capture_output=True,errors='replace',timeout=2000)
    (OUT/f'{i+1:02d}.stdout.txt').write_text(r.stdout,encoding='utf-8')
    (OUT/f'{i+1:02d}.stderr.txt').write_text(r.stderr,encoding='utf-8')
    if r.returncode: raise RuntimeError(f'run {i+1}: rc={r.returncode}')
    ps=perf_re.findall(r.stdout)
    keys=key_re.findall(r.stderr)
    if not ps: raise RuntimeError(f'run {i+1}: no perf')
    if KEY not in keys: raise RuntimeError(f'run {i+1}: wrong key {keys[-4:]}')
    rate=float(ps[-1][1]); rates.append(rate)
    print(f'qp_tb {i+1}: {rate:.2f} t/s key={keys[-1]}',flush=True)
summary={'rates':rates,'median':statistics.median(rates),'mean':statistics.fmean(rates),'min':min(rates),'max':max(rates)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
