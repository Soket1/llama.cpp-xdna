import os, re, subprocess, sys, json, hashlib, statistics
from pathlib import Path

MODEL=Path(r'C:/llama.cpp-xdna/models/llama-3.2-1b-instruct-Q4_0-vocabQ4.gguf')
PROMPT='What is the capital of France? Explain in detail.'
N=192
OUT=Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/bench_raw449_vs_current_rr')
OUT.mkdir(parents=True,exist_ok=True)
BASE=Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/raw449-source')
CUR=Path(r'C:/llama.cpp-xdna')
CONFIGS={
 'raw449': {'root':BASE,'exe':BASE/'build/bin/Release/llama-cli.exe','cache':BASE/'npu_kernels_win_8col','key':'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_decouple','toggles':{'F3BEST_MT_DECOUPLE':'1'}},
 'current_rr': {'root':CUR,'exe':CUR/'build/bin/Release/llama-cli.exe','cache':CUR/'npu_kernels_win_8col','key':'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_mxp_ub_nokv_decouple_tb_rr','toggles':{'F3BEST_MT_DECOUPLE':'1','F3BEST_TRIPLE_B':'1','F3BEST_HANDASM_RR':'1'}},
}
FLAGS=['XDNA_ENABLE_GEMV','XDNA_ENABLE_SWIGLU','XDNA_ENABLE_QKV','XDNA_ENABLE_DECODE_BATCH','XDNA_ENABLE_TRANSFORMER_BLOCK','XDNA_ENABLE_FLOWKV_DECODE','XDNA_ENABLE_RMS_NORM','XDNA_ENABLE_GEMV_INT4','XDNA_ENABLE_SWIGLU_INT4','XDNA_ENABLE_FUSED_LAYER','XDNA_LAYER_FUSED','XDNA_ENABLE_LAYER_F3BEST','XDNA_ATTN_SUPPORTS','XDNA_LAYER_F3BEST_LIVE','XDNA_F3BEST_LOOP']
CLEAR=['XDNA_F3BEST_NPU_KV','XDNA_F3BEST_GLUETIME','XDNA_F3BEST_TIME','XDNA_FW_PROF','XDNA_FW_PROF_VERBOSE','XDNA_FW_PROF_SYNC','F3BEST_MT_DECOUPLE','F3BEST_TRIPLE_B','F3BEST_HANDASM_RR','F3BEST_FFN_DIV','F3BEST_MT_RELAY','F3BEST_MT_DEPTH','F3BEST_RL_FIX','XDNA_F3BEST_MT_DECOUPLE','XDNA_F3BEST_TRIPLE_B']
perf_re=re.compile(r'\[\s*Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s\s*\]')
key_re=re.compile(r'loaded kernel for (decode_layer_f3best\S+)')
for n,c in CONFIGS.items():
 for ext in ('.xclbin','.insts'):
  p=c['cache']/(c['key']+ext)
  if not p.is_file():raise RuntimeError(f'{n}: missing {p}')
 if not c['exe'].is_file():raise RuntimeError(f'{n}: missing {c["exe"]}')
if not MODEL.is_file():raise RuntimeError(MODEL)
def env_for(c):
 e=os.environ.copy()
 for k in CLEAR:e.pop(k,None)
 e.update({k:'1' for k in FLAGS})
 e.update(c['toggles']);e['GGML_XDNA_CACHE_DIR']=str(c['cache'])
 return e
args=lambda c:[str(c['exe']),'-m',str(MODEL),'-n',str(N),'-c','512','-ngl','100','--no-mmap','-fa','off','--temp','0','-s','42','-p',PROMPT,'--single-turn']
manifest={'model':str(MODEL),'prompt':PROMPT,'n_predict':N,'configs':{}}
for n,c in CONFIGS.items():
 x=c['cache']/(c['key']+'.xclbin');i=c['cache']/(c['key']+'.insts')
 manifest['configs'][n]={'root':str(c['root']),'exe':str(c['exe']),'key':c['key'],'toggles':c['toggles'],'xclbin_sha256':hashlib.sha256(x.read_bytes()).hexdigest(),'xclbin_bytes':x.stat().st_size,'insts_sha256':hashlib.sha256(i.read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
order=['raw449','current_rr']*5
results={n:[] for n in CONFIGS}
for ix,n in enumerate(order,1):
 c=CONFIGS[n]
 print(f'=== {ix}/{len(order)} {n} ===',flush=True)
 r=subprocess.run(args(c),cwd=c['root'],env=env_for(c),capture_output=True,text=True,errors='replace',timeout=2000)
 prefix=OUT/f'{ix:02d}_{n}'
 prefix.with_suffix('.stdout.txt').write_text(r.stdout,encoding='utf-8')
 prefix.with_suffix('.stderr.txt').write_text(r.stderr,encoding='utf-8')
 if r.returncode:raise RuntimeError(f'{n} run{ix} rc={r.returncode}')
 ps=perf_re.findall(r.stdout);keys=key_re.findall(r.stderr)
 if not ps:raise RuntimeError(f'{n} run{ix}: no perf')
 if c['key'] not in keys:raise RuntimeError(f'{n} run{ix}: expected {c["key"]}, got {keys[-4:]}')
 rate=float(ps[-1][1]);results[n].append(rate)
 print(f'{n} {rate:.2f} t/s key={keys[-1]}',flush=True)
summary={n:{'rates':v,'median':statistics.median(v),'mean':statistics.fmean(v),'min':min(v),'max':max(v)} for n,v in results.items()}
summary['median_delta_current_rr_minus_raw449']=summary['current_rr']['median']-summary['raw449']['median']
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
