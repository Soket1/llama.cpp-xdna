import os,re,subprocess,json,hashlib,statistics
from pathlib import Path
MODEL=Path(r'C:/llama.cpp-xdna/models/llama-3.2-1b-instruct-Q4_0-vocabQ4.gguf'); PROMPT='What is the capital of France? Explain in detail.'; N=192
OUT=Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/bench_qp_vs_qpmxp');OUT.mkdir(parents=True,exist_ok=True)
configs={
 'qp_tb':{'root':Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/qp-tb-source'),'key':'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_tb','artifact':Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/qp_tb_exact/f3best.xclbin'),'insts':Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/qp_tb_exact/f3best.insts'),'toggles':{'F3BEST_TRIPLE_B':'1'}},
 'qpmxp_tb':{'root':Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/qpmxp-tb-source'),'key':'decode_layer_f3best_K2048_H8192_sl256_d64_ag4_kv8_g32_mc_preq_vexp_vreg_dq8_qp_mxp_tb','artifact':Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/qpmxp_tb_exact/f3best.xclbin'),'insts':Path(r'C:/llama.cpp-xdna/.historical-artifact-sweep/qpmxp_tb_exact/f3best.insts'),'toggles':{}},
}
FLAGS=['XDNA_ENABLE_GEMV','XDNA_ENABLE_SWIGLU','XDNA_ENABLE_QKV','XDNA_ENABLE_DECODE_BATCH','XDNA_ENABLE_TRANSFORMER_BLOCK','XDNA_ENABLE_FLOWKV_DECODE','XDNA_ENABLE_RMS_NORM','XDNA_ENABLE_GEMV_INT4','XDNA_ENABLE_SWIGLU_INT4','XDNA_ENABLE_FUSED_LAYER','XDNA_LAYER_FUSED','XDNA_ENABLE_LAYER_F3BEST','XDNA_ATTN_SUPPORTS','XDNA_LAYER_F3BEST_LIVE','XDNA_F3BEST_LOOP']
CLEAR=['XDNA_F3BEST_NPU_KV','XDNA_F3BEST_GLUETIME','XDNA_F3BEST_TIME','XDNA_FW_PROF','XDNA_FW_PROF_VERBOSE','XDNA_FW_PROF_SYNC','F3BEST_MT_DECOUPLE','F3BEST_TRIPLE_B','F3BEST_HANDASM_RR','F3BEST_FFN_DIV','F3BEST_MT_RELAY','F3BEST_MT_DEPTH','F3BEST_RL_FIX','XDNA_F3BEST_MT_DECOUPLE','XDNA_F3BEST_TRIPLE_B']
perf=re.compile(r'\[\s*Prompt:\s+([\d.]+)\s+t/s\s*\|\s*Generation:\s+([\d.]+)\s+t/s\s*\]'); keyre=re.compile(r'loaded kernel for (decode_layer_f3best\S+)')
for n,c in configs.items():
 c['exe']=c['root']/'build/bin/Release/llama-cli.exe';c['cache']=c['root']/'npu_kernels_win_8col';c['cache'].mkdir(exist_ok=True)
 for p in(c['exe'],c['artifact'],c['insts'],MODEL):
  if not p.is_file():raise RuntimeError(f'{n}: missing {p}')
 target=c['cache']/(c['key']+'.xclbin');targeti=c['cache']/(c['key']+'.insts');target.write_bytes(c['artifact'].read_bytes());targeti.write_bytes(c['insts'].read_bytes())
def env(c):
 e=os.environ.copy()
 for k in CLEAR:e.pop(k,None)
 e.update({k:'1'for k in FLAGS});e.update(c['toggles']);e['GGML_XDNA_CACHE_DIR']=str(c['cache']);return e
def args(c):return[str(c['exe']),'-m',str(MODEL),'-n',str(N),'-c','512','-ngl','100','--no-mmap','-fa','off','--temp','0','-s','42','-p',PROMPT,'--single-turn']
man={n:{'root':str(c['root']),'key':c['key'],'xclbin_sha256':hashlib.sha256(c['artifact'].read_bytes()).hexdigest(),'xclbin_bytes':c['artifact'].stat().st_size,'insts_sha256':hashlib.sha256(c['insts'].read_bytes()).hexdigest(),'toggles':c['toggles']}for n,c in configs.items()};(OUT/'manifest.json').write_text(json.dumps(man,indent=2),encoding='utf-8')
res={n:[]for n in configs}
for ix,n in enumerate(['qp_tb','qpmxp_tb']*4,1):
 c=configs[n];print(f'=== {ix}/8 {n} ===',flush=True)
 r=subprocess.run(args(c),cwd=c['root'],env=env(c),capture_output=True,text=True,errors='replace',timeout=2000)
 pre=OUT/f'{ix:02d}_{n}';pre.with_suffix('.stdout.txt').write_text(r.stdout,encoding='utf-8');pre.with_suffix('.stderr.txt').write_text(r.stderr,encoding='utf-8')
 if r.returncode:raise RuntimeError(f'{n} rc={r.returncode}')
 ps=perf.findall(r.stdout);ks=keyre.findall(r.stderr)
 if not ps:raise RuntimeError(f'{n}: no perf')
 if c['key'] not in ks:raise RuntimeError(f'{n}: wrong key {ks[-4:]}')
 rate=float(ps[-1][1]);res[n].append(rate);print(f'{n} {rate:.2f} t/s key={ks[-1]}',flush=True)
sumry={n:{'rates':v,'median':statistics.median(v),'mean':statistics.fmean(v),'min':min(v),'max':max(v)}for n,v in res.items()};sumry['delta_qpmxp_minus_qp']=sumry['qpmxp_tb']['median']-sumry['qp_tb']['median'];(OUT/'summary.json').write_text(json.dumps(sumry,indent=2),encoding='utf-8');print(json.dumps(sumry),flush=True)
