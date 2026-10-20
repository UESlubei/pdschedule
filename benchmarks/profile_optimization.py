"""Post-validation node-level GPU captures, excluded from performance matrices."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from apb.common import DATA,ROOT,PYTHON,clean_env,dump,digest
from benchmarks.analyze import audit_session
from benchmarks.profile_summary import extract

def main():
    # Both registered performance cohorts run before diagnostic GPU sampling.
    while not (DATA/'manifests/match1536-completed.json').exists():time.sleep(2)
    locks=[]
    for p in [DATA/'gpu.lock',DATA.with_name('adaptive_prefill_budget_v0320')/'gpu.lock',
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    root=DATA/'diagnostic_profiles';root.mkdir(exist_ok=True)
    configurations={'balanced':{'mode':'joint','minimum':2048},'fixed1536':{'fixed_idle_relax':True},'equalized':{'cap':2048},
                    'equalized256':{'mode':'equalized','cap':2048,'chunk_alignment':256},
                    'equalized1536':{'mode':'equalized','cap':1536,'chunk_alignment':512}}
    frozen={str(p):digest(p) for p in [ROOT/'apb/controller.py',ROOT/'apb/runtime.py',ROOT/'benchmarks/session.py']}
    receipt=[]
    for policy,cfg in configurations.items():
        directory=root/('node-'+policy);directory.mkdir(exist_ok=True);dest=directory/'raw';report=directory/'capture'
        saved=directory/'receipt.json'
        if saved.exists() and json.loads(saved.read_text()).get('complete'):
            receipt.append(json.loads(saved.read_text()));continue
        for p,sha in frozen.items():
            if digest(p)!=sha:raise RuntimeError('Online source changed during profiling')
        cmd=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
            '--force-overwrite=true','--output='+str(report),str(PYTHON),'-m','benchmarks.session',
            '--policy',policy,'--controller-json',json.dumps(cfg),'--track','default','--profile',
            '--seeds','919','--scenarios','interference','--output',str(dest)]
        print('PROFILE_START',policy,flush=True)
        with (directory/'capture.log').open('w') as log:
            code=subprocess.call(cmd,cwd=ROOT,env=clean_env('stock'),stdout=log,stderr=subprocess.STDOUT)
        r=dict(policy=policy,command=cmd,returncode=code,frozen=frozen,classification='diagnostic only; outside performance matrices')
        if code==0:
            with (directory/'export.log').open('w') as log:
                r['export_returncode']=subprocess.call(['/usr/local/cuda/bin/nsys','export','--type=sqlite',
                    '--force-overwrite=true','--output='+str(report)+'.sqlite',str(report)+'.nsys-rep'],stdout=log,stderr=subprocess.STDOUT)
            _,audit=audit_session(dest);r['audit_errors']=audit['errors']
            if r['export_returncode']==0:
                r['gpu']=extract(report.with_suffix('.sqlite'),'node',policy)
            steps={};completed=[]
            for line in (dest/'scheduler.jsonl').open():
                item=json.loads(line)
                if item['type']=='step':steps[item['step']]=item
                else:completed.append(item)
            first=min(sid for sid,s in steps.items() if any(rid.startswith('cmpl-interference') for rid in s['allocations']))
            window=[steps[c['step']] for c in completed if c['step']>=first][:128]
            r['completed_work']=dict(callbacks=len(window),prompt_tokens=sum(s['prompt'] for s in window),
                replay_tokens=sum(s['replay'] for s in window),decode_tokens=sum(s['decode_tokens'] for s in window),
                mixed_steps=sum(bool(s['prompt']) and bool(s['decode_tokens']) for s in window),
                note='Logical first 128 completion callbacks; profiler can include edge work and changes execution timing.')
            r['artifact_hashes']={str(p):digest(p) for p in [report.with_suffix('.nsys-rep'),report.with_suffix('.sqlite'),dest/'scheduler.jsonl'] if p.exists()}
        r['complete']=code==0 and r.get('export_returncode')==0 and not r.get('audit_errors') and r.get('gpu',{}).get('exposed_kernel_ms',{}).get('count',0)>0
        dump(saved,r);receipt.append(r);dump(root/'receipt.json',receipt)
        print('PROFILE_COMPLETE',policy,r['complete'],flush=True)
        if not r['complete']:raise SystemExit(1)
    dump(ROOT/'reports/optimization_v2/profile_summary.json',dict(captures=receipt,
        interpretation='Node trace includes graph kernels. Function-name categories are heuristic; activity sums are not wall time, utilization, or a causal speedup estimate. Different completed work is recorded explicitly.'))

if __name__=='__main__':main()
