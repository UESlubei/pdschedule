"""Additional pre-registered equal-base comparison on three untouched seeds."""
import fcntl
import json
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time
from apb.common import DATA,ROOT,dump,digest
from benchmarks.pipeline import run_job

def main():
    stage='opt_match1536'
    policies={'stock':{},'fixed1536':{'fixed_idle_relax':True},
        'equalized1536':{'mode':'equalized','cap':1536,'chunk_alignment':512}}
    jobs=[]
    rate=json.loads((DATA/'manifests/load-selection.json').read_text())['formal_mixed_rate']
    for seed in [401,409,419]:
        order=list(policies);random.Random(seed).shuffle(order)
        for policy in order:
            jobs.append(dict(policy=policy,track='default',seeds=[seed],
                scenarios=['correctness','interference','transition','mixed'],suffix=f's{seed}',
                controller=policies[policy],rate=rate,requests=200))
    frozen={str(p):digest(p) for p in [ROOT/'apb/controller.py',ROOT/'apb/runtime.py',ROOT/'apb/common.py',
        ROOT/'benchmarks/session.py',Path(__file__).resolve(),ROOT/'reports/optimization_v2/cost_fit.json']}
    protocol=dict(created_unix=time.time(),jobs=jobs,frozen=frozen,
        reason='The new fixed1536 comparator was stronger in the first completed validation seed. Test equalization at the same base, on untouched seeds, without modifying the ongoing five-seed matrix.',
        goals={'primary':'Decode P99 versus fresh fixed1536 on interference/transition',
               'guardrails':'Prefill P95 <=+10%, throughput >=-5% versus fresh stock in all scenes',
               'secondary':'Mixed P99; report all negative results. Three repeats, engineering evidence.'})
    plan=DATA/'manifests'/f'{stage}-protocol.json'
    if plan.exists():
        old=json.loads(plan.read_text())
        if old['jobs']!=jobs or old['frozen']!=frozen:raise RuntimeError('Cannot change the registered equal-base experiment')
    else:dump(plan,protocol)
    snapshot=DATA/'snapshots'/stage
    for absolute in frozen:
        source=Path(absolute);target=snapshot/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
    locks=[]
    for p in [DATA/'gpu.lock',DATA.with_name('adaptive_prefill_budget_v0320')/'gpu.lock',
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    stop=threading.Event()
    def sample():
        with (DATA/'logs'/f'{stage}-hardware.jsonl').open('a') as f:
            while not stop.is_set():
                s=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                f.write(json.dumps({'time':time.time(),'sample':s})+'\n');f.flush();stop.wait(2)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start();status={}
    try:
        for job in jobs:
            for p,sha in frozen.items():
                if digest(p)!=sha:raise RuntimeError('Frozen equal-base source changed: '+p)
            key=job['policy']+'-'+job['suffix'];status[key]={'state':'running','started':time.time()}
            dump(DATA/f'{stage}-status.json',status);run_job(stage,job)
            status[key].update(state='completed',ended=time.time());dump(DATA/f'{stage}-status.json',status)
        dump(DATA/'manifests/match1536-completed.json',{'sessions':len(jobs),'trials':len(jobs)*4})
    finally:stop.set();monitor.join(5)

if __name__=='__main__':main()
