"""Separate, resumable optimization experiment; never rewrites the first matrix."""
import argparse
import fcntl
import json
from pathlib import Path
import random
import subprocess
import threading
import time
from apb.common import DATA,ROOT,dump,digest
from benchmarks.pipeline import run_job

POLICIES={
    'stock':{},
    'balanced':{'mode':'joint','minimum':2048},
    'fixed1536':{'fixed_idle_relax':True},
    'equalized':{'cap':2048},
    'equalized256':{'mode':'equalized','cap':2048,'chunk_alignment':256},
    'fixed1792':{'fixed_idle_relax':True},
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='opt_screen')
    ap.add_argument('--seeds',nargs='+',type=int,default=[227])
    ap.add_argument('--policies',nargs='+',default=list(POLICIES))
    ap.add_argument('--scenarios',nargs='+',default=['interference','transition','mixed'])
    ap.add_argument('--requests',type=int,default=200)
    args=ap.parse_args()
    assert DATA.name.endswith('_opt2'),'Use APB_DATA for the separate optimization run'
    locks=[]
    for path in [DATA/'gpu.lock',DATA.with_name('adaptive_prefill_budget_v0320')/'gpu.lock',
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    jobs=[]
    rate=json.loads((DATA/'manifests/load-selection.json').read_text())['formal_mixed_rate']
    for seed in args.seeds:
        order=list(args.policies);random.Random(seed).shuffle(order)
        for policy in order:
            jobs.append(dict(policy=policy,track='default',seeds=[seed],scenarios=args.scenarios,
                suffix=f's{seed}',controller=POLICIES[policy],rate=rate,requests=args.requests))
    protocol=dict(args=vars(args),created_unix=time.time(),jobs=jobs,
        goals={'prefill95_change_vs_stock_max_pct':10,'throughput_change_vs_stock_min_pct':-5,
               'primary':'Reduce Decode cohort P99 versus balanced while retaining the stock guardrails.',
               'mixed':'All-request token-gap P99; finite offered-load throughput, not serving capacity.'},
        frozen={str(p):digest(p) for p in [ROOT/'apb/controller.py',ROOT/'apb/runtime.py',ROOT/'apb/common.py',
            ROOT/'benchmarks/session.py',ROOT/'benchmarks/optimize.py',ROOT/'reports/optimization_v2/cost_fit.json']})
    plan=DATA/'manifests'/f'{args.stage}-protocol.json'
    if plan.exists():
        old=json.loads(plan.read_text())
        if old['jobs']!=jobs or old['frozen']!=protocol['frozen']:raise RuntimeError('Cannot change a registered experiment')
    else:dump(plan,protocol)
    snapshot=DATA/'snapshots'/args.stage
    import shutil
    for name in protocol['frozen']:
        source=Path(name);target=snapshot/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
    stop=threading.Event()
    def sample():
        with (DATA/'logs'/f'{args.stage}-hardware.jsonl').open('a') as f:
            while not stop.is_set():
                text=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                f.write(json.dumps({'time':time.time(),'sample':text})+'\n');f.flush();stop.wait(2)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    status={}
    try:
        for job in jobs:
            for p,sha in protocol['frozen'].items():
                if digest(p)!=sha:raise RuntimeError('Frozen experiment source changed: '+p)
            key=job['policy']+'-'+job['suffix'];status[key]={'state':'running','started':time.time()}
            dump(DATA/f'{args.stage}-status.json',status)
            run_job(args.stage,job)
            status[key].update(state='completed',ended=time.time());dump(DATA/f'{args.stage}-status.json',status)
    finally:
        stop.set();monitor.join(5)

if __name__=='__main__':main()
