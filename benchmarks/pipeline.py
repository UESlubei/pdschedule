"""Resumable single GPU pipeline; explicit stage manifests and failure records."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from apb.common import DATA,ROOT,PYTHON,dump,clean_env,digest
from benchmarks.analyze import audit_session,calibration

def specs(stage):
    calibration_scenarios=[f'calib-{n}-{c}' for c in [512,2048,4096] for n in [1,2,4,8]]
    if stage=='pilot':return [
        dict(policy='observe',track='eager',seeds=[823],scenarios=calibration_scenarios+['correctness','interference','transition']),
        dict(policy='load',track='eager',seeds=[823],scenarios=['correctness','interference','transition'],cancel=True),
        dict(policy='fixed512',track='eager',seeds=[823],scenarios=['interference','transition']),
        dict(policy='fixed4096',track='eager',seeds=[823],scenarios=['interference','transition']),
        dict(policy='joint',track='eager',seeds=[823],scenarios=['interference','transition'])]
    if stage=='default_pilot':return [
        dict(policy='observe',track='default',seeds=[827],scenarios=calibration_scenarios+['correctness','interference']),
        dict(policy='stock',track='default',seeds=[827],scenarios=['correctness','interference']),
        dict(policy='joint',track='default',seeds=[827],scenarios=['correctness','interference'],cancel=True)]
    if stage=='calibration_extension':
        return [dict(policy='observe',track=t,seeds=[839],scenarios=[f'calib-16-{c}' for c in [512,2048,4096]]+[f'calib-{n}-6144' for n in [1,2,4,8,16]]) for t in ['eager','default']]
    if stage=='validation':
        return [dict(policy=p,track='default',seeds=[829],scenarios=['interference'],requests=200)
            for p in ['stock','fixed256','fixed512','fixed1024','fixed2048','fixed4096']]
    if stage=='idle':
        return [dict(policy=p,track='default',seeds=[947],scenarios=['prefill']) for p in ['stock','fixed512','joint']]
    if stage=='balanced':
        # Exploratory engineering correction after the original confirmatory
        # matrix was frozen. New seeds, unchanged controller, separate tables.
        rate=json.loads((DATA/'manifests/load-selection.json').read_text())['formal_mixed_rate']
        jobs=[]
        import random
        for seed in [151,163,179,191,211]:
            order=['stock','fixed2048','joint'];random.Random(seed).shuffle(order)
            for p in order:
                jobs.append(dict(policy=p,track='default',seeds=[seed],
                    scenarios=['interference','transition','mixed'],suffix=f'min2048-s{seed}',
                    controller={'minimum':2048} if p=='joint' else {},rate=rate))
        return jobs
    if stage=='load_calibration':
        return [dict(policy='stock',track='default',seeds=[853],scenarios=['mixed'],requests=200,rate=4.)]
    if stage=='formal':
        import random
        jobs=[]
        policies=['stock','observe','fixed256','fixed512','fixed1024','fixed2048','fixed4096','load','aimd','joint','aging']
        for seed in [17,29,43,71,101]:
            order=list(policies);random.Random(seed).shuffle(order)
            for p in order:jobs.append(dict(policy=p,track='default',seeds=[seed],scenarios=['interference','transition','mixed'],suffix=f'seed{seed}',rate=json.loads((DATA/'manifests/load-selection.json').read_text())['formal_mixed_rate']))
        return jobs
    if stage=='mechanism':
        return [dict(policy=p,track='eager',seeds=[seed],scenarios=['interference'],suffix=f'seed{seed}')
            for seed in [17,29,43,71,101] for p in ['stock','fixed512','joint']]
    if stage=='starvation':
        return [dict(policy=p,track='default',seeds=[seed],scenarios=['starvation'],suffix=f'seed{seed}')
            for seed in [17,29,43,71,101] for p in ['fixed512','joint','aging']]
    if stage=='compatibility':
        return [dict(policy=p,track=t,seeds=[907],scenarios=['correctness','cache'],cancel=True,prefix_cache=True)
            for t in ['eager','default'] for p in ['stock','off','fixed512','aging']]+[dict(policy='aging',track=t,seeds=[907],scenarios=['pressure'],blocks=640,relaxed_admission=True,suffix='pressure') for t in ['eager','default']]
    if stage=='native':
        return [dict(policy='stock',track='default',seeds=[seed],scenarios=['interference','mixed'],threshold=threshold,
            threshold_adaptive=adaptive,suffix=f'th{threshold}-a{int(adaptive)}-s{seed}',rate=json.loads((DATA/'manifests/load-selection.json').read_text())['formal_mixed_rate'])
            for seed in [17,29,43,71,101] for threshold,adaptive in [(512,False),(1024,False),(1024,True)]]
    if stage=='native2048':
        return [dict(policy='stock',track='default',seeds=[seed],scenarios=['interference','mixed'],threshold=2048,
            suffix=f'th2048-a0-s{seed}',rate=json.loads((DATA/'manifests/load-selection.json').read_text())['formal_mixed_rate'])
            for seed in [17,29,43,71,101]]
    if stage=='ablation':
        settings=[('reserve-off','fixed4096',{'reserve':False}),('idle-relax','fixed512',{'fixed_idle_relax':True}),('aging-every-step','aging',{'rescue_interval':1}),('ema01','joint',{'alpha':.1}),
            ('ema04','joint',{'alpha':.4}),('k1','joint',{'interval':1}),('k4','joint',{'interval':4}),
            ('k16','joint',{'interval':16}),('min512','joint',{'minimum':512}),('min1024','joint',{'minimum':1024}),
            ('rho01','joint',{'margin':.1}),('rho03','joint',{'margin':.3})]
        return [dict(policy=p,track='default',seeds=[seed],scenarios=['interference'],controller=cfg,suffix=f'{name}-s{seed}')
            for seed in [17,29,43] for name,p,cfg in settings]
    raise ValueError(stage)

def run_job(stage,job):
    key=f"{stage}/{job['track']}-{job['policy']}"+(('-'+job['suffix']) if job.get('suffix') else '');dest=DATA/'raw'/key
    if (dest/'completed.json').exists():
        trials,receipt=audit_session(dest)
        if receipt['errors']:raise RuntimeError('existing run failed audit '+key)
        return
    cmd=[str(PYTHON),'-m','benchmarks.session','--policy',job['policy'],'--track',job['track'],'--output',str(dest),
        '--seeds',*map(str,job['seeds']),'--scenarios',*job['scenarios']]
    for opt in ['requests','rate','blocks','threshold']:
        if opt in job:cmd+=['--'+opt.replace('_','-'),str(job[opt])]
    if 'controller' in job:cmd+=['--controller-json',json.dumps(job['controller'])]
    for opt in ['cancel','prefix_cache','threshold_adaptive','relaxed_admission']:
        if job.get(opt):cmd+=['--'+opt.replace('_','-')]
    log=DATA/'logs'/(f'{stage}-{job["track"]}-{job["policy"]}'+(('-'+job['suffix']) if job.get('suffix') else '')+'.log')
    if stage in {'formal','mechanism','starvation','native','native2048','ablation','compatibility','balanced'}:
        frozen=json.loads((DATA/'manifests/formal-source.json').read_text())['hashes']
        for path,sha in frozen.items():
            if digest(path)!=sha:raise RuntimeError('frozen implementation changed: '+path)
    print('SESSION_START',key,flush=True)
    with log.open('w') as f:
        code=subprocess.call(cmd,cwd=ROOT,env=clean_env('stock'),stdout=f,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'{key} exited {code}: {log}')
    trials,receipt=audit_session(dest)
    if receipt['errors']:raise RuntimeError(f'{key} audit: {receipt["errors"]}')
    if job['policy']=='observe' and any(x.startswith('calib-') for x in job['scenarios']):calibration(dest,job['track'])
    print('SESSION_COMPLETE',key,len(trials),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['pilot']);args=ap.parse_args()
    locks=[]
    for path in [DATA/'gpu.lock',Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if gpu.strip():raise RuntimeError('GPU occupied '+gpu)
    status={}
    for stage in args.stages:
        jobs=specs(stage);dump(DATA/'manifests'/f'{stage}-plan.json',jobs)
        for job in jobs:
            key=f'{stage}/{job["track"]}-{job["policy"]}'+(('-'+job['suffix']) if job.get('suffix') else '');status[key]={'state':'running','started':time.time()};dump(DATA/'status.json',status)
            try:run_job(stage,job)
            except Exception as exc:
                status[key].update(state='failed',error=repr(exc));dump(DATA/'status.json',status);raise
            status[key].update(state='completed',ended=time.time());dump(DATA/'status.json',status)

if __name__=='__main__':main()
