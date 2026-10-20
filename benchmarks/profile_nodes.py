"""One explicit CUDA-graph node capture to complement the whole-graph profiles."""
import fcntl
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
from apb.common import DATA,ROOT,PYTHON,clean_env,dump
from benchmarks.analyze import audit_session


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=['joint','observe'],default='joint');args=parser.parse_args()
    locks=[]
    for path in [DATA/'gpu.lock',Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    directory=DATA/'profiles_node' if args.policy=='joint' else DATA/'profiles_node'/args.policy
    directory.mkdir(parents=True,exist_ok=True)
    dest=directory/('raw-'+args.policy);report=directory/(args.policy+'-node')
    cmd=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
         '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
         '--force-overwrite=true','--output='+str(report),str(PYTHON),'-m','benchmarks.session',
         '--policy',args.policy,'--track','default','--profile','--seeds','919',
         '--scenarios','interference','--output',str(dest)]
    with (directory/'capture.log').open('w') as log:
        code=subprocess.call(cmd,cwd=ROOT,env=clean_env('stock'),stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(command=cmd,returncode=code,granularity='node',classification='diagnostic only; outside performance matrices')
    if code==0:
        with (directory/'stats.txt').open('w') as out:
            receipt['stats_returncode']=subprocess.call(['/usr/local/cuda/bin/nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum',str(report)+'.nsys-rep'],stdout=out,stderr=subprocess.STDOUT)
        with sqlite3.connect(report.with_suffix('.sqlite')) as connection:
            count,total,maximum=connection.execute('SELECT COUNT(*), SUM(end-start), MAX(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
        receipt.update(kernel_count=count,kernel_duration_sum_ms=total/1e6,kernel_max_ms=maximum/1e6)
        rows,audit=audit_session(dest);receipt['audit_errors']=audit['errors']
    receipt['complete']=code==0 and receipt.get('kernel_count',0)>0 and receipt.get('stats_returncode')==0 and not receipt.get('audit_errors')
    dump(directory/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    if not receipt['complete']:raise SystemExit(1)


if __name__=='__main__':main()
