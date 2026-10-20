"""Bounded, post-warmup CUDA profiler capture. Profile results are diagnostic."""
import json
import fcntl
from pathlib import Path
import os
import subprocess
import sqlite3
from apb.common import DATA,ROOT,PYTHON,clean_env,dump

def main():
    locks=[]
    for path in [DATA/"gpu.lock",Path("/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock"),Path("/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock")]:
        f=path.open("a");fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    results=[]
    # Observe preserves stock decisions while providing the bounded CUDA trigger.
    # Include the independently selected balanced static comparator as well.
    for policy in ['observe','fixed512','fixed2048','joint','aging']:
        dest=DATA/'raw/profiles'/policy;report=DATA/'profiles'/policy
        cmd=['/usr/local/cuda/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
            '--output='+str(report),str(PYTHON),'-m','benchmarks.session','--policy',policy,'--track','default',
            '--profile','--seeds','919','--scenarios','interference','--output',str(dest)]
        with (DATA/'logs'/f'profile-{policy}.log').open('w') as log:
            code=subprocess.call(cmd,cwd=ROOT,env=clean_env('stock'),stdout=log,stderr=subprocess.STDOUT)
        record={'policy':policy,'returncode':code,'command':cmd,'report_exists':report.with_suffix('.nsys-rep').exists()}
        if record['report_exists']:
            with (DATA/'profiles'/f'{policy}-stats.txt').open('w') as f:
                record['stats_returncode']=subprocess.call(['/usr/local/cuda/bin/nsys','stats','--report','cuda_gpu_kern_sum,cuda_api_sum',str(report)+'.nsys-rep'],stdout=f,stderr=subprocess.STDOUT)
            db=report.with_suffix('.sqlite')
            if db.exists():
                try:
                    with sqlite3.connect(db) as connection:
                        n,total,maximum=connection.execute('SELECT COUNT(*), SUM(end-start), MAX(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
                        record.update(kernel_count=n,kernel_duration_sum_ms=total/1e6 if total else 0.,kernel_max_ms=maximum/1e6 if maximum else 0.)
                except sqlite3.Error as exc:record['sqlite_error']=str(exc)
        results.append(record);dump(DATA/'profiles/receipt.json',results)
    print(json.dumps(results))

if __name__=='__main__':main()
