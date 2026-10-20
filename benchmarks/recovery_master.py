"""Detached recovery runner; completes experiments and reports across chat interruptions."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from apb.common import DATA,ROOT,PYTHON,dump,digest

ANALYSIS_PY=Path('/home/sdn/anaconda3/envs/ljc/bin/python')

def main():
    mutex=(DATA/'recovery-master.lock').open('a')
    fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(pid=os.getpid(),started_unix=time.time(),state='running')
    status_path=DATA/'recovery-master-status.json';dump(status_path,status)
    env=os.environ.copy();env.update(APB_DATA=str(DATA),PYTHONPATH=str(ROOT))
    phases=[('validation',PYTHON,['-m','benchmarks.optimize','--stage','opt_validation','--seeds','307','311','313','317','331',
        '--policies','stock','balanced','fixed1536','equalized256','--scenarios','correctness','interference','transition','mixed'],'opt_validation-runner.log'),
        ('equal-base',PYTHON,['-m','benchmarks.match_budget'],'opt_match1536-runner.log'),
        ('gpu-diagnostics',PYTHON,['-m','benchmarks.profile_optimization'],'opt_profiles-runner.log'),
        ('analysis',ANALYSIS_PY,['-m','benchmarks.optimization_report'],'report-final.log'),
        ('source-quality',ANALYSIS_PY,['-m','benchmarks.optimization_diagnostics','--weights'],'diagnostics-final.log'),
        ('report',ANALYSIS_PY,['-m','benchmarks.finalize_optimization'],'finalize.log'),
        ('project-report',ANALYSIS_PY,['-m','benchmarks.project_report'],'project-report.log')]
    for name,interpreter,args,log_name in phases:
        status.update(phase=name,phase_started_unix=time.time());dump(status_path,status)
        print('RECOVERY_PHASE',name,flush=True)
        with (DATA/'logs'/log_name).open('a') as log:
            log.write('\nRECOVERY_RESUME '+str(time.time())+'\n');log.flush()
            code=subprocess.call([str(interpreter),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if code:
            status.update(state='failed',returncode=code,log=str(DATA/'logs'/log_name));dump(status_path,status)
            raise SystemExit(code)
    status.update(phase='delivery');dump(status_path,status)
    completion=json.loads((ROOT/'reports/optimization_v2/completion.json').read_text())
    selected=completion['selected_policy']
    snapshot=DATA/'snapshots/final-project'
    shutil.copytree(ROOT,snapshot,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    traces={str(p):digest(p) for p in (DATA/'raw').glob('*/*/*/trace.json')}
    dump(DATA/'manifests/optimization-final-delivery.json',dict(created_unix=time.time(),project_hashes=hashes,
        trace_hashes=traces,snapshot=str(snapshot),completion=completion))
    status.update(state='complete',phase='complete',ended_unix=time.time(),completion=completion);dump(status_path,status)
    print('RECOVERY_COMPLETE',selected,len(hashes),len(traces),flush=True)

if __name__=='__main__':main()
