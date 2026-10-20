"""Full experiment driver with source freeze, resumability and raw-data audits."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from apb.common import DATA,ROOT,PYTHON,clean_env,dump

def call(module,args=()):
    cmd=[str(PYTHON),'-m',module,*args]
    print('DRIVER_START',module,list(args),flush=True)
    code=subprocess.call(cmd,cwd=ROOT,env=clean_env('stock'))
    if code:raise RuntimeError(f'{module} failed ({code})')

def main():
    # Pipeline waits for the preceding pilot's GPU locks, never competes with it.
    call('benchmarks.pipeline',['--stages','calibration_extension','validation'])
    # Verify both paths and all extended target bins are present before freezing.
    for track in ['eager','default']:
        table=json.loads((DATA/'calibration'/f'decode-{track}.json').read_text())
        assert len(table['rows'])>=20
    env=clean_env('candidate')
    with (DATA/'logs/tests-frozen.log').open('w') as f:
        code=subprocess.call([str(PYTHON),'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('frozen regression tests failed')
    call('benchmarks.select_static')
    call('scripts.freeze')
    call('benchmarks.pipeline',['--stages','formal','mechanism','starvation','native','ablation','compatibility'])
    call('benchmarks.analyze')
    call('benchmarks.profile')
    call('benchmarks.analyze')
    print('EXPERIMENT_DRIVER_COMPLETE',flush=True)
    dump(DATA/'driver-completed.json',{'ended_unix':time.time()})

if __name__=='__main__':main()
