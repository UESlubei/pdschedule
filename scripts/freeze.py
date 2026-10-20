"""Freeze implementation/target tables before confirmatory measurements."""
from pathlib import Path
import shutil
import json
import subprocess
from apb.common import ROOT,DATA,PYTHON,clean_env,digest,dump

def files():
    return sorted([*(ROOT/'apb').glob('*.py'),ROOT/'benchmarks/session.py',ROOT/'scripts/prepare.py',
        ROOT/'patches/v0.32.0/scheduler.patch',DATA/'manifests/static-selection.json',DATA/'manifests/load-selection.json',*(ROOT/'configs').glob('*.json'),*(DATA/'calibration').glob('*.json')])

def main():
    if not (DATA/'manifests/static-selection.json').exists():
        from benchmarks.select_static import main as select_static
        select_static()
    if not (DATA/'manifests/load-selection.json').exists():
        subprocess.run([str(PYTHON),'-m','benchmarks.pipeline','--stages','load_calibration'],cwd=ROOT,env=clean_env('stock'),check=True)
        trial=DATA/'raw/load_calibration/default-stock/mixed-seed853'
        m=json.loads((trial/'metrics.json').read_text())
        rate=round(min(4.,.8*m['requests']/m['wall_s']),3)
        dump(DATA/'manifests/load-selection.json',{'pilot_rate':4.,'pilot_requests':m['requests'],'pilot_wall_s':m['wall_s'],'formal_mixed_rate':rate,'rule':'80% of finite-trace achieved RPS, capped at 4; conservative offered rate, not a maximum capacity estimate','validation_seed':853})
        print('FORMAL_MIXED_RATE',rate,flush=True)
    rows={str(p):digest(p) for p in files()}
    existing=DATA/'manifests/formal-source.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old['hashes']!=rows:raise RuntimeError('Refusing to overwrite frozen measurement sources. Use a new APB_DATA directory.')
        print('FROZEN_UNCHANGED',len(rows));return
    dest=DATA/'snapshots/formal-source';dest.mkdir(parents=True,exist_ok=True)
    for p in files():
        rel=p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.relative_to(DATA)
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    dump(DATA/'manifests/formal-source.json',{'hashes':rows,'snapshot':str(dest)})
    print('FROZEN',len(rows))

if __name__=='__main__':main()
