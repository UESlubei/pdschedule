"""Idempotent detached launcher for the local optimization pipeline."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('APB_DATA','/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2'))

def main():
    (DATA/'logs').mkdir(parents=True,exist_ok=True)
    lock=(DATA/'launch-master.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    pidfile=DATA/'recovery-master-pid.json'
    if pidfile.exists():
        pid=json.loads(pidfile.read_text())['pid']
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'benchmarks.recovery_master' in command and Path(f'/proc/{pid}/cwd').resolve()==ROOT:
                print('ALREADY_RUNNING',pid);return
        except FileNotFoundError:pass
    env=os.environ.copy();env.update(APB_DATA=str(DATA),PYTHONPATH=str(ROOT))
    with (DATA/'logs/recovery-master.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-m','benchmarks.recovery_master'],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    pidfile.write_text(json.dumps({'pid':p.pid,'launched_unix':time.time()})+'\n')
    print('DETACHED_MASTER_PID',p.pid)

if __name__=='__main__':main()
