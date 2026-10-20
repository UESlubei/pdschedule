"""Compact read-only status for the long single-GPU experiment."""
import json
from pathlib import Path
import time
from apb.common import DATA

def tail(path,size=1500):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-size));return f.read().decode(errors='replace')

def main():
    status=json.loads((DATA/'status.json').read_text()) if (DATA/'status.json').exists() else {}
    active=[(k,v) for k,v in status.items() if v['state']=='running']
    failed=[(k,v) for k,v in status.items() if v['state']=='failed']
    counts={stage:len(list((DATA/'raw'/stage).glob('*/completed.json'))) for stage in ['formal','mechanism','starvation','native','native2048','ablation','compatibility','idle','balanced','profiles']}
    out=dict(completed_sessions=counts,active=[k for k,v in active],failures=failed,driver_completed=(DATA/'driver-completed.json').exists())
    if active:
        k,v=active[-1];log=DATA/'raw'/k/'server.log'
        out['active_seconds']=round(time.time()-v['started'])
        lines=[line for line in tail(log).splitlines() if 'Avg prompt throughput' in line]
        out['engine_progress']=lines[-1].split('Engine 000: ')[-1] if lines else '正在加载/预热或关闭服务'
    out['driver_tail']=tail(DATA/'logs/complete-project.log',800).splitlines()[-2:]
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
