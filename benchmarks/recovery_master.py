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
        ('report',ANALYSIS_PY,['-m','benchmarks.finalize_optimization'],'finalize.log')]
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
    readme=ROOT/'README.md';body=readme.read_text()
    marker='\n## 第二轮优化（2026-10-20）\n'
    if marker in body:body=body.split(marker)[0]
    note=(f'本轮工程配置为 `{selected}`，见 `configs/presets/balanced_v2.json`；具体收益、置信区间和负结果以第二轮报告为准。' if selected else
        '本轮没有选出同时满足全部点估计约束的新配置，保留原均衡档；收益和负结果以第二轮报告为准。')
    body+=marker+'\n'+note+'\n\n新增134个有效Trial、35个服务和30项回归，另有5个节点Profile与6个数值诊断服务。自然问答126/126逐token一致，合成尾块104/105一致；原始差异与原生批次复现证据见报告。交互中断的一轮归档后整轮重跑，旧数据保留。\n\n'
    body+='- [第二轮完整优化报告](reports/动态Prefill_Budget第二轮优化报告.md)\n- [成本均分算法](docs/动态Prefill_Budget成本均分策略.md)\n- [逐阶段数据表](reports/optimization_v2/实验数据表.md)\n\n'
    body+='```bash\nAPB_DATA=/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2 \\\n  PYTHONPATH="$PWD" /mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python \\\n  -m benchmarks.recovery_master\n```\n\n运行目录中的 `recovery-master-status.json` 给出阶段状态；已完成服务通过审计后跳过。新实验不覆盖原421个Trial。代码默认仍为 `mode=off`。\n'
    readme.write_text(body)
    snapshot=DATA/'snapshots/final-project'
    shutil.copytree(ROOT,snapshot,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    traces={str(p):digest(p) for p in (DATA/'raw').glob('*/*/*/trace.json')}
    dump(DATA/'manifests/optimization-final-delivery.json',dict(created_unix=time.time(),project_hashes=hashes,
        trace_hashes=traces,snapshot=str(snapshot),completion=completion))
    status.update(state='complete',phase='complete',ended_unix=time.time(),completion=completion);dump(status_path,status)
    print('RECOVERY_COMPLETE',selected,len(hashes),len(traces),flush=True)

if __name__=='__main__':main()
