from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('APB_DATA', '/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320'))
PYTHON = Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python')
INSTALLED = PYTHON.parent.parent / 'lib/python3.11/site-packages/vllm'
MODEL = Path('/mnt/ondeviceinfer/models/Qwen3-1.7B')

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def stats(xs):
    xs = sorted(xs)
    if not xs: return dict(count=0, mean=None, p50=None, p90=None, p95=None, p99=None, max=None)
    def q(p):
        a=(len(xs)-1)*p; i=int(a)
        return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(a-i)
    return dict(count=len(xs),mean=sum(xs)/len(xs),p50=q(.5),p90=q(.9),p95=q(.95),p99=q(.99),max=xs[-1])

def clean_env(flavor='candidate'):
    env = os.environ.copy()
    for key in ['PYTHONPATH','KV_SCHED_CONFIG','KV032_POLICY','VLLM_ENABLE_V1_MULTIPROCESSING','APB_CONFIG','APB_PROFILE_PREFIX']:
        env.pop(key, None)
    for sub in ['tmp','cache','cache/torchinductor','cache/triton']: (DATA/sub).mkdir(parents=True,exist_ok=True)
    env.update(HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', PYTHONHASHSEED='17',
        TMPDIR=str(DATA/'tmp'), XDG_CACHE_HOME=str(DATA/'cache'),
        TORCHINDUCTOR_CACHE_DIR=str(DATA/'cache/torchinductor'), TRITON_CACHE_DIR=str(DATA/'cache/triton'),
        PYTHONPATH=os.pathsep.join([str(DATA/'overlays'/flavor),str(ROOT)]))
    return env
