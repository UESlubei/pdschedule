"""Copy Python sources; link unmodified binaries; exact-anchor pinned patch."""
from __future__ import annotations
import difflib
import json
import shutil
from pathlib import Path
from apb.common import ROOT,DATA,INSTALLED,MODEL,dump,digest

def patch(text):
    def replace(old,new):
        nonlocal text
        old="\n"+old;new="\n"+new
        if text.count(old)!=1:raise ValueError(f'expected unique patch anchor ({text.count(old)}): {old[:100]}')
        text=text.replace(old,new)
    replace('        self._kv_fetch_counts: dict[str, int] = dict.fromkeys(KV_FETCH_STAGES, 0)',
        '        self._kv_fetch_counts: dict[str, int] = dict.fromkeys(KV_FETCH_STAGES, 0)\n\n'
        '        from apb.runtime import create as create_apb_runtime\n        self._apb = create_apb_runtime(self)')
    replace('        self.current_step += 1\n', '        self.current_step += 1\n        if self._apb is not None:\n            self._apb.begin()\n')
    replace('            request = self.running[req_index]\n','            request = self.running[req_index]\n            if self._apb is not None:\n                self._apb.enter(request)\n')
    replace('            if num_new_tokens == 0:\n                # The request cannot be scheduled because',
        '            if self._apb is not None:\n                num_new_tokens = self._apb.clip(request, request.num_computed_tokens, num_new_tokens, token_budget, input_budget)\n\n'
        '            if num_new_tokens == 0:\n                # The request cannot be scheduled because')
    replace('                            restored = num_scheduled_tokens.pop(preempted_req_id)\n',
        '                            restored = num_scheduled_tokens.pop(preempted_req_id)\n                            if self._apb is not None:\n                                self._apb.refund(preempted_req_id)\n')
    replace('            num_scheduled_tokens[request_id] = num_new_tokens\n',
        '            if self._apb is not None:\n                self._apb.charge(request, request.num_computed_tokens, num_new_tokens)\n'
        '            num_scheduled_tokens[request_id] = num_new_tokens\n')
    replace('                    num_new_tokens = min(num_new_tokens, request_token_budget)\n                    assert num_new_tokens > 0',
        '                    num_new_tokens = min(num_new_tokens, request_token_budget)\n'
        '                    if self._apb is not None:\n                        num_new_tokens = self._apb.clip(request, num_computed_tokens, num_new_tokens, token_budget, input_budget)\n'
        '                        if num_new_tokens == 0:\n                            break\n                    assert num_new_tokens > 0')
    replace('                num_scheduled_tokens[request_id] = num_new_tokens\n',
        '                if self._apb is not None:\n                    self._apb.charge(request, num_computed_tokens, num_new_tokens)\n'
        '                num_scheduled_tokens[request_id] = num_new_tokens\n')
    replace('        with record_function_or_nullcontext("schedule: update_after_schedule"):',
        '        if self._apb is not None:\n            self._apb.scheduled(scheduler_output)\n\n'
        '        with record_function_or_nullcontext("schedule: update_after_schedule"):')
    replace('        return engine_core_outputs\n',
        '        if self._apb is not None:\n            self._apb.completed(scheduler_output, engine_core_outputs)\n        return engine_core_outputs\n')
    replace('        logger.debug_once("[shutdown] Scheduler: start")',
        '        logger.debug_once("[shutdown] Scheduler: start")\n        if self._apb is not None:\n            self._apb.close()')
    compile(text,'<apb-patched-scheduler>','exec')
    return text

def main():
    reference=json.loads((ROOT/'docs/evidence/local_inventory_20261019.json').read_text())
    for rel,row in reference['source_files'].items():
        if digest(INSTALLED/rel)!=row['sha256']:raise RuntimeError('installed source changed: '+rel)
    for sub in ['overlays','raw','logs','profiles','manifests','cache','tmp']:(DATA/sub).mkdir(parents=True,exist_ok=True)
    for flavor in ['stock','candidate']:
        target=DATA/'overlays'/flavor/'vllm'
        if not target.exists():
            for p in INSTALLED.rglob('*'):
                if '__pycache__' in p.parts:continue
                q=target/p.relative_to(INSTALLED)
                if p.is_dir():q.mkdir(parents=True,exist_ok=True)
                elif p.suffix=='.py':shutil.copy2(p,q)
                else:q.symlink_to(p)
    rel='v1/core/sched/scheduler.py';original=(INSTALLED/rel).read_text();changed=patch(original)
    (DATA/'overlays/candidate/vllm'/rel).write_text(changed)
    diff=''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='a/vllm/'+rel,tofile='b/vllm/'+rel))
    patchpath=ROOT/'patches/v0.32.0/scheduler.patch';patchpath.write_text(diff)
    model_files={p.name:{'sha256':digest(p),'size':p.stat().st_size} for p in MODEL.iterdir() if p.is_file()}
    dump(DATA/'manifests/prepared.json',{'source':reference['source_files'],'patch_sha256':digest(patchpath),
        'model_files':model_files,'model_revision':MODEL.resolve().name,'model':str(MODEL),'strategy':'copied Python; read-only native binary symlinks'})
    print('PREPARED',DATA,flush=True)

if __name__=='__main__':main()
