"""Quality, chunk fragmentation, CPU cost, and source provenance for round two."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import time
import subprocess
from apb.common import DATA,ROOT,MODEL,INSTALLED,PYTHON,dump,digest,stats,clean_env
from scripts.prepare import patch
from benchmarks.analyze import audit_session

OUT=ROOT/'reports/optimization_v2'

def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',action='store_true');args=ap.parse_args()
    sessions=[p.parent for p in sorted((DATA/'raw').glob('*/*/completed.json'))]
    quality=[];cpu=[];fragmentation=[];ledgers=[]
    for dest in sessions:
        manifest=read(dest/'manifest.json');policy=manifest['args']['policy'];stage=dest.parent.name
        summary=dest/'scheduler.summary.json'
        if summary.exists():
            s=read(summary);cpu.append(dict(session=str(dest),stage=stage,policy=policy,
                **{k:s[k] for k in ['scheduler_us','decision_us','feedback_us','pending_steps','active_tracked','log_dropped']}))
        windows=[]
        for trial in sorted(dest.glob('*/metrics.json')):
            row=read(trial)
            if row['scenario']=='correctness' and policy!='stock':
                base=next((d for d in sessions if d.parent==dest.parent and read(d/'manifest.json')['args']['policy']=='stock'
                    and (d/trial.parent.name/'requests.json').exists()),None)
                if base is not None:
                    reference={r['id']:r for r in read(base/trial.parent.name/'requests.json')}
                    comparisons=[]
                    for r in read(trial.parent/'requests.json'):
                        b=reference[r['id']];exact=r['token_ids']==b['token_ids']
                        first=next((i for i,(a,z) in enumerate(zip(r['token_ids'],b['token_ids'])) if a!=z),None)
                        comparisons.append(dict(id=r['id'],cohort=r['cohort'],exact=exact,first_mismatch_1based=None if first is None else first+1,
                            output_length=len(r['token_ids']),baseline_length=len(b['token_ids']),
                            text=r['text'] if not exact else None,baseline_text=b['text'] if not exact else None))
                    quality.append(dict(stage=stage,policy=policy,seed=row['seed'],session=str(dest),baseline=str(base),comparisons=comparisons))
            if row['scenario'] in ['interference','transition','mixed']:
                windows.append((row,Counter(),Counter()))
        ledger=dest/'scheduler.jsonl'
        if not ledger.exists():continue
        ledgers.append(dict(path=str(ledger),sha256=digest(ledger)))
        for line in ledger.open():
            r=json.loads(line)
            if r['type']!='step' or not r['prompt']+r['replay']:continue
            for row,counts,shapes in windows:
                if not row['started_ns']<=r['scheduled_ns']<=row['ended_ns']:continue
                n=sum(bool(p+replay) for p,replay,d,pos,total in r['allocations'].values())
                counts['prefill_steps']+=1;counts['multi_prefill_request_steps']+=n>1
                counts['mixed_steps']+=bool(r['decode_tokens']);counts['prompt_tokens']+=r['prompt']
                shapes[r['prompt']+r['replay']]+=1
        for row,counts,shapes in windows:
            analysis=read(dest/f"{row['scenario']}-seed{row['seed']}"/'analysis.json')
            fragmentation.append(dict(stage=stage,policy=policy,seed=row['seed'],scenario=row['scenario'],
                counts=dict(counts),shapes=dict(shapes),prefill_timing=analysis['telemetry']['prefill_timing_by_cohort']))
    prepared=read(DATA/'manifests/prepared.json');checks={}
    for source in read(OUT/'cost_fit.json')['sources']:
        checks['cost-fit-input:'+source['path']]=digest(source['path'])==source['sha256']
    for dest in sessions:
        protocol=DATA/'manifests'/f'{dest.parent.name}-protocol.json'
        if not protocol.exists():continue
        manifest=read(dest/'manifest.json')
        for absolute,sha in read(protocol)['frozen'].items():
            source=Path(absolute)
            rel=str(source.relative_to(ROOT))
            if rel.startswith(('apb/','benchmarks/')) and source.suffix=='.py':
                checks['session-source:'+str(dest)+':'+rel]=manifest['source_hashes'].get(rel)==sha
    for rel,r in prepared['source'].items():
        checks['original:'+rel]=digest(r['path'])==r['sha256']
        checks['stock:'+rel]=digest(DATA/'overlays/stock/vllm'/rel)==r['sha256']
        if rel!='v1/core/sched/scheduler.py':checks['candidate:'+rel]=digest(DATA/'overlays/candidate/vllm'/rel)==r['sha256']
    rel='v1/core/sched/scheduler.py'
    checks['candidate:patched-scheduler']=(DATA/'overlays/candidate/vllm'/rel).read_text()==patch((INSTALLED/rel).read_text())
    for protocol in sorted((DATA/'manifests').glob('opt_*-protocol.json')):
        p=read(protocol)
        for absolute,sha in p['frozen'].items():
            snapshot=DATA/'snapshots'/protocol.name.removesuffix('-protocol.json')/Path(absolute).relative_to(ROOT)
            checks['snapshot:'+str(snapshot)]=digest(snapshot)==sha
            # Later stages can add policies to the runner; current online code
            # must still match every stage, while each runner snapshot is retained.
            if Path(absolute).name!='optimize.py':checks['current:'+absolute]=digest(absolute)==sha
    old=DATA.with_name('adaptive_prefill_budget_v0320')
    for absolute,sha in read(old/'manifests/formal-source.json')['hashes'].items():
        source=Path(absolute)
        archived=DATA/'baseline_source'/source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
        checks['original-round-archive:'+str(archived)]=digest(archived)==sha
    if args.weights:
        for name,r in prepared['model_files'].items():checks['model:'+name]=digest(MODEL/name)==r['sha256']
    hardware=[]
    for path in sorted((DATA/'logs').glob('opt_*-hardware.jsonl')):
        samples=[]
        for line in path.open():
            fields=json.loads(line)['sample'].split(',')
            samples.append([float(v.strip()) for v in fields[1:]])
        active=[r for r in samples if r[3]>=90]
        hardware.append(dict(stage=path.stem.removesuffix('-hardware'),samples=len(samples),active_samples=len(active),
            active={k:stats(r[i] for r in active) for i,k in enumerate(['temperature_C','power_W','sm_clock_MHz','utilization_pct','memory_MiB'])}))
    interruptions=[]
    for receipt in sorted((DATA/'interruptions').glob('*/receipt.json')):
        record=read(receipt);archived_audits=[]
        for done in sorted(receipt.parent.glob('*/completed.json')):
            _,audit=audit_session(done.parent);archived_audits.append(audit)
            checks['interrupted-archive-audit:'+str(done.parent)]=not audit['errors']
        interruptions.append(dict(receipt=str(receipt),record=record,archived_audits=archived_audits))
    with (DATA/'logs/batch-fragmentation-probe.log').open('w') as log:
        subprocess.run([str(PYTHON),'-m','benchmarks.batch_fragmentation_probe'],cwd=ROOT,env=clean_env('candidate'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
    batch_probe=read(OUT/'batch_probe.json')
    for p,sha in batch_probe['source_hashes'].items():checks['batch-probe-source:'+p]=digest(p)==sha
    numerical=None
    if any(not c['exact'] for q in quality for c in q['comparisons']):
        with (DATA/'logs/quality-probe-optimization.log').open('a') as log:
            subprocess.run([str(PYTHON),'-m','benchmarks.quality_probe_optimization'],cwd=ROOT,env=clean_env('stock'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        numerical=read(DATA/'quality_drift_opt/receipt.json');checks['numerical-diagnostics-complete']=numerical['complete']
        checks['numerical-diagnostics-source']=numerical['source_sha256']==digest(ROOT/'benchmarks/quality_probe_optimization.py')
        checks['numerical-request-integrity']=all(not r['integrity_audit_errors'] for r in numerical['runs'])
    source=dict(checked_unix=time.time(),weights_verified=args.weights,passed=all(checks.values()),checks=checks)
    dump(DATA/'manifests/optimization-source-verification.json',source)
    dump(OUT/'diagnostics.json',dict(quality=quality,cpu=cpu,fragmentation=fragmentation,hardware=hardware,source=source,ledger_hashes=ledgers,interruptions=interruptions,batch_probe=batch_probe,numerical_probe=numerical))
    dump(OUT/'interruptions.json',interruptions)
    print(json.dumps(dict(sessions=len(sessions),quality_trials=len(quality),quality_exact=sum(c['exact'] for q in quality for c in q['comparisons']),
        quality_cases=sum(len(q['comparisons']) for q in quality),source_passed=source['passed'],checks=len(checks))))
    if not source['passed']:raise SystemExit(1)

if __name__=='__main__':main()
