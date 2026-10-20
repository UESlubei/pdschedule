"""Additional reproducible evidence, separate from the frozen online implementation."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from apb.common import DATA, ROOT, dump, stats, digest


def read(path):
    return json.loads(path.read_text())


def main():
    rows = read(ROOT / 'reports/results.json')
    quality, cache, cancellation, cpu, pressure, starvation = [], [], [], [], [], []
    sessions = sorted({Path(r['session']) for r in rows})
    for dest in sessions:
        manifest = read(dest / 'manifest.json')
        summary = dest / 'scheduler.summary.json'
        if summary.exists():
            s = read(summary)
            cpu.append(dict(session=str(dest), stage=dest.parent.name,
                            policy=manifest['args']['policy'], track=manifest['args']['track'],
                            **{k:s[k] for k in ['scheduler_us','decision_us','feedback_us',
                                              'pending_steps','active_tracked','log_dropped']}))
        if (dest / 'cancellation.json').exists():
            cancellation.append(dict(session=str(dest), **read(dest / 'cancellation.json')))
    for r in rows:
        trial = Path(r['session']) / f"{r['scenario']}-seed{r['seed']}"
        if r['scenario'] == 'correctness':
            group = trial.parent.parent
            baseline = next((p for p in group.glob('*')
                             if (p/'manifest.json').exists()
                             and read(p/'manifest.json')['args']['policy']=='stock'
                             and read(p/'manifest.json')['args']['track']==r['track']
                             and (p/trial.name/'requests.json').exists()), None)
            if baseline is not None:
                ref = {q['id']:q for q in read(baseline/trial.name/'requests.json')}
                reqs = read(trial/'requests.json')
                comparisons = []
                for q in reqs:
                    b = ref[q['id']]
                    first = next((i for i,(a,z) in enumerate(zip(q['token_ids'],b['token_ids'])) if a!=z), None)
                    comparisons.append(dict(id=q['id'],cohort=q['cohort'],success=q['success'],
                        exact=q['token_ids']==b['token_ids'],first_mismatch=first,
                        expected_length=b['output'],actual_length=len(q['token_ids']),
                        text=q['text'],baseline_text=b['text']))
                quality.append(dict(session=r['session'],baseline=str(baseline),track=r['track'],
                    policy=r['policy'],seed=r['seed'],comparisons=comparisons))
        if r['scenario']=='cache':
            reqs = read(trial/'requests.json')
            cache.append(dict(session=r['session'],track=r['track'],policy=r['policy'],
                hits_from_server_counter=r.get('prefix_cache_hits_total_delta'),queries_from_server_counter=r.get('prefix_cache_queries_total_delta'),
                actual_prompt_compute=r['telemetry'].get('actual_prompt_tokens'),first_compute_offsets=r['telemetry'].get('first_compute_offset_by_request',{}),
                requests=[dict(id=q['id'],input=q['input'],
                    cached_tokens=(q.get('usage',{}).get('prompt_tokens_details') or {}).get('cached_tokens'),
                    success=q['success']) for q in reqs]))
        if r['scenario']=='pressure':
            counts=defaultdict(int)
            for line in (Path(r['session'])/'scheduler.jsonl').open():
                s=json.loads(line)
                if s['type']=='step' and r['started_ns']<=s['scheduled_ns']<=r['ended_ns']:
                    for k in ['prompt','replay','decode_tokens','preemptions']:counts[k]+=s[k]
            pressure.append(dict(session=r['session'],track=r['track'],
                                 native_preemptions=r['native_preemptions'],work=dict(counts)))
        if r['stage']=='starvation':
            reqs=read(trial/'requests.json');bins=defaultdict(list)
            for q in reqs:bins[int(q['offset']//30)*30].append(q)
            waiting=[]
            for line in (Path(r['session'])/'scheduler.jsonl').open():
                s=json.loads(line)
                if s['type']=='step' and r['started_ns']<=s['scheduled_ns']<=r['ended_ns']:
                    waiting.append(((s['scheduled_ns']-r['started_ns'])/1e9,s['waiting']))
            weighted={}
            for a,b in [(0,60),(120,180)]:
                numerator=denominator=0.
                for (t,n),(u,_) in zip(waiting,waiting[1:]):
                    dt=max(0.,min(u,b)-max(t,a));numerator+=n*dt;denominator+=dt
                weighted[f'{a}-{b}s']=numerator/denominator if denominator else None
            starvation.append(dict(session=r['session'],policy=r['policy'],seed=r['seed'],
                requests=r['requests'],completed=r['completed'],errors=r['errors'],
                wall_s=r['wall_s'],last_injection_s=max(q['offset'] for q in reqs),
                drain_s=r['wall_s']-max(q['offset'] for q in reqs),
                max_waiting=max((n for t,n in waiting),default=None),time_weighted_waiting=weighted,
                arrival_bins=[dict(start_s=k,count=len(v),ttft_ms=stats(q['ttft_ms'] for q in v),
                    long_prefill_ttft_ms=stats(q['ttft_ms'] for q in v if q['input']==6144))
                    for k,v in sorted(bins.items())],telemetry=r['telemetry']))
    hardware=[]
    hardware_path=DATA/'logs/hardware.csv'
    if hardware_path.exists():
        for row in csv.reader(hardware_path.open()):
            if len(row)!=7:continue
            try:hardware.append([row[0].strip(), *[float(v) for v in row[1:]]])
            except ValueError:continue
    h=dict(samples=len(hardware),first=hardware[0][0] if hardware else None,last=hardware[-1][0] if hardware else None,
           scope='Sampling began after the first formal seed; clocks were not locked.',
           **{name:stats(row[i] for row in hardware) for i,name in enumerate(
               ['temperature_C','power_W','sm_clock_MHz','memory_clock_MHz','utilization_pct','memory_MiB'],1)})
    active_hardware=[row for row in hardware if row[5]>=90]
    h['active_gpu_samples']=len(active_hardware)
    h['active_gpu_statistics']={name:dict(min=min((row[i] for row in active_hardware),default=None),
        **stats(row[i] for row in active_hardware)) for i,name in enumerate(
        ['temperature_C','power_W','sm_clock_MHz','memory_clock_MHz','utilization_pct','memory_MiB'],1)}
    profile=DATA/'profiles/receipt.json'
    numeric_path=DATA/'quality_drift/receipt.json'
    numerical=read(numeric_path) if numeric_path.exists() else None
    if numerical and numerical.get('complete'):
        base=DATA/'raw/compatibility/default-stock/correctness-seed907'
        original={r['id']:r for r in read(base/'requests.json')}
        paired_runs=[];drift=[];hashes={str(base/'trace.json'):digest(base/'trace.json')}
        for repeat in [1,2]:
            request_sets={}
            for policy in ['stock','off']:
                path=DATA/'quality_drift'/f'{policy}-repeat{repeat}'/'correctness-seed907'
                request_sets[policy]={r['id']:r for r in read(path/'requests.json')}
                hashes[str(path/'trace.json')]=digest(path/'trace.json')
                for rid,r in request_sets[policy].items():
                    ref=original[rid]
                    if r['token_ids']!=ref['token_ids']:
                        first=next(i for i,(a,b) in enumerate(zip(ref['token_ids'],r['token_ids'])) if a!=b)
                        drift.append(dict(policy=policy,repeat=repeat,id=rid,first_difference_1based=first+1,
                            original_token=ref['token_ids'][first],new_token=r['token_ids'][first]))
            paired_runs.append(dict(repeat=repeat,exact=sum(request_sets['stock'][rid]['token_ids']==r['token_ids'] for rid,r in request_sets['off'].items()),total=len(request_sets['off'])))
        numerical['assessment']=dict(stock_off_pairs=paired_runs,drift_vs_original=drift,
            input_trace_hashes=hashes,all_input_traces_equal=len(set(hashes.values()))==1)
    dump(ROOT/'reports/diagnostics.json',dict(quality=quality,cache=cache,cancellation=cancellation,
         cpu=cpu,pressure=pressure,starvation=starvation,hardware=h,
         profiles=read(profile) if profile.exists() else [],
         numerical_probe=numerical,
         node_profile=read(DATA/'profiles_node/receipt.json') if (DATA/'profiles_node/receipt.json').exists() else None,
         observe_node_profile=read(DATA/'profiles_node/observe/receipt.json') if (DATA/'profiles_node/observe/receipt.json').exists() else None,
         analysis_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/'benchmarks').glob('*.py'))}))
    print('DIAGNOSTICS',len(quality),len(starvation),len(hardware))


if __name__=='__main__':main()
