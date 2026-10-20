"""Recompute all trial metrics/ledger invariants from saved raw artifacts."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from apb.common import DATA,ROOT,dump,stats

def jsonl(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)

def audit_session(dest):
    manifest=json.loads((dest/'manifest.json').read_text());mode=manifest['apb_config']['mode']
    ledger=dest/'scheduler.jsonl';steps={};complete={};errors=[]
    if dest.parent.name in {'formal','mechanism','starvation','native','native2048','ablation','compatibility','balanced'}:
        frozen=json.loads((DATA/'manifests/formal-source.json').read_text())['hashes']
        for absolute,sha in frozen.items():
            source=Path(absolute)
            if source.is_relative_to(ROOT):
                rel=str(source.relative_to(ROOT))
                if rel in manifest['source_hashes'] and manifest['source_hashes'][rel]!=sha:errors.append('frozen measurement source mismatch: '+rel)
    if ledger.exists():
        for row in jsonl(ledger):
            if row['type']=='step':
                sid=row['step'];steps[sid]=row
                if row['total']>4096:errors.append('global budget')
                if mode!='observe' and row['prompt']+row['replay']>row['cap']:errors.append('prefill budget')
                if row['prompt']+row['replay']+row['decode_tokens']!=row['total']:errors.append('partition')
            else:
                if row['step'] in complete:errors.append('duplicate completed step')
                complete[row['step']]=row
        if any(sid not in steps for sid in complete):errors.append('completed unknown step')
        nonempty={s for s,r in steps.items() if r['total']}
        if nonempty!=set(complete):errors.append('incomplete step ledger')
        summary=json.loads((dest/'scheduler.summary.json').read_text())
        if summary['log_dropped'] or summary['log_error'] or summary['pending_steps']:errors.append('telemetry loss or pending')
        if summary.get('active_tracked') or summary.get('counts',{}).get('missing_ledger'):errors.append('request state leak or unmatched completion')
    fault_path=dest/'cancellation.json'
    if fault_path.exists():
        fault=json.loads(fault_path.read_text())
        if fault.get('followup_output_tokens')!=8 or not fault.get('reset_cache',{}).get('success'):errors.append('cancellation recovery or cache reset failed')
    trials=[]
    for path in sorted(dest.glob('*/metrics.json')):
        metrics=json.loads(path.read_text());reqs=json.loads((path.parent/'requests.json').read_text())
        if metrics['errors'] or any(not r['success'] for r in reqs):errors.append(str(path)+' failed requests')
        if any(len(r['token_ids'])!=r['output'] for r in reqs if r['success']):errors.append('output length mismatch')
        def metric_total(text,name):
            return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name+'{') or line.startswith(name+' '))
        before=(path.parent/'metrics-before.txt').read_text();after=(path.parent/'metrics-after.txt').read_text()
        metrics['native_preemptions']=metric_total(after,'vllm:num_preemptions_total')-metric_total(before,'vllm:num_preemptions_total')
        for key in ['prefix_cache_hits_total','prefix_cache_queries_total']:
            metrics[key+'_delta']=metric_total(after,'vllm:'+key)-metric_total(before,'vllm:'+key)
        usage_cache_supported=any('cached_tokens' in (r.get('usage',{}).get('prompt_tokens_details') or {}) for r in reqs)
        metrics['cached_tokens_from_usage']=sum((r.get('usage',{}).get('prompt_tokens_details') or {}).get('cached_tokens',0) for r in reqs if r.get('success')) if usage_cache_supported else None
        metrics['active_after_trial']=metric_total(after,'vllm:num_requests_running')+metric_total(after,'vllm:num_requests_waiting')
        tag=reqs[0]['tag'];start=metrics['started_ns'];end=metrics['ended_ns']
        ss=[r for r in steps.values() if start<=r['scheduled_ns']<=end]
        cs=[r for r in complete.values() if start<=r['completed_ns']<=end]
        req_lookup={r['request_id']:r for r in reqs}
        # API may add a per-completion suffix to user supplied request IDs.
        cohort_cache={}
        def cohort(rid):
            if rid in cohort_cache:return cohort_cache[rid]
            original=rid;rid=rid.removeprefix('cmpl-')
            value=req_lookup[rid]['cohort'] if rid in req_lookup else next((r['cohort'] for k,r in req_lookup.items() if rid.startswith(k+'-')),None)
            cohort_cache[original]=value
            return value
        server=defaultdict(list);per_request=defaultdict(list)
        for r in cs:
            for rid,n,interval in r['events']:
                c=cohort(rid)
                if c is not None and n==1 and interval is not None:
                    server[c].append(interval);per_request[rid].append(interval)
        progress=defaultdict(list);first_schedule={};first_output={};first_offsets={}
        for r in ss:
            for rid,(p,replay,d,pos,n) in r['allocations'].items():
                if p+replay and cohort(rid) is not None:
                    first_schedule.setdefault(rid,r['scheduled_ns'])
                    first_offsets.setdefault(rid,pos)
                    if r['step'] in complete:progress[rid].append(complete[r['step']]['completed_ns'])
        for c in cs:
            for rid,n,dt in c['events']:
                if n and cohort(rid) is not None:first_output.setdefault(rid,c['completed_ns'])
        worksteps=[r for r in ss if r['prompt']+r['replay']]
        service=[(first_output[rid]-t)/1e6 for rid,t in first_schedule.items() if rid in first_output]
        queue_ms=[];timing_by_cohort=defaultdict(lambda:defaultdict(list))
        for rid,t in first_schedule.items():
            c=cohort(rid)
            if rid in first_output:timing_by_cohort[c]['service_ms'].append((first_output[rid]-t)/1e6)
            advances=progress[rid]
            timing_by_cohort[c]['prefill_steps'].append(len(advances))
            timing_by_cohort[c]['max_progress_gap_ms'].append(max(((b-a)/1e6 for a,b in zip(advances,advances[1:])),default=0.))
        for rid,t in first_schedule.items():
            norm=rid.removeprefix('cmpl-')
            match=next((r for k,r in req_lookup.items() if norm==k or norm.startswith(k+'-')),None)
            if match is not None:
                gap=(t-match['sent_ns'])/1e6;queue_ms.append(gap)
                timing_by_cohort[cohort(rid)]['queue_ms'].append(gap)
        all_itl=[x for xs in server.values() for x in xs]
        excess=[]
        for c in cs:
            sr=steps.get(c['step'])
            if sr is not None:
                excess.extend(dt>sr['target_ms'] for rid,n,dt in c['events'] if cohort(rid) is not None and n==1 and dt is not None)
        telemetry=dict(ledger_available=ledger.exists(),prefill_service_to_first_output_ms=stats(service),client_to_first_prefill_schedule_ms=stats(queue_ms),
            first_compute_offset_by_request=first_offsets,
            prefill_timing_by_cohort={c:{k:stats(v) for k,v in values.items()} for c,values in timing_by_cohort.items()},
            completed_lag_ms=stats(c['lag_ms'] for c in cs),
            actual_prompt_tokens=sum(r['prompt'] for r in ss) if ledger.exists() else None,
            actual_replay_tokens=sum(r['replay'] for r in ss) if ledger.exists() else None,
            completed_lag_by_work={kind:stats(c['lag_ms'] for c in cs if c['step'] in steps and (
                'mixed' if steps[c['step']]['prompt']+steps[c['step']]['replay'] and steps[c['step']]['decode_tokens'] else
                'prefill' if steps[c['step']]['prompt']+steps[c['step']]['replay'] else 'decode')==kind)
                for kind in ['mixed','prefill','decode']},
            active_prefill_caps=dict(Counter(r['cap'] for r in worksteps)),
            aging_rescue_steps=sum(r['reason']=='aging_rescue' for r in ss),
            aging_rescue_with_compute_steps=sum(r['reason']=='aging_rescue' for r in worksteps),
            min_cap_prefill_step_fraction=sum(r['cap']<=manifest['apb_config'].get('minimum',256) for r in worksteps)/len(worksteps) if worksteps else None,
            decode_count=stats([r['decode'] for r in ss]),waiting_count=stats([r['waiting'] for r in ss]),
            high_decode_step_fraction=sum(r['decode']>=7 for r in ss)/len(ss) if ss else None,
            target_extrapolated_steps=sum(r.get('target_extrapolated',False) for r in ss),
            target_interval_exceed_fraction=sum(excess)/len(excess) if excess else None,steps=len(ss),server_itl_ms=stats(all_itl),server_by_cohort={k:stats(v) for k,v in server.items()},
            caps=dict(Counter(r['cap'] for r in ss)),reasons=dict(Counter(r['reason'] for r in ss)),
            scheduler_us=stats([r['scheduler_us'] for r in ss]),
            mixed_steps=sum(bool(r['prompt']+r['replay']) and bool(r['decode_tokens']) for r in ss),
            max_inflight=max((r['inflight'] for r in ss),default=0),
            prefill_gap_ms=stats([(b-a)/1e6 for xs in progress.values() for a,b in zip(xs,xs[1:])]),
            mean_actual_prefill=stats([r['prompt']+r['replay'] for r in ss]),
            per_request_p99_ms=stats([stats(xs)['p99'] for xs in per_request.values()]))
        metrics['engine_args']=manifest['args'];metrics['stage']=dest.parent.name;metrics['session']=str(dest);metrics['controller_config']=manifest['apb_config']
        metrics['telemetry']=telemetry;dump(path.parent/'analysis.json',metrics);trials.append(metrics)
        if metrics['scenario']=='cache' and ledger.exists():
            if telemetry['actual_prompt_tokens']+metrics['prefix_cache_hits_total_delta']!=sum(r['input'] for r in reqs):
                errors.append('prefix-hit versus actual prompt compute accounting mismatch')
    jit_warnings=[line.strip() for line in (dest/'server.log').read_text().splitlines() if 'during inference:' in line]
    if jit_warnings:errors.append('measurement-time JIT warning; whole session requires investigation/rerun')
    receipt={'jit_warnings':jit_warnings,'session':str(dest),'ledger_present':ledger.exists(),'steps':len(steps),'completed':len(complete),'errors':errors,'trials':len(trials)}
    dump(dest/'audit.json',receipt)
    return trials,receipt

def calibration(dest,track):
    rows=[];scheduled={};completed=[]
    for row in jsonl(dest/'scheduler.jsonl'):
        if row['type']=='step':scheduled[row['step']]=row
        else:completed.append(row)
    for path in sorted(dest.glob('calib-*/analysis.json')):
        r=json.loads(path.read_text());_,n,context=r['scenario'].split('-');n=int(n)
        window={};windows=[];valid=0;sample_count=0
        for event in completed:
            step=scheduled[event['step']]
            if not r['started_ns']<=event['completed_ns']<=r['ended_ns']:continue
            if step['prompt']+step['replay'] or step['decode']!=n:continue
            vals={rid:dt for rid,count,dt in event['events'] if count==1 and dt is not None}
            if len(vals)!=n:continue
            window.update(vals);valid+=1;sample_count+=len(vals)
            if valid==8:
                windows.append(stats(list(window.values()))['p90']);window={};valid=0
        if len(windows)<5:raise ValueError('insufficient stable pure decode calibration windows: '+str(path))
        rows.append(dict(decode=n,context=int(context)+256,reference_p90_ms=stats(windows)['p90'],
            sample_count=sample_count,windows=len(windows),source_session=str(dest)))
    target=DATA/'calibration'/f'decode-{track}.json'
    if target.exists():
        oldobj=json.loads(target.read_text());old=[dict(r,source_session=r.get('source_session',oldobj.get('source_session'))) for r in oldobj['rows']];keys={(r['decode'],r['context']) for r in rows}
        rows.extend(r for r in old if (r['decode'],r['context']) not in keys)
    dump(target,{'track':track,'source_session':str(dest),'model':'Qwen3-1.7B','rows':rows,
        'statistic':'P90 of stable 8-completed-step windows; each window is P90 of latest per-request server intervals; prefill and incomplete cohorts excluded',
        'margin_applied_online':.2})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session');ap.add_argument('--calibration-track');args=ap.parse_args()
    if args.session:
        trials,receipt=audit_session(Path(args.session));print(json.dumps(receipt))
        if args.calibration_track:calibration(Path(args.session),args.calibration_track)
        if receipt['errors']:raise SystemExit(1)
    else:
        rows=[];receipts=[]
        for p in sorted((DATA/'raw').rglob('completed.json')):
            if 'failures' in p.parts:continue
            rr,rc=audit_session(p.parent);rows.extend(rr);receipts.append(rc)
        dump(ROOT/'reports/results.json',rows);dump(ROOT/'reports/audits.json',receipts)
        print('TRIALS',len(rows),'AUDIT_ERRORS',sum(len(r['errors']) for r in receipts))

if __name__=='__main__':main()
