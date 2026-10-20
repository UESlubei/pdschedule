"""Compact read-only progress snapshot; completed audits remain authoritative."""
import json
from apb.common import DATA
from benchmarks.report import getmetric

def main():
    stage='opt_validation';root=DATA/'raw'/stage
    trials=[];sessions=0;quality=[]
    for dest in sorted(root.glob('*')):
        if not dest.is_dir():continue
        sessions+=int((dest/'completed.json').exists())
        for path in dest.glob('*/metrics.json'):
            r=json.loads(path.read_text());trials.append(r)
            if r['scenario']=='correctness':
                quality.append((r,json.loads((path.parent/'requests.json').read_text())))
    comparisons=[]
    for r in trials:
        if r['policy'] not in ['equalized256','fixed1536'] or r['scenario']=='correctness':continue
        for base in ['stock','balanced']:
            b=next((b for b in trials if b['policy']==base and b['scenario']==r['scenario'] and b['seed']==r['seed']),None)
            if b:
                assert b['trace_sha256']==r['trace_sha256']
                comparisons.append(dict(seed=r['seed'],scenario=r['scenario'],policy=r['policy'],baseline=base,
                    **{k:round(100*(getmetric(r,k)/getmetric(b,k)-1),2) for k in ['decode99','prefill95','output_tok_s']}))
    mismatches=[];quality_exact=quality_total=0
    for r,reqs in quality:
        if r['policy']=='stock':continue
        ref=next((qs for b,qs in quality if b['policy']=='stock' and b['seed']==r['seed']),None)
        if ref is None:continue
        lookup={q['id']:q for q in ref}
        for q in reqs:
            exact=q['token_ids']==lookup[q['id']]['token_ids'];quality_exact+=exact;quality_total+=1
            if not exact:mismatches.append(dict(seed=r['seed'],policy=r['policy'],id=q['id'],cohort=q['cohort']))
    print(json.dumps(dict(completed_sessions=sessions,registered_sessions=20,finished_trials=len(trials),
        request_errors=sum(r['errors'] for r in trials),quality_exact=quality_exact,quality_total=quality_total,
        quality_mismatches=mismatches,comparisons=comparisons),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
