"""Fresh native/off/fixed repeats and common-prefix logits for the observed drift."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import time
import aiohttp
from apb.common import DATA,ROOT,dump,digest
from benchmarks import session
from benchmarks.analyze import audit_session

def read(p):return json.loads(p.read_text())

async def queries(port,prompt,prefix):
    results=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client:
        async def post(payload):
            async with client.post(f'http://127.0.0.1:{port}/v1/completions',json=payload) as response:
                value=await response.json()
                if response.status!=200:raise RuntimeError(str(value))
                return value
        for kind in ['teacher_forced','autoregressive']:
            for batch in [1,2,8]:
                target=dict(model='apb-qwen3',prompt=prompt+prefix if kind=='teacher_forced' else prompt,
                    max_tokens=1 if kind=='teacher_forced' else 32,temperature=0,ignore_eos=True,
                    return_token_ids=True,logprobs=20,request_id=f'probe-{kind}-{batch}')
                distractors=[dict(model='apb-qwen3',prompt=session.tokens(8123+i,512),max_tokens=32,
                    temperature=0,ignore_eos=True,request_id=f'noise-{kind}-{batch}-{i}') for i in range(batch-1)]
                replies=await asyncio.gather(*(post(p) for p in [target,*distractors]))
                results.append(dict(kind=kind,offered_batch=batch,target_payload=target,response=replies[0]))
    return results

def main():
    root=DATA/'quality_drift_opt';root.mkdir(exist_ok=True);receipt_path=root/'receipt.json'
    if receipt_path.exists() and read(receipt_path).get('complete'):
        print('QUALITY_PROBE_ALREADY_COMPLETE');return
    locks=[]
    for p in [DATA/'gpu.lock',DATA.with_name('adaptive_prefill_budget_v0320')/'gpu.lock',
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),
        Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    base=DATA/'raw/opt_match1536/default-stock-s419/correctness-seed419'
    changed=DATA/'raw/opt_match1536/default-fixed1536-s419/correctness-seed419'
    native={r['id']:r for r in read(base/'requests.json')};fixed={r['id']:r for r in read(changed/'requests.json')}
    target='tail1';a=native[target]['token_ids'];b=fixed[target]['token_ids']
    mismatch=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    prefix=a[:mismatch];prompt=next(r['prompt'] for r in read(base/'trace.json') if r['id']==target)
    original_measure=session.measure;runs=[]
    try:
        for repeat in [1,2]:
            for policy in ['stock','off','fixed1536']:
                dest=root/f'{policy}-repeat{repeat}'
                async def measured(port,rows,tag):
                    records,metrics=await original_measure(port,rows,tag)
                    if tag.startswith('correctness'):dump(dest/'logprob-probes.json',await queries(port,prompt,prefix))
                    return records,metrics
                session.measure=measured
                if not (dest/'completed.json').exists():
                    args=argparse.Namespace(policy=policy,track='default',output=str(dest),scenarios=['correctness'],
                        seeds=[419],requests=200,rate=2.422,blocks=8192,relaxed_admission=False,profile=False,
                        controller_json=json.dumps({'fixed_idle_relax':True}) if policy=='fixed1536' else '{}',
                        cancel=False,prefix_cache=False,threshold=0,threshold_adaptive=False)
                    print('QUALITY_REPEAT_START',policy,repeat,flush=True);session.run(args)
                records=read(dest/'correctness-seed419/requests.json');probes=read(dest/'logprob-probes.json')
                _,audit=audit_session(dest)
                # Logprob Top-K compilation is retained in the unmodified
                # performance audit. These services are content diagnostics,
                # never latency evidence; all other integrity errors still fail.
                integrity_errors=[e for e in audit['errors'] if e!='measurement-time JIT warning; whole session requires investigation/rerun']
                assert not integrity_errors
                assert digest(dest/'correctness-seed419/trace.json')==digest(base/'trace.json')
                margins=[]
                for p in probes:
                    choice=p['response']['choices'][0];pos=0 if p['kind']=='teacher_forced' else mismatch
                    top=choice['logprobs']['top_logprobs'][pos];ranked=sorted(top.items(),key=lambda x:x[1],reverse=True)
                    ids=choice.get('token_ids')
                    margins.append(dict(kind=p['kind'],offered_batch=p['offered_batch'],position_0based=pos,
                        common_prefix_aligned=True if p['kind']=='teacher_forced' else ids[:mismatch]==prefix,
                        top5=ranked[:5],top1_minus_top2=ranked[0][1]-ranked[1][1],generated_token_ids=ids))
                runs.append(dict(policy=policy,repeat=repeat,session=str(dest),integrity_audit_errors=integrity_errors,
                    performance_audit_errors=audit['errors'],jit_warnings=audit['jit_warnings'],
                    exact_vs_original_native={r['id']:r['token_ids']==native[r['id']]['token_ids'] for r in records},
                    exact_vs_original_fixed={r['id']:r['token_ids']==fixed[r['id']]['token_ids'] for r in records},
                    logprob_margins=margins,trace_sha256=digest(dest/'correctness-seed419/trace.json')))
                dump(receipt_path,dict(complete=False,runs=runs,classification='Independent numerical diagnostics; excluded from performance matrices.'))
                print('QUALITY_REPEAT_COMPLETE',policy,repeat,flush=True)
    finally:session.measure=original_measure
    pairs=[]
    for repeat in [1,2]:
        requests={p:{r['id']:r for r in read(root/f'{p}-repeat{repeat}/correctness-seed419/requests.json')} for p in ['stock','off','fixed1536']}
        pairs.append(dict(repeat=repeat,stock_off_exact=sum(requests['stock'][rid]['token_ids']==r['token_ids'] for rid,r in requests['off'].items()),
            stock_fixed_exact=sum(requests['stock'][rid]['token_ids']==r['token_ids'] for rid,r in requests['fixed1536'].items()),total=len(requests['stock'])))
    result=dict(complete=True,runs=runs,pairs=pairs,ended_unix=time.time(),target=target,seed=419,
        first_difference_1based=mismatch+1,original_native_token=a[mismatch],original_fixed_token=b[mismatch],
        common_prefix_tokens=prefix,original_prompt_tokens=prompt,baseline=str(base),changed=str(changed),
        source_sha256=digest(Path(__file__)),classification='Independent numerical diagnostics; excluded from performance matrices.')
    dump(receipt_path,result);dump(ROOT/'reports/optimization_v2/numerical_probe.json',result)
    print('QUALITY_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
