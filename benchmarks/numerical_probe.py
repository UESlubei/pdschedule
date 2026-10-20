"""Independent stock/off repeat and near-tie logprob diagnosis; never performance data."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import time
import aiohttp
from apb.common import DATA, ROOT, dump, digest
from benchmarks import session


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
                session.http(port,'/reset_prefix_cache',{})
                target=dict(model='apb-qwen3',prompt=prompt+prefix if kind=='teacher_forced' else prompt,
                    max_tokens=1 if kind=='teacher_forced' else 32,temperature=0,ignore_eos=True,
                    return_token_ids=True,logprobs=20,request_id=f'probe-{kind}-{batch}')
                distractors=[dict(model='apb-qwen3',prompt=session.tokens(8123+i,512),
                    max_tokens=32,temperature=0,ignore_eos=True,request_id=f'noise-{kind}-{batch}-{i}') for i in range(batch-1)]
                replies=await asyncio.gather(*(post(p) for p in [target,*distractors]))
                results.append(dict(kind=kind,offered_batch=batch,target_payload=target,response=replies[0]))
    return results


def main():
    locks=[]
    for path in [DATA/'gpu.lock',Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock'),Path('/mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock')]:
        f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX);locks.append(f)
    base=DATA/'raw/compatibility/default-stock/correctness-seed907'
    baseline={r['id']:r for r in json.loads((base/'requests.json').read_text())}
    trace={r['id']:r for r in json.loads((base/'trace.json').read_text())}
    target='tail1';prompt=trace[target]['prompt'];prefix=baseline[target]['token_ids'][:17]
    destroot=DATA/'quality_drift';destroot.mkdir(exist_ok=True)
    original_measure=session.measure;receipts=[]
    for repeat in [1,2]:
        for policy in ['stock','off']:
            dest=destroot/f'{policy}-repeat{repeat}'
            async def measured(port,rows,tag):
                records,metrics=await original_measure(port,rows,tag)
                if tag.startswith('correctness'):
                    dump(dest/'logprob-probes.json',await queries(port,prompt,prefix))
                return records,metrics
            session.measure=measured
            args=argparse.Namespace(policy=policy,track='default',output=str(dest),scenarios=['correctness'],
                seeds=[907],requests=200,rate=2.422,blocks=8192,relaxed_admission=False,profile=False,
                controller_json='{}',cancel=False,prefix_cache=True,threshold=0,threshold_adaptive=False)
            session.run(args)
            records=json.loads((dest/'correctness-seed907/requests.json').read_text())
            receipt=dict(policy=policy,repeat=repeat,session=str(dest),
                exact_by_id={r['id']:r['token_ids']==baseline[r['id']]['token_ids'] for r in records})
            probes=json.loads((dest/'logprob-probes.json').read_text());margins=[]
            for p in probes:
                choice=p['response']['choices'][0];pos=0 if p['kind']=='teacher_forced' else 17
                top=choice['logprobs']['top_logprobs'][pos]
                ranked=sorted(top.items(),key=lambda item:item[1],reverse=True)
                margins.append(dict(kind=p['kind'],offered_batch=p['offered_batch'],
                    position=pos,top5=ranked[:5],top1_minus_top2=ranked[0][1]-ranked[1][1],
                    generated_token_ids=choice.get('token_ids')))
            receipt['logprob_margins']=margins;receipts.append(receipt)
            dump(destroot/'receipt.json',dict(complete=False,runs=receipts,classification='numerical diagnosis, excluded from performance matrices'))
    session.measure=original_measure
    dump(destroot/'receipt.json',dict(complete=True,runs=receipts,ended_unix=time.time(),
        classification='numerical diagnosis, excluded from performance matrices',
        source_sha256=digest(Path(__file__)),common_prefix_tokens=prefix,original_prompt_tokens=prompt))
    print('NUMERICAL_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
