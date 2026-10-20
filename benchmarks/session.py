"""Single policy server, reproducible drained trials with raw token/SSE events."""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.request
import aiohttp
from apb.common import ROOT,DATA,PYTHON,MODEL,clean_env,dump,digest,stats

def http(port,path,payload=None):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r:return r.read().decode()

def tokens(seed,length):
    rng=random.Random(seed);return [rng.randrange(100,30000) for _ in range(length)]

def trace(kind,seed,count=200,rate=4.):
    rng=random.Random(seed)
    rows=[]
    def add(name,cohort,length,output,offset):
        rows.append(dict(id=name,cohort=cohort,input=length,output=output,offset=offset,prompt=tokens(seed*10000+len(rows),length)))
    if kind=='prefill':
        for i in range(8):add(f'p{i}','prefill',6144,128,i*5.)
    elif kind=='interference':
        for i in range(8):add(f'd{i}','decode',512,1024,0.)
        for i in range(24):add(f'p{i}','prefill',6144,128,.4+i*.12)
    elif kind=='transition':
        for i in range(2):add(f'l{i}','decode',512,1536,0.)
        for i in range(8):add(f'h{i}','decode',2048,768,2.)
        for i in range(50):add(f'p{i}','prefill',6144,128,.2+i*.8)
    elif kind in {'mixed','burst'}:
        for i in range(count):
            length=rng.choices([512,2048,6144],[.5,.3,.2])[0]
            output=rng.choices([128,256,512],[.4,.4,.2])[0]
            offset=i/rate if kind=='mixed' else (i//16)*16/rate
            add(f'm{i}','mixed',length,output,offset)
    elif kind=='starvation':
        # Fixed open-loop, bounded outputs; pressure is measured, never assumed constant.
        for i in range(8):add(f'd{i}','decode',2048,2048,0.)
        for i in range(180):
            add(f'p{i}','prefill',6144,128,float(i))
        for i in range(120):add(f'dx{i}','decode',2048,2048,1.5*i+.5)
    elif kind=='pressure':
        for i in range(8):add(f'd{i}','decode',1024,384,0.)
        for i in range(2):add(f'p{i}','prefill',6144,64,.2+i*.1)
    elif kind=='cache':
        add('seed','cache_seed',6144,1,0.)
        add('repeat','cache_repeat',6144,32,2.)
        rows[-1]['prompt']=list(rows[0]['prompt'])
        add('partial','cache_partial',6144,32,3.)
        rows[-1]['prompt']=rows[0]['prompt'][:4096]+tokens(seed+999,2048)
    elif kind=='correctness':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
        questions=['What is 7 plus 5? Answer briefly.','请用一句话介绍 KV Cache 的作用。','What is the capital of France?',
            '写出数字 1 到 5。','Explain FIFO in one sentence.','计算 9 乘以 8。']
        for i,q in enumerate(questions):
            ids=tok.apply_chat_template([{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if hasattr(ids,'keys'):ids=ids['input_ids']
            add(f'n{i}','natural',len(ids),32,0.);rows[-1]['prompt']=ids;rows[-1]['question']=q
        for i,n in enumerate([1,17,255,513,6144]):add(f'tail{i}','tail',n,32,.2)
    elif kind.startswith('calib-'):
        _,n,l=kind.split('-')
        for i in range(int(n)):add(f'c{i}','calibration',int(l),256,0.)
    else:raise ValueError(kind)
    return rows

async def measure(port,rows,tag):
    started=time.perf_counter_ns()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600),connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(row):
            due=started+int(row['offset']*1e9);await asyncio.sleep(max(0,(due-time.perf_counter_ns())/1e9))
            sent=time.perf_counter_ns();events=[];ids=[];usage=None;text=[]
            record={k:v for k,v in row.items() if k!='prompt'};record.update(tag=tag,sent_ns=sent,planned_ns=due,request_id=f'{tag}-{row["id"]}')
            payload=dict(model='apb-qwen3',prompt=row['prompt'],max_tokens=row['output'],temperature=0,ignore_eos=True,
                stream=True,stream_interval=1,return_token_ids=True,request_id=record['request_id'],stream_options={'include_usage':True})
            try:
                async with session.post(f'http://127.0.0.1:{port}/v1/completions',json=payload) as response:
                    record['status']=response.status
                    if response.status!=200:raise RuntimeError(await response.text())
                    async for line in response.content:
                        if not line.startswith(b'data: '):continue
                        raw=line[6:].strip()
                        if raw==b'[DONE]':break
                        received=time.perf_counter_ns();obj=json.loads(raw)
                        if obj.get('error'):raise RuntimeError(str(obj['error']))
                        if obj.get('usage'):usage=obj['usage']
                        for choice in obj.get('choices',[]):
                            out=choice.get('token_ids') or []
                            if out:
                                ids.extend(out);events.append([received,len(out)])
                            if choice.get('text'):text.append(choice['text'])
                end=time.perf_counter_ns()
                if not events or usage is None or usage['completion_tokens']!=row['output'] or len(ids)!=row['output']:
                    raise RuntimeError(f'token/usage mismatch: ids={len(ids)}, usage={usage}')
                record.update(success=True,events=events,token_ids=ids,text=''.join(text),usage=usage,end_ns=end,
                    ttft_ms=(events[0][0]-sent)/1e6,e2e_ms=(end-sent)/1e6,
                    tpot_ms=(events[-1][0]-events[0][0])/1e6/(len(ids)-1) if len(ids)>1 else None,
                    intervals_ms=[(b[0]-a[0])/1e6 for a,b in zip(events,events[1:])],single_token_events=all(e[1]==1 for e in events))
            except Exception as exc:record.update(success=False,error=repr(exc),end_ns=time.perf_counter_ns())
            return record
        records=await asyncio.gather(*(one(r) for r in rows))
    end=time.perf_counter_ns();ok=[r for r in records if r['success']]
    all_single=all(r['single_token_events'] for r in ok)
    return records,dict(started_ns=started,ended_ns=end,wall_s=(end-started)/1e9,requests=len(records),completed=len(ok),errors=len(records)-len(ok),
        output_tokens=sum(len(r['token_ids']) for r in ok),output_tok_s=sum(len(r['token_ids']) for r in ok)/((end-started)/1e9),
        ttft_ms=stats([r['ttft_ms'] for r in ok]),interval_ms=stats([x for r in ok for x in r['intervals_ms']]),
        interval_label='client_token_event_gap' if all_single else 'client_chunk_gap',all_events_single_token=all_single,
        by_cohort={c:dict(ttft_ms=stats([r['ttft_ms'] for r in ok if r['cohort']==c]),
            interval_ms=stats([x for r in ok if r['cohort']==c for x in r['intervals_ms']])) for c in {r['cohort'] for r in ok}},
        submission_lag_ms=stats([(r['sent_ns']-r['planned_ns'])/1e6 for r in records]),
        tpot_ms=stats([r['tpot_ms'] for r in ok if r['tpot_ms'] is not None]))

async def cancellation(port):
    async with aiohttp.ClientSession() as session:
        async def one(i):
            async with session.post(f'http://127.0.0.1:{port}/v1/completions',json=dict(model='apb-qwen3',prompt=tokens(999+i,6144),max_tokens=512,stream=True,temperature=0,ignore_eos=True,request_id=f'cancel-{i}')) as r:
                async for _ in r.content:pass
        tasks=[asyncio.create_task(one(i)) for i in range(8)]
        await asyncio.sleep(.1)
        for t in tasks:t.cancel()
        results=await asyncio.gather(*tasks,return_exceptions=True)
        return {'attempted':len(tasks),'cancelled':sum(isinstance(r,asyncio.CancelledError) for r in results)}

def stop(child):
    if child.poll() is None:
        child.send_signal(signal.SIGTERM)
        try:child.wait(timeout=25)
        except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait()

def policy_cfg(name,track,output):
    mode='fixed' if name.startswith('fixed') else {'stock':'off','off':'off','observe':'observe','load':'load','aimd':'aimd','joint':'joint','aging':'aging','fixed4096-noreserve':'fixed'}.get(name,name)
    cap=int(name.removeprefix('fixed').split('-')[0]) if name.startswith('fixed') else 1024
    cfg=dict(mode=mode,cap=cap,output=str(output),reserve='noreserve' not in name)
    table=DATA/'calibration'/f'decode-{track}.json'
    if table.exists():cfg['target_table']=str(table)
    return cfg

def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    flavor='stock' if args.policy=='stock' else 'candidate';env=clean_env(flavor)
    cfg=policy_cfg(args.policy,args.track,dest/'scheduler');cfg.update(json.loads(args.controller_json))
    from apb.controller import Config
    Config(**cfg)
    env.update(VLLM_SERVER_DEV_MODE='1',VLLM_LOG_STATS_INTERVAL='1',APB_CONFIG=json.dumps(cfg))
    if args.profile:env['APB_PROFILE_PREFIX']='interference'
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd=[str(PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',str(MODEL),'--served-model-name','apb-qwen3',
        '--host','127.0.0.1','--port',str(port),'--dtype','float16','--max-model-len','8192','--max-num-seqs','16',
        '--max-num-batched-tokens','4096','--max-num-scheduled-tokens','4096','--gpu-memory-utilization','0.85',
        '--block-size','16','--num-gpu-blocks-override',str(args.blocks),'--enable-chunked-prefill',
        '--enable-prefix-caching' if args.prefix_cache else '--no-enable-prefix-caching','--generation-config','vllm',
        '--stream-interval','1','--shutdown-timeout','20','--disable-uvicorn-access-log']
    if args.track=='eager':cmd+=['--enforce-eager','--no-async-scheduling']
    if args.relaxed_admission:cmd+=['--no-scheduler-reserve-full-isl']
    if args.threshold:cmd+=['--long-prefill-token-threshold',str(args.threshold)]
    if args.threshold_adaptive:cmd+=['--long-prefill-token-threshold-adaptive']
    manifest={'args':vars(args),'command':cmd,'apb_config':json.loads(env['APB_CONFIG']),'started_unix':time.time(),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for folder in ['apb','benchmarks','scripts'] for p in (ROOT/folder).glob('*.py')},
        'prepare_manifest_sha256':digest(DATA/'manifests/prepared.json'),'fresh_server':True,
        'trial_isolation':'server reused for drained seed/scenario trials; prefix cache disabled or explicitly reset; warmup shared'}
    dump(dest/'manifest.json',manifest);child=None;results=[]
    with (dest/'server.log').open('w') as log:
        try:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+600
            while True:
                if child.poll() is not None:raise RuntimeError('server exited; see '+str(dest/'server.log'))
                try:http(port,'/health');break
                except Exception:
                    if time.monotonic()>deadline:raise TimeoutError('startup timeout')
                    time.sleep(.5)
            manifest['startup_s']=time.time()-manifest['started_unix']
            resolved=json.loads(http(port,'/server_info?config_format=json'));dump(dest/'resolved_config.json',resolved)
            sc=resolved['vllm_config']['scheduler_config']
            assert sc['max_num_batched_tokens']==sc['max_num_scheduled_tokens']==4096
            assert sc['async_scheduling']==(args.track=='default')
            # Warm all relevant query lengths before measurements.
            for n in [256,512,1024,2048,4096,6144]:
                result=json.loads(http(port,'/v1/completions',dict(model='apb-qwen3',prompt=tokens(99000+n,n),max_tokens=4,temperature=0,ignore_eos=True)))
                assert result['usage']['completion_tokens']==4
            warm=trace('calib-8-512',991);warm=[dict(r,output=8) for r in warm]
            asyncio.run(measure(port,warm,'warm-batch'))
            for seed in args.seeds:
                for scenario in args.scenarios:
                    trial=dest/f'{scenario}-seed{seed}';trial.mkdir(exist_ok=True)
                    rows=trace(scenario,seed,args.requests,args.rate)
                    dump(trial/'trace.json',rows)
                    before=http(port,'/metrics');(trial/'metrics-before.txt').write_text(before)
                    records,metrics=asyncio.run(measure(port,rows,f'{scenario}-s{seed}'))
                    dump(trial/'requests.json',records);dump(trial/'metrics.json',metrics)
                    (trial/'metrics-after.txt').write_text(http(port,'/metrics'))
                    metrics.update(scenario=scenario,seed=seed,policy=args.policy,track=args.track,trace_sha256=digest(trial/'trace.json'))
                    dump(trial/'metrics.json',metrics);results.append(metrics)
                    print('TRIAL',args.policy,args.track,scenario,seed,'errors',metrics['errors'],'ITL99',metrics['interval_ms']['p99'],'TTFT95',metrics['ttft_ms']['p95'],'tok_s',metrics['output_tok_s'],flush=True)
                    if metrics['errors']:raise RuntimeError('failed requests')
                    if args.prefix_cache:
                        state=json.loads(http(port,'/reset_prefix_cache',{}));assert state['success']
            if args.cancel:
                fault=asyncio.run(cancellation(port));time.sleep(.5)
                proof=json.loads(http(port,'/v1/completions',dict(model='apb-qwen3',prompt=tokens(123,128),max_tokens=8,ignore_eos=True,temperature=0)))
                fault['followup_output_tokens']=proof['usage']['completion_tokens']
                time.sleep(.5);fault['reset_cache']=json.loads(http(port,'/reset_prefix_cache',{}))
                dump(dest/'cancellation.json',fault)
        finally:
            if child:stop(child)
            manifest['ended_unix']=time.time();dump(dest/'manifest.json',manifest)
    dump(dest/'completed.json',{'trials':len(results),'errors':sum(r['errors'] for r in results)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',default='load');p.add_argument('--track',choices=['eager','default'],default='eager')
    p.add_argument('--output',required=True);p.add_argument('--scenarios',nargs='+',default=['interference']);p.add_argument('--seeds',nargs='+',type=int,default=[811])
    p.add_argument('--requests',type=int,default=200);p.add_argument('--rate',type=float,default=4.);p.add_argument('--blocks',type=int,default=8192)
    p.add_argument('--relaxed-admission',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--controller-json',default='{}');p.add_argument('--cancel',action='store_true');p.add_argument('--prefix-cache',action='store_true');p.add_argument('--threshold',type=int,default=0);p.add_argument('--threshold-adaptive',action='store_true')
    args=p.parse_args();run(args)

if __name__=='__main__':main()
