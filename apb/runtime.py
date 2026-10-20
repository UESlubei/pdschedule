"""Scheduler hooks: charge after allocation; completed outputs drive feedback.

Only plain dense text autoregressive TP=PP=DP=1 is supported. Model buffers,
KV allocation and native request order are never modified here.
"""
from __future__ import annotations
from dataclasses import asdict
import atexit
from collections import Counter,deque
import json
import os
from pathlib import Path
import queue
import threading
import time
from .controller import Config,Controller,equalized_chunk
from .common import dump,stats

class BufferedLog:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(64);self.buf=[];self.dropped=0;self.error=None
        self.worker=threading.Thread(target=self._run,daemon=True);self.worker.start()
    def _run(self):
        try:
            with self.path.open('w') as f:
                while True:
                    rows=self.queue.get()
                    if rows is None:break
                    f.write(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows));f.flush()
        except Exception as e:self.error=repr(e)
    def add(self,row):
        self.buf.append(row)
        if len(self.buf)>=256:self.flush()
    def flush(self):
        if not self.buf:return
        rows=self.buf;self.buf=[]
        try:self.queue.put_nowait(rows)
        except queue.Full:self.dropped+=len(rows)
    def close(self):
        self.flush()
        if self.worker.is_alive():self.queue.put(None);self.worker.join(30)


def partition(request,start,n):
    """Prompt takes precedence; replayed prior outputs stop before last input."""
    prompt=max(0,min(start+n,request.num_prompt_tokens)-min(start,request.num_prompt_tokens))
    work_end=max(request.num_prompt_tokens,request.num_tokens-1)
    work=max(0,min(n,work_end-start))
    return prompt,work-prompt,n-work

class Runtime:
    def __init__(self,scheduler,raw):
        self.s=scheduler;self.cfg=Config(**raw);self.controller=Controller(self.cfg)
        s=scheduler;c=s.vllm_config
        if c.speculative_config is not None or s.lora_config or s.connector or s.ec_connector:
            raise ValueError('APB does not support spec/LoRA/KV/EC connectors')
        if any(getattr(s.parallel_config,k)!=1 for k in ['tensor_parallel_size','pipeline_parallel_size','data_parallel_size']):
            raise ValueError('APB requires TP=PP=DP=1')
        if s.policy.value!='fcfs' or s.has_mamba_layers or s.is_encoder_decoder or s.num_sampled_tokens_per_step!=1:
            raise ValueError('APB requires dense text FCFS autoregressive scheduler')
        if not s.scheduler_config.enable_chunked_prefill:raise ValueError('APB requires chunked prefill')
        self.cfg.maximum=min(self.cfg.maximum,s.max_num_scheduled_tokens,s.scheduler_config.max_num_batched_tokens)
        if self.cfg.minimum+s.max_num_running_reqs>self.cfg.maximum:raise ValueError('insufficient global budget for decode reservation')
        self.out=Path(self.cfg.output) if self.cfg.output else None
        self.log=BufferedLog(str(self.out)+'.jsonl') if self.out and self.cfg.telemetry else None
        self.counts=Counter();self.pending={};self.seq=0;self.tracked={};self.closed=False
        self.scheduler_us=deque(maxlen=100000);self.decision_us=deque(maxlen=100000);self.feedback_us=deque(maxlen=100000)
        self.table=json.loads(Path(self.cfg.target_table).read_text()) if self.cfg.target_table else None
        if self.out:
            dump(str(self.out)+'.config.json',{'config':asdict(self.cfg),'scheduler_class':type(s).__name__,
                'async':s.scheduler_config.async_scheduling,'vllm_module':__import__('vllm').__file__,
                'gpu_blocks':s.kv_cache_config.num_blocks,'max_num_scheduled_tokens':s.max_num_scheduled_tokens})
        self.profile_prefix=os.environ.get('APB_PROFILE_PREFIX','');self.profile_started=False;self.profile_done=False;self.profile_steps=0
        atexit.register(self.close)

    def emit(self,row):
        if self.log:self.log.add(row)

    def target_for(self,requests):
        self.target_extrapolated=False
        if not self.table:return self.cfg.target_ms
        n=len(requests);context=max((r.num_computed_tokens for r in requests),default=0)
        candidates=self.table['rows']
        counts=sorted({r['decode'] for r in candidates});nn=next((x for x in counts if x>=n),counts[-1])
        contexts=sorted({r['context'] for r in candidates if r['decode']==nn});cc=next((x for x in contexts if x>=context),contexts[-1])
        self.target_extrapolated=n>counts[-1] or context>contexts[-1]
        return next(r['reference_p90_ms'] for r in candidates if r['decode']==nn and r['context']==cc)*(1+self.cfg.margin)

    def eligible(self,r):
        if r.is_finished() or self.s.current_step<r.next_decode_eligible_step:return False
        if r.num_computed_tokens < max(r.num_prompt_tokens,r.num_tokens-1):return False
        need=r.num_tokens_with_spec+r.num_output_placeholders-r.num_computed_tokens
        if need<=0:return False
        if r.num_output_placeholders>0 and r.num_computed_tokens+2-r.num_output_placeholders>=r.num_prompt_tokens+r.max_tokens:return False
        return True

    def begin(self):
        self.started=time.perf_counter_ns();self.seq+=1;self.charges={};self.reserve=set()
        if self.profile_prefix and not self.profile_started and any(rid.startswith('cmpl-'+self.profile_prefix) for rid in self.s.requests):
            import torch
            torch.cuda.profiler.start();torch.cuda.nvtx.range_push('apb-measured-steps');self.profile_started=True
        now=self.started/1e6
        dec=[r for r in self.s.running if self.eligible(r)]
        if self.cfg.reserve:self.reserve={r.request_id for r in dec}
        # Cleanup catches abort, finish and request ID reuse without retaining history.
        self.tracked={k:v for k,v in self.tracked.items() if v['identity'] is self.s.requests.get(k)}
        running_prefill=[]
        for r in self.s.running:
            if r.mm_features or r.use_structured_output or r.resumable:raise ValueError('unsupported request type')
            if r.num_computed_tokens<max(r.num_prompt_tokens,r.num_tokens-1):running_prefill.append(r)
        oldest=max((now-self.tracked[r.request_id]['progress_ms'] for r in running_prefill if r.request_id in self.tracked),default=0.)
        # Native Request.arrival_time is Unix time; never subtract a monotonic clock.
        head=next(iter(self.s.waiting),None)
        if head is not None:oldest=max(oldest,max(0.,time.time()-head.arrival_time)*1000)
        stall=max((now-self.tracked[r.request_id]['output_ms'] for r in dec if r.request_id in self.tracked and self.tracked[r.request_id]['output_ms'] is not None),default=0.)
        self.controller.target=self.target_for(dec)
        t=time.perf_counter_ns();self.cap,self.reason=self.controller.decide(len(dec),now,stall if running_prefill or head is not None else 0.,oldest)
        self.decision_us.append((time.perf_counter_ns()-t)/1000)
        self.remaining=self.cap;self.decode_count=len(dec)
        self.chunk_selected=False
        self.state={'type':'step','step':self.seq,'scheduled_ns':self.started,'decode':len(dec),'running_prefill':len(running_prefill),
            'waiting':len(self.s.waiting),'cap':self.cap,'load_cap':4096 if not dec else 2048 if len(dec)<=2 else 1024 if len(dec)<=6 else 512,
            'target_extrapolated':self.target_extrapolated,'aimd_cap':self.controller.budget,'ema_ms':self.controller.ema,'target_ms':self.controller.target,'reason':self.reason,
            'prefill_wait_ms':oldest,'stall_ms':stall,'inflight':len(self.pending)}

    def enter(self,r):self.reserve.discard(r.request_id)

    def clip(self,r,start,n,token_budget,input_budget):
        if self.cfg.mode=='observe':return n
        p,replay,d=partition(r,start,n)
        if p+replay==0:return n
        if self.cfg.mode=='equalized' and self.decode_count and not self.chunk_selected:
            end=max(r.num_prompt_tokens,r.num_tokens-1)
            # Small requests retain the base aggregate cap for batching. Long
            # requests use a context-aware cap shared by all requests this step.
            cap=(self.cfg.cap if end <= self.cfg.cap else
                 equalized_chunk(start,end-start,self.cfg.cap,self.cfg.context_weight,self.cfg.chunk_alignment))
            self.cap=min(self.cfg.maximum,cap);self.remaining=self.cap;self.chunk_selected=True
            self.reason='equalized_chunk'
            self.state.update(cap=self.cap,reason=self.reason,chunk_start=start,chunk_end=end)
        available=max(0,min(self.remaining,token_budget-len(self.reserve),input_budget-len(self.reserve)))
        # Sequential work cannot jump over an unscheduled prefill suffix.
        return min(n,available)

    def charge(self,r,start,n):
        p,replay,d=partition(r,start,n)
        self.charges[r.request_id]=(p,replay,d,start,n,id(r))
        if self.cfg.mode!='observe':self.remaining-=p+replay;assert self.remaining>=0

    def refund(self,rid):
        row=self.charges.pop(rid,None)
        if row is not None and self.cfg.mode!='observe':self.remaining+=row[0]+row[1]
        self.reserve.discard(rid);self.counts['refunds']+=1

    def scheduled(self,out):
        row=self.state
        row.update(prompt=sum(x[0] for x in self.charges.values()),replay=sum(x[1] for x in self.charges.values()),decode_tokens=sum(x[2] for x in self.charges.values()),
            total=out.total_num_scheduled_tokens,preemptions=len(out.preempted_req_ids),scheduler_us=(time.perf_counter_ns()-self.started)/1000,
            allocations={rid:list(v[:5]) for rid,v in self.charges.items()},kv_usage=self.s.kv_cache_manager.usage)
        assert set(self.charges)==set(out.num_scheduled_tokens)
        assert row['prompt']+row['replay']+row['decode_tokens']==row['total']
        if self.cfg.mode!='observe':assert row['prompt']+row['replay']<=self.cap
        self.scheduler_us.append(row['scheduler_us']);self.counts['scheduled_steps']+=1
        # Match by object identity: SchedulerOutput stays alive in native queue.
        if out.total_num_scheduled_tokens:self.pending[id(out)]=(row,dict(self.charges))
        self.emit(row)

    def completed(self,out,outputs):
        start=time.perf_counter_ns();item=self.pending.pop(id(out),None)
        if item is None:
            if out.total_num_scheduled_tokens:self.counts['missing_ledger']+=1
            return
        row,charges=item;now=start/1e6;intervals={};events=[]
        for rid,(p,replay,d,pos,n,identity) in charges.items():
            r=self.s.requests.get(rid)
            if r is not None and id(r)!=identity:continue
            tracked=self.tracked.get(rid)
            if tracked is None and r is not None:
                tracked={'identity':r,'progress_ms':now,'output_ms':None};self.tracked[rid]=tracked
            if tracked is not None and p+replay:tracked['progress_ms']=now
        for eco in outputs.values():
            for o in eco.outputs:
                ids=o.new_token_ids
                if not ids:continue
                rid=o.request_id
                # Tokens accepted by native output update only. Finished requests
                # can already have been removed; use existing per-request state.
                t=self.tracked.get(rid)
                if t is None:
                    r=self.s.requests.get(rid)
                    t={'identity':r,'progress_ms':now,'output_ms':None}
                    if r is not None:self.tracked[rid]=t
                prev=t['output_ms'];t['output_ms']=now
                if prev is not None and len(ids)==1:intervals[rid]=now-prev
                if len(ids)>1:self.counts['multi_token_events']+=1
                events.append([rid,len(ids),None if prev is None else now-prev])
        self.controller.observe(row['step'],now,intervals,self.controller.target,had_prefill=bool(row['prompt']+row['replay']))
        self.counts['completed_steps']+=1;self.counts['output_events']+=len(events)
        self.emit({'type':'complete','step':row['step'],'completed_ns':start,'lag_ms':(start-row['scheduled_ns'])/1e6,
            'events':events,'windows':self.controller.windows,'feedback_us':(time.perf_counter_ns()-start)/1000})
        self.feedback_us.append((time.perf_counter_ns()-start)/1000)
        if self.profile_started and not self.profile_done:
            self.profile_steps+=1
            if self.profile_steps>=128:self.stop_profile()

    def stop_profile(self):
        import torch
        torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop();self.profile_done=True

    def close(self):
        if self.closed:return
        if self.profile_started and not self.profile_done:self.stop_profile()
        self.closed=True
        if self.log:self.log.close()
        if self.out:dump(str(self.out)+'.summary.json',{'counts':dict(self.counts),'config':asdict(self.cfg),
            'scheduler_us':stats(self.scheduler_us),'decision_us':stats(self.decision_us),'feedback_us':stats(self.feedback_us),
            'pending_steps':len(self.pending),'active_tracked':sum(k in self.s.requests for k in self.tracked),
            'log_dropped':self.log.dropped if self.log else 0,'log_error':self.log.error if self.log else None})


def create(scheduler):
    raw=json.loads(os.environ.get('APB_CONFIG','{}'))
    if raw.get('mode','off')=='off':return None
    return Runtime(scheduler,raw)
