"""Real native Request, KV manager and sync/async schedulers. No model weights."""
import copy
import importlib.util
import time
import os
import json
import unittest
from types import SimpleNamespace as NS
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm import SamplingParams
from vllm.v1.request import Request,RequestStatus
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from apb.common import MODEL,DATA
_spec=importlib.util.spec_from_file_location("vllm.v1.core.sched.apb_stock_reference",DATA/"overlays/stock/vllm/v1/core/sched/scheduler.py")
_native=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_native)
NativeScheduler=_native.Scheduler

class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=EngineArgs(model=str(MODEL),dtype='float16',max_model_len=8192,max_num_seqs=16,
            max_num_batched_tokens=4096,max_num_scheduled_tokens=4096,
            enforce_eager=True,async_scheduling=False,enable_prefix_caching=False).create_engine_config()
    def make(self,mode='fixed',cap=512,async_=False,cache=False,blocks=10000,reserve=True,native=False,alignment=16):
        c=copy.deepcopy(self.cfg);c.scheduler_config.async_scheduling=async_;c.cache_config.enable_prefix_caching=cache;c.cache_config.num_gpu_blocks=blocks;c.cache_config.num_cpu_blocks=0
        spec=FullAttentionSpec(block_size=16,num_kv_heads=8,head_size=128,dtype=torch.float16)
        kv=KVCacheConfig(num_blocks=blocks,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(['layer'],spec)])
        os.environ['APB_CONFIG']=json.dumps(dict(mode=mode,cap=cap,telemetry=False,reserve=reserve,chunk_alignment=alignment))
        s=(NativeScheduler if native else AsyncScheduler if async_ else Scheduler)(c,kv,NS(accept_tokens=lambda *a:True),16,16)
        self.addCleanup(s.shutdown);return s
    def req(self,s,rid,prompt=100,max_tokens=8):
        r=Request(rid,[42]*prompt,SamplingParams(temperature=0,max_tokens=max_tokens),None)
        s.add_request(r);return r
    def complete(self,s,o):
        ids=list(o.num_scheduled_tokens)
        generated=[[50] if r in s.requests and not s.requests[r].is_prefill_chunk else [] for r in ids]
        s.update_from_output(o,ModelRunnerOutput(req_ids=ids,req_id_to_index={r:i for i,r in enumerate(ids)},sampled_token_ids=generated))
    def drain(self,s,limit=2000):
        for _ in range(limit):
            if not s.requests:return
            o=s.schedule();self.complete(s,o)
        self.fail('scheduler failed to drain')
    def test_shared_running_and_waiting(self):
        s=self.make();a=self.req(s,'a',6144);o=s.schedule();self.assertEqual(sum(o.num_scheduled_tokens.values()),512);self.complete(s,o)
        self.req(s,'b',2048)
        for _ in range(30):
            o=s.schedule();self.assertLessEqual(s._apb.state['prompt']+s._apb.state['replay'],512);self.complete(s,o)
        self.drain(s)

    def test_equalized_real_shared_budget_and_decode(self):
        for asynchronous in [False,True]:
            s=self.make(mode='equalized',cap=2048,async_=asynchronous,alignment=256)
            self.req(s,'d',16,max_tokens=32);o=s.schedule();self.complete(s,o)
            self.req(s,'long',6144);self.req(s,'next',6144)
            chunks=[]
            for _ in range(3):
                o=s.schedule()
                self.assertEqual(o.num_scheduled_tokens['d'],1)
                self.assertLessEqual(s._apb.state['prompt']+s._apb.state['replay'],s._apb.cap)
                chunks.append(o.num_scheduled_tokens['long']);self.complete(s,o)
            self.assertEqual(sum(chunks),6144)
            self.assertGreater(chunks[0],chunks[-1])
            self.drain(s);self.assertEqual(s._apb.pending,{})

    def test_equalized_1536_aligned_four_step_plan(self):
        for asynchronous in [False,True]:
            s=self.make(mode='equalized',cap=1536,async_=asynchronous,alignment=512)
            self.req(s,'d',16,max_tokens=32);o=s.schedule();self.complete(s,o)
            self.req(s,'long',6144);chunks=[]
            for _ in range(4):
                o=s.schedule();chunks.append(o.num_scheduled_tokens['long'])
                self.assertEqual(o.num_scheduled_tokens['d'],1)
                self.assertLessEqual(s._apb.state['prompt']+s._apb.state['replay'],s._apb.cap)
                self.complete(s,o)
            self.assertEqual(sum(chunks),6144)
            self.assertTrue(all(n%512==0 for n in chunks))
            self.assertGreater(chunks[0],chunks[-1]);self.drain(s)
    def test_decode_reserved_behind_running_prefill(self):
        s=self.make(cap=4096);p=self.req(s,'p',6144);o=s.schedule();self.complete(s,o)
        d=self.req(s,'d',16);o=s.schedule();self.complete(s,o)
        # Put a fresh partial prefill before the now-decoding request, preserving native list thereafter.
        p2=self.req(s,'p2',6144);o=s.schedule();self.complete(s,o)
        if p2 in s.running:s.running.remove(p2);s.running.insert(0,p2)
        o=s.schedule();self.assertIn('d',o.num_scheduled_tokens);self.assertEqual(o.num_scheduled_tokens['d'],1)
        self.assertLessEqual(o.total_num_scheduled_tokens,4096);self.complete(s,o);self.drain(s)
    def test_tail_and_cancel(self):
        for n in [1,17,255,513]:
            s=self.make();self.req(s,'tail',n);self.drain(s)
        s=self.make();self.req(s,'cancel',6144);o=s.schedule()
        s.finish_requests('cancel',RequestStatus.FINISHED_ABORTED);self.complete(s,o)
        self.req(s,'after',17);self.drain(s);self.assertEqual(s._apb.pending,{})
    def test_sync_async_two_inflight(self):
        s=self.make(async_=True);self.req(s,'a',6144)
        a=s.schedule();b=s.schedule();self.assertEqual(len(s._apb.pending),2)
        self.complete(s,a);self.complete(s,b);self.drain(s)
        self.assertEqual(s._apb.counts['missing_ledger'],0)
    def test_allocation_failure_no_charge(self):
        s=self.make(blocks=2);self.req(s,'too-big',6144);o=s.schedule()
        self.assertEqual(o.total_num_scheduled_tokens,0);self.assertEqual(s._apb.remaining,512)
        s.finish_requests(None,RequestStatus.FINISHED_ABORTED)
    def test_refund_idempotent(self):
        s=self.make();r=self.req(s,'a',100);s.current_step+=1;s._apb.begin()
        s._apb.charge(r,0,100);self.assertEqual(s._apb.remaining,412)
        s._apb.refund('a');s._apb.refund('a');self.assertEqual(s._apb.remaining,512)
    def test_off_matches_native_without_runtime(self):
        a=self.make(mode='off');b=self.make(mode='off',native=True)
        for s in [a,b]:self.req(s,'x',6144);self.req(s,'y',17)
        for _ in range(10):
            oa=a.schedule();ob=b.schedule();self.assertEqual(oa.num_scheduled_tokens,ob.num_scheduled_tokens)
            self.complete(a,oa);self.complete(b,ob)
        self.assertIsNone(a._apb)
    def test_preemption_replay(self):
        s=self.make();r=self.req(s,'x',17,max_tokens=16);o=s.schedule();self.complete(s,o)
        for _ in range(3):o=s.schedule();self.complete(s,o)
        s.running.remove(r);s._preempt_request(r,0.,drop_stale_output=False)
        o=s.schedule();self.assertGreater(s._apb.state['replay'],0);self.complete(s,o);self.drain(s)

    def test_equalized_preemption_replay_and_cleanup(self):
        s=self.make(mode='equalized',cap=2048)
        r=self.req(s,'x',17,max_tokens=16);o=s.schedule();self.complete(s,o)
        for _ in range(3):o=s.schedule();self.complete(s,o)
        s.running.remove(r);s._preempt_request(r,0.,drop_stale_output=False)
        o=s.schedule();self.assertGreater(s._apb.state['replay'],0)
        self.assertLessEqual(s._apb.state['prompt']+s._apb.state['replay'],s._apb.cap)
        self.complete(s,o);self.drain(s)
        self.assertEqual(s._apb.pending,{})

    def test_equalized_partial_cache_uses_computed_frontier(self):
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher,init_none_hash
        from vllm.utils.hashing import get_hash_fn_by_name
        s=self.make(mode='equalized',cap=2048,cache=True)
        init_none_hash(get_hash_fn_by_name(s.cache_config.prefix_caching_hash_algo))
        hasher=get_request_block_hasher(16,get_hash_fn_by_name(s.cache_config.prefix_caching_hash_algo))
        def req(rid,ids,outputs):
            r=Request(rid,ids,SamplingParams(temperature=0,max_tokens=outputs),None,block_hasher=hasher)
            s.add_request(r);return r
        req('seed',[42]*4096,2);self.drain(s)
        req('d',[44]*17,16);o=s.schedule();self.complete(s,o)
        req('partial',[42]*4096+[43]*2048,2)
        o=s.schedule()
        self.assertEqual(s._apb.state['chunk_start'],4096)
        self.assertEqual(s._apb.state['prompt'],2048)
        self.assertEqual(o.num_scheduled_tokens['d'],1)
        self.complete(s,o);self.drain(s)

    def test_aging_uses_native_unix_arrival_clock(self):
        s=self.make(mode='aging');r=self.req(s,'old',6144);r.arrival_time=time.time()-.5
        s.current_step+=1;s._apb.begin()
        self.assertGreater(s._apb.state['prefill_wait_ms'],400)
    def test_prefix_hit_excluded_from_cap(self):
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher,init_none_hash
        from vllm.utils.hashing import get_hash_fn_by_name
        s=self.make(cache=True)
        init_none_hash(get_hash_fn_by_name(s.cache_config.prefix_caching_hash_algo))
        hasher=get_request_block_hasher(16,get_hash_fn_by_name(s.cache_config.prefix_caching_hash_algo))
        def req(rid):
            r=Request(rid,[42]*1024,SamplingParams(temperature=0,max_tokens=2),None,block_hasher=hasher)
            s.add_request(r);return r
        req('seed');self.drain(s)
        r=req('hit');o=s.schedule()
        self.assertGreaterEqual(r.num_computed_tokens,1008)
        self.assertLessEqual(s._apb.state['prompt'],16)
        self.complete(s,o);self.drain(s)

if __name__=='__main__':unittest.main()
