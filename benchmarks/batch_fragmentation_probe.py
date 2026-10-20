"""CPU state experiment with real V1 Scheduler/KV manager; no GPU timing claims."""
import importlib.util
from apb.common import ROOT,dump,digest

def main():
    spec=importlib.util.spec_from_file_location('apb_test_fixtures',ROOT/'tests/test_scheduler.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cls=module.SchedulerTests;cls.setUpClass();results=[]
    for asynchronous in [False,True]:
        for mode in ['fixed','equalized']:
            case=cls();s=case.make(mode=mode,cap=1536,async_=asynchronous,alignment=512)
            try:
                case.req(s,'anchor',512,max_tokens=128);o=s.schedule();case.complete(s,o)
                prompts=[case.req(s,f'h{i}',2048,max_tokens=32) for i in range(8)]
                trace=[]
                for step in range(64):
                    if all(r.num_computed_tokens>=r.num_prompt_tokens for r in prompts):break
                    o=s.schedule();row=s._apb.state
                    trace.append(dict(step=step,cap=s._apb.cap,prompt=row['prompt'],decode_tokens=row['decode_tokens'],
                        prompt_requests=sum(bool(v[0]) for v in row['allocations'].values()),
                        allocations=row['allocations']))
                    case.complete(s,o)
                assert all(r.num_computed_tokens>=r.num_prompt_tokens for r in prompts)
                assert sum(r['prompt'] for r in trace)==8*2048
                results.append(dict(mode=mode,async_scheduling=asynchronous,prefill_steps=sum(bool(r['prompt']) for r in trace),
                    multi_prefill_request_steps=sum(r['prompt_requests']>1 for r in trace),
                    prompt_tokens=sum(r['prompt'] for r in trace),trace=trace))
                case.drain(s)
            finally:case.doCleanups()
    sources=[ROOT/'benchmarks/batch_fragmentation_probe.py',ROOT/'tests/test_scheduler.py',ROOT/'apb/runtime.py',ROOT/'apb/controller.py']
    result=dict(scene='One existing Decode anchor, then eight 2048-token prompts; base cap 1536; equalized alignment 512.',
        interpretation='Real native Scheduler and KV manager with deterministic fake ModelRunner outputs. Counts prove scheduling behavior, not GPU latency or throughput.',
        results=results,source_hashes={str(p):digest(p) for p in sources})
    dump(ROOT/'reports/optimization_v2/batch_probe.json',result)
    print([(r['mode'],r['async_scheduling'],r['prefill_steps'],r['multi_prefill_request_steps']) for r in results])

if __name__=='__main__':main()
