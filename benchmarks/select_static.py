"""Choose a static policy on held-out validation, before formal seeds."""
import json
from apb.common import DATA,dump,digest
from benchmarks.report import getmetric

def main():
    base=DATA/'raw/validation/default-stock/interference-seed829/metrics.json'
    stock=json.loads(base.read_text());eligible=[];rows=[]
    for cap in [256,512,1024,2048,4096]:
        path=DATA/f'raw/validation/default-fixed{cap}/interference-seed829/metrics.json'
        r=json.loads(path.read_text())
        row=dict(policy=f'fixed{cap}',decode99=getmetric(r,'decode99'),prefill95=getmetric(r,'prefill95'),output_tok_s=r['output_tok_s'],trace_sha256=r['trace_sha256'])
        assert row['trace_sha256']==stock['trace_sha256']
        row['eligible']=row['prefill95']<=getmetric(stock,'prefill95')*1.10 and row['output_tok_s']>=stock['output_tok_s']*.95
        rows.append(row)
        if row['eligible']:eligible.append(row)
    chosen=min(eligible,key=lambda r:r['decode99'])['policy'] if eligible else None
    dump(DATA/'manifests/static-selection.json',{'validation_seed':829,'scenario':'interference','constraints':{'prefill_p95_ttft_max_change_pct':10,'fixed_trace_throughput_min_change_pct':-5},
        'selection':chosen,'validation_static_pareto':rows,'baseline':stock,'frozen_before_formal':True})
    print('SELECTED_STATIC',chosen)

if __name__=='__main__':main()
