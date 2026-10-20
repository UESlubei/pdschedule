"""Scientific timeline plots; no smoothing of actual token gaps."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from apb.common import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--scenario',default='transition');p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(args.session);trial=dest/f'{args.scenario}-seed{args.seed}'
    m=json.loads((trial/'metrics.json').read_text());start=m['started_ns'];end=m['ended_ns']
    steps=[];complete={}
    for line in (dest/'scheduler.jsonl').open():
        r=json.loads(line)
        if r['type']=='step' and start<=r['scheduled_ns']<=end:steps.append(r)
        elif r['type']=='complete':complete[r['step']]=r
    reqs=json.loads((trial/'requests.json').read_text())
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,constrained_layout=True)
    ts=[(r['scheduled_ns']-start)/1e9 for r in steps]
    axes[0].step(ts,[r['decode'] for r in steps],where='post',label='eligible decode');axes[0].set_ylabel('Decode count')
    axes[1].step(ts,[r['cap'] for r in steps],where='post',label='cap',lw=1)
    axes[1].plot(ts,[r['prompt']+r['replay'] for r in steps],'.',ms=2,label='actual prefill');axes[1].set_ylabel('Tokens / step');axes[1].legend()
    axes[2].plot(ts,[r['target_ms'] for r in steps],label='target')
    axes[2].plot(ts,[r['ema_ms'] if r['ema_ms'] is not None else float('nan') for r in steps],label='EWMA');axes[2].set_ylabel('Feedback (ms)');axes[2].legend()
    for req in reqs:
        if req['cohort']!='decode':continue
        ev=req['events'];axes[3].scatter([(b[0]-start)/1e9 for a,b in zip(ev,ev[1:])],[(b[0]-a[0])/1e6 for a,b in zip(ev,ev[1:])],s=2,alpha=.25)
    axes[3].set_ylabel('Decode gap (ms)');axes[3].set_xlabel('Time from trial start (s)')
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle(dest.name+' / '+args.scenario+f' / seed {args.seed}')
    out=ROOT/'reports/figures';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{dest.name}-{args.scenario}-{args.seed}-timeline.png',dpi=170)

if __name__=='__main__':main()
