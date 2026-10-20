"""Continuous-load arrivals, completion, queue and per-request TTFT."""
import argparse
import json
from apb.common import DATA,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True,constrained_layout=True)
    for i,policy in enumerate(['fixed512','joint','aging']):
        dest=DATA/'raw/starvation'/f'default-{policy}-seed{args.seed}'
        trial=dest/f'starvation-seed{args.seed}'
        m=json.loads((trial/'metrics.json').read_text());start=m['started_ns'];end=m['ended_ns']
        reqs=json.loads((trial/'requests.json').read_text())
        ts=[];decode=[];waiting=[]
        for line in (dest/'scheduler.jsonl').open():
            r=json.loads(line)
            if r['type']=='step' and start<=r['scheduled_ns']<=end:
                ts.append((r['scheduled_ns']-start)/1e9);decode.append(r['decode']);waiting.append(r['waiting'])
        color=f'C{i}'
        axes[0].step(ts,decode,where='post',label=policy,color=color,lw=.8)
        axes[1].step(ts,waiting,where='post',label=policy,color=color,lw=1)
        finished=sorted((q['end_ns']-start)/1e9 for q in reqs)
        axes[2].step(finished,range(1,len(finished)+1),where='post',label=policy,color=color)
        if i==0:
            arrivals=sorted(q['offset'] for q in reqs)
            axes[2].step(arrivals,range(1,len(arrivals)+1),where='post',label='offered arrivals',color='black',ls='--')
        qs=[q for q in reqs if q['input']==6144]
        axes[3].scatter([q['offset'] for q in qs],[q['ttft_ms']/1000 for q in qs],s=6,alpha=.65,color=color,label=policy)
    for ax,label in zip(axes,['Eligible decode count','Waiting requests','Cumulative requests','Long-prompt TTFT (s)']):
        ax.set_ylabel(label);ax.axvline(180,color='grey',ls=':',label='injection ends' if ax is axes[0] else None)
        ax.grid(alpha=.25);ax.legend(loc='best')
    axes[-1].set_xlabel('Time from trial start / request arrival (s)')
    fig.suptitle(f'Finite continuous-load stress, default async / seed {args.seed}')
    out=ROOT/'reports/figures';out.mkdir(exist_ok=True)
    fig.savefig(out/f'starvation-seed{args.seed}.png',dpi=170)
    fig.savefig(out/f'starvation-seed{args.seed}.pdf')


if __name__=='__main__':main()
