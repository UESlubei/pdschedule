"""Paired run bootstrap and standalone scientific plots from audited raw data."""
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
from apb.common import DATA,ROOT,dump

ORDER=['stock','observe','fixed256','fixed512','fixed1024','fixed2048','fixed4096','load','aimd','joint','aging']

def getmetric(row,key):
    if key=='decode99':return row['by_cohort'].get('decode',{'interval_ms':row['interval_ms']})['interval_ms']['p99']
    if key=='decode95':return row['by_cohort'].get('decode',{'interval_ms':row['interval_ms']})['interval_ms']['p95']
    if key=='prefill_mean':return row['by_cohort'].get('prefill',{'ttft_ms':row['ttft_ms']})['ttft_ms']['mean']
    if key=='tpot95':return row['tpot_ms']['p95']
    if key=='tpot_mean':return row['tpot_ms']['mean']
    if key=='prefill95':return row['by_cohort'].get('prefill',{'ttft_ms':row['ttft_ms']})['ttft_ms']['p95']
    if key=='ttft95':return row['ttft_ms']['p95']
    if key=='itl99':return row['interval_ms']['p99']
    return row[key]

def paired(rows,candidate,base,key,track='default',scenario='interference'):
    a={r['seed']:getmetric(r,key) for r in rows if r['policy']==candidate and r['track']==track and r['scenario']==scenario}
    b={r['seed']:getmetric(r,key) for r in rows if r['policy']==base and r['track']==track and r['scenario']==scenario}
    changes=[100*(a[s]/b[s]-1) for s in sorted(set(a)&set(b))]
    if not changes:return None
    rng=random.Random(2048);bs=sorted(sum(rng.choices(changes,k=len(changes)))/len(changes) for _ in range(5000))
    return dict(pairs=len(changes),mean_change_pct=statistics.mean(changes),ci95=[bs[125],bs[4874]],changes=changes)

def main():
    rows=json.loads((ROOT/'reports/results.json').read_text())
    trace_groups=defaultdict(list)
    for r in rows:
        if r['stage'] in {'formal','balanced','mechanism','starvation','native','native2048','ablation','compatibility','idle','profiles'}:
            trace_groups[(r['scenario'],r['seed'])].append(r)
    trace_audit=[dict(stages=sorted({r['stage'] for r in v}),scenario=k[0],seed=k[1],sessions=len(v),
        hashes=sorted({r['trace_sha256'] for r in v})) for k,v in sorted(trace_groups.items())]
    dump(ROOT/'reports/trace_comparability.json',trace_audit)
    if any(len(r['hashes'])!=1 for r in trace_audit):raise RuntimeError('paired policies received different traces')
    # Only independently restarted formal seed sessions enter confirmatory tables.
    mainrows=[r for r in rows if r.get('stage')=='formal']
    groups=defaultdict(list)
    for r in mainrows:groups[(r['track'],r['scenario'],r['policy'])].append(r)
    summaries=[]
    for (track,scenario,policy),rs in sorted(groups.items()):
        summaries.append(dict(track=track,scenario=scenario,policy=policy,repeats=len(rs),
            **{k:statistics.mean(getmetric(r,k) for r in rs) for k in ['decode99','decode95','prefill95','prefill_mean','ttft95','itl99','output_tok_s','tpot95','tpot_mean']},
            errors=sum(r['errors'] for r in rs),intervals=sum(r['interval_ms']['count'] for r in rs)))
    dump(ROOT/'reports/summary.json',summaries)
    comparisons=[]
    for scenario in ['interference','transition','mixed']:
        for policy in ORDER[1:]:
            for metric in ['decode99','prefill95','ttft95','output_tok_s']:
                value=paired(mainrows,policy,'stock',metric,scenario=scenario)
                if value:comparisons.append(dict(scenario=scenario,policy=policy,baseline='stock',metric=metric,**value))
    selection=json.loads((DATA/'manifests/static-selection.json').read_text())['selection']
    if selection:
        for scenario in ['interference','transition','mixed']:
            for policy in ['load','aimd','joint','aging']:
                for metric in ['decode99','prefill95','prefill_mean','output_tok_s']:
                    value=paired(mainrows,policy,selection,metric,scenario=scenario)
                    if value:comparisons.append(dict(scenario=scenario,policy=policy,baseline=selection,metric=metric,**value))
    dump(ROOT/'reports/paired_comparisons.json',comparisons)
    balanced=[r for r in rows if r['stage']=='balanced']
    balanced_summary=[];balanced_comparisons=[]
    for scenario in ['interference','transition','mixed']:
        for policy in ['stock','fixed2048','joint']:
            rs=[r for r in balanced if r['scenario']==scenario and r['policy']==policy]
            if rs:balanced_summary.append(dict(scenario=scenario,policy=policy,repeats=len(rs),
                **{k:statistics.mean(getmetric(r,k) for r in rs) for k in ['decode99','prefill95','output_tok_s']}))
        for baseline in ['stock','fixed2048']:
            for metric in ['decode99','prefill95','output_tok_s']:
                value=paired(balanced,'joint',baseline,metric,scenario=scenario)
                if value:balanced_comparisons.append(dict(scenario=scenario,policy='joint-min2048',baseline=baseline,metric=metric,**value))
    dump(ROOT/'reports/balanced_summary.json',balanced_summary)
    dump(ROOT/'reports/balanced_comparisons.json',balanced_comparisons)
    acceptance=[]
    for scenario in ['interference','transition','mixed']:
        values={r['metric']:r for r in balanced_comparisons if r['scenario']==scenario and r['baseline']=='stock'}
        if len(values)==3:
            decode,prefill,throughput=(values[k] for k in ['decode99','prefill95','output_tok_s'])
            acceptance.append(dict(scenario=scenario,pairs=decode['pairs'],
                point_constraints_met=decode['mean_change_pct']<0 and prefill['mean_change_pct']<=10 and throughput['mean_change_pct']>=-5,
                ci_constraints_met=decode['ci95'][1]<0 and prefill['ci95'][1]<=10 and throughput['ci95'][0]>=-5))
    dump(ROOT/'reports/balanced_acceptance.json',dict(protocol=str(DATA/'manifests/balanced-protocol.json'),scenarios=acceptance))
    extra=[];extra_groups=defaultdict(list)
    for r in rows:
        if r['stage'] in {'ablation','native','native2048'}:
            label=Path(r['session']).name.rsplit('-s',1)[0]
            extra_groups[(r['stage'],label,r['scenario'])].append(r)
    for (stage,label,scenario),rs in sorted(extra_groups.items()):
        baseline=('fixed4096' if 'reserve-off' in label else 'fixed512' if 'idle-relax' in label else
                  'aging' if 'aging-every-step' in label else 'joint') if stage=='ablation' else 'stock'
        candidates=[dict(r,policy='candidate') for r in rs]
        reference=[r for r in mainrows if r['policy']==baseline]
        for metric in ['decode99','prefill95','output_tok_s']:
            value=paired(candidates+reference,'candidate',baseline,metric,scenario=scenario)
            if value:extra.append(dict(stage=stage,configuration=label,scenario=scenario,
                baseline=baseline,metric=metric,**value))
    dump(ROOT/'reports/additional_comparisons.json',extra)
    # Optional plotting uses the existing analysis interpreter, not inference env.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    figdir=ROOT/'reports/figures';figdir.mkdir(exist_ok=True)
    for scenario in ['interference','transition','mixed']:
        ss=[r for r in summaries if r['scenario']==scenario]
        ss.sort(key=lambda r:ORDER.index(r['policy']))
        if not ss:continue
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for ax,key,label in zip(axes,['decode99','prefill95','output_tok_s'],['Decode cohort P99 gap (ms)','Prefill cohort P95 TTFT (ms)','Fixed-trace output tok/s']):
            low=[];high=[];rng=random.Random(2048)
            for r in ss:
                xs=[getmetric(v,key) for v in mainrows if v['scenario']==scenario and v['policy']==r['policy']]
                boots=sorted(sum(rng.choices(xs,k=len(xs)))/len(xs) for _ in range(5000))
                low.append(max(0.,r[key]-boots[125]));high.append(max(0.,boots[4874]-r[key]))
            ax.bar([r['policy'] for r in ss],[r[key] for r in ss],yerr=[low,high],capsize=2)
            ax.tick_params(axis='x',rotation=65);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Default async / CUDA Graph: '+scenario);fig.savefig(figdir/(scenario+'.png'),dpi=170);fig.savefig(figdir/(scenario+'.pdf'));plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,6),constrained_layout=True)
        offsets={'stock':(12,-20),'observe':(12,6),'fixed4096':(-72,-28),'fixed512':(-62,-20),
                 'load':(12,12),'aimd':(-52,15),'joint':(10,-14)}
        throughputs=[r['output_tok_s'] for r in ss]
        for r in ss:
            scatter=ax.scatter(r['prefill95'],r['decode99'],s=60,c=[r['output_tok_s']],cmap='viridis',
                vmin=min(throughputs),vmax=max(throughputs),marker='^' if r['policy'] in {'load','aimd','joint','aging'} else 'o')
            ax.annotate(r['policy'],(r['prefill95'],r['decode99']),xytext=offsets.get(r['policy'],(8,8)),
                textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','lw':.5,'alpha':.5})
        fig.colorbar(scatter,ax=ax,label='Fixed-trace output tok/s')
        ax.margins(.15);ax.set_xlabel('Prefill P95 TTFT (ms)');ax.set_ylabel('Decode P99 token event gap (ms)');ax.grid(alpha=.3)
        fig.savefig(figdir/(scenario+'-pareto.png'),dpi=170);fig.savefig(figdir/(scenario+'-pareto.pdf'));plt.close(fig)

if __name__=='__main__':main()
