"""Auditable, paired summaries for the separate optimization experiments."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import random
from apb.common import DATA,ROOT,dump,digest
from benchmarks.analyze import audit_session
from benchmarks.report import getmetric,paired

OUT=ROOT/'reports/optimization_v2'
KEYS=['decode99','prefill95','output_tok_s']

def main():
    rows=[];audits=[]
    for done in sorted((DATA/'raw').glob('*/*/completed.json')):
        rr,receipt=audit_session(done.parent);rows.extend(rr);audits.append(receipt)
    if any(a['errors'] for a in audits):raise RuntimeError('audit failed')
    groups=defaultdict(list)
    traces=defaultdict(set)
    for row in rows:
        groups[(row['stage'],row['scenario'],row['policy'])].append(row)
        traces[(row['stage'],row['scenario'],row['seed'])].add(row['trace_sha256'])
    if any(len(h)!=1 for h in traces.values()):raise RuntimeError('unmatched traces')
    summaries=[];comparisons=[]
    for (stage,scenario,policy),rs in sorted(groups.items()):
        if scenario not in ['interference','transition','mixed']:continue
        summaries.append(dict(stage=stage,scenario=scenario,policy=policy,repeats=len(rs),
            **{k:statistics.mean(getmetric(r,k) for r in rs) for k in KEYS}))
    for stage,scenario in sorted({(r['stage'],r['scenario']) for r in summaries}):
        rr=[r for r in rows if r['stage']==stage and r['scenario']==scenario]
        for policy in sorted({r['policy'] for r in rr}):
            for base in ['stock','balanced','fixed1536','fixed1792']:
                if policy==base:continue
                for metric in KEYS:
                    value=paired(rr,policy,base,metric,scenario=scenario)
                    if value:comparisons.append(dict(stage=stage,scenario=scenario,policy=policy,baseline=base,metric=metric,**value))
    artifacts={'results':rows,'audits':audits,'summary':summaries,'comparisons':comparisons,
        'trace_audit':[dict(stage=k[0],scenario=k[1],seed=k[2],hashes=sorted(v)) for k,v in sorted(traces.items())]}
    for name,value in artifacts.items():dump(OUT/(name+'.json'),value)
    text=['# 动态 Prefill Budget 第二轮优化实验', '',
          '本轮将历史数据用于定位瓶颈和拟合成本模型，新种子用于初筛与独立验证。所有数值来自实际 GPU 请求；原始 421 个 Trial 保持独立。', '',
          '## 实验汇总', '',f'当前完成 {len(rows)} 个 Trial / {len(audits)} 个独立服务；审计错误 0。', '',
          '| 阶段 | 场景 | 策略 | 重复 | Decode P99 ms | Prefill P95 ms | 输出 tok/s |',
          '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in summaries:
        text.append(f"| {r['stage']} | {r['scenario']} | {r['policy']} | {r['repeats']} | {r['decode99']:.2f} | {r['prefill95']:.2f} | {r['output_tok_s']:.2f} |")
    text+=['','## 配对变化','','先逐种子计算变化，再按种子重采样 5000 次给出 95% bootstrap 区间。单种子初筛的退化区间没有统计推断意义。', '',
           '| 阶段 | 场景 | 策略 / 对照 | 指标 | 变化 % | 95% 区间 % |',
           '| --- | --- | --- | --- | ---: | --- |']
    for r in comparisons:
        if not r['policy'].startswith('equalized'):continue
        lo,hi=r['ci95']
        text.append(f"| {r['stage']} | {r['scenario']} | {r['policy']} / {r['baseline']} | {r['metric']} | {r['mean_change_pct']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    (OUT/'实验数据表.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'trials':len(rows),'sessions':len(audits),'summary':summaries},ensure_ascii=False,indent=2))
    plot(summaries,rows)

def plot(summaries,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in sorted({r['stage'] for r in summaries}):
        fig,axes=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
        for i,scenario in enumerate(['interference','transition','mixed']):
            ss=[r for r in summaries if r['stage']==stage and r['scenario']==scenario]
            order=['stock','balanced','fixed1536','fixed1792','equalized','equalized256','equalized1536']
            ss.sort(key=lambda r:order.index(r['policy']) if r['policy'] in order else 99)
            for ax,key,label in zip(axes[i],KEYS,['Decode P99 gap (ms)','Prefill P95 TTFT (ms)','Fixed-trace output tok/s']):
                low=[];high=[];rng=random.Random(2048)
                for s in ss:
                    xs=[getmetric(r,key) for r in rows if r['stage']==stage and r['scenario']==scenario and r['policy']==s['policy']]
                    boots=sorted(statistics.mean(rng.choices(xs,k=len(xs))) for _ in range(5000))
                    low.append(max(0,s[key]-boots[125]));high.append(max(0,boots[4874]-s[key]))
                ax.bar([r['policy'] for r in ss],[r[key] for r in ss],yerr=[low,high],capsize=3)
                ax.set_title(scenario);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        fig.savefig(OUT/(stage+'.png'),dpi=160);fig.savefig(OUT/(stage+'.pdf'));plt.close(fig)

if __name__=='__main__':main()
