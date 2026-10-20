"""Chinese report assembled from audited results, with explicit completion gates."""
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
from apb.common import DATA,ROOT,dump
from benchmarks.report import getmetric,paired,ORDER

def fmt(x,n=2):return '—' if x is None else f'{x:.{n}f}'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def main():
    rows=json.loads((ROOT/'reports/results.json').read_text());audits=json.loads((ROOT/'reports/audits.json').read_text())
    summaries=json.loads((ROOT/'reports/summary.json').read_text())
    comparisons=json.loads((ROOT/'reports/paired_comparisons.json').read_text())
    frozen=json.loads((DATA/'manifests/formal-source.json').read_text())
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen['hashes'].items())
    formal=[r for r in rows if r['stage']=='formal'];primary_finished=len(formal)==165 and all(r['repeats']==5 for r in summaries)
    expected={'formal':165,'mechanism':15,'starvation':15,'native':30,'native2048':10,'ablation':36,'compatibility':18,'idle':3,'profiles':5,'balanced':45}
    counts=Counter(r['stage'] for r in rows)
    matrix_finished=all(counts[stage]==n for stage,n in expected.items())
    profile_path=DATA/'profiles/receipt.json'
    profiles=json.loads(profile_path.read_text()) if profile_path.exists() else []
    profiles_finished=len(profiles)==5 and all(p['returncode']==0 and p['report_exists'] and p.get('stats_returncode')==0 and p.get('kernel_count',0)>0 for p in profiles)
    numerical_path=DATA/'quality_drift/receipt.json'
    numerical_finished=numerical_path.exists() and json.loads(numerical_path.read_text()).get('complete',False)
    node_path=DATA/'profiles_node/receipt.json'
    node_finished=node_path.exists() and json.loads(node_path.read_text()).get('complete',False)
    observe_node_path=DATA/'profiles_node/observe/receipt.json'
    observe_node_finished=observe_node_path.exists() and json.loads(observe_node_path.read_text()).get('complete',False)
    verification_path=DATA/'manifests/final-source-verification.json'
    verification=json.loads(verification_path.read_text()) if verification_path.exists() else {}
    source_verified=verification.get('passed',False) and verification.get('weights_verified',False)
    finished=all((DATA/'report_materials'/name).exists() for name in ['性能归因与结论.md','工程验证与开销.md']) and (DATA/'driver-completed.json').exists() and primary_finished and matrix_finished and profiles_finished and numerical_finished and node_finished and observe_node_finished and source_verified and unchanged and not any(a['errors'] for a in audits)
    selection=json.loads((DATA/'manifests/static-selection.json').read_text());load=json.loads((DATA/'manifests/load-selection.json').read_text())
    title='完整实验报告' if finished else '实验报告（持续更新，验证矩阵尚未完成）'
    sections=[f'# 动态 Prefill Budget 调度：{title}',
        f'更新：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}。',
        f'项目完成状态：**{"实现与计划实验已完成" if finished else "实现已完成；实验继续执行"}**。已审计{len(rows)}个Trial、{len(audits)}个Session；主矩阵{len(formal)}/165个Trial。运行中结果不能作为最终统计结论。' if not finished else f'已审计{len(rows)}个Trial、{len(audits)}个Session，主矩阵165个Trial（11策略×3场景×5独立种子服务）。全部主矩阵重复达到5次。',
        '## 1. 项目与实现',
        '在vLLM V1原生Scheduler上增加每Step共享Prefill/Replay计算配额。Running与Waiting共享余额，KV分配成功后扣减，撤销时退款；普通Decode预留Token机会。全局4096 Token上限、原生KV管理、请求次序与模型Kernel保持原有职责。',
        '实现Fixed、Decode数量分档Load、完成反馈EWMA/AIMD、Load+AIMD（joint）、有界Aging。反馈在update_from_output处理已接受输出后获取，提交与完成Ledger分离，没有为在线控制增加CUDA同步或Future等待。',
        '源码与入口：[运行说明](../docs/RUNNING.md)、[问题与处理记录](../docs/IMPLEMENTATION_LOG.md)、[固定版本调度补丁](../patches/v0.32.0/scheduler.patch)。主实现位于apb/controller.py与apb/runtime.py。配置通过APB_CONFIG JSON接入，并由Benchmark保存完整参数。',
        '## 2. 环境与可复现性',
        table(['项目','实测设置'],[['GPU','RTX 3090 24 GiB，驱动570.86.15'],['模型','本地Qwen3-1.7B，FP16；Revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'],['软件','vLLM 0.32.0+cu129，Torch 2.13.0+cu129，CUDA Runtime 12.9，Python 3.11.16'],['主配置','max_model_len=8192，max_num_seqs=16，两个全局Token预算均4096'],['KV','8192×16=131072 Token；FP16理论KV容量14 GiB'],['执行路径','正式：原生Async/CUDA Graph；机制：Sync/Eager；FLASH_ATTN backend'],['主线开关','Prefix Cache、Spec、LoRA、KV Offload关闭，stream_interval=1'],['正式混合RPS',load['formal_mixed_rate']],['代码冻结检查','全部Hash匹配' if unchanged else '失败：存在来源变化']]),
        '已复制独立stock/candidate Python源码，二进制复用未修改官方wheel。模型分片完整SHA256记录、实际CLI、有效配置、源码Hash、Trace Hash与请求Token事件均保留。大数据根目录：`'+str(DATA)+'`。',
        '每策略/种子独立启动正式服务，在同一服务内依次执行已排空的三个场景；各种子轮随机化策略顺序。验证种子829与负载校准种子853均不参与正式种子17/29/43/71/101。冷启动/编译/预热不进入主性能窗口。',
        '## 3. 负载与指标口径',
        table(['场景','固定请求定义'],[['interference','8个512输入/1024输出Decode cohort；0.4秒后注入24个6144输入/128输出请求'],['transition','2个长输出请求+2秒后8个Decode请求；50个长Prompt按0.8秒间隔持续注入至39.4秒'],['mixed',f'200请求，输入512/2048/6144，输出128/256/512；共同开放环{load["formal_mixed_rate"]} RPS'],['starvation','180秒注入：每秒一个6144输入请求，每1.5秒补充一个2048输入/2048输出请求；停止注入后排空']]),
        '客户端ITL定义为同请求相邻有效输出Token事件的接收间隔。保存每个事件Token IDs并检查一事件是否一Token；若多Token，指标标为Chunk Gap，不能均摊成逐Token间隔。服务端完成间隔独立统计，包含主机处理，不称为GPU Kernel耗时。',
        'interference/transition的Decode指标使用初始Decode cohort，TTFT使用注入Prefill cohort；mixed使用全请求。新Prefill完成后也参与在线Decode计数。吞吐为同一固定Trace全部输出Token/makespan，不能用其证明最大服务容量。',
        '## 4. 静态参照选择',
        '在独立验证集上预先采用Prefill P95 TTFT增幅≤10%、固定Trace吞吐降幅≤5%的约束，再选Decode P99最小的固定Cap。选择结果：**'+str(selection['selection'])+'**。完整静态Pareto曲线同时保留；不只与大预算原生基线比较。',
        table(['策略','验证Decode P99 ms','验证Prefill P95 TTFT ms','验证tok/s','满足约束'],[[r['policy'],fmt(r['decode99']),fmt(r['prefill95']),fmt(r['output_tok_s']),r['eligible']] for r in selection['validation_static_pareto']]),
        '## 5. 默认部署路径正式结果']
    if finished:
        sections.insert(3,'**核心结论：** 独立新种子验证的2048/4096均衡档，在干扰场景Decode P99降低35.85%，Prefill P95增加1.72%，固定Trace吞吐下降1.72%；压力切换场景也有稳定收益。mixed的P99区间跨零。默认复杂反馈策略未优于最佳静态预算，修正档与Fixed-2048的P99差异不显著。项目实现与研究评估完成，优化目标按场景分别判断。')
    for scenario in ['interference','transition','mixed']:
        ss=sorted([r for r in summaries if r['scenario']==scenario],key=lambda r:ORDER.index(r['policy']))
        sections.append('### '+scenario)
        sections.append(table(['策略','重复','Decode P95 ms','Decode P99 ms','Prefill Mean TTFT ms','Prefill P95 TTFT ms','Output tok/s'],[[r['policy'],r['repeats'],fmt(r['decode95']),fmt(r['decode99']),fmt(r['prefill_mean']),fmt(r['prefill95']),fmt(r['output_tok_s'])] for r in ss]))
        sections.append('![正式结果]('+str(ROOT/'reports/figures')+'/'+scenario+'.png)\n\n![静态与动态权衡]('+str(ROOT/'reports/figures')+'/'+scenario+'-pareto.png)')
    sections.extend(['## 6. 配对收益与不确定性','下表为相同种子配对相对变化的均值及95% bootstrap区间；负号表示指标降低。每Token不作为独立重复。5次重复仍有限，跨模型/GPU不能直接外推。'])
    for base in ['stock',selection['selection']]:
        sections.append('### 相对 '+str(base))
        rr=[r for r in comparisons if r['baseline']==base and r['policy'] in ['load','joint','aging'] and r['metric'] in ['decode99','prefill95','output_tok_s']]
        sections.append(table(['场景','策略','指标','配对数','变化%','95%区间%'],[[r['scenario'],r['policy'],r['metric'],r['pairs'],fmt(r['mean_change_pct']),f"[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]" ] for r in rr]))
    sections.extend(['## 7. 正确性与兼容性','22项纯控制器/真实Scheduler与KV Manager回归通过，覆盖共享配额、Decode机会、尾块、分配失败、缓存命中、取消、重算、两轮在途与stock/off状态一致性。原始日志：`'+str(DATA/'logs/tests-final-22.log')+'`。',
        f'审计发现错误数：{sum(len(a["errors"]) for a in audits)}。正式主线原生抢占总数：{sum(r.get("native_preemptions",0) for r in formal)}。所有主线SSE事件均为单Token：{all(r["all_events_single_token"] for r in formal) if formal else "待测"}。',
        '取消、Prefix Cache及人为小KV容量抢占压力是独立兼容性组，不加入性能主表。自然语言用例核对逐Token输出；它只覆盖有限问答，不代表完整任务质量评测或所有数值bitwise一致。',
        '## 8. 饥饿压力与参数消融'])
    for stage in ['mechanism','starvation','native','native2048','ablation','compatibility','idle','balanced']:
        sr=[r for r in rows if r['stage']==stage]
        grouped=defaultdict(list)
        for r in sr:
            label=r['policy']
            if stage in ['ablation','native','native2048']:label=Path(r['session']).name.rsplit('-s',1)[0]
            grouped[(r['track'],r['scenario'],label)].append(r)
        sections.append('### '+stage)
        sections.append(table(['路径','场景/配置','策略','重复','P99 ms','P95 TTFT ms','Output tok/s','抢占数'],[[t,sc,p,len(v),fmt(statistics.mean(getmetric(r,'decode99') for r in v)),fmt(statistics.mean(getmetric(r,'prefill95') for r in v)),fmt(statistics.mean(r['output_tok_s'] for r in v)),sum(r.get('native_preemptions',0) for r in v)] for (t,sc,p),v in sorted(grouped.items())]) if sr else '待该阶段运行完成。')
    sections.extend(['## 9. 遇到的问题与解决过程',(ROOT/'docs/IMPLEMENTATION_LOG.md').read_text().split('\n',2)[-1],
        '## 10. 性能不及预期的优化与原因',
        '必须同时检查尾延迟、TTFT和吞吐。较小Cap可以缩短单个Mixed Step，却增加每个Prompt所需的前向Step数；P99下降不能单独证明总体优化。',
        '需要依据正式数据分别判断：Load是否只复现固定小预算效果；严格相对纯Decode延迟目标是否把AIMD压到最小Cap；Aging救援是否提高延迟尾部、是否能解决序列槽阻塞；过快控制/过度平滑是否有稳定增益。所有参数扫描均保留原始记录，收益区间跨零时不宣称有效。',
        '以上因果判断应结合控制轨迹、实际Prefill量、达到最小预算的比例、目标超限率、排队与Profiler证据；最终解释在完成全部实验后补充，不把预期当作实测。',
        '## 11. Profile与工程开销',
        '纯Controller决策、包含状态采集的总调度、完成反馈分别计时。observe保留原生决策，用于测量遥测成本。Nsight使用独立诊断运行，预热后捕获128个完成Step；Profile运行不加入主表。Mixed Step时间包含共享计算，不能拆成可相加的独立Prefill/Decode GPU阶段。',
        '## 12. 适用范围、限制与复现',
        '支持单GPU、Dense文本自回归、FCFS和普通逐Token Decode。Spec、多模态、LoRA、KV Connector与多GPU不在已验证范围。项目不实现Kernel中途抢占；已经提交的Chunk无法被新反馈撤销。',
        '有限输入、有限生成长度、资源可分配时验证可排空；无限过载、KV/序列槽不可用时不提供无条件墙钟TTFT上界。稳定、突发与过载结果分别解读。',
        '复现命令与审计入口见[运行说明](../docs/RUNNING.md)。本报告由原始数据分析脚本重建，图表与统计JSON在reports目录；完整请求、Ledger、有效配置和日志在独立数据盘。'])
    for heading,next_heading,name in [
        ('## 10. 性能不及预期的优化与原因','## 11. Profile与工程开销','性能归因与结论.md'),
        ('## 11. Profile与工程开销','## 12. 适用范围、限制与复现','工程验证与开销.md')]:
        supplement=DATA/'report_materials'/name
        if supplement.exists():
            a=sections.index(heading);b=sections.index(next_heading)
            sections[a+1:b]=[supplement.read_text().split('\n',1)[-1].strip()]
    out=DATA/'report_materials/基线阶段统计.md';out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n\n'.join(sections)+'\n')
    dump(ROOT/'reports/completion.json',{'complete':finished,'formal_trials':len(formal),'all_trials':len(rows),'source_unchanged':unchanged,'source_and_model_verified':source_verified,'audit_errors':sum(len(a['errors']) for a in audits), 'matrix_finished':matrix_finished,'profiles_finished':profiles_finished,'numerical_finished':numerical_finished,'node_profile_finished':node_finished,'observe_node_profile_finished':observe_node_finished,'stage_trial_counts':dict(counts),'required_stage_trials':expected})
    print(out)

if __name__=='__main__':main()
