"""Build the single project report from the preserved, audited JSON summaries.

This command needs only the Python standard library and does not run inference.
Stage writers keep their working prose under APB_DATA/report_materials instead
of overwriting the project's final report.
"""
import json
import statistics
from apb.common import ROOT

REPORT = ROOT / 'reports/动态Prefill_Budget完整实验报告.md'
OUT = ROOT / 'reports/optimization_v2'
SCENARIOS = ['interference', 'transition', 'mixed']
NAMES = dict(interference='长 Prompt 干扰', transition='压力切换', mixed='混合流量')


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return ('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
            + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows))


def main():
    old = read(ROOT / 'reports/results.json')
    rows = read(OUT / 'results.json')
    audits = read(ROOT / 'reports/audits.json') + read(OUT / 'audits.json')
    completion = read(OUT / 'completion.json')
    comparisons = read(OUT / 'comparisons.json')
    summaries = read(OUT / 'summary.json')
    diag = read(OUT / 'diagnostics.json')
    fit = read(OUT / 'cost_fit.json')
    profiles = read(OUT / 'profile_summary.json')
    selection = read(OUT / 'selection.json')
    preset = read(ROOT / 'configs/presets/balanced_v2.json')
    assert len(old) == 421 and len(rows) == 134 and len(audits) == 227
    assert all(not a['errors'] for a in audits)
    assert completion['selected_policy'] == selection['selected_policy'] == 'fixed1536'
    assert completion['source_passed'] and diag['source']['passed']
    assert preset == dict(mode='fixed', cap=1536, maximum=4096, fixed_idle_relax=True, reserve=True)

    def comparison(policy, baseline, scenario, metric, stage='opt_validation'):
        return next(c for c in comparisons if c['stage'] == stage and c['policy'] == policy
                    and c['baseline'] == baseline and c['scenario'] == scenario and c['metric'] == metric)

    def change(policy, baseline, scenario, metric, stage='opt_validation'):
        c = comparison(policy, baseline, scenario, metric, stage)
        lo, hi = c['ci95']
        return f"{c['mean_change_pct']:+.2f}% [{lo:+.2f}, {hi:+.2f}]"

    def changes(policy, baseline, stage='opt_validation'):
        return table(['场景', 'Decode P99', 'Prefill P95 TTFT', '输出 tok/s'],
                     [[NAMES[s]] + [change(policy, baseline, s, k, stage)
                       for k in ['decode99', 'prefill95', 'output_tok_s']] for s in SCENARIOS])

    def absolute(stage, policies):
        result = []
        for s in SCENARIOS:
            for p in policies:
                r = next(r for r in summaries if r['stage'] == stage and r['scenario'] == s and r['policy'] == p)
                result.append([NAMES[s], p, r['repeats']] + [f'{r[k]:.2f}' for k in ['decode99', 'prefill95', 'output_tok_s']])
        return table(['场景', '策略', '独立种子数', 'Decode P99 ms', 'Prefill P95 ms', '输出 tok/s'], result)

    parts = ['''# PDSchedule：动态 Prefill Budget 调度项目完整实验报告

更新：2026-10-21。本文是项目唯一的完整实验报告，按当前实现和最终配置组织；阶段协议与原始统计继续独立保存，便于追溯，不把不同种子集合混成同一组重复。

## 1. 当前结论与最新改动

最终工程配置为 **1536/4096 两档**：有可推进 Decode 请求时，整轮共享 Prefill/Replay 配额为1536；无 Decode 时放宽到4096。继续保留原生全局 token 上限和 Decode token 预留。配置见 [balanced_v2.json](../configs/presets/balanced_v2.json)。

在预先登记的五种子验证中，相对同轮新鲜启动的原生 vLLM，三个场景 Decode P99 分别降低49.80%、43.91%、21.20%；Prefill P95 增加1.48%、7.81%、4.67%，固定 Trace 输出吞吐下降1.13%、0.04%、0.03%。这批验证满足 TTFT≤+10%、吞吐≥−5%的约束，区间见第5节。额外三种子中，mixed 的 TTFT 约束区间未完全通过，不能承诺对其他 Trace 也稳定控制在10%以内。

项目实现了负载分档、EWMA/AIMD、Joint、Aging和上下文成本均分，但复杂策略未稳定优于最终两档规则。当前收益不能表述为 AIMD 或在线学习的增量收益，也不代表最大稳定服务容量提升。

最新实现与优化包括：

- 在已有共享配额和异步记账上增加上下文成本拟合、`equalized`求解器以及256/512对齐验证；保留不达标结果。
- 补充1536、1792中间预算，独立验证后将工程配置从2048/4096更新为1536/4096。
- 增加同基础预算对照和真实Scheduler批处理状态实验，定位尾块收紧共享配额导致的批处理损失。
- 回归扩展到30项，补齐部分缓存前沿、同步/异步共享配额、抢占Replay及清理检查。
- 完成节点粒度Nsight与合成输入数值诊断；后台运行器支持中断恢复、来源冻结与完整交付快照。
- 本次仅整合报告与生成入口，未改动在线调度参数、模型权重或既有测量数据，未新增性能实验。

## 2. 针对的问题与实现改进

长Prompt的Prefill一次处理大量token，与已有Decode共用执行批次时会拉长输出间隔。切块过小又会增加前向次数、Prefill排队与首token延迟，项目目标是约束这三类指标的取舍。

原生vLLM已经提供Chunked Prefill、连续批处理、KV Cache、Async与CUDA Graph。本项目在其Scheduler上增加一层整轮计算配额控制。原生长Prefill阈值作用于单请求，本项目限制Running与Waiting合计的Prompt/Replay计算量；二者不是同一控制量，原生阈值也作为独立对照测试。

| 改进 | 作用与边界 |
| --- | --- |
| 每轮共享Prefill配额 | Running/Waiting共用余额，限制多个Prefill的合计工作量；不替换原生两个全局预算 |
| Decode预留 | 为本轮符合条件的普通Decode保留token机会，防止先遍历的Partial Prefill耗尽预算；不构成硬实时保证 |
| 实际计算分类 | Prompt、恢复重算Replay、新Decode分开记账；Prefix Cache命中不扣计算量 |
| 分配后扣费与退款 | KV分配成功才扣减，撤销或抢占已排入的分配时同步退款 |
| 完成反馈 | 用已接受输出的完成事件更新观测，提交与完成分别保存Ledger，避免重复或过期反馈 |
| 自适应策略 | Load按Decode数量分档，AIMD按完成间隔调整预算，Joint组合，Aging周期救援；无Decode时放宽 |
| 可观测与审计 | 保存预算、实际计算区间、队列、输出事件、CPU计时、来源Hash和完成状态 |

主要实现：[controller.py](../apb/controller.py)、[runtime.py](../apb/runtime.py)、[Scheduler补丁](../patches/v0.32.0/scheduler.patch)。沿用原生请求顺序、KV管理、抢占受害者规则和模型kernel。在线反馈没有新增CUDA强制同步或Future等待；独立Profiler组使用诊断API。

## 3. 实验环境、负载与指标

| 项目 | 实测设置 |
| --- | --- |
| 硬件 | 单RTX 3090 24 GiB，驱动570.86.15 |
| 模型 | Qwen3-1.7B，FP16；revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| 软件 | vLLM 0.32.0+cu129，Torch 2.13.0+cu129，CUDA Runtime 12.9，Python 3.11.16 |
| 主配置 | max_model_len=8192，max_num_seqs=16，两个全局token预算均4096 |
| KV配置 | 8192×16=131072 token容量；FP16理论KV容量14 GiB |
| 执行路径 | 主要性能结果使用默认Async/CUDA Graph；机制实验另测Sync/Eager |
| 主线开关 | Prefix Cache、Spec、LoRA、KV Offload关闭，stream_interval=1 |

| 场景 | 请求定义与指标cohort |
| --- | --- |
| interference | 8个512输入/1024输出请求；0.4秒后注入24个6144输入/128输出请求；Decode指标看原有8请求，TTFT看注入组 |
| transition | 2个长输出请求，2秒后加入8个Decode请求；50个长Prompt每0.8秒注入至39.4秒；使用对应Decode与Prefill cohort |
| mixed | 200请求，输入512/2048/6144、输出128/256/512；共同开放环2.422 RPS，指标看全部请求 |
| 持续压力 | 注入180秒：每秒一个6144输入请求，每1.5秒补充一个2048输入/2048输出请求；停止后排空，单独分析 |

ITL是同请求相邻有效输出token事件的客户端接收间隔，保存token IDs并检查单事件是否单token。多token事件不均摊伪造ITL。服务端完成间隔单独保存，包含主机处理，不能称为GPU kernel时间。TTFT是请求发送到首token的时延。

输出吞吐=固定Trace全部输出token数/包含注入与排空的makespan，不能据此证明最大容量、稳定系统吞吐或SLO goodput。mixed与固定Decode cohort的P99也不是同一统计总体。

每个正式策略/种子单独启动服务，服务内依次执行已排空场景；轮内随机化策略顺序。冷启动、编译、预热不计入主窗口。逐服务计算指标，再按相同Trace/种子配对；95%区间采用5000次按种子bootstrap，未作多重比较或选择校正，token不作为独立重复。

## 4. 实验规模与选择规则
''']
    parts.append(table(['材料', '有效Trial', '服务数', '用途'], [
        ['基线与策略评估', len(old), 192, '开发/标定/静态选择、165个正式Trial、机制/兼容性/消融等'],
        ['追加优化矩阵', len(rows), 35, '105个性能Trial、29个输出质量Trial'],
        ['累计', len(old) + len(rows), len(audits), '不是555个独立性能重复']]))
    parts.append('''旧归档的421个Trial含其既定Profile阶段计数；追加优化的5个Nsight节点服务和6个数值诊断服务独立于134个Trial。统计口径沿用原始归档，不为统一报告改写历史计数。

主要验证为种子307/311/313/317/331，原生、旧均衡档、Fixed-1536、Equalized-256各5个独立服务，共80个Trial。初筛seed227只用于探索；参数与正式种子在结果产生前登记。额外验证使用401/409/419，是根据前一阶段发现提出的独立工程对照，不并入原五种子。

选择规则：三个场景相对原生的Prefill P95点估计≤+10%、吞吐≥−5%，再比较两个预定主场景的配对Decode P99变化。区间验收与点估计验收分别列出。最终选择还检查同基础预算候选；没有把多个局部最优参数拼接为未经验证的组合。原有2048静态参考来自独立验证集，不是只和大预算原生比较。

## 5. 最终方案的性能效果

### 5.1 最新五种子验证的绝对指标
''')
    parts.append(absolute('opt_validation', ['stock', 'balanced', 'fixed1536', 'equalized256']))
    parts.append('''`balanced`是先前2048/4096两档，`fixed1536`是当前1536/4096两档。表中先对各服务取指标再取均值；配对百分比先逐种子计算，因此不一定等于两个均值的直接比值。

### 5.2 当前配置相对同轮原生

单元格为配对变化均值 [95%区间]；负号表示下降。
''')
    parts.append(changes('fixed1536', 'stock'))
    guard_rows = []
    for s in SCENARIOS:
        d, p, t = [comparison('fixed1536', 'stock', s, k) for k in ['decode99', 'prefill95', 'output_tok_s']]
        guard_rows.append([NAMES[s], '是' if p['mean_change_pct'] <= 10 and t['mean_change_pct'] >= -5 else '否',
                           '是' if p['ci95'][1] <= 10 and t['ci95'][0] >= -5 else '否',
                           '是' if d['ci95'][1] < 0 else '否'])
    parts.append(table(['场景', '点估计约束通过', '95%区间约束通过', 'P99改善区间不跨零'], guard_rows))
    parts.append('### 5.3 当前配置相对旧2048/4096均衡档')
    parts.append(changes('fixed1536', 'balanced'))
    parts.append('''主要收益是Decode尾延迟进一步下降约两成，Prefill TTFT变化较小，固定Trace吞吐基本持平。这是工作划分的延迟取舍，不是模型每token计算量减少。

![独立验证](optimization_v2/opt_validation.png)

### 5.4 额外三种子对当前配置的复核
''')
    parts.append(changes('fixed1536', 'stock', 'opt_match1536'))
    parts.append(r'''额外种子的mixed TTFT增幅点估计8.89%仍在10%内，但95%区间上界15.10%越界。它不推翻五种子那批已测结果，也不能被五种子的通过掩盖。当前配置是本机工程选择，尚不是对任意Trace都满足TTFT约束的保证。

## 6. 瓶颈、尝试的优化与未采用原因

### 6.1 原反馈目标过紧，切块与排队损失占主导

原Joint将纯Decode参考P90的1.2倍作为目标，在混合计算中经常不可达。interference/transition实际含Prefill步骤分别有90.31%/97.95%停在256下限。6144-token Prompt按256理论上需24次推进，按2048只需3次；更多前向与排队增加累计成本。

原正式干扰场景，Joint相对原生Decode P99降低81.17%，但Prefill P95增加58.21%、输出吞吐下降17.88%，不满足均衡约束。Load的对应变化为−74.65%、+27.02%、−10.32%。这类结果不能只报告尾延迟下降。

原单因素消融中，提高Joint下限至512/1024分别降低其Prefill P95约19.78%/27.48%，但Decode P99增加33.29%/111.50%；调alpha=0.1或更频繁Aging救援只改变取舍，没有修复目标可行性。旧均衡档虽改善两个主场景，mixed的P99区间跨零，因此补测中间预算。

### 6.2 上下文成本模型与均分策略

历史2048-token切块的Decode输出间隔中位数，在起点0/2048/4096时约123/138/152 ms，说明token数量相同不意味着步骤成本相同。用历史固定512/1024/2048/4096的seed17/29/43拟合，71/101留出验证：

$$
\widehat T=6.302+41.679(P/1024)+5.026(A/10^6)+0.091N_d+3.320(C_d/10^4)\quad\mathrm{ms}
$$

其中 $A=\sum_i n_i(s_i+(n_i+1)/2)$，P是实际Prefill token，N_d是实际Decode token数，C_d是Decode上下文总量。模型预测主机观测的完成间隔，不是算子或GPU独立计时。
''')
    parts.append(f"训练{fit['train_steps']}步，稳健重拟合保留{fit['retained_steps']}步；留出{fit['valid_steps']}步，MAE={fit['validation_mae_ms']:.2f} ms，绝对误差P95={fit['validation_absolute_error_p95_ms']:.2f} ms。完整输入Hash见[cost_fit.json](optimization_v2/cost_fit.json)。")
    parts.append(r'''定义 $W(s,n)=n[1+w(s+(n+1)/2)]$，w=0.000123475。剩余R、基础预算B，令k=ceil(R/B)，每轮在实际计算前沿求解W(s,n)=W(s,R)/k并向上对齐。共享配额由首个实际Prefill确定；缓存前沿、Replay与Decode预留继续按实际工作记账。该模式是离线标定的前馈控制，AIMD不决定其有效预算。

6144-token Prompt的典型切块：B2048/对齐16为2464/1984/1696，对齐256为2560/2048/1536；B1536/对齐512为2048/1536/1536/1024。

### 6.3 规整对齐和中间固定预算的筛选

seed227探索中，16对齐均分的transition Prefill P95为505.90 ms，原生439.74 ms，超出约束；Fixed-1536为487.75 ms也略超。改为256对齐后降至466.70 ms，故进入新种子验证。Fixed-1792则达到627.27 ms，未采用。

1792切块的典型形状为1792/1792/1792/768，长Prompt平均推进4.125次，跨请求组合与排队更不利；transition首次调度前排队P95为78.64 ms，Fixed-1536为22.99 ms。预算处在两个数值之间不意味着性能也在二者之间。分位数不相加，不能把全部损失精确归因于单一kernel。

初筛与修正探索复用seed227基线，存在运行间隔，只用于候选筛选。最终结论来自新鲜启动的独立种子基线。

### 6.4 成本均分与更强静态对照

Equalized-256相对Fixed-1536的五种子结果：
''')
    parts.append(changes('equalized256', 'fixed1536'))
    parts.append('该候选在两个主场景优于旧均衡档，但未超过Fixed-1536，mixed还出现退化。进一步用三个新种子比较同为1536基础预算的候选：')
    parts.append(changes('equalized1536', 'fixed1536', 'opt_match1536'))
    parts.append('Equalized-1536相对原生的三种子结果：')
    parts.append(changes('equalized1536', 'stock', 'opt_match1536'))
    parts.append('''该候选在两个主场景进一步降低P99，但transition/mixed的TTFT增加15.48%/14.45%，均超出点估计约束；mixed P99还比同轮Fixed-1536增加0.76%。最终未晋级，保留为实验模式。

为核查批处理机制，使用真实V1 Scheduler/KV Manager，配置一个存量Decode和八个2048-token Prompt，模型输出用确定性模拟器提供。以下是CPU状态实验，不测GPU性能：
''')
    parts.append(table(['模式', 'Async', '含Prefill步骤', '含多个Prefill请求步骤', 'Prompt token总量'],
                       [[r['mode'], r['async_scheduling'], r['prefill_steps'], r['multi_prefill_request_steps'], r['prompt_tokens']]
                        for r in diag['batch_probe']['results']]))
    parts.append('''均分模式在请求尾部收紧整轮共享配额，固定模式可以把剩余额度给后续请求，因此相同总量需要更多全局步骤。该机制与GPU Trace中的尾块、批处理和排队记录共同解释负结果；CPU步骤数不直接等于GPU加速比。

![同基础预算验证](optimization_v2/opt_match1536.png)

## 7. 正确性、兼容性与输出差异

### 7.1 预算与状态

当前30项回归通过，覆盖Controller、真实Request/Scheduler/KV Manager、同步/异步共享配额、Decode机会、尾块、部分缓存前沿、分配失败、退款、取消、Replay及两轮在途。stock/off的固定事件序列调度与KV状态对照通过；未运行上游全部测试套件。

两份有效审计表共227个服务均无错误。带Ledger的组检查：全局预算≤4096、Prompt+Replay≤有效配额、分区之和等于提交量、非空提交与完成一一对应、关闭后无在途状态或遥测丢失。observe只记录，不改变原生分配。

### 7.2 Prefix Cache、取消与真实抢占

缓存专组每组总输入18432 token，命中10224、实际计算8208，满足对账。usage未返回cached_tokens细节时保留缺失值，结合服务端计数器与实际前沿验证，不能把缺失误填为0。

取消组共80次取消尝试，后续验证请求均完成，缓存重置通过。专门缩小到640个KV Blocks并关闭完整长度预留的GPU组，在默认/同步路径分别触发115/106次原生抢占，每条路径10请求全部完成；Replay各240 token受配额约束。它们改变了KV配置，不混入主性能收益。

### 7.3 最新输出一致性与数值诊断
''')
    quality_rows = []
    natural = []; tail = []
    for stage, policy in [('opt_validation', p) for p in ['balanced', 'fixed1536', 'equalized256']] + [('opt_match1536', p) for p in ['fixed1536', 'equalized1536']]:
        qs = [c for q in diag['quality'] if q['stage'] == stage and q['policy'] == policy for c in q['comparisons']]
        n = [c for c in qs if c['cohort'] == 'natural']; t = [c for c in qs if c['cohort'] == 'tail']
        natural += n; tail += t
        quality_rows.append([stage, policy, f"{sum(c['exact'] for c in n)}/{len(n)}", f"{sum(c['exact'] for c in t)}/{len(t)}"])
    assert sum(c['exact'] for c in natural) == len(natural) == 126
    assert sum(c['exact'] for c in tail) == 104 and len(tail) == 105
    parts.append(table(['数据阶段', '策略', '自然问答逐token一致', '合成尾块逐token一致'], quality_rows))
    parts.append('''自然问答126/126一致，合成尾块104/105一致，总计230/231。Fixed-1536/seed419的17-token输入在第11个输出分歧：原生429（“ that”），固定档323（“ and”），共同前10个token不变。原始差异保留，不改为全部一致。

随后相同Trace、关闭Prefix Cache的新鲜stock/off/Fixed-1536各两次，共六个诊断服务，原始correctness重复均回到原stock分支，每次stock/off及stock/Fixed均11/11一致。额外自回归探测中，两次未修改stock在offered batch=2时均完整复现原Fixed的32-token分支，第11位候选logprob并列，支持原生批次/数值敏感性的解释。

offered batch是同时提供的请求数，不等于实际每步GPU批大小；诊断开启logprobs，不能断言还原了原始运行的精确数值路径。未发现新增调度记账错误，有限一致性检查不替代任务质量基准，也不保证任意FP16异步批形逐token相同。

此前stock/off兼容性组也保留一例第18个token差异，后续两次未修改stock复现该分支。该历史诊断与上述最新诊断独立，不混算231次比较。

数值诊断首次logprobs触发Top-K内核JIT，原性能审计的拒绝项与日志保留。诊断单独检查请求、长度、Trace与响应完整性，不使用其时延/吞吐作证据，未放宽主矩阵JIT审计。

证据：[原始比较汇总](optimization_v2/diagnostics.json)、[公共前缀与logprobs](optimization_v2/numerical_probe.json)、[完整32-token复核](optimization_v2/numerical_batch_reproduction.json)。

## 8. CPU开销、GPU诊断与持续压力

### 8.1 CPU与遥测

旧五服务统计中，纯Joint决策P95均值约11.90 µs，反馈约64.87 µs。observe相对stock的配对吞吐变化最差区间下界为−0.425%，支持该批Trace约2%的遥测预算。没有为未修改stock建立同构CPU内部计时，因此不能宣称新增CPU P95小于原生的10%。

最新组先对每服务取P95，再取服务均值：
''')
    cpu_rows = []
    for stage, policy in [('opt_validation', p) for p in ['balanced', 'fixed1536', 'equalized256']] + [('opt_match1536', p) for p in ['fixed1536', 'equalized1536']]:
        rs = [r for r in diag['cpu'] if r['stage'] == stage and r['policy'] == policy]
        cpu_rows.append([stage, policy] + [f"{statistics.mean(r[k]['p95'] for r in rs):.2f}" for k in ['decision_us', 'scheduler_us', 'feedback_us']])
    parts.append(table(['阶段', '策略', 'decide P95 µs', '完整调度 P95 µs', '反馈 P95 µs'], cpu_rows))
    parts.append('''decide不含clip中的成本求解，完整调度包含求解与原生状态处理。不同策略的请求状态和推进次数不同，完整调度差值不能直接视为纯算法开销。现有证据更支持GPU工作划分、重复前向与排队是主要损失来源。

### 8.2 GPU节点采集

最新五组独立Nsight节点采集使用seed919，预热后128个完成回调；全部通过记账审计，不纳入性能表。
''')
    profile_rows = []
    for c in profiles['captures']:
        g = c['gpu']; cats = g['name_based_categories']; w = c['completed_work']
        profile_rows.append([c['policy'], w['prompt_tokens'], w['decode_tokens'], g['exposed_kernel_ms']['count'], g['graph_internal_kernel_count'],
                             f"{cats.get('GEMM', {}).get('duration_ms', 0):.2f}", f"{cats.get('Attention', {}).get('duration_ms', 0):.2f}"])
    parts.append(table(['策略', '逻辑完成Prompt token', 'Decode token', '可见kernel数', '图内kernel数', 'GEMM活动ms', 'Attention活动ms'], profile_rows))
    parts.append('''函数名分类是启发式归类。不同窗口的请求状态、Decode工作量和边缘在途工作不同，Profiler也会扰动执行；kernel累计活动时间不等于墙钟、利用率或可相加的Prefill/Decode成本，不能按其比值宣称收益。

早期graph粒度采集遗漏图内kernel，后续显式node采集补足覆盖；没有图节点ID的窗口不能包装为图内算子验证。当前没有充分证据宣称带宽饱和、Tensor Core利用率提升或完整Roofline结论。时钟未锁定，随机化与新鲜基线只能降低漂移影响。

### 8.3 持续压力与Aging边界

180秒持续注入中三种策略的308请求均完成，但Fixed-512/Joint/Aging测量时长约363.70/371.38/365.11秒，停止注入后仍需184.70/192.38/186.11秒排空。等待队列持续增长，是有限过载，不是稳定容量证明。

Aging五次共触发10680个救援步骤，仅1623个有实际Prefill计算。提高token配额不能直接释放序列槽或改变FCFS准入，未证明Aging优于合理静态，也未提供无限过载下的墙钟等待上界。

## 9. 遇到的问题与处理

| 问题 | 处理与保留证据 |
| --- | --- |
| 源码补丁锚点误匹配Running/Waiting | 增加唯一匹配与源码Hash断言，失败时不修改原安装 |
| 调度类型/缓存前沿导致记账错误风险 | 按实际计算区间分类；缓存命中不收费，Replay计费，分配撤销退款；真实Scheduler回归 |
| Request到达时间使用Unix时钟 | 等待年龄改用同一时基，输出间隔仍用单调时钟 |
| API请求ID扩展与cohort无法关联 | 规范化前缀并验证后缀边界，保留原始ID，不按时间近似匹配 |
| 默认立即关停丢失末尾遥测 | 统一shutdown_timeout=20并重跑，失败产物保留，不补写summary伪造通过 |
| 相对纯Decode目标不可行 | 保留负结果，转向预算工作点、成本前馈和更强静态对照 |
| Fixed-1792不及相邻预算 | 检查尾块、跨请求组合和排队；未按预算数值线性推断性能 |
| 部分缓存测试缺少原生block_hasher | 修正所有测试请求的构造，保留真实KV断言，不修改原生缓存逻辑 |
| API缺少cached_tokens详情 | 缺失保留null，使用Prometheus与实际Prompt计算量交叉核验 |
| Profiler目录名balanced撞旧审计阶段 | 改node-前缀重采，旧失败保留，未混入有效Profile或性能表 |
| logprobs首次内核JIT | 保留性能拒绝项，数值诊断与性能矩阵分开，不放宽主性能审计 |
| 交互中断后运行器退出、API服务孤立 | 核实本项目服务空闲后清理；采用独立Session后台主进程、锁和阶段状态 |
| seed317轮内出现约五小时间隔 | 两个已完成服务的8个Trial整轮归档，按原顺序重跑该种子四组；不按性能筛选 |
| 阶段报告重复、旧结论容易混淆 | 统一为本文，历史版本留Git记录与本地归档；阶段生成器不再覆盖最终报告 |

详细实施日志见[IMPLEMENTATION_LOG.md](../docs/IMPLEMENTATION_LOG.md)。恢复前后在线代码与模型冻结不变；失败与中断归档不计入主表有效样本。

## 10. 复现、来源与适用范围

仓库包含代码、补丁、配置、测试、本文、统计JSON和图表，不含模型、完整原始请求/服务日志、环境二进制与Nsight数据库。源码覆盖层复用未修改官方二进制，原安装与模型文件Hash已校验。

原实验机数据位置：

- `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320`：基线、策略矩阵与兼容性原始记录。
- `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2`：追加优化、协议、冻结源码、诊断与中断归档。

`APB_DATA`可选择数据目录，解释器/模型与部分辅助脚本仍使用原机器路径。其他机器需准备匹配环境并适配路径，不能直接继承原机器的验收标记。已完成数据的冻结来源不得被覆盖为新来源。

仅从仓库内已审计JSON重建本文，无需启动推理：

```bash
PYTHONPATH="$PWD" python -m benchmarks.project_report
```

完整推理实验的准备与运行见[RUNNING.md](../docs/RUNNING.md)、[恢复说明](../docs/优化实验运行与恢复.md)、[项目设计](../docs/动态Prefill_Budget调度项目设计_V2.md)。阶段性文字材料写入实验数据目录的`report_materials/`，最终项目报告统一由上述命令生成。

主要统计与审计入口：

- [基线与策略评估结果](results.json)、[配对比较](paired_comparisons.json)、[原始审计汇总](audits.json)。
- [最新优化结果](optimization_v2/results.json)、[配对比较](optimization_v2/comparisons.json)、[审计汇总](optimization_v2/audits.json)、[选择规则](optimization_v2/selection.json)。
- [成本拟合](optimization_v2/cost_fit.json)、[批处理状态实验](optimization_v2/batch_probe.json)、[工程诊断](optimization_v2/diagnostics.json)、[节点采集](optimization_v2/profile_summary.json)。

已验证范围是单GPU、Dense文本、FCFS、普通逐token生成。Spec、多模态、LoRA、KV Connector和多GPU未验收。没有kernel中途抢占；已提交的chunk不能被后续反馈撤销。样本、自然任务和负载有限，未锁频且无多重比较校正，不能外推到其他模型/GPU、无限时域或生产SLO。
''')
    REPORT.write_text('\n\n'.join(p.strip() for p in parts) + '\n')
    print('PROJECT_REPORT', REPORT, 'trials', len(old) + len(rows), 'tests', completion['tests'])


if __name__ == '__main__':
    main()
