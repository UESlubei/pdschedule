"""Finalize only after all registered trials, audits, and GPU diagnostics finish."""
import json
from pathlib import Path
import statistics
import re
import time
from apb.common import ROOT,DATA,dump,digest

OUT=ROOT/'reports/optimization_v2'

def read(path):return json.loads(path.read_text())

def main():
    rows=read(OUT/'results.json');audits=read(OUT/'audits.json')
    summaries=read(OUT/'summary.json');comparisons=read(OUT/'comparisons.json')
    diag=read(OUT/'diagnostics.json');profiles=read(OUT/'profile_summary.json')
    validation=[r for r in rows if r['stage']=='opt_validation']
    protocol=read(DATA/'manifests/opt_validation-protocol.json')
    matched=[r for r in rows if r['stage']=='opt_match1536']
    assert len(protocol['jobs'])==20 and len(validation)==80 and len(audits)==35
    assert len(rows)==134 and len(matched)==36 and all(not a['errors'] for a in audits)
    assert diag['source']['passed'] and diag['source']['weights_verified']
    assert len(profiles['captures'])==5 and all(c['complete'] for c in profiles['captures'])
    assert (DATA/'logs/tests-30.log').read_text().rstrip().endswith('OK')
    keys=['decode99','prefill95','output_tok_s'];scenarios=['interference','transition','mixed']
    def comparison(policy,baseline,scenario,metric,stage='opt_validation'):
        return next(r for r in comparisons if r['stage']==stage and r['policy']==policy
            and r['baseline']==baseline and r['scenario']==scenario and r['metric']==metric)
    def change(policy,baseline,scenario,metric,stage='opt_validation'):
        r=comparison(policy,baseline,scenario,metric,stage);lo,hi=r['ci95']
        return f"{r['mean_change_pct']:+.2f}% [{lo:+.2f}, {hi:+.2f}]"
    acceptance=[]
    for policy in ['balanced','fixed1536','equalized256']:
        for scenario in scenarios:
            d,p,t=(comparison(policy,'stock',scenario,k) for k in keys)
            acceptance.append(dict(stage='opt_validation',policy=policy,scenario=scenario,
                point_guardrails=p['mean_change_pct']<=10 and t['mean_change_pct']>=-5,
                ci_guardrails=p['ci95'][1]<=10 and t['ci95'][0]>=-5,
                point_decode_improvement=d['mean_change_pct']<0,ci_decode_improvement=d['ci95'][1]<0))
    feasible=[p for p in ['fixed1536','equalized256'] if all(r['point_guardrails'] for r in acceptance if r['policy']==p)]
    score=lambda p:statistics.mean(comparison(p,'balanced',s,'decode99')['mean_change_pct'] for s in scenarios[:2])
    selected=min(feasible,key=score) if feasible else None
    presets={'fixed1536':dict(mode='fixed',cap=1536,maximum=4096,fixed_idle_relax=True,reserve=True),
        'equalized256':dict(mode='equalized',cap=2048,maximum=4096,chunk_alignment=256,
            context_weight=0.00012347502462458387,reserve=True),
        'equalized1536':dict(mode='equalized',cap=1536,maximum=4096,chunk_alignment=512,
            context_weight=0.00012347502462458387,reserve=True)}
    for name,config in presets.items():dump(ROOT/'configs/presets'/(name+'.json'),config)
    if selected:dump(ROOT/'configs/presets/balanced_v2.json',presets[selected])
    selection=dict(selected_policy=selected,criterion='Among the two pre-specified candidates meeting point TTFT/throughput guardrails in all three scenes, minimize mean paired primary-scene Decode P99 change versus balanced.',
        candidate_scores={p:score(p) for p in ['fixed1536','equalized256']},feasible=feasible,
        note='Engineering selection from the registered comparisons; intervals are descriptive, without selection or multiplicity adjustment.')
    text=['独立验证使用预先登记的新种子 **307/311/313/317/331**。四组各五个独立服务，轮内随机化顺序。下表是五次均值，变化与区间采用逐种子配对计算。', '',
        '| 场景 | 策略 | Decode P99 ms | Prefill P95 ms | 固定 Trace 输出 tok/s |',
        '| --- | --- | ---: | ---: | ---: |']
    for scenario in scenarios:
        for policy in ['stock','balanced','fixed1536','equalized256']:
            r=next(r for r in summaries if r['stage']=='opt_validation' and r['scenario']==scenario and r['policy']==policy)
            text.append(f"| {scenario} | {policy} | {r['decode99']:.2f} | {r['prefill95']:.2f} | {r['output_tok_s']:.2f} |")
    if selected:
        text+=['',f'五种子阶段的工程选择为 **{selected}**，单独配置保存为 `configs/presets/{selected}.json`。选择规则：先要求三个场景的 TTFT/吞吐点估计全部满足约束，再比较两个预定主场景的配对 P99 变化均值。区间约束是否通过，仍须看下方验收表；不将点估计通过写成区间保证。最终别名 `balanced_v2.json` 的选择还包括后文额外验证。', '',
            '阶段选择相对原均衡档：', '',
            '| 场景 | Decode P99 | Prefill P95 | 输出 tok/s |','| --- | --- | --- | --- |']
        for scenario in scenarios:
            text.append('| '+scenario+' | '+' | '.join(change(selected,'balanced',scenario,k) for k in keys)+' |')
        text+=['','阶段选择相对新鲜原生：','',
            '| 场景 | Decode P99 | Prefill P95 | 输出 tok/s |','| --- | --- | --- | --- |']
        for scenario in scenarios:
            text.append('| '+scenario+' | '+' | '.join(change(selected,'stock',scenario,k) for k in keys)+' |')
    else:text+=['','两个新候选均未在三个场景同时满足 TTFT/吞吐点估计约束，保留原均衡档；本轮不能宣称得到全面达标的新配置。']
    text+=['','相对原均衡档，成本均分的配对变化及 95% 区间：','',
        '| 场景 | Decode P99 | Prefill P95 | 输出 tok/s |','| --- | --- | --- | --- |']
    for scenario in scenarios:
        text.append('| '+scenario+' | '+' | '.join(change('equalized256','balanced',scenario,k) for k in keys)+' |')
    text+=['','相对新鲜原生基线：','',
        '| 场景 | Decode P99 | Prefill P95 | 输出 tok/s |','| --- | --- | --- | --- |']
    for scenario in scenarios:
        text.append('| '+scenario+' | '+' | '.join(change('equalized256','stock',scenario,k) for k in keys)+' |')
    text+=['','相对更强的 Fixed-1536 静态对照：','',
        '| 场景 | Decode P99 | Prefill P95 | 输出 tok/s |','| --- | --- | --- | --- |']
    for scenario in scenarios:
        text.append('| '+scenario+' | '+' | '.join(change('equalized256','fixed1536',scenario,k) for k in keys)+' |')
    text+=['','约束验收（相对原生：Prefill P95≤+10%，吞吐≥−5%；区间验收要求对应区间全部位于边界内）：','',
        '| 策略 | 场景 | 点估计满足约束 | 95% 区间满足约束 | P99 改善区间不跨零 |',
        '| --- | --- | --- | --- | --- |']
    yes=lambda b:'是' if b else '否'
    for r in acceptance:text.append(f"| {r['policy']} | {r['scenario']} | {yes(r['point_guardrails'])} | {yes(r['ci_guardrails'])} | {yes(r['ci_decode_improvement'])} |")
    primary=[r for r in acceptance if r['policy']=='equalized256' and r['scenario']!='mixed']
    robust=all(r['point_guardrails'] and r['ci_guardrails'] and r['ci_decode_improvement'] for r in primary)
    incremental=all(comparison('equalized256','balanced',s,'decode99')['ci95'][1]<0 for s in scenarios[:2])
    text+=['','2048 基础预算候选的判断：'+('成本均分在两个预定主场景同时通过区间约束，并有明确的 P99 改善。' if robust else
        '成本均分未在两个预定主场景同时通过全部区间约束，应按上表保留收益和不达标项，不能宣称全面达标。'),
        '相对原均衡档的两个主场景 P99 改善'+('均获得本批配对区间支持。' if incremental else '未全部获得本批配对区间支持。'),
        'Fixed-1536 提供了另一组时延/TTFT 取舍；成本均分的价值须结合 TTFT 和吞吐评价，不能仅凭策略复杂度宣称优于最佳静态。', '',
        '五次重复仅描述本机本批实验变异。Mixed 的 P99 包括全部请求，不等同于固定 Decode cohort；吞吐包含固定到达时序和排空时间，不是最大稳定容量。区间采用 5000 次按种子 bootstrap，未作多重比较校正。', '',
        '![独立验证]('+str(OUT/'opt_validation.png')+')', '',
        '## 同基础预算的额外验证', '',
        '第一组新种子显示 Fixed-1536 比 2048 基础预算候选更强，随后登记三个未使用的新种子 **401/409/419**，比较原生、Fixed-1536 与基础预算同为 1536 的成本均分。此阶段在前一阶段结果之后提出，单独保留协议和结论，不合并成原先的五次重复。', '',
        '新候选仍使用同一求解器与标定系数，改为 512 对齐。6144-token Prompt 的典型切块为 2048/1536/1536/1024，与静态 1536 的四次推进对齐；由此检验同基础预算下的成本均分收益。', '',
        '| 场景 | 策略 | Decode P99 ms | Prefill P95 ms | 输出 tok/s |','| --- | --- | ---: | ---: | ---: |']
    for scenario in scenarios:
        for policy in ['stock','fixed1536','equalized1536']:
            r=next(r for r in summaries if r['stage']=='opt_match1536' and r['scenario']==scenario and r['policy']==policy)
            text.append(f"| {scenario} | {policy} | {r['decode99']:.2f} | {r['prefill95']:.2f} | {r['output_tok_s']:.2f} |")
    for baseline in ['stock','fixed1536']:
        text+=['',f'额外候选相对 {baseline} 的三种子配对变化：','',
            '| 场景 | Decode P99 | Prefill P95 | 输出 tok/s |','| --- | --- | --- | --- |']
        for scenario in scenarios:text.append('| '+scenario+' | '+' | '.join(change('equalized1536',baseline,scenario,k,'opt_match1536') for k in keys)+' |')
    matched_acceptance=[]
    for scenario in scenarios:
        d,p,t=(comparison('equalized1536','stock',scenario,k,'opt_match1536') for k in keys)
        matched_acceptance.append(dict(stage='opt_match1536',policy='equalized1536',scenario=scenario,
            point_guardrails=p['mean_change_pct']<=10 and t['mean_change_pct']>=-5,
            ci_guardrails=p['ci95'][1]<=10 and t['ci95'][0]>=-5,
            point_decode_improvement=d['mean_change_pct']<0,ci_decode_improvement=d['ci95'][1]<0))
    matched_incremental=all(comparison('equalized1536','fixed1536',s,'decode99','opt_match1536')['ci95'][1]<0 for s in scenarios[:2])
    matched_feasible=all(a['point_guardrails'] for a in matched_acceptance)
    final_selected='equalized1536' if matched_feasible and matched_incremental else selected
    selection.update(base_stage_selection=selected,selected_policy=final_selected,
        matched_guardrails=matched_acceptance,matched_incremental_p99=matched_incremental,
        matched_selection_rule='Promote the new equal-base candidate only if all three scene point guardrails pass and primary-scene P99 improvement versus fresh fixed1536 has a negative upper CI bound.')
    dump(OUT/'selection.json',selection);dump(OUT/'acceptance.json',acceptance+matched_acceptance)
    if final_selected:dump(ROOT/'configs/presets/balanced_v2.json',presets[final_selected])
    if final_selected=='equalized1536':
        text+=['','额外候选在三个场景通过 TTFT/吞吐点估计约束，并在两个主场景获得相对 Fixed-1536 的 P99 改善区间支持，最终配置更新为 `equalized1536`。此阶段仅三次独立重复，区间不能视为跨环境保证。']
    else:text+=['','额外候选未同时满足全部晋级条件，最终保留前阶段选择。成本均分可能降低峰值，却增加短 Prompt 批处理、尾块或排队代价；具体负结果保留在表中，不把该候选宣称为全面优于静态。']
    text+=['','为核查批处理机制，使用真实 V1 Scheduler/KV Manager 做独立 CPU 状态实验：一个存量 Decode 请求，加八个 2048-token Prompt，基础预算均为 1536。模型输出由确定性模拟器提供，以下只比较调度次数，不测 GPU 时间。', '',
        '| 模式 | Async | 含 Prefill 步骤 | 含多个 Prefill 请求的步骤 | 总 Prompt token |',
        '| --- | --- | ---: | ---: | ---: |']
    for r in diag['batch_probe']['results']:
        text.append(f"| {r['mode']} | {r['async_scheduling']} | {r['prefill_steps']} | {r['multi_prefill_request_steps']} | {r['prompt_tokens']} |")
    text+=['','成本均分模式在长于基础预算的请求尾部，按剩余量收紧全局共享配额；固定模式可以把剩余配额交给后续请求。相同 token 总量可能因此需要不同数量的全局步骤。完整 GPU Trace 的尾块、批处理和队列数据保存在 diagnostics.json，可与上述受控状态实验交叉核对。']
    text+=['','![同基础预算验证]('+str(OUT/'opt_match1536.png')+')', '', '## 输出、记账与来源检查', '',
        f'本轮主矩阵纳入 **134 个 Trial / 35 个服务**：105 个性能 Trial、29 个质量 Trial；另有五个有效节点 Profile 服务，独立保存。中断轮次的8个已完成Trial归档后整轮重跑，不计入主矩阵。全部主矩阵原始数据审计错误为 0。30 项回归通过，包含 256/512 对齐的同步/异步共享配额、部分缓存前沿、抢占 Replay 和状态清理。', '',
        '| 策略 | 自然问答逐 token 一致 | 合成尾块逐 token 一致 |','| --- | ---: | ---: |']
    mismatch=[]
    for stage,policy in [('opt_validation',p) for p in ['balanced','fixed1536','equalized256']]+[('opt_match1536',p) for p in ['fixed1536','equalized1536']]:
        qs=[c for q in diag['quality'] if q['stage']==stage and q['policy']==policy for c in q['comparisons']]
        natural=[c for c in qs if c['cohort']=='natural'];tail=[c for c in qs if c['cohort']=='tail']
        text.append(f"| {policy} ({stage}) | {sum(c['exact'] for c in natural)}/{len(natural)} | {sum(c['exact'] for c in tail)}/{len(tail)} |")
        mismatch.extend(dict(stage=stage,policy=policy,**c) for c in qs if not c['exact'])
    if mismatch:
        text+=['','检测到输出差异，原始 ID、首个差异位置和文本保存在 diagnostics.json。独立重复与公共前缀 logprob 诊断如下；原始差异保留，不改写为全部一致。']
        numerical=diag['numerical_probe'];assert numerical and numerical['complete']
        text+=['',f"补充阶段 Fixed-1536 的 seed{numerical['seed']} / {numerical['target']} 在第 {numerical['first_difference_1based']} 个输出产生差异：原生 token={numerical['original_native_token']}，固定档 token={numerical['original_fixed_token']}，此前 {len(numerical['common_prefix_tokens'])} 个输出一致。", '',
            '随后使用相同 Trace 和关闭 Prefix Cache 的新鲜 stock/off/Fixed-1536 各两次，共六个诊断服务，独立于性能矩阵。', '',
            '| 新鲜重复 | stock/off 逐 token 一致 | stock/Fixed-1536 逐 token 一致 |',
            '| --- | ---: | ---: |']
        for p in numerical['pairs']:text.append(f"| {p['repeat']} | {p['stock_off_exact']}/{p['total']} | {p['stock_fixed_exact']}/{p['total']} |")
        jit_runs=[r for r in numerical['runs'] if r['jit_warnings']]
        if jit_runs:text+=['',f'其中 {len(jit_runs)} 个诊断服务记录了 logprob Top-K 内核首次编译警告。原始性能审计中的 JIT 拒绝项与日志保留；诊断仅要求请求、长度、Trace 和响应完整，不使用这些服务的时延或吞吐作性能证据。主性能矩阵的 JIT 审计未放宽。']
        text+=['','Teacher-forced 将共同前缀强制追加到输入，自回归检查原始输入的对应位置；offered batch 为1/2/8。自回归公共前缀不一致的样本不用于同上下文比较。', '',
            '| 策略 | 检查 | 同前缀样本/总样本 | Top1−Top2 logprob差最小/最大 |',
            '| --- | --- | ---: | ---: |']
        for policy in ['stock','off','fixed1536']:
            for kind in ['teacher_forced','autoregressive']:
                vals=[p for r in numerical['runs'] if r['policy']==policy for p in r['logprob_margins'] if p['kind']==kind]
                aligned=[p for p in vals if p['common_prefix_aligned']];gaps=[p['top1_minus_top2'] for p in aligned]
                margin=f'{min(gaps):.6f}/{max(gaps):.6f}' if gaps else '无同前缀样本'
                text.append(f'| {policy} | {kind} | {len(aligned)}/{len(vals)} | {margin} |')
        native_drift=[r for r in numerical['runs'] if r['policy']=='stock' and not r['exact_vs_original_native'][numerical['target']]]
        reproduced=[r for r in native_drift if r['exact_vs_original_fixed'][numerical['target']]]
        changed_requests=Path(numerical['changed'])/'requests.json'
        original_fixed=next(r['token_ids'] for r in read(changed_requests) if r['id']==numerical['target'])
        native_probes=[]
        for run in numerical['runs']:
            if run['policy']!='stock':continue
            for probe in run['logprob_margins']:
                if probe['kind']!='autoregressive' or not probe['common_prefix_aligned']:continue
                ids=probe['generated_token_ids'];position=numerical['first_difference_1based']-1
                native_probes.append(dict(repeat=run['repeat'],offered_batch=probe['offered_batch'],
                    common_prefix_aligned=True,original_fixed_full_output_exact=ids==original_fixed,
                    original_fixed_first_difference_token_exact=len(ids)>position and ids[position]==original_fixed[position],
                    top1_minus_top2=probe['top1_minus_top2'],generated_token_ids=ids))
        dump(OUT/'numerical_batch_reproduction.json',dict(
            numerical_probe_sha256=digest(OUT/'numerical_probe.json'),
            changed_requests=str(changed_requests),changed_requests_sha256=digest(changed_requests),
            original_fixed_token_ids=original_fixed,stock_autoregressive_probes=native_probes))
        probe_reproduced=[p for p in native_probes if p['original_fixed_full_output_exact']]
        if reproduced:
            text+=['','新鲜未修改 stock 也复现了原 Fixed-1536 的该续写分支，说明这类分歧可以由原生自身产生。结合共同前缀 logprob 检查和 stock/off 对照，证据支持原生批次/数值敏感性的解释；未发现新增调度记账错误。']
        elif probe_reproduced:
            text+=['','六个新鲜服务的原始 correctness 重复均回到原 stock 分支；另外的自回归批次探测中，未修改 stock 完整复现了原 Fixed-1536 的全部 32 个输出 token：', '',
                '| stock 重复 | offered batch | 原 Fixed 全部32 token一致 | 第11位 Top1−Top2差 |',
                '| --- | ---: | --- | ---: |']
            for p in probe_reproduced:
                text.append(f"| {p['repeat']} | {p['offered_batch']} | 是 | {p['top1_minus_top2']:.6f} |")
            text+=['','这些样本的前10个输出与原始共同前缀一致，第11位候选“ that”和“ and”的返回 logprob 并列。直接证据说明未修改 stock 在改变提供的并发批次后也会产生同一续写分支，支持原生批次/数值敏感性的解释。offered batch 表示同时提交的请求数，不等同于每步实际 GPU 批大小；诊断还启用了 logprobs，不能据此断言已还原原始运行的精确数值路径。未发现新增调度记账错误，原始差异仍计为不一致。完整 token 比较见 numerical_batch_reproduction.json。']
        else:text+=['','本批重复没有观察到新鲜 stock 完整复现原 Fixed-1536 分支；应按上述真实重复和 logprob 结果保留解释边界，不能仅凭“FP16”就断言差异无害。']
        text+=['','自然问答与合成尾块的结果分别列出。有限一致性检查和候选 logprob 诊断不替代完整任务质量评估，也不保证任意 FP16 异步批形逐 token 一致。完整响应与原始 logprobs 见 numerical_probe.json 和运行目录 quality_drift_opt/。']
    else:text+=['','上述有限用例全部与同种子的新鲜原生服务一致；它们不替代完整任务质量基准，也不构成任意 FP16 批形的逐位一致保证。']
    text+=['',f"来源检查共 {len(diag['source']['checks'])} 项通过，包括原安装的固定源码、源码覆盖层、旧轮归档、每个服务的冻结来源及模型文件 SHA256。全局 token 上限始终为 4096，实际 Prompt+Replay 不超过当轮配额；非空提交与完成一一对应，无在途残留或遥测丢失。", '',
        '新组 CPU 计时先对每个服务取 P95，再按阶段取服务均值：', '',
        '| 策略 | decide 阶段 P95 µs | 完整调度 P95 µs | 完成反馈 P95 µs |',
        '| --- | ---: | ---: | ---: |']
    for stage,policy in [('opt_validation',p) for p in ['balanced','fixed1536','equalized256']]+[('opt_match1536',p) for p in ['fixed1536','equalized1536']]:
        rs=[r for r in diag['cpu'] if r['stage']==stage and r['policy']==policy]
        values=[statistics.mean(r[k]['p95'] for r in rs) for k in ['decision_us','scheduler_us','feedback_us']]
        text.append('| '+policy+' ('+stage+') | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    text+=['','decide 计时不含 clip 中的成本求解，完整调度计时包含求解与原生调度。策略改变请求状态和推进次数，完整调度差值不能直接解释为纯算法开销；也没有同构的 stock 内部计时作为相对开销分母。', '',
        '## GPU 节点诊断', '',
        '五个独立 Nsight 节点采集使用 seed919、预热后 128 个完成回调。所有采集均通过记账审计，未加入性能表。', '',
        '| 策略 | 逻辑完成 Prompt token | Decode token | 可见 kernel 数 | 图内 kernel 数 | GEMM 活动合计 ms | Attention 活动合计 ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in profiles['captures']:
        g=c['gpu'];w=c['completed_work'];cats=g['name_based_categories']
        text.append(f"| {c['policy']} | {w['prompt_tokens']} | {w['decode_tokens']} | {g['exposed_kernel_ms']['count']} | {g['graph_internal_kernel_count']} | {cats.get('GEMM',{}).get('duration_ms',0):.2f} | {cats.get('Attention',{}).get('duration_ms',0):.2f} |")
    text+=['','另保留一次采集完成后审计失败的诊断：最初目录名 balanced 被旧审计逻辑当成原有冻结阶段，查找不存在的 formal-source.json。统一改为 node- 前缀目录后重采并通过，原始失败产物保存在 diagnostic_profiles/balanced/；该失败不影响性能矩阵，也未加入上述五组有效采集。']
    text+=['','函数名分类仅为启发式归类。不同窗口的 Decode 工作量、请求状态和边缘在途工作可能不同，且 Profiler 会扰动执行；活动时长之和不等于墙钟时间、利用率或可加的阶段成本。这些数据用于核查主要 GPU 活动与图内覆盖，不按合计时间比值宣称优化收益。', '',
        '新策略未改变模型 kernel、KV 分配或全局编译缓冲区。本轮主要机制证据为实际切块、完成间隔与请求排队的受控对照。时钟未锁定，随机化轮内顺序和新鲜基线降低漂移影响，但不能消除全部系统偏差。', '']
    path=DATA/'report_materials/优化阶段统计.md';path.parent.mkdir(parents=True,exist_ok=True)
    body=path.read_text() if path.exists() else '# 动态 Prefill Budget 第二轮优化报告\n\n## 独立验证与最终选择\n\n## 复现与证据\n\n阶段统计保存在实验数据目录；项目完整报告由 benchmarks.project_report 统一生成。\n'
    a=body.index('## 独立验证与最终选择');b=body.index('## 复现与证据',a)
    recovery=body[body.index('## 交互中断与恢复',a):b] if '## 交互中断与恢复' in body[a:b] else ''
    body=body[:a]+'## 独立验证与最终选择\n\n'+'\n'.join(text)+'\n'+recovery+body[b:]
    body=body.replace('实验运行中，最终结论以文末独立验证结果为准。','本轮已完成独立验证，收益、代价和适用边界见下文。')
    body=re.sub(r'\n本轮(?:更好的工程配置是|选择 \*\*)[^\n]*完整收益和约束区间见独立验证部分。\n','\n',body)
    if final_selected:
        statement=('本轮更好的工程配置是 **1536/4096 两档**：有 Decode 时共享 Prefill 配额为 1536，无 Decode 时放宽到 4096。' if final_selected=='fixed1536' else
            '本轮选择 **'+('512 对齐、1536 基础预算' if final_selected=='equalized1536' else '256 对齐、2048 基础预算')+'的成本均分**作为新工程配置，按实际上下文动态分配 Prefill token。')
        body=body.replace('# 动态 Prefill Budget 第二轮优化报告\n','# 动态 Prefill Budget 第二轮优化报告\n\n'+statement+'完整收益和约束区间见独立验证部分。\n',1)
    path.write_text(body)
    selected_stage='opt_match1536' if final_selected=='equalized1536' else 'opt_validation'
    selected_baseline='fixed1536' if selected_stage=='opt_match1536' else 'balanced'
    selected_guards=[r for r in acceptance+matched_acceptance if r['stage']==selected_stage and r['policy']==final_selected]
    selected_robust=bool(selected_guards) and all(r['ci_guardrails'] and r['ci_decode_improvement'] for r in selected_guards if r['scenario'] in scenarios[:2])
    selected_incremental=bool(final_selected) and all(comparison(final_selected,selected_baseline,s,'decode99',selected_stage)['ci95'][1]<0 for s in scenarios[:2])
    dump(OUT/'completion.json',dict(completed_unix=time.time(),trials=len(rows),performance_trials=105,quality_trials=29,
        sessions=len(audits),profile_sessions=5,tests=30,audit_errors=0,primary_robust_guardrails=selected_robust,
        primary_incremental_p99=selected_incremental,selected_policy=final_selected,selected_stage=selected_stage,
        incremental_baseline=selected_baseline,quality_mismatches=mismatch,
        numerical_diagnostic_sessions=len(diag['numerical_probe']['runs']) if diag.get('numerical_probe') else 0,source_passed=True))
    print('OPTIMIZATION_COMPLETE',len(rows),len(audits),'selected',final_selected,'robust',selected_robust,'incremental',selected_incremental)

if __name__=='__main__':main()
