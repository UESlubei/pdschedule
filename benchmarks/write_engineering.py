"""Final engineering evidence assembled only after every required artifact exists."""
import json
import statistics
from apb.common import DATA,ROOT
from benchmarks.write_report import table,fmt


def read(path):return json.loads(path.read_text())


def main():
    d=read(ROOT/'reports/diagnostics.json');rows=read(ROOT/'reports/results.json')
    audits=read(ROOT/'reports/audits.json');profiles=read(ROOT/'reports/profile_summary.json')
    verified=read(DATA/'manifests/final-source-verification.json')
    assert len(rows)==421 and not any(a['errors'] for a in audits)
    assert verified['passed'] and verified['weights_verified']
    assert d['numerical_probe']['complete'] and d['node_profile']['complete'] and len(d['profiles'])==5
    assert len(profiles['captures'])==7
    formal=[r for r in rows if r['stage']=='formal']
    sections=['# 工程验证与开销',
      f'有效数据共 **{len(rows)} 个Trial、{len(audits)} 个Session**，原始请求、Ledger、配置、Trace Hash和服务日志可重算。另有4个数值诊断服务和2个节点Profile服务，独立存储，不加入性能矩阵。',
      '### 预算、状态与回退',
      '22项纯Controller/真实Request、Scheduler、KV Manager回归通过，覆盖共享配额、Decode机会、分配失败与退款、尾块、缓存命中、生成历史Replay、取消及两轮在途。stock/off在固定事件序列下的调度与KV状态对照通过。未运行上游全部测试套件。',
      f'全部有效Session审计错误为0；正式主线原生抢占总数为{sum(r["native_preemptions"] for r in formal):.0f}。有Ledger的运行均验证全局预算≤4096、Prompt+Replay≤有效配额（observe保留原生决策）、分区之和等于实际提交、非空提交与完成一一对应，以及关停后无pending/active状态或日志丢失。',
      'Async提交与接受输出分别记账，正式观测的最大在途Ledger数为'+str(max(r['telemetry']['max_inflight'] for r in formal))+ '。正式SSE事件全部为单Token；客户端接收间隔与服务端完成间隔独立保存。完成lag包含异步队列和主机处理，不标为GPU Kernel时间。在线控制没有新增CUDA强制同步；CUDA Profiler API只用于诊断组。',
      '回归日志：`'+str(DATA/'logs/tests-final-22.log')+'`；审计汇总：`'+str(ROOT/'reports/audits.json')+'`。',
      '### 输出一致性与原生数值漂移']
    quality=[q for q in d['quality'] if '/compatibility/' in q['session'] and q['policy']!='stock']
    def fraction(q,cohort):
        xs=[r for r in q['comparisons'] if (r['cohort']=='natural')==cohort]
        return f"{sum(r['exact'] for r in xs)}/{len(xs)}"
    sections.append(table(['路径','策略','自然问答逐Token一致','合成尾块逐Token一致'],
        [[q['track'],q['policy'],fraction(q,True),fraction(q,False)] for q in quality]))
    sections.extend(['每个配置包含6个自然问答与5个尾块合成输入，均固定生成32Token。默认路径Pilot Joint的11个用例也与stock一致。该有限用例不是完整任务质量基准。',
      '原始default-off的17Token合成输入在第18个输出产生分歧：原stock为15860（“ involving”），off为264（“ a”）；前17个输出相同，长度均为32。此记录保持10/11，不改写为全一致。',
      '随后使用完全相同输入Trace重启stock/off各两次。两次未修改stock均复现原off的同一续写分支，四个新服务的该用例均与原off完整32Token相同；每次新stock/off之间11/11一致。四次诊断的自然问答仍全部一致。',
      table(['新鲜配对重复','stock/off逐Token一致'],[[r['repeat'],f"{r['exact']}/{r['total']}"] for r in d['numerical_probe']['assessment']['stock_off_pairs']])])
    assert d['numerical_probe']['assessment']['all_input_traces_equal']
    margins=[q for r in d['numerical_probe']['runs'] for q in r['logprob_margins']]
    sections.extend([table(['对齐方式','观测数','Top1−Top2 logprob差最小/最大'],
        [[kind,len(xs:= [q for q in margins if q['kind']==kind]),f"{min(q['top1_minus_top2'] for q in xs):.6f} / {max(q['top1_minus_top2'] for q in xs):.6f}"] for kind in ['teacher_forced','autoregressive']]),
      '公共前缀Top-20检查的前两候选为“ involving”和“ a”。teacher-forced追加相同前17个输出作为输入，自回归检查第18个输出；分别改变offered batch为1/2/8。候选接近、原生自身复现，以及off不创建Runtime和CPU状态对照的证据，支持批次/数值敏感性的解释，未发现新增调度状态错误。这不是对所有FP16误差的形式证明，也不承诺任意异步批形的bitwise一致。',
      '原始响应和logprobs：`'+str(DATA/'quality_drift')+'`；评估：`reports/diagnostics.json`的numerical_probe。',
      '### 缓存与取消恢复',
      '本版本usage未返回cached_tokens细节，缺失值记为null。缓存验证使用服务端Prometheus计数增量及实际计算Ledger；stock/off未启用Ledger，实际计算列为未观测，不能填0。',
      table(['路径','策略','服务端命中Token','实际Prompt计算Token'],[[c['track'],c['policy'],fmt(c['hits_from_server_counter'],0),fmt(c['actual_prompt_compute'],0)] for c in d['cache']]),
      '每组3次输入合计18432Token，命中10224Token。候选组实际计算8208Token，满足8208+10224=18432；近全命中的重复请求从6128开始计算，部分命中从4096开始计算。命中量不扣入Prefill计算配额。'])
    faults=d['cancellation']
    sections.append(f'取消组共{len(faults)}个Session、{sum(f["attempted"] for f in faults)}次取消尝试，记录{sum(f["cancelled"] for f in faults)}个取消任务；每组后续验证请求均输出8Token且reset_prefix_cache成功。预期取消单独记录，不混为性能请求失败。')
    sections.extend(['### 真实GPU抢占与恢复重算',
      table(['路径','原生抢占','累计Prompt计算','Replay历史输出','新Decode输入'],[[r['track'],fmt(r['native_preemptions'],0),r['work']['prompt'],r['work']['replay'],r['work']['decode_tokens']] for r in d['pressure']]),
      '专用压力组仅640个KV Blocks，并关闭完整输入长度预留。每条路径10请求全完成；Prompt重复计算量远大于原始输入，符合原生抢占重算。Replay各240Token受共享配额约束，没有冒充新Decode输出。这组配置与主线不同，不能用于主性能收益。',
      '### CPU与遥测成本'])
    cpu=[]
    for policy in ['observe','load','joint','aging']:
        rs=[r for r in d['cpu'] if r['stage']=='formal' and r['policy']==policy]
        cpu.append([policy,len(rs),*[fmt(statistics.mean(r[k]['p95'] for r in rs)) for k in ['decision_us','feedback_us','scheduler_us']]])
    sections.extend([table(['策略','服务数','决策P95 µs','反馈P95 µs','调度P95 µs'],cpu),
      '上表先计算每Session有界计时窗口的P95，再取五次均值，包含该Session预热与三个场景。调度计时覆盖状态采集、预算裁剪及原生调度，后台日志序列化/写盘另行执行；不能将各分位数相加。不同策略的请求状态与每步工作不同，不能用总调度差值直接估计纯算法成本。',
      'observe相对stock的五种子配对P99变化分别为+0.263%、+0.503%、+0.455%，各95%区间上界≤1.075%；吞吐变化分别为−0.262%、−0.014%、−0.027%，最差区间下界−0.425%。这批正式Trace支持约2%的遥测性能预算。',
      '没有对未修改stock建立同构CPU内部计时，因此未验证“新增CPU P95小于原生P95的10%”这一初拟门槛。纯决策约12µs的事实不能代替该相对门槛验收。',
      '### Nsight采集与GPU归因边界'])
    sections.append(table(['粒度','策略','GPU图活动数','图活动P95 ms','可见Kernel数','含Graph节点ID的Kernel','Kernel最大ms'],
       [[r['granularity'],r['policy'],r['gpu_graph_ms']['count'],fmt(r['gpu_graph_ms']['p95']),r['exposed_kernel_ms']['count'],r['graph_internal_kernel_count'],fmt(r['exposed_kernel_ms']['max'])] for r in profiles['captures']]))
    sections.extend(['五个graph粒度采集为observe、F512、F2048、Joint、Aging；observe保持原生决策，并提供采集触发。Joint的128步窗口没有记录GPU图活动，node粒度下也无Graph节点ID，不能据此声称已验证图内算子。补充observe节点粒度采集核对实际图内活动。全部采集独立保留，节点组另通过预算与完成Ledger审计。',
      '本机Nsight默认graph粒度，图活动与图外Kernel分别统计；GPU图活动可能是模型的一个子图，不是完整Scheduler Step。node粒度的函数名归类如下，属于启发式分类：'])
    observe_node=next(r for r in profiles['captures'] if r['granularity']=='node' and r['policy']=='observe')
    sections.append('补充observe采集中带Graph节点ID的Kernel活动数为'+str(observe_node['graph_internal_kernel_count'])+'，用于核对图内算子覆盖；Joint窗口与它的工作组成不同。')
    for node in [r for r in profiles['captures'] if r['granularity']=='node']:
        sections.append('节点采集策略：'+node['policy'])
        sections.append(table(['函数名类别','Kernel活动数','活动时长合计ms'],[[k,v['count'],fmt(v['duration_ms'])] for k,v in sorted(node['name_based_categories'].items(),key=lambda item:-item[1]['duration_ms'])]))
    sections.extend(['采集在预热后开启，目标为128个完成回调，异步边缘可能包含额外在途工作。不同策略的128步完成的Prefill量不同，Profile还会扰动反馈和吞吐；这些数据不进入主性能表，也不按累计GPU时间比值计算收益。Kernel活动时长之和不是墙钟时间、GPU利用率或可加的Prefill/Decode独立阶段。没有据此宣称Tensor Core占用率、带宽提升或完整Roofline结论。',
      '原始整体图Profile：`'+str(DATA/'profiles')+'`；节点Profile：`'+str(DATA/'profiles_node')+'`；SQL重算结果：`reports/profile_summary.json`。源码、提交量/完成轨迹、受控Cap干预和CUDA活动共同支持“小配额缩短单步、更多步骤及排队增加代价”的解释；精确算子瓶颈仍受这些采集边界限制。',
      '### 硬件采样与来源校验'])
    h=d['hardware'];a=h['active_gpu_statistics']
    sections.append(f'连续采样{h["samples"]}条，范围{h["first"]}至{h["last"]}；其中GPU利用率≥90%的采样{h["active_gpu_samples"]}条。活跃采样SM时钟P50为{a["sm_clock_MHz"]["p50"]:.0f}MHz、P90为{a["sm_clock_MHz"]["p90"]:.0f}MHz，温度最高{a["temperature_C"]["max"]:.0f}℃，功率最高{a["power_W"]["max"]:.2f}W。')
    sections.extend(['未锁频，采样从正式实验中途开始，首轮前段及最后补充的observe节点诊断没有连续覆盖。主矩阵轮内随机顺序和修正阶段的新鲜stock配对降低漂移影响，不能消除所有系统偏差。',
      f'原安装、stock源码副本、候选可重放补丁、12项冻结输入及模型全部文件共{len(verified["checks"])}项最终Hash检查通过。完整权重分片SHA256再次一致；没有改写原安装或权重。源码检查：`'+str(DATA/'manifests/final-source-verification.json')+'`。',
      '开发阶段另保留1个关停遥测未落盘的失败Session：`raw/failures/eager-observe-abort`。统一shutdown_timeout=20后重跑通过；失败数据未伪造summary，也未计入有效性能统计。',
      '支持范围为已验证的单GPU、Dense文本、FCFS、普通逐Token生成。Spec、多模态、LoRA、KV Connector及多GPU不在本轮验收范围。有限自然问答、有限负载和本机环境的结论不能代替完整模型质量或生产规模验证。'])
    path=ROOT/'reports/工程验证与开销.md';path.write_text('\n\n'.join(sections)+'\n');print(path)


if __name__=='__main__':main()
