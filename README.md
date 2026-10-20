# PDSchedule — vLLM V1 动态 Prefill Budget 调度

在 vLLM V1 调度层限制每轮共享 Prefill/Replay 计算量，减少长 Prompt 对已有 Decode 请求的尾延迟干扰，并显式评估 TTFT 与吞吐代价。

实测环境：**vLLM 0.32.0+cu129 · Qwen3-1.7B FP16 · 单 RTX 3090 24 GiB**。累计完成 **555 个有效 Trial**，当前 **30 项回归通过**。实验结论限于本机模型、硬件和有限负载。

## 最终结果

推荐配置为 [`configs/presets/balanced_v2.json`](configs/presets/balanced_v2.json)：**有 Decode 时共享 Prefill 预算 1536，无 Decode 时放宽到 4096**。保留全局 4096-token 约束与 Decode token 预留。

相对最新五种子验证中同轮新鲜启动的原生 vLLM，五个独立种子配对均值为：

| 场景 | Decode P99 | Prefill P95 TTFT | 固定 Trace 输出吞吐 |
| --- | ---: | ---: | ---: |
| 干扰 interference | −49.80% | +1.48% | −1.13% |
| 压力切换 transition | −43.91% | +7.81% | −0.04% |
| 混合 mixed | −21.20% | +4.67% | −0.03% |

这批五种子验证的三个场景均通过相对原生 TTFT 增幅≤10%、吞吐降幅≤5%的约束。相对旧 2048/4096 均衡档，Decode P99 进一步降低 21.40% / 21.36% / 18.66%。置信区间、重复次数、选择规则和限制见完整项目报告；额外三种子中 mixed 的 TTFT 区间上界超过10%，不构成任意负载的保证。固定 Trace 吞吐不代表最大服务容量。

复杂反馈和上下文成本均分均已实现，但未稳定优于最终两档规则。收益不能归因于 AIMD 的增量价值。最新质量矩阵的自然问答126/126逐token一致，合成尾块104/105一致；差异与原生批次复现证据原样保留。

## 报告与设计

- [完整项目设计 V2](docs/动态Prefill_Budget调度项目设计_V2.md)
- [完整项目实验报告](reports/动态Prefill_Budget完整实验报告.md)：最终实现、最新改动、555个Trial、性能收益、负结果、正确性与问题解决记录。
- [实施问题与解决记录](docs/IMPLEMENTATION_LOG.md)
- [上下文成本均分算法](docs/动态Prefill_Budget成本均分策略.md)
- [最新优化数据明细](reports/optimization_v2/实验数据表.md)
- [发布内容、来源与复现边界](PUBLICATION.md)

## 最新文档与优化记录

2026-10-21：将阶段报告、性能归因和工程验证整合为一份完整项目报告，突出当前1536/4096方案及未采用的优化；补充三种子mixed TTFT区间越界的限制。未改动在线策略和既有测量数据。

只用仓库内已审计JSON重建完整报告，无需GPU或启动推理：

```bash
PYTHONPATH="$PWD" python -m benchmarks.project_report
```

阶段生成器将工作材料写到实验数据目录的`report_materials/`，不会覆盖最终报告。历史报告可通过Git记录追溯。

## 代码结构

| 目录 | 内容 |
| --- | --- |
| `apb/` | 控制器、共享配额、Decode预留、实际计算记账、异步完成反馈与遥测 |
| `patches/v0.32.0/` | 对固定vLLM Scheduler源码的可审计补丁 |
| `configs/` | 策略配置与最终preset |
| `tests/` | 控制器与真实Scheduler/KV Manager的30项回归 |
| `benchmarks/` | Trace生成、实验运行、统计、Profile与数值诊断 |
| `scripts/` | 源码覆盖层准备、冻结、来源校验和断点恢复 |
| `reports/` | 统一项目报告、JSON汇总数据、PNG/PDF图表与审计结果 |
| `requirements/` | 实验环境依赖快照 |

## 本机复现

实验脚本保留原实验机的解释器、模型和数据路径；**本仓库不是安装后即可在任意机器运行的通用vLLM插件**。其他机器需先准备匹配的环境，并适配 `apb/common.py` 及运行脚本中的路径，保留版本与源码校验。

在已准备好的原实验机上：

```bash
cd /path/to/pdschedule
export PYTHONPATH="$PWD"
APB_PY=/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python
"$APB_PY" scripts/prepare.py
PYTHONPATH=/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320/overlays/candidate:$PWD \
  "$APB_PY" -m unittest discover -s tests -v
```

重放实验的准备条件和命令见 [运行说明](docs/RUNNING.md)与[实验恢复说明](docs/优化实验运行与恢复.md)。已有实验目录受冻结Hash保护；不得覆盖旧来源清单来伪装为原始重复。完整原始运行数据、模型、二进制与Nsight大文件位于原实验机，未随Git仓库发布，不能仅凭汇总JSON重算所有原始事件。

已验证范围为单GPU、Dense文本、FCFS、普通逐token生成，覆盖默认Async/Graph和同步Eager的相应实验。Spec、多模态、LoRA、KV Connector及多GPU未验收。Aging未提供无限过载下的墙钟等待上界。
