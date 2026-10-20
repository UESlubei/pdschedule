# Adaptive Prefill Budget Scheduling

固定 vLLM 0.32.0+cu129，Qwen3-1.7B FP16，单 RTX 3090。为 Scheduler 每轮 Running/Waiting Prefill 增加共享计算配额与 Decode Token 预留，支持 Fixed、Load、EWMA/AIMD、Joint、Aging。原生 KV Manager、请求顺序、抢占与模型 Kernel 保持原有职责。

- [项目设计](../docs/动态Prefill_Budget调度项目设计_V2.md)
- [完整实验报告](../reports/动态Prefill_Budget完整实验报告.md)
- [性能归因与修正结论](../reports/性能归因与结论.md)
- [工程验证与开销](../reports/工程验证与开销.md)
- [实施问题与解决记录](../docs/IMPLEMENTATION_LOG.md)
- `patches/v0.32.0/scheduler.patch`：基于固定源码的可审计补丁。
- `tests/`：纯控制器与真实 Scheduler/KV Manager 回归。
- `benchmarks/`：固定 Trace、Token SSE 事件、服务端完成记录及原始数据审计。

本批完成421个有效Trial与22项回归，原始数据审计错误为0。独立新种子验证的2048/4096均衡档在干扰场景降低Decode P99约35.85%，Prefill P95增加1.72%，固定Trace吞吐下降1.72%；mixed尾延迟区间跨零。默认复杂反馈策略未优于最佳静态，完整报告保留收益、代价和负结果。该均衡档有Decode时恒2048、无Decode时4096，不构成AIMD增量收益证据。

## 本机运行

```bash
cd /home/sdn/wjh/Ondeviceinfer/source/adaptive_prefill_budget
export PYTHONPATH="$PWD"
APB_PY=/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python
"$APB_PY" scripts/prepare.py
PYTHONPATH=/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320/overlays/candidate:$PWD \
  "$APB_PY" -m unittest discover -s tests -v
"$APB_PY" -m benchmarks.pipeline --stages pilot default_pilot
"$APB_PY" -m benchmarks.analyze
```

运行根目录 `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320` 存储完整模型校验、源码副本、编译缓存、请求/Token日志和原始指标。源码准备不会改写已安装的 vLLM 文件，Python 源码复制，未修改二进制只读链接。`mode=off` 不创建控制器。未支持 Spec、多模态、LoRA、KV Connector、非 FCFS 或多 GPU。

单 GPU 流水线同时占用新项目锁与旧项目两把锁；每策略启动独立服务，在服务内复用已完全排空的场景/种子试验，减少重复加载。各 Trial 保存独立 Trace、请求事件、指标快照和统计；此隔离粒度在 manifest 中明确记录。

## 第二轮优化（2026-10-20）

本轮工程配置为 `fixed1536`，见 `configs/presets/balanced_v2.json`；具体收益、置信区间和负结果以第二轮报告为准。

新增134个有效Trial、35个服务和30项回归，另有5个节点Profile与6个数值诊断服务。自然问答126/126逐token一致，合成尾块104/105一致；原始差异与原生批次复现证据见报告。交互中断的一轮归档后整轮重跑，旧数据保留。

- [第二轮完整优化报告](../reports/动态Prefill_Budget第二轮优化报告.md)
- [成本均分算法](../docs/动态Prefill_Budget成本均分策略.md)
- [逐阶段数据表](../reports/optimization_v2/实验数据表.md)

```bash
APB_DATA=/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2 \
  PYTHONPATH="$PWD" /mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python \
  -m benchmarks.recovery_master
```

运行目录中的 `recovery-master-status.json` 给出阶段状态；已完成服务通过审计后跳过。新实验不覆盖原421个Trial。代码默认仍为 `mode=off`。
