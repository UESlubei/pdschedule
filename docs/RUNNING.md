# 本机复现与结果审计

主环境使用 `/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python`；完整版本在 `requirements/vllm0320-cu129-lock.txt`。该锁是依赖版本快照，不是可跨GPU/驱动直接安装的完整二进制包。权重在 `/mnt/ondeviceinfer/models/Qwen3-1.7B`，模型各文件SHA256由prepare保存于数据根目录。

```bash
cd /home/sdn/wjh/Ondeviceinfer/source/adaptive_prefill_budget
export PYTHONPATH="$PWD"
APB_PY=/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python
"$APB_PY" scripts/prepare.py
PYTHONPATH=/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320/overlays/candidate:$PWD \
  "$APB_PY" -m unittest discover -s tests -v
```

`prepare.py`先验证设计阶段保存的已安装源码Hash，再复制Python源码并应用精确补丁。没有修改原环境。已安装环境发生变化时，必须重新审阅版本并建立新的环境清单，不能关闭Hash检查。准备源码时不要同时运行依赖该副本的实验。

完整执行与续跑：

```bash
"$APB_PY" -m benchmarks.pipeline --stages pilot default_pilot
"$APB_PY" -m scripts.complete_project
"$APB_PY" -m benchmarks.pipeline --stages idle balanced native2048
"$APB_PY" -m benchmarks.numerical_probe
"$APB_PY" -m benchmarks.profile_nodes
"$APB_PY" -m benchmarks.profile_nodes --policy observe
"$APB_PY" -m benchmarks.analyze
"$APB_PY" -m benchmarks.diagnostics
"$APB_PY" -m scripts.verify_sources --weights
```

流水线使用GPU及旧项目两把锁，发现其他计算进程时退出。已完成且通过原始数据审计的Session可续跑跳过。失败原始产物保留；修改控制器、关键Benchmark或标定表后，正式来源Hash检查会阻止混用，必须使用新的运行目录/来源快照重新实验。默认数据根目录为 `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320`，可通过环境变量`APB_DATA`指定新目录；需重新prepare和标定。

单独运行一组（先确认GPU可用，并持有同样的锁）：

```bash
flock -n /mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320/gpu.lock \
flock -n /mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/performance.lock \
flock -n /mnt/ondeviceinfer/experiments/vllm_kv_scheduling/pipeline_v3/background.lock \
  "$APB_PY" -m benchmarks.session --policy joint --track default \
  --scenarios interference transition mixed --seeds 17 \
  --output /mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320/raw/manual/default-joint-seed17
```

`--controller-json`接收项目配置覆盖，例如`'{"alpha":0.1,"interval":4}'`。实际vLLM进程通过`APB_CONFIG`加载已校验JSON，避免给官方CLI添加未支持选项。该接入与设计中的原生SchedulerConfig扩展有所不同；全部解析值和实际导入路径均保存于manifest/config文件。`stock`使用未修改源码；`off`使用补丁但不创建Runtime；`observe`保留原生决策并记录遥测。

每个Trial的`trace.json`、`requests.json`、`metrics-before.txt`、`metrics-after.txt`是原始证据；`analysis.json`和`reports/*.json`可由分析脚本重算。`scheduler.jsonl`按提交/完成分别记录Step，`audit.json`验证配额、分区、完成对应关系和输出长度。`scheduler.summary.json`必须由实际EngineCore关闭时生成，缺失时不能手工补写为通过。

绘图使用已有分析环境（不向推理环境额外安装包）：

```bash
/home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.report
/home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.diagnostics
/home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.profile_summary
/home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.write_engineering
/home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.write_report
/home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.plot_timeline \
  --session /path/to/audited/session --scenario transition --seed 17
```

Nsight数据是独立诊断运行，不加入性能主表。`benchmarks.profile`在预热后通过CUDA Profiler API收集128个完成Step；该开关只存在于Profile运行，不加入在线控制所用计时路径。

`balanced`是原主矩阵冻结后的配置修正实验：不修改在线代码，将Joint的`minimum`设为2048，并使用151/163/179/191/211五个新种子与stock、Fixed-2048配对。选择理由和约束在运行前写入`manifests/balanced-protocol.json`。该结果单独报告，不能并入原主矩阵的五次确认性重复或用于事后改写其结论。`diagnostics.json`保存自然用例输出比较、缓存命中、取消恢复、抢占Replay、压力队列及硬件采样；分析源码Hash也在其中。

`numerical_probe`在数据根目录的`quality_drift`保留4个stock/off新服务与公共前缀logprobs，`profile_nodes`在`profiles_node`保留独立节点采集；这些均不加入421个标准Trial。整体图采集与节点采集的Kernel统计覆盖不同，重算用`profile_summary`读取SQLite，不需要重跑GPU。

现有解释正文与验收数字绑定本批原始数据。使用同一数据执行分析命令可重建表图与工程证据；若换模型、环境或新的APB_DATA重新实验，需复核性能正文、数值漂移解释和新实验的验收结论，不能直接沿用旧报告中的收益。

## 统一项目报告

最终验收与最新优化统一记录于[完整项目报告](../reports/动态Prefill_Budget完整实验报告.md)。从仓库已有统计JSON重建，无需启动推理：

```bash
PYTHONPATH="$PWD" python -m benchmarks.project_report
```

`write_report`、`write_engineering`、`finalize_optimization`的阶段文字材料写到相应`APB_DATA/report_materials/`。后台主流水线在阶段验收后调用`project_report`生成唯一项目报告，不再向README追加独立阶段报告入口。
