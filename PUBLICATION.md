# 发布说明

项目于2026-10-20首次发布，2026-10-21统一项目报告与文档入口。来源为本地 `adaptive_prefill_budget` 项目。此次调整仅涉及文档和离线报告生成路径，未改动在线调度、模型或既有实验统计。

## 发布内容

包含调度控制代码、vLLM补丁、配置、测试、实验与分析脚本、设计文档、唯一完整项目报告、PNG/PDF图表、JSON汇总与诊断、依赖快照。排除Python缓存、模型权重、安装环境、编译缓存和外部实验目录的完整原始日志及Nsight数据库。

`publication_manifest.json`逐文件记录本地源文件与发布文件的SHA256及展示链接转换。代码、配置、测试、补丁与JSON实验数据保持与本地源文件字节一致。Markdown中的项目内绝对链接转换为相对链接，以便GitHub阅读。

阶段报告、性能归因与工程验证已整合到[完整项目报告](reports/动态Prefill_Budget完整实验报告.md)，重复文档从当前树移除，历史发布版本可在Git记录中追溯。本地原文另存于实验目录的`document_archive/20261021-report-consolidation/`。报告生成器的工作文字输出到实验目录`report_materials/`，不再覆盖最终项目报告。

## 实验来源与边界

- 基线与策略评估421个有效Trial，见 `reports/completion.json`。
- 追加优化134个有效Trial、35个服务，见 `reports/optimization_v2/completion.json`。
- 当前回归30项，不将早期22项与当前30项相加；555个累计Trial也不是555次独立性能重复。
- 独立诊断不作为性能收益证据；各阶段计数口径见完整报告。
- 历史交接文档保留当时状态，以唯一完整项目报告为准。
- 完整原始数据保存在原实验机的 `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320` 与 `adaptive_prefill_budget_v0320_opt2`。
- JSON中的绝对路径和Hash作为来源保留，不表示这些路径可在GitHub下载；不覆盖历史冻结来源。

## 复现要求

本项目依赖固定vLLM版本、匹配CUDA环境、单GPU和本地模型。`APB_DATA`可选择数据目录，但解释器、模型与部分辅助脚本路径针对原机器，其他机器需适配后重新校验，不能继承原验收标记。

README中的源码准备与回归命令不运行完整性能矩阵。仅重建完整项目报告可用 `PYTHONPATH="$PWD" python -m benchmarks.project_report`，只读取仓库内已有审计JSON，无需GPU。完整推理实验条件见[运行说明](docs/RUNNING.md)与[完整项目报告](reports/动态Prefill_Budget完整实验报告.md)。
