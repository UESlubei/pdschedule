# 发布说明

本仓库发布于2026-10-20，来源为本地 `adaptive_prefill_budget` 项目。只整理发布副本；原实验代码、安装环境、模型及测量数据不因发布而修改。

## 发布内容

包含调度控制代码、vLLM补丁、配置、测试、实验与分析脚本、设计文档、两轮报告、PNG/PDF图表、JSON汇总与诊断、依赖快照。排除Python缓存、模型权重、安装环境、编译缓存及位于外部实验目录的完整原始请求/服务日志和Nsight数据库。

`publication_manifest.json` 逐文件记录本地源文件与发布文件的SHA256及展示链接转换；其中README的发布版另行整理，原README存入 `docs/本地项目原始README.md`。代码、配置、测试、补丁和JSON实验数据与本地源文件保持字节一致。Markdown中的项目内绝对链接改为相对链接，以便GitHub展示。

## 实验来源与边界

- 第一轮：421个有效Trial，见 `reports/completion.json`。
- 第二轮：134个有效Trial，35个服务，见 `reports/optimization_v2/completion.json`。
- 当前回归：30项；不是第一轮22项与第二轮30项相加。
- 独立Profile与数值诊断未混入主性能矩阵。
- 历史交接文档保留当时状态，以两轮最终报告为准，不代表仍有待运行工作。
- 完整原始数据仍在原实验机的 `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320` 与 `adaptive_prefill_budget_v0320_opt2`。
- JSON中的原始绝对路径和Hash保留作为来源记录，不表示这些路径是GitHub可下载资源。发布副本的SHA与原冻结源码SHA分别记录，不覆盖历史冻结来源。

## 复现要求

本项目依赖固定版本vLLM及匹配CUDA环境、单GPU和本地模型。`APB_DATA`可选择实验数据目录，但解释器、模型与部分辅助脚本路径仍针对原机器；其他机器须显式适配后重新校验，不能直接继承原机器的实验验收标记。

README中的本机命令用于源码覆盖层准备和30项回归，不启动完整GPU性能矩阵。完整矩阵成本较高，详细运行条件见 `docs/RUNNING.md` 和两轮报告。
