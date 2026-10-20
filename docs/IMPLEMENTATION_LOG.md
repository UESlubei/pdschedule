# 实施问题与处理记录

日期：2026-10-20。此文件记录实际发生的问题；最终报告引用对应日志和结果。

| 问题 | 发现方式 | 处理与验证 |
| --- | --- | --- |
| 精确补丁锚点只匹配缩进时，Running 与 Waiting 两处被同时识别 | prepare首次运行主动拒绝，未修改原环境 | 为每个锚点加行首边界，并对源码Hash与唯一匹配做断言 |
| 真实Scheduler测试需要设置cache_config.num_gpu_blocks | 初版8个测试在构造时失败 | 补齐真实缓存配置；不跳过原生构造断言 |
| 取消后Request已从requests移除，测试模拟器仍访问它 | 取消回归KeyError | 模拟器对已取消请求返回空结果，再让原生完成处理释放在途状态 |
| Native Request.arrival_time使用Unix时间 | 代码审查与新增500ms等待用例 | 等待年龄改用同一时基；完成与输出间隔继续用单调时钟 |
| 独立Prefix Cache测试未初始化NONE_HASH | 缓存命中回归NameError | 调用原生init_none_hash后测试真实缓存命中，不绕过Hash/KV逻辑 |
| 带Replay尾部Decode的分配可能多占一枚预留Token | 预算边界审查 | 只要分配包含Prefill/Replay，整个连续候选区间都先受剩余配额与Decode预留限制，尾部Decode可在下一步推进 |
| 原始纯Decode参考开始阶段仍可能包含其他请求的Prefill | 基线校准日志出现首段高尾延迟 | 标定只使用实际Prompt/Replay均为零、全部N个Decode都有有效样本的8步窗口 |

实现与设计的明确调整：每个策略/执行路径使用新服务，服务内部复用已排空的种子和场景试验，以减少重复模型加载；Prefix Cache主线关闭，记录此差异，不将其称为“每个trial新服务”。纯决策状态与时间样本有界；详细日志由后台批量写入，审计检查丢失与未完成Step。

API实际将传入的请求ID扩展为`cmpl-{user_id}-0-{suffix}`。服务端/客户端关联时显式归一化前缀并验证后缀边界；若只做原始ID直接比较会得到空的cohort样本。原始ID全部保留，避免通过时间近似匹配。

第一次observe GPU Pilot正常完成请求，但关停时原生默认`shutdown_timeout=0`采用立即abort，管理器在Scheduler关闭日志前强杀EngineCore，末尾遥测和summary未落盘。审计因此拒绝该批次。原始产物保存在`raw/failures/eager-observe-abort`；所有后续组统一设置`--shutdown-timeout 20`，重跑该Pilot，不能通过补写summary使失败记录伪装通过。

后续标定增加N=16与6144上下文档位，并记录目标表范围外的请求。正式实验每个策略/种子单独启动服务，在同一服务内测三个已排空场景；各种子轮内随机化策略顺序。Pilot中在服务内复用多个种子/标定档仅用于开发与选型，不冒充独立启动的正式重复。

饥饿压力Trace改为180秒开放环：每秒一个6144Token Prefill，加每1.5秒一个2048输出Token请求。原先每4秒补充一个1024输出请求，首批长Decode结束后无法保持足够高的Decode数量，因此不能声称验证了持续高压。正式日志以实际Decode数量证明压力，而不是按注入计划推断。

GPU兼容性另设640 Blocks、关闭完整输入长度预留的专用压力场景，触发原生抢占/恢复，避免只在CPU模拟器上验证Replay。该组主动改变KV容量与准入配置，严格排除在性能主表之外。

正式混合流量在冻结前另跑原生200请求、4 RPS的开放环校准，以该有限Trace完成RPS的80%选共同正式到达率，并明确这只是保守负载选择，不是最大稳定容量估计。若策略在该共同到达率下仍积压，队列与排空代价保留，不作为稳定系统吞吐收益。

Pilot时间线发现原压力切换Trace的Prefill注入在7.2秒就结束，Decode压力下降后已无新Prefill，无法评估低压恢复时扩大预算的实际价值。冻结前将正式transition改为50个长Prompt、0.8秒间隔，注入持续至39.4秒；原Pilot数据保留为开发用突发场景，与新Trace正式结果分开。

复现流程审查中补强了来源保护：重复freeze只有Hash完全一致时才幂等返回，禁止覆盖已有正式来源快照；原始数据审计同时核对各Session保存的在线代码Hash。报告与分析脚本可以更新，在线调度、请求生成与标定表的12项冻结输入保持不变。

前三个正式种子的Joint均大量停留在256配额下限，Decode尾延迟改善伴随Prefill排队和固定Trace吞吐退化。因此增加独立的`balanced`配置修正实验：预算下限取自既有独立验证集选中的2048，不修改控制器或主矩阵，冻结五个新种子做单独验证。该组属于观察结果后的工程修正，不包装为事先指定的确认性主结果。原配置和所有负结果继续保留。

兼容性实验发现当前API的usage不返回prompt_tokens_details.cached_tokens。将缺失值按0处理会误判缓存未命中。离线分析改为同时保留usage缺失状态、Prometheus prefix_cache_hits_total/queries_total增量、候选Scheduler实际Prompt计算量及首次计算位置；缓存场景要求“实际Prompt计算+命中Token=输入Token总数”。默认Aging缓存组已通过该对账。

默认Async/Graph的patched-off在17Token合成输入的第18个输出出现一次与stock不同的续写；前17个输出相同，所有自然问答及长度检查通过。此差异原始数据保留，另安排stock/off各两次新服务重复、对齐公共前缀的teacher-forced和自回归Top-20 logprob诊断，不能未经核查就归因于浮点误差。该诊断独立于性能矩阵，最终解释以实际探测记录为准。

数值诊断的两次全新stock服务均在同一合成用例、同一第18个Token复现原off续写分支；共同前17个输出不变。Top-20检查中“ involving”与“ a”的logprob差约0.015625–0.03125，off与stock均表现为接近候选。四次诊断均保留原始响应与logprobs，原始兼容性结果不覆盖。该证据支持原生批次/数值敏感性，输出完全一致不作为FP16异步服务的通用保证；有限自然任务仍一致。

Nsight本机默认cuda-graph-trace=graph，逐Kernel表不含图内节点。保留五策略整体图采集，并增加一个显式node粒度Joint诊断，避免以图外Kernel统计冒充完整模型Kernel统计。所有Profile均独立于性能主表，采集本身的时延扰动不计入优化收益。

复核Joint的128步窗口发现其中没有GPU图活动或非零Graph节点ID，不能将它称为图内算子验证。随后选择实际有图活动的observe补充节点采集：得到48228个Kernel活动，其中41975个带Graph节点ID，补齐图内覆盖证据。Joint窗口有41393个Kernel但无图内ID；不同策略窗口内的Prefill进度和Step组成不同，不能将这些数量或累计GPU时间的比值当作性能收益。

## 第二轮瓶颈优化（2026-10-20）

新实验根目录为 `/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320_opt2`。旧代码完整归档到 `baseline_source/`，旧原始数据和 421 个 Trial 不覆盖；新阶段分别保存协议、在线代码快照、请求/Token 记录和审计。原安装、模型、Scheduler 补丁和二进制保持原样；新增的是 `equalized` 模式与独立分析/实验脚本。

| 问题或负结果 | 证据 | 处理 |
| --- | --- | --- |
| 原 Joint 的纯 Decode 相对目标在混合步骤中不可行 | 原正式 Ledger 长期停在 256，Prefill TTFT 和吞吐代价过大 | 本轮尝试按实际前沿和因果 Attention 工作量均分，而非继续调整 EWMA |
| 16 对齐的成本均分没有达到 TTFT 约束 | seed227 transition Prefill P95 505.90 ms，原生 439.74 ms；Mixed P99 也未改善 | 保留负结果，测试 256 对齐，并独立比较固定中间档 |
| Fixed-1792 的结果没有位于相邻预算之间 | seed227 transition Prefill P95 627.27 ms，明显高于 Fixed-1536/2048 | 检查尾块、跨请求步骤和排队；采用更强的 Fixed-1536 作独立验证的静态对照 |
| 新的部分缓存回归中出现 block_pool IndexError | 测试启用了 Prefix Cache，但辅助 Decode 请求未提供原生 block_hasher | 让缓存组的所有请求都按原生方式构造 Hash；修改测试夹具后通过，不修改原生缓存代码 |

独立验证在看新种子结果之前登记五个种子 307/311/313/317/331，参数冻结，原生/旧均衡档/Fixed-1536/256 对齐均分各启动独立服务。另安排节点粒度 Nsight 诊断，独立保存，不加入性能收益表。最终数据、测试数量和输出一致性以第二轮报告为准。

执行中断后发现运行器/客户端退出、独立Session的API服务存活但空闲。18:30清理本项目孤立服务，归档seed317已完成的两个服务（8个Trial）与未开始测量的stock服务，按原顺序重跑该种子整轮以避免约五小时的同轮间隔。前三个完整种子保留。恢复脚本 `benchmarks.recovery_master` 以独立Session脱离交互终端，串行完成剩余GPU实验、节点采集、分析、模型校验和交付快照；阶段状态与PID写入运行目录。

节点采集首组的目录名balanced与旧阶段名碰撞，API试验完成后审计误读formal-source.json并失败。改为node-前缀的独立诊断目录后重采，原失败文件保留，不计入性能或有效Profile组。补充阶段Fixed-1536/seed419的17Token合成输入在第11个输出分歧（429与323，前10个相同），安排stock/off/Fixed-1536各两次全新服务及公共前缀logprob检查；原始差异不覆盖，最终解释以诊断实际结果为准。

六个数值诊断服务已完成：原始correctness重复均返回原stock分支，stock/off与stock/Fixed各次均为11/11一致。额外自回归探测中，两次未修改stock在offered batch=2时均完整复现原Fixed的32-token分支；共同前10个输出一致，第11位“ that”与“ and”的返回logprob并列。该证据支持原生批次/数值敏感性，不能把offered batch当成实际每步GPU批大小，也不代表logprobs模式精确还原原运行路径。原134个Trial中230/231输出比较一致，自然问答126/126、合成尾块104/105；原始不一致保持不变。逐token复核与来源Hash保存为numerical_batch_reproduction.json。

数值探测首次请求logprobs时触发Top-K内核JIT，通用性能审计拒绝该诊断服务。保留拒绝项与原始日志，诊断单独检查请求、长度、Trace与响应完整性；六个服务不提供时延或吞吐证据，性能主矩阵的JIT审计标准未修改。最终工程配置为有Decode时共享1536、无Decode时4096；成本均分在同基础预算下暴露短请求配额利用不足，受控真实Scheduler实验由11次含Prefill步骤增至16次，因此保留为实验模式，不替换最终配置。

## 项目文档整合（2026-10-21）

阶段报告、性能归因和工程验证合并为`reports/动态Prefill_Budget完整实验报告.md`，以1536/4096当前配置、最新验证与负结果为主。原始报告移入实验数据目录`document_archive/20261021-report-consolidation`，Git历史保留发布版本。新增`benchmarks.project_report`从已有审计JSON重建唯一项目报告；阶段文字生成改写到`APB_DATA/report_materials/`，后台流水线不再追加重复报告入口。同步明确补充三种子mixed TTFT配对增幅8.89%、95%区间[5.75,15.10]的约束限制。本次未新增GPU实验，未改动在线调度、模型或历史统计。
