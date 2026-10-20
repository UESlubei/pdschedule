# 当前实施状态与续跑事项

**项目已完成。** 最新验收以`reports/completion.json`和`reports/动态Prefill_Budget完整实验报告.md`为准：421个有效Trial、零审计错误、22项回归通过、39项源码/模型最终校验通过。另完成4个数值诊断服务及2个节点Profile，合计7份Nsight采集。硬件采样已停止，GPU计算进程已退出。以下内容是长实验过程中的历史交接记录，不代表仍有待跑矩阵。

本文件用于长实验的工作衔接；最终事实以原始结果和状态文件为准。

- 用户要求完成V2项目，交付实验报告，明确问题/修复/收益/负收益。不能在实验尚未完成时声称项目完成。
- 主目录：`/home/sdn/wjh/Ondeviceinfer/source/adaptive_prefill_budget`。
- 大数据：`/mnt/ondeviceinfer/experiments/adaptive_prefill_budget_v0320`。
- 解释器：`/mnt/ondeviceinfer/experiments/vllm_kv_scheduling_latest_v0320/environment/bin/python`。
- 所有模块命令需要`PYTHONPATH=/home/sdn/wjh/Ondeviceinfer/source/adaptive_prefill_budget`；CPU回归额外把`$DATA/overlays/candidate`放在PYTHONPATH最前。
- 实现完成，22项控制器/真实Scheduler/KV回归通过，Sync/Async GPU Pilot、取消恢复、自然用例6/6逐Token一致。
- `apb/*.py`、`benchmarks/session.py`、`scripts/prepare.py`、调度补丁、策略清单、两份延迟标定表和两份验证选择文件共12项已经冻结；不要修改这些文件。`scripts.freeze`会拒绝覆盖。分析/报告/监控脚本可更新，不影响已测在线实现。
- 后台主驱动：`python -m scripts.complete_project`，统一exec session_id=76394；主流水线在GPU锁内依次运行formal、mechanism、starvation、native、ablation、compatibility。日志`logs/complete-project.log`。
- 另一后台进程`python -m benchmarks.pipeline --stages idle`，session_id=33949，等待相同GPU锁，主流水线结束后做3组空闲Prefill。主驱动的Profile也使用同样GPU锁，不会并发抢卡。
- GPU硬件监控PID保存在`manifests/hardware-monitor.json`，每5秒采样至`logs/hardware.csv`。最终应停止自己的该监控进程。采样从首轮正式种子之后开始，首轮缺少连续硬件数据，须如实说明。
- 初始observe因默认shutdown_timeout=0导致遥测未落盘被审计拒绝，原始记录保留`raw/failures/eager-observe-abort`。已统一shutdown-timeout=20并成功重跑，不能把失败目录并入主结果。
- 第一正式种子17的11策略/33 Trial已完成并通过审计，随后开始种子29；完整formal预计55独立服务/165 Trial。每服务做interference、transition、mixed三场景，5种子17/29/43/71/101，轮内随机化策略顺序。
- 验证集独立选择Fixed-2048（Prefill P95 TTFT≤+10%，吞吐≥-5%），保存于`manifests/static-selection.json`。
- 4RPS Pilot积压，正式mixed共同到达率选2.422RPS，保存于`manifests/load-selection.json`。这是保守负载，不是最大容量测量。
- `scripts.watch_status`可用紧凑JSON读取当前进度；直接读取日志/状态即可，不要重复启动GPU服务。
- 审计入口：`python -m benchmarks.analyze`，生成`reports/results.json`与`audits.json`。已支持核对冻结代码、全局/Prefill配额、提交/完成、Token长度、单Token SSE、队列/服务分解、目标超限率和测量JIT告警。
- 绘图/统计：`PYTHONPATH=... /home/sdn/anaconda3/envs/ljc/bin/python -m benchmarks.report`。
- 报告：同一分析解释器`-m benchmarks.write_report`，输出`reports/动态Prefill_Budget完整实验报告.md`，目前明确标为持续运行。最终还需要写入基于全部数据的`reports/性能归因与结论.md`，把真正的结论合入完整报告，不能保留“待补充最终解释”。
- `benchmarks.write_report`的complete门槛要求driver完成标记、165正式Trial、来源不变、审计无错误及上述结论文件存在；还应检查idle和Nsight是否完成，补充完整解释和链接。
- 待完成：等全部矩阵结束；检查任何失败并修复/重跑；汇总原始报告与5对种子CI；画最终曲线、控制时间线、Pareto；分析AIMD压到最小Cap的TTFT/吞吐代价、Aging救援代价、原生阈值与最佳静态对照；检查Nsight产物并记录失败或结论；自然输出兼容性核对；核验原安装源码和模型/私有副本；写最终报告与README链接；输出真实数值，不能只报P99下降。

首种子仅用于观察：原生default interference Decode P99约252ms，Load约64ms，Joint约48ms；Prefill P95 TTFT原生约9.57s、Load约12.14s、Joint约15.08s；固定Trace吞吐约596/535/491 tok/s。正式结论必须等5个种子，不能把这些单次数字包装为最终收益。

首种子Joint在interference约90%的实际Prefill Step处于最小配额，transition约98%；主要TTFT代价是排队放大，服务端完成代理的目标并不总可达。Mixed吞吐接近同一到达率给定的约600 tok/s，不能宣称最大吞吐提升。

## 2026-10-20 03时续跑补充

- 已完成正式44/55服务、132/165 Trial，开始种子101。此前刷新到99正式Trial/全部178 Trial均零审计错误。
- 新增`benchmarks.diagnostics`：可重算自然输出逐Token对照、cache使用、取消、Replay、CPU统计、持续压力到达时间分箱/队列增长/排空、硬件采样。非冻结分析脚本，不影响主测。
- 新增`scripts.verify_sources`：原安装/stock副本/候选补丁/12项冻结来源，30项检查通过；最终带`--weights`核验完整模型。
- `benchmarks.profile`最终改为5策略observe/F512/F2048/joint/aging（原生stock没有采集触发，observe保留原生决策作为诊断代理）。主驱动尚未调用，届时会加载新脚本。
- 增加独立`balanced`配置修正实验：仍用冻结在线实现，只把Joint minimum改2048，与stock/F2048对照；5新种子151/163/179/191/211，3场景，共15服务45 Trial。原因是原Joint最小256带来严重TTFT/吞吐代价；2048来自既有独立静态验证选择，修正前注册在DATA/manifests/balanced-protocol.json。严格分开报告，不并入原主矩阵。
- `balanced`后台已启动并等待GPU锁，统一exec session_id22452，日志logs/balanced-driver.log；原idle session33949仍在等待；主driver session76394仍运行。预计主stage释放锁后idle→balanced→profile（实际以锁顺序为准）。
- 完成门槛已增强：原165正式+15mechanism+15starvation+30native+36ablation+18compatibility+3idle+5profiles+45balanced共332后冻结Trial，另有79开发/校准/选择Trial，总目标411 Trial。Profiles必须5份返回0、nsys-rep存在、stats返回0，才算complete；完整结论文件依然必须写。
- 分析脚本现将“during inference:” JIT警告记为审计错误；旧常驻pipeline已经导入较早分析模块，最终必须重新跑全部analyze核对新检查。
- 默认路径Pilot natural+tail共11/11逐Token一致（stock/observe/joint）；尚未完成兼容性全矩阵。前三种子observe相对stock性能开销约<2%，最终五对确认。
- 最终仍需修改write_report的10/11段占位解释，集成真正`性能归因与结论.md`；报告无需假称多RPS容量搜索或上游全套测试已做。预算较小增加每Prompt步数与排队代价，不能只报P99收益。

## 2026-10-20 05:38续跑

- 正式55服务165Trial、mechanism15、starvation15均已全部完成，无pipeline失败。开始native阈值15服务30Trial，之后ablation36、compat10服务18Trial，锁释放后idle/balanced/profile。
- `benchmarks.analyze`新加cohort分别的排队/首Prefill到首Token/Chunk步数/最大进度间隔，以及完成lag按mixed/prefill/decode区分；完成lag含Async在途，不能称GPU Kernel时间。新审计同时拒绝active_tracked泄漏和missing_ledger。
- 首seed压力：F512/Joint/Aging三者308请求全成功；wall363.54/371.97/364.48s，PF95约154.59/163.35/155.21s，最大等待144/152/146，高Decode步占比94.6/95.1/94.6%。Aging2140救援中320有实际PF。有限过载，不是稳态或有界TTFT。图已生成reports/figures/starvation-seed17.png/pdf。
- `benchmarks.report`新增trace_comparability.json（formal/balanced相同场景种子跨策略Trace Hash必须一致）、balanced_summary/comparisons独立配对统计。
- `benchmarks.profile`新增SQLite kernel_count/sum/max，完成报告现在要求每份Profile捕获kernel_count>0，不仅文件存在。
- balanced-protocol已补充验收规则：每场景对stock的配对均值 PF95≤+10%、吞吐≥-5%、P99<0，并报告CI；强证据需CI边界也满足。数学上minimum2048使Joint有Decode时恒2048、无Decode时4096，因此是保守二档回退，不证明AIMD增量。配置已放configs/presets/balanced.json（子目录，不在原12项冻结输入中）。
- 最终必须将这一退化机制和默认Joint负结果写清楚；不能把新配置当成复杂反馈胜过最佳静态的证据。新种子优化实验数据尚未开始，没有事后按其结果调参。

- 05:47补充：native最初3种配置首种子已完成，实际解析adaptive false/true正确；Threshold512 mixed P99约52.62ms、PF95约643.72ms，原生也有竞争力，不能预设劣于项目。
- 已增加设计原列出的native2048阈值，作为单独stage `native2048`，5种原正式种子×2场景=10 Trial、5服务。后台session75212等待锁，日志logs/native2048-driver.log；由watch_status计数，完成门槛也含该10Trial。总目标现为421 Trial、191个成功Session（以最终实际审计为准），而非旧411。
- 全量刷新到274Trial、全部0错误；持续压力五重复均有限过载，F512/Joint/Aging平均测量363.70/371.38/365.11s、停止注入后平均排空184.70/192.38/186.11s；早0–60s时间加权等待24.1/25.7/24.7，晚120–180s120.0/125.8/121.9。Aging实际带PF的救援总1623/10680，约15.2%。

- 06:29：native15服务30Trial全部完成；ablation第一seed17的12配置完成，开始seed29。min1024将P99/Prefill95/tokps变为约101.11ms/10966.5ms/554.55，较默认Joint恢复TTFT/吞吐但牺牲尾延迟；参数扫描仍是权衡。
- write_report现在需要两个最终解释文件：reports/性能归因与结论.md、reports/工程验证与开销.md；生成时分别完整插入第10/11节，覆盖之前的待补充段。两份必须等数据齐全再写实际结论。完成门槛也要求两份文件存在。
- 原native5均值：th512 interference P99/PF95/tokps=201.72/10176.12/590.49，mixed=52.25/631.84/597.46；th1024 fixed=219.89/9901.16/588.88，mixed79.56/502.34/598.18；adaptive1024=220.05/9891.42/589.21，mixed79.50/501.96/598.19。原生在mixed也有强竞争力，不能宣称项目处处更好。

## 07:06重要续跑：兼容性已完成，数值诊断待完成

- 原主流水线formal/mechanism/starvation/native/ablation/compatibility全部完成，driver已全量分析358Trial、0错误，现等待GPU锁做Profile。
- idle已1组完成、正在F512；然后锁队列预计balanced→native2048→numerical_probe→profile（实际以进程/receipt为准）。balanced15服务45Trial、native2048五服务10Trial仍未开始。
- 小KV压力真实GPU：Eager106抢占、Default115抢占，均Replay240Token、10请求全完成。调度Ledger Prompt累计115984/108560（含重复重算）、Decode3190，压力容量640Blocks，不能并入性能主表。
- 8组compat自然问答均6/6一致，全部取消恢复proof8Token成功。合成尾块默认off有1例分歧：tail1输入17，前17输出相同，第18个stock生成15860(involving)、off生成264(a)。default-off总10/11一致，其余组11/11。必须保留并解释，不能最终宣称所有逐Token完全一致。
- 新增benchmarks.numerical_probe，后台统一exec session19529，日志logs/numerical-probe.log，已排队同三把GPU锁；DATA/quality_drift下存4个独立服务（stock/off×2重复），原correctness重跑+公共17输出前缀的teacher-forced/自回归top20 logprobs，offered batch1/2/8。该目录在raw之外，不计入421性能/集成Trial，也不改变冻结实现。最终receipt.json complete=true是报告新完成门槛。诊断必须检查实际分歧是否也在stock重复/批形下出现，以及近似平局margin，不能仅凭猜测归因浮点。
- Cache usage当前版本缺cached_tokens细节，已把缺失记None而非0。analyze增加prefix_cache_hits_total_delta/queries_delta和first_compute_offset_by_request；cache若有Ledger强制实际Prompt计算+缓存命中=总输入。default-aging单独新分析已通过。最终需要全量新analyze和diagnostics刷新该证据。
- 最终报告需要numerical_probe完成、5份Profile有非零kernel_count、全部矩阵421Trial、source/模型校验、两个解释文件齐全才能complete。
- 原生与消融附加配对bootstrap在report.py新写additional_comparisons.json；Trace Hash公平性检查扩大到所有后冻结阶段，同scenario/seed跨stage也必须一致。最终若报不一致，应查真实Trace区别，不能删除检查。
- 四份正式transition时间线已生成load/joint/aging/F2048 seed17，可最终报告使用。pressure进度图starvation-seed17.png/pdf已生成。

- 07:17：idle3服务完成：stock/F512/Joint 单Prefill P95 TTFT394.52/489.76/407.14ms；Joint无Decode放宽仍有约3.2%观测差异，只有1服务8请求，不应声称零开销或精确系统因果。
- balanced第1新种子151三服务完成：Stock/F2048/Joint-min2048 interference P99=249.38/161.80/160.97ms，PF95=9510.37/9718.20/9696.08ms，tokps597.91/586.30/587.00；transition Stock/F/J P99=227.74/163.91/163.23，PF95=442.88/469.33/465.50；mixed P99=113.95/113.36/113.05，PF95=415.48/431.28/428.91。新档满足首seed均衡边界，但mixed尾部只差0.8%，必须等5个seed且对照F2048，不夸大。
- report.py图表已新增每策略Run bootstrap95%均值误差棒；Pareto以颜色表示吞吐、三角形动态/圆形其他，标签加位移与连线，并导出PDF。最终全量report时再生成所有图。

## 07:56：balanced全5种子完成，接近最终汇总

- balanced15服务45Trial完成，开始native2048。其后仍待数值诊断4服务及Profile5（都已排队，不能重复启动）。最终完整矩阵仍421Trial，另4数值诊断服务不计入主统计。
- 新鲜配对均衡档Joint-min2048 vs stock：interference P99 -35.853% CI[-36.011,-35.645]，PF95 +1.718% [1.528,1.908]，tokps -1.721%[-1.815,-1.627]；transition P99 -28.668%[-28.860,-28.477]，PF95+5.810%[5.425,6.130]，tokps-.051%[-.059,-.042]；mixed P99-3.535%[-10.451,+.365]（跨0，不能宣称显著），PF95+3.609%[3.064,4.200]，tokps-.021%[-.034,-.010]。三场景点估计满足原均衡边界，只有前两场景CI也证明P99下降。
- vs同一轮F2048：interference P99+.164%[-.379,.694]、PF95-.066%[-.243,.112]、TPS+.052%[-.090,.195]；transition P99-.196%[-.502,.106]；mixed P99-.156%[-.413,.238]。基本等效，不能声称复杂反馈优于最佳静态。该preset本质有Decode=2048、无Decode=4096。
- 全量分析/绘图/diagnostics/write_report后台refresh session37754（07:56启动），需poll完成；预计406+初始native2048若干Trial。最新缓存缺失字段修复、缓存compute+hits对账、跨stage Trace hash、新fig bootstrap/Pareto都会应用。最终全部421齐后还要再跑一次全量刷新。
- 未来最终报告性能正文应写：默认Joint/aging负结果，min512/min1024提高TTFT/TPS但P99损失，修正档新种子验证，mixed无稳健增益，低载Joint TTFT约+3.2%只1服务8请求，不称零开销；纯Decode目标不可达与Queue放大、实际Decode内生、有限过载/Aging槽位限制。
- 必须待numerical receipt complete，检查stock-vs-stock/shape漂移及Top20近似平局，之后才能写工程验证与开销.md；不要提前将Off合成分歧归因浮点。不得编辑运行中已经加载的numerical_probe.py或profile.py源码，以保持诊断来源一致。

## 08:08：数值诊断已复现原生漂移，追加节点粒度Profile

- native2048五服务10Trial全部完成；数值诊断正在运行（session19529），第一个stock重复已完成，原生stock自身也是10/11，唯一差异为同一tail1！公共前缀teacher-forced top2为“ involving”与“ a”，logprob差0.015625，autoregressive offered batch1/2差0.03125、batch8差0.015625。最终应检查全部4服务后再定稿。
- 本机Nsight help确认driver>=11.7默认cuda-graph-trace=graph，即整体图，不收集图内节点。原profile.py进程已在7:04加载并等待锁，不编辑其源码。另新增benchmarks.profile_nodes独立短采集joint，显式--cuda-graph-trace=node，后台session12378，日志logs/node-profile-driver.log，等待锁；DATA/profiles_node存data/receipt，完全在raw之外，不计入421主/集成Trial。完成门槛新增node_profile_finished，必须kernel_count>0、stats0、预算审计0。
- 性能归因与结论.md已写入实际完整主矩阵/均衡/消融/压力/低载结论，会被write_report插入第10节。工程验证与开销.md尚未写，必须等全部数值及Profile数据。
- 最终all-stage421Trial仍待Profiles5（native2048已齐）。额外诊断是4数值服务+1节点Profile服务，不混入性能表。
