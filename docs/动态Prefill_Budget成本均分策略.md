# 按计算成本均分的 Prefill Budget

本策略在 vLLM V1 Scheduler 内按请求的实际计算前沿选择每轮共享 Prefill/Replay 配额。它使用历史实验标定的前馈成本模型；AIMD 的反馈状态不决定此模式的预算。性能结论见 [第二轮优化报告](../reports/动态Prefill_Budget第二轮优化报告.md)。

## 规则

没有可执行 Decode 请求时，使用全局上限 4096。有 Decode 时，在本轮第一个实际需要 Prefill/Replay 的请求处确定共享配额：

1. 读取已经扣除 KV Cache 命中的实际计算位置 s，待重算前沿 e=max(prompt_tokens, num_tokens−1)，剩余量 R=e−s。
2. 对总长度不超过基础预算 B 的短请求，使用 B 的共享配额，以保留批处理机会。
3. 对长请求，令 k=ceil(R/B)，将剩余计算成本均分到 k 步，计算本轮 token 数；向上对齐后，不超过 R 和全局预算。
4. 全部 Running/Waiting Prefill 和 Replay 共用该配额；每个尚未遍历的可执行 Decode 请求预留一个 token 空间。成功分配 KV 后扣费，撤销选择时退款。

当 R≤B 时直接使用剩余量，避免为了对齐虚增工作量。计划每轮重算，取消、抢占和请求 ID 复用不会遗留切块计划。对 Replay 使用生成历史的实际重算前沿；纯 Decode 的单个新输入 token 不计为 Prefill。

## 成本与求解

定义归一化计算成本：

\[
W(s,n)=n\left[1+w\left(s+\frac{n+1}{2}\right)\right]
\]

其中 w 来自线性 token 成本与因果 Attention pair 成本之比；本机标定为 0.00012347502462458387。令 U=W(s,R)/k、a=1+w(s+0.5)，求解正根：

\[
n=\frac{2U}{a+\sqrt{a^2+2wU}}
\]

此形式避免相近大数相减。在线只包含少量标量运算，不执行 GPU 同步或在线训练。

对于 6144-token Prompt、B=2048：16 对齐得到 2464/1984/1696；256 对齐得到 2560/2048/1536。两者都保持三次推进，但真实执行成本和端到端表现必须实测。Decoder 的请求数、上下文与排队成本没有完全包含在这个相对成本公式中，因此它不提供单 token 的硬时延上界。

## 接入和配置

`Config.mode="equalized"` 启用该模式，`cap` 表示基础预算 B，`chunk_alignment` 控制对齐，`context_weight` 表示 w。初筛候选为：

```json
{
  "mode": "equalized",
  "cap": 2048,
  "maximum": 4096,
  "chunk_alignment": 256,
  "context_weight": 0.00012347502462458387,
  "reserve": true
}
```

`apb/controller.py::equalized_chunk` 是无状态求解器；`apb/runtime.py::clip` 在缓存前沿已知后选择共享配额；`charge/refund` 保持原有事务式记账。`cap` 和 `reason` 在实际选择后更新到 Ledger，而非把预先的全局上限误记为最终预算。

代码默认仍为 `mode=off`。模型缓冲区、全局 token 上限、原生 KV 分配、FCFS 顺序和抢占职责保持原有实现。基线模型和硬件为 Qwen3-1.7B FP16 / RTX 3090，系数不应直接当作其他硬件和模型的通用最优值。
