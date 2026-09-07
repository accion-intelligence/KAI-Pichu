# GPT-5.6 Luna LayerNorm 实测：2026-09-07

本报告记录发行包中保留的 LayerNorm-1024 示例。模型实际生成候选、诊断错误、
读取 NCU 反馈并完成独立验收；这不是 replay 模拟。

## 任务与结果

输入为 FP32 `[16384,1024]`，epsilon=1e-5。性能基线是原 FlashAttention CUDA
实现，正确性参考为独立的 PyTorch 实现，z、mu、rs 均按 atol=rtol=1e-4 检查。

速度比定义为 baseline 耗时 / candidate 耗时，大于 1 表示改进。

| 阶段 | 结果 | 95% 置信区间 |
| --- | --- | --- |
| 候选 1 | 正确性失败 | 不计性能结果 |
| 候选 2 | 修复后正确，0.804317× | [0.802642, 0.805934] |
| 候选 3 | 正确，1.139411× | [1.137549, 1.141189] |
| 独立验收 1 | 通过，1.128998× | [1.127264, 1.130718] |
| 独立验收 2 | 通过，1.129226× | [1.128056, 1.130412] |

第一次独立验收的平均算子耗时从 110.96 µs 降到 98.28 µs。速度比采用配对
block 统计，仅适用于声明的 GPU 算子计时边界，不包含主机调用、编译或图捕获。

## 循环行为

首个候选遗漏了求均值时的列数归一化。正确性检查拒绝它，judge 指出具体
表达式，下一轮修复后通过，但性能仍落后于基线。随后真实 NCU 反馈进入优化
judge，生成器将 gamma/beta 暂存到共享内存，产生通过验收的候选。

最终候选还通过了随机 gamma/beta 检查，以及图捕获后更新同一缓冲区、再重放
的检查。这些是额外正确性检查，不构成对所有输入范围的穷尽证明。

## 测量方法

最初的外层 CUDA event 计时未通过 A/A，模型调用数为 0。该尝试的报告保留在
本地运行产物中。随后使用 `graph_events` 模式：两个 `external=True` 的事件
记录节点在同一 CUDA Graph 内包围一次完整算子执行，`operator_latency_ms`
是优化目标，SDK 原有外层 `latency_ms` 作为诊断数据保留。

这种边界排除 Python 提交间隙，仍可能受设备调度、频率和资源干扰影响。
校准阈值保持 0.5%，置信度保持 95%；40 blocks、每个位置 5 次独立恢复后的
执行，每个 block 有四个交叉配对位置。没有删除离群值；每次候选评测和最终
验收都重新执行 A/A。

每次执行前毒化输出并冲刷独立的 256 MiB buffer。编译、输入准备、图捕获、
恢复状态和正确性验证均在评分区间外。NCU 数据用于诊断，不用于速度比评分。

## 环境与复现

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition，SM120。
- 驱动：595.58.03；PyTorch：2.11.0+cu130；CUDA：13.0；Python：3.13.9。
- 模型：`gpt-5.6-luna`，Responses API，`reasoning.effort=high`，temperature 不传，
  每次最多 16384 output tokens，`store=false`。
- 预算：3 轮、5 次模型调用、1800 秒总预算、300 秒单次评测、2 次最终验收。
- 不锁频；评测前、中、后检查外来 GPU 进程。

从 KAI-light 目录运行，提供自己的 dotenv 文件、GPU UUID 和新输出目录：

```bash
python scripts/run_with_env.py --env-file /path/to/.env --gpu GPU-YOUR-UUID -- \
  optimize examples/layernorm/benchmark-graph-events.yaml \
  --config configs/gpt56_luna_smoke.yaml --output runs/my-layernorm
```

该任务的 5 次优化调用记录 input 126512、output 25213、total 151725 tokens。
这是 API 返回的 usage 汇总，不是账单金额。

本地原始产物位于 `runs/luna-smoke-20260907/layernorm1024-graph-events/`，
包括源码快照、请求/响应、全部评测、NCU、checkpoint、独立验收和 patch。
额外检查位于同级 `layernorm1024-affine-probe.json` 和
`layernorm1024-replay-probe.json`。运行产物不进入源码发行包。

这是一次小规模集成实验，不能作为跨任务、跨模型或跨硬件的普遍性能结论。
