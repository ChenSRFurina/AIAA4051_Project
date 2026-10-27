# Explore：成功–失败对照驱动的自我纠正

对应 plan 第 4 节主方案（课程 Option A）。本目录复用相邻 `Baseline` 的数据、模型调用、进化循环、抽样、选择、指标、预算与冻结流程；唯一方法差异是父代的 `train_reflection_context`。

借鉴 [JitRL](https://arxiv.org/html/2601.18510v1#S4) 的经验记忆与局部反馈思想，实现的是**技能指令层面的自我纠正**。JitRL 原文直接估计动作优势并修改 logits；这里没有把指令反馈冒充 logit 更新，也没有实现 plan 第 13 节明确列为后续扩展的动作数值惩罚。

## 已实现

- `explore/memory.py`：只允许本条件固定 train ID 的真实输出进入记忆；拒绝 dev/test、B 正式分叉后的记录及缓存输出。每题最多 2 份不同成功和 2 份不同失败；去重不删除原调用账本。
- `explore/pairing.py`：相同题目 ID 与相同可见上下文 hash 的成功优先；否则按固定问题规则划分粗类型，并以 Jaccard>=0.2 找同类参照；没有则 unpaired。优先当前轮，其次最近历史，固定 provenance 顺序打破并列。任务类型不使用 HotpotQA 的 gold type。
- `explore/feedback.py`：给原抽样器选中的 k=3 失败补充成功参照、格式检查、证据状态、诊断不确定性以及“保留有效行为”的约束。同题成功不等于每一步正确；相似题只支持抽象策略比较；没有节点执行日志时始终标为诊断假设。
- 两个父代的训练执行全部完成后才构造反馈，因此第二个父代的正常成功输出也可被第一个父代引用。不额外调用成功采样器或评审模型。
- 完整 packet 和模型实际收到的有界反馈分别保存。通过逐字段缩短保持实际 JSON 完整，避免直接截掉后半段成功参照。完全没有可比成功时回退原 B 反馈。

输出 `memory.json`、`feedback/*.json` 包含来源技能 hash、round、visible_context hash、匹配类型与实际文本。配对覆盖统计应依据实际被选父代对应的反馈；没有失败的父代记为 0 个样例，不伪造分母。

## 运行

从 `AIAA4051_Project` 根目录执行；先按 `Baseline/README.md` 配置模型、跑 pilot，并完成 B 的全部 dev 失败整理。

```powershell
python Explore/run.py freeze-design --config Baseline/configs/searchqa.json --baseline-run Baseline/runs/searchqa/B --output Explore/runs/searchqa/feedback_design.json
python Explore/run.py train --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --initialization Baseline/runs/searchqa/shared_init --design Explore/runs/searchqa/feedback_design.json --output Explore/runs/searchqa/C
```

`freeze-design` 保存 B 完整 dev 失败清单 hash、配对规则、反馈 prompt 和代码 hash；这是实现规格快照，不代表人工诊断已经完成。`train` 检查该快照仍一致，然后与 B 从同一共享初始化分叉。

HotpotQA 同样使用 `hotpotqa.json`，课程版准备好后改用 `hotpotqa_course` 数据目录，并独立初始化与演化。两套数据的记忆完全分离。

最终使用 `Baseline/run.py freeze` 同时冻结 N/I/B/C，再统一 `test`。测试仅执行冻结技能，不根据测试 gold 更新记忆或纠正答案。所谓“自我纠正”发生在 train 经验驱动的技能迭代阶段。

## 验证

```powershell
python -m unittest discover -s Baseline/tests -v
python Baseline/run.py demo --output Baseline/runs/demo
```

测试覆盖共享失败抽样、数据隔离、记忆来源/去重、同题与同类配对、有界反馈结构、调用预留、失败计费、reasoning 统计、变异范围、冻结后篡改拒绝，以及两数据集 N/I/B/C 的合成端到端运行。合成成绩不用于回答方法是否有效。
