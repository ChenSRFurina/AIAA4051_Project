# Explore 成功与失败对照反馈

C 与 Baseline B 共用数据、模型、初始化、抽样、选择、预算和评测流程，唯一方法差异是父代的 `train_reflection_context`。它在训练经验驱动的技能迭代阶段形成对照反馈，不在测试时根据 gold 修改回答，也不训练模型权重。

借鉴 JitRL 的经验记忆与局部反馈思想，当前实现修改技能优化器的输入，不执行 logits 更新或数值奖励惩罚。

## 已实现的反馈

- `memory.py` 只接受本条件、固定 train ID 的真实输出，拒绝 dev/test、其他条件分叉后的输出和缓存结果。每题最多两份不同成功和两份不同失败。
- `pairing.py` 优先同题、同可见上下文的成功；其次按问题类型和 Jaccard≥0.2 寻找同类参照；没有则标为 unpaired。忽略未来轮经验，不使用 gold 类型标签。
- `feedback.py` 给原抽样器选中的 k=3 失败补充成功参照、格式检查、可见证据、诊断不确定性和保留有效行为的要求。
- 两个父代的 train 执行都完成后才构造反馈，避免先后顺序造成不对称。没有额外成功采样器或评审模型。
- 全部没有可比成功时，回退到 B 的反馈。成功答案不保证每一步推理正确，同类参照也不直接证明失败的因果来源。

`memory.json`、`feedback/*.json` 保存来源、轮数、上下文 hash、匹配类型与实际文本。图导出反映技能定义，不冒充模型内部的节点执行轨迹。

## 从已经完成的 API Baseline 运行 C

先完成 N/I/B，检查 B 的完整 `dev_analysis/failures.json`、原始输出及技能图。下面的 `run_dir` 必须改为一个已经完成的实际 Baseline 目录；保持其代码、配置、数据和分词器不变。

```bash
conda activate NLP_project

# 修改此行中的运行目录名称
run_dir="Baseline/runs/api/searchqa/替换为已完成的运行目录"
config_path="Baseline/configs/searchqa_api.json"
data_dir="Baseline/data/prepared/searchqa"
explore_dir="Explore/runs/api/searchqa/$(basename "${run_dir}")"

python Explore/run.py freeze-design \
  --config "${config_path}" --baseline-run "${run_dir}/B" \
  --output "${explore_dir}/feedback_design.json"

python Explore/run.py train \
  --config "${config_path}" --data "${data_dir}" \
  --initialization "${run_dir}/shared_init" \
  --design "${explore_dir}/feedback_design.json" \
  --output "${explore_dir}/C"
```

`freeze-design` 固定 B 的完整 dev 失败清单、反馈规则与代码指纹，只记录规格，不代表人工语义诊断已经完成。C 只复用共享初始化，不读取 B 分叉后积累的 train 成功记忆。

HotpotQA 使用 `hotpotqa_api.json` 和相应数据目录，两数据集独立运行。本地模型把配置换为 `*_local.json`，并保持原本地服务和 GPU 账本。若需要修改代码、模型、数据或其他协议项，重新建立共同初始化并重跑受影响条件。

## 四组冻结与最终测试

在同一终端沿用上面的变量，C 完成后执行：

```bash
python Baseline/run.py freeze \
  --N "${run_dir}/N" --I "${run_dir}/I" --B "${run_dir}/B" \
  --C "${explore_dir}/C" --data "${data_dir}" \
  --output "${run_dir}/frozen.json"

python Baseline/run.py test \
  --frozen "${run_dir}/frozen.json" \
  --output "${run_dir}/test"
```

输出 `test/results.json`、`test/comparison.json` 和各条件的逐题结果。冻结校验覆盖技能、运行证据、代码、配置、tokenizer 和数据。最终测试不更新经验记忆，不把 test gold 或测试成绩送回优化器。

公开 HotpotQA 数据仍为 `test_pending=true`，不能进行最终冻结。接入课程划分须在正式初始化前完成，不能临时把 validation 改作 test。

## 验证

在根目录执行 `python -m unittest discover -s Baseline/tests -v`。`python Baseline/run.py demo --output Baseline/runs/demo-check` 使用合成数据和模拟响应，验证流程，不代表方法有效性的实验结论。
