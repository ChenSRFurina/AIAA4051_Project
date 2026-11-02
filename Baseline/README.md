# Baseline 运行与公共实验协议

Baseline 包含 N（无技能）、I（原始初始技能）、B（原版失败反馈优化器），以及数据、调用、预算、评分、图导出和冻结流程。Explore 的 C 复用这些公共代码，只替换优化器输入反馈。

环境安装与后端选择见 [主 README](../README.md)。当前 Mac 使用 API；原 `configs/searchqa.json`、`hotpotqa.json` 是旧 Windows 本地配置，不应与 `*_api.json` 混用。

## 一键执行 API Baseline

在仓库根目录：

```bash
conda activate NLP_project
bash api/run_baseline.sh
```

只跑一个数据集：`bash api/run_baseline.sh searchqa` 或 `hotpotqa`。脚本依次下载数据、pilot、初始化、N、I、B，结果在 `Baseline/runs/api/<dataset>/<时间戳>-<PID>/`。进度查询为 `python api/status.py`，不产生 API 请求。

## 分阶段执行

以下以 SearchQA API 为例，适合检查某个阶段或诊断后重跑。先在同一 Bash 终端设置路径，所有输出目录必须为空或不存在：

```bash
config_path="Baseline/configs/searchqa_api.json"
data_dir="Baseline/data/prepared/searchqa"
run_dir="Baseline/runs/api/searchqa/manual-$(date +%Y%m%d_%H%M%S)"

python Baseline/download_data.py --dataset searchqa
python Baseline/run.py pilot --config "${config_path}" --data "${data_dir}" --output "${run_dir}/pilot"
python Baseline/run.py initialize --config "${config_path}" --data "${data_dir}" --output "${run_dir}/shared_init"
python Baseline/run.py static --condition N --config "${config_path}" --data "${data_dir}" --output "${run_dir}/N"
python Baseline/run.py static --condition I --config "${config_path}" --data "${data_dir}" --output "${run_dir}/I"
python Baseline/run.py train --config "${config_path}" --data "${data_dir}" --initialization "${run_dir}/shared_init" --output "${run_dir}/B"
```

HotpotQA 同步替换配置、数据目录和输出路径。本地模型使用 `*_local.json` 和 `runs/local/`，先按 [本地模型说明](../local_model/README.md) 启动实际服务和 GPU 账本。API 模式不需要 GPU 账本。

`initialize` 产生原模板和模型自动生成的第二个个体，并在完整 dev 上评分。N/I 的 `static` 评估完整 dev，I 始终使用上游原模板。B 的 `train` 优化技能文档，不训练模型权重。B/C 必须使用同一个已完成的 `shared_init`；代码、配置、数据和 tokenizer 签名不匹配时拒绝复用。

## 数据准备

克隆仓库不含实际数据，下载脚本固定来源 revision 和划分 ID，已有准备数据先校验再跳过。

| 数据集 | train | dev | test | 边界 |
| --- | --- | --- | --- | --- |
| SearchQA | 100 | 30 | 50 | 使用固定上游划分中的 ID |
| 公开 HotpotQA distractor | 100 | 30 | 0 | 课程 test 待补，禁止最终冻结 |

课程 HotpotQA 文件到齐后，在正式初始化之前准备独立目录：

```bash
python Baseline/run.py prepare --dataset hotpotqa --train COURSE_TRAIN.json --dev COURSE_DEV.json --test COURSE_TEST.json --output Baseline/data/prepared/hotpotqa_course
```

`COURSE_*.json` 是实际课程文件路径的占位符。支持官方 `_id/question/answer/context/supporting_facts` 格式及统一的 `id/question/context/answers/metadata` JSON/JSONL。当前仅实现提供上下文的单轮 QA；若课程版本需要检索，必须另接检索。不得把 dev 改名为 test。数据 manifest 改变后需重跑受影响条件。

## 预算和异常

- population=2，最多三轮，每轮最多两个新候选；算子为 `mutation_other`、`mutation_graph`。
- 每轮两个父代使用同一 20 题 train batch；候选评估完整 30 题 dev。
- 失败抽样沿用上游 k=3、固定种子。B/C 使用同一初始化、预算、评分及选择机制。
- 每个阶段运行目录最多 600 次逻辑调用、700,000 生成 tokens、9 小时，生成总尝试次数最多两次。
- 调用前预留生成上限，失败且无 usage 时保守计账；reasoning 是 completion 的子集，不重复累计。
- API 失败后的下一次请求等待 `api_retry_delay_seconds`，默认 2 秒，不增加隐式调用。网络或未知 HTTP 故障耗尽两次尝试后保存诊断并停止。明确的 `data_inspection_failed` 内容拒答在两次尝试后仍未恢复时计零分并继续，保留在分母中，单独记录 `provider_refusals`。
- 输出格式或技能图无效属于模型生成失败，仍受同一候选尝试上限约束；保留所有被拒绝候选。

N/I 完成状态是 `static_complete`，B 是 `completed` 且 `completed_rounds=3`。`budget_stopped:...` 是提前终止。`calls.jsonl` 中的请求错误可能已在第二次尝试恢复，应同时看逐题结果；最终完整 dev 的 `infrastructure_errors` 应为 0。

B 的成本包含共享初始化的逻辑消耗。汇总实际 API 请求时只计算 `physical_calls>0` 的记录，不把 B/C 导入的初始化账本重复计费。

## 结果与后续 C/test

各阶段的 `summary.json` 保存 dev 指标；B 的 `best_skill.md` 是按 dev 选出的最终技能，`history.json`、`operators/`、`skills/`、`evolution.mmd` 记录进化过程，`dev_analysis/failures.json` 包含完整 dev 失败清单。

B 完成后检查完整 dev 错误和技能变化，再按 [Explore 说明](../Explore/README.md) 冻结反馈设计、运行 C。最终 `freeze` 要求 N/I/B/C 全部完成且共享协议一致；随后 `test` 只读取冻结技能，不更新训练记忆。

## 离线正确性验证

```bash
python -m unittest discover -s Baseline/tests -v
python Baseline/run.py demo --output Baseline/runs/demo-check
```

合成流程覆盖两数据集和四个条件，不能报告为真实模型或 benchmark 成绩。

上游固定为 [GraphSkillEvo 提交 722a8a9](https://github.com/ruisun7/GraphSkillEvo/tree/722a8a913b05dc9d1b4078f0bec604cde4dc7529)，源码保持在 `vendor/GraphSkillEvo`。共同包装层记录见 `UPSTREAM.json`；手工诊断、课程划分等未完成工作不能由合成验证替代。
