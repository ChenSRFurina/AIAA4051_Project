# Project 5 Qwen API 实施方案

> **历史方案说明（2026-11-10）：** 以下正文描述早期阿里云接入，当前以队友部署的 `qwen3.5-9b`、环境 `qwen3.5`、[api/README.md](api/README.md) 及固定 API 配置为准。已完成 [Part 1/2](api/PART12.md) 和 [封存 C 的 Part 3 对照](api/PART3.md)：SearchQA 50 题 test 的 N/I/B/C EM 为 70% / 72% / 68% / 64%；HotpotQA dev B/C 均 19/30，课程 test 仍缺。两套 C 均完成三轮但保留初始化最佳技能，本次结果不支持 C 优于 B。报告为 `Baseline/runs/reports/part3_20261110.html`，当前方法与配置保持 test 前封存版本。

本方案使用阿里云 Qwen API 完成现有 GraphSkillEvo 实验流程，本机只负责数据、图结构校验、经验记忆和评测，无需部署模型或配备 NVIDIA 显卡。原本的 [本地模型方案](Project5_Implementation_Plan_zh.md) 保留；本方案的运行目录与本地实验分开。

研究问题仍是：在相同模型与预算上限下，成功与失败对照能否帮助技能优化器修复错误、减少退化。API 负责生成答案和修改技能文档；训练记忆、配对、反馈构造、EM/F1 评分及选择规则继续使用仓库代码。这里不训练模型权重，也不实现 logits 修改或数值奖励惩罚。

## 1 固定 API 模型与运行环境

| 项目 | 本方案 |
| --- | --- |
| conda 环境 | `NLP_project`，Python 3.11 |
| 服务地址 | `https://llm-ps7f58n2fo6d73fm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| 模型 | `qwen3.5-flash-2026-02-23` |
| 密钥 | 仓库根目录 `.env` 的 `QWEN_API_KEY`，仅在运行时加载 |
| 思考模式 | `enable_thinking=false` |
| QA | temperature=0，最多生成 1024 tokens |
| 技能生成 | temperature=0.6，最多生成 3072 tokens，JSON object 模式 |
| 共同设置 | seed=42，top_p=1，并发 1，超时 300 秒 |
| 本地模型权重和显存 | 不需要 |

2026 年 11 月 2 日，该端点返回的模型列表包含上述带日期版本，但没有 `qwen3.5-9b`。因此此方案是 **Qwen3.5 Flash API 上的实验**，不能写成 Qwen3.5 9B Q4_K_M 的复现，也不能沿用原本 4070 Ti SUPER 的速度、显存或 GPU 小时结果。带日期模型 ID 用于固定请求目标，服务商未公开的权重 revision、量化方式和部署版本如实记为未知。若 API 返回的模型 ID 与配置不符，客户端会报错。

API 凭据通过程序加载，不放入命令行、配置快照或日志；已有进程环境变量优先于 `.env`。不修改现有 `.env`，也不要求手工 `source .env`。

为保留技能和反馈的长度约束，本地只下载固定 revision 的 Qwen3.5 9B 分词器文件。它是 **Flash API 的长度代理**，不能声称与服务商分词完全一致。B/C 使用相同代理及相同 token 上限；API 返回的 `usage.prompt_tokens`、`usage.completion_tokens` 才是请求用量的依据。

## 2 环境与连通性检查

所有命令在仓库根目录执行：

```bash
conda activate NLP_project
python -m pip install -r requirements-api.txt
python api/setup_tokenizer.py
python api/check.py --models
python api/check.py
python -m unittest discover -s Baseline/tests -v
```

新机器可先执行 `conda env create -f environment.yml`。当前安装的确切依赖另存于 `api/requirements-lock.txt`，可用 `python -m pip install -r api/requirements-lock.txt` 复现 Python 包版本。

`setup_tokenizer.py` 仅下载 `tokenizer.json` 与 `tokenizer_config.json`，不下载权重；重复执行会校验已有文件。普通 `check.py` 发出两次短请求，验证文字回答、JSON 模式、模型 ID 和 token 记账。需要额外验证图结构技能生成时执行：

```bash
python api/check.py --operators
```

这个命令共四次调用，后两次自动生成并校验初始技能和图变异。它使用合成提示，不是 SearchQA 或 HotpotQA 成绩。证据保存在 `api/runs/check-*/`；每次运行生成新目录，避免覆盖历史消耗。网络、权限或服务限流导致的失败会显式停止，不自动切换模型。

## 3 数据与实验条件

沿用现有课程规模：每个数据集最多 100 train、30 dev、50 test，最多三轮，每轮两个新候选；当前共同配置每轮 train batch=20，失败抽样 k=3，生成总尝试次数最多两次。

| 条件 | 内容 |
| --- | --- |
| N | 不使用技能文档 |
| I | 使用上游原始初始技能文档 |
| B | 原版失败反思输入的 GraphSkillEvo 包装流程 |
| C | 同一流程中使用成功与失败对照反馈 |

B/C 从同一共享初始化分叉。C 仅使用自己在 train 中产生的经验，优先找同题同上下文成功，再找同类任务成功；缺少可比成功时保留失败，全部无配对时回退到 B 的反馈。偏离成功答案并不自动说明某段推理错误：反馈要关联可见证据、指出有害行为，并明确保留有效步骤。

API 更换不改变评分、失败抽样、候选选择或测试纪律。测试时冻结技能，不把 test gold、测试输出或测试评分加入经验记忆。JSON 模式保证输出形式，生成结果仍经过现有 JSON 字段、图结构和长度校验。

克隆仓库不含实际数据，可恢复公开数据：

```bash
python Baseline/download_data.py --dataset searchqa
python Baseline/download_data.py --dataset hotpotqa
```

SearchQA 使用固定上游 ID 清单恢复 100/30/50。公开 HotpotQA 路径只准备 train/dev，test 标记为待补。完整 HotpotQA 实验需要课程规定的划分和带标签测试材料；不能临时拿 dev 当 test。接入最终课程划分应在正式初始化之前完成，后续数据 manifest 变化需要重跑受影响条件。

## 4 SearchQA 从 pilot 到正式评测

先在固定 train 内运行 pilot，检查长上下文、截断、答案格式、候选有效性及 API 时延：

```bash
python Baseline/run.py pilot --config Baseline/configs/searchqa_api.json --data Baseline/data/prepared/searchqa --output Baseline/runs/api/searchqa/pilot
```

pilot 通常包含 40 次 QA 和两次技能生成，失败重试可能增加调用。pilot 通过后固定配置；以下各输出目录必须为空或不存在。正式实验需要主动执行这些命令，环境安装和连通检查不会自动启动它们。

```bash
# 共享初始化
python Baseline/run.py initialize --config Baseline/configs/searchqa_api.json --data Baseline/data/prepared/searchqa --output Baseline/runs/api/searchqa/shared_init

# N 与 I 的 dev 基线
python Baseline/run.py static --condition N --config Baseline/configs/searchqa_api.json --data Baseline/data/prepared/searchqa --output Baseline/runs/api/searchqa/N
python Baseline/run.py static --condition I --config Baseline/configs/searchqa_api.json --data Baseline/data/prepared/searchqa --output Baseline/runs/api/searchqa/I

# B 优化与完整 dev 诊断
python Baseline/run.py train --config Baseline/configs/searchqa_api.json --data Baseline/data/prepared/searchqa --initialization Baseline/runs/api/searchqa/shared_init --output Baseline/runs/api/searchqa/B
```

检查 B 的 `dev_analysis/failures.json`、技能图和候选日志，记录失败类型、证据及改进依据，再冻结对照反馈设计并运行 C：

```bash
python Explore/run.py freeze-design --config Baseline/configs/searchqa_api.json --baseline-run Baseline/runs/api/searchqa/B --output Explore/runs/api/searchqa/feedback_design.json
python Explore/run.py train --config Baseline/configs/searchqa_api.json --data Baseline/data/prepared/searchqa --initialization Baseline/runs/api/searchqa/shared_init --design Explore/runs/api/searchqa/feedback_design.json --output Explore/runs/api/searchqa/C

# 四组全部完成后冻结，再做 held-out test
python Baseline/run.py freeze --N Baseline/runs/api/searchqa/N --I Baseline/runs/api/searchqa/I --B Baseline/runs/api/searchqa/B --C Explore/runs/api/searchqa/C --data Baseline/data/prepared/searchqa --output Baseline/runs/api/searchqa/frozen.json
python Baseline/run.py test --frozen Baseline/runs/api/searchqa/frozen.json --output Baseline/runs/api/searchqa/test
```

HotpotQA 使用 `Baseline/configs/hotpotqa_api.json`，将命令中的数据集路径统一替换为 `hotpotqa`；最终冻结要求课程 test 数据已准备好。若诊断后修改了实现、模型、配置、tokenizer 或数据，需要重新建立共同初始化及受影响条件，不能将本地 9B、API 或不同版本的产物混合比较。

## 5 API 成本与公平性

现有预算对每个运行目录分别生效：最多 600 次逻辑调用、700,000 个生成 tokens、9 小时运行时间。这些是上限，不是目标消耗，也不是整个项目共用的人民币额度。共享初始化被 B/C 各自计入逻辑成本，但共享记录的 `physical_calls=0`，汇总实际网络请求时避免重复计算。

每次请求先按最大生成量预留预算；失败且无可靠 usage 时按全部预留量保守计账。不启用 SDK 隐式重试。reasoning tokens 是 completion tokens 的子集，不重复相加。输入 tokens 会记录，但当前没有单独的输入 token 或人民币硬上限；金额以账户实际费率、输入长度档位、缓存计费与服务商账单为准。

费用估算可写为：`输入 tokens / 1,000,000 × 输入单价 + 输出 tokens / 1,000,000 × 输出单价`，再按缓存和实际计费规则调整。先用 pilot 的实测平均输入/输出量估算全量，不把请求的最大输出额度当成实际输出量。

API 模式不启动本地 GPU 账本。报告本机推理 GPU 需求为零、服务商 GPU 占用未知；不能把服务商 GPU 小时写成零，或用 API wall time 替代课程 40 GPU 小时。课程硬件条件与 API 替代条件的差异应在报告实验设置中披露。

## 6 结果和交付

每个数据集报告四组 EM/F1、正确题数、平均/P50/P95 时延、实际调用数、输入/生成 tokens、失败/重试数、完成轮数和停止原因。B/C 另比较修复、退化、同题配对、同类配对与无配对情况，附自动生成技能图、逐轮变化和失败清单。

保留配置、模型 ID、请求日期、代码与数据 hash、tokenizer hash、调用日志、冻结清单及服务商用量。seed 和 temperature=0 不能保证托管服务逐字确定性，现有 dev 缓存规则需在报告说明。延迟包含网络和云端排队，不能直接与原本本机 GPU 延迟比较。

最终完成正式 N/I/B/C、两数据集测试、课程报告和演示后，才能声称完成项目；连通性检查或合成流程通过只说明软件接入可运行。

接口行为参考：[阿里云兼容 Chat API](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)、[Qwen 思考模式](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)。
