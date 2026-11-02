# AIAA4051 Project 5

本项目复现课程规模的 GraphSkillEvo，并研究成功与失败对照反馈能否改善图结构技能。模型负责回答问题和生成技能文档，代码负责评分、选优和经验记忆，不训练模型权重。

| 条件 | 含义 | 入口 |
| --- | --- | --- |
| N | 无技能文档 | Baseline |
| I | 上游原始初始技能 | Baseline |
| B | 原版失败反馈优化器，三轮进化 | Baseline |
| C | 成功与失败对照反馈优化器 | Explore |

**当前这台 Mac 使用 Qwen API 路线**，环境名是 `NLP_project`。完整 Baseline 指 pilot、共享初始化和 N/I/B 的训练及 dev 评估；最终 test 要等 C 完成后四组统一冻结。

## 1 两种模型运行方式

| 项目 | Qwen API | 本地模型服务 |
| --- | --- | --- |
| 模型 | `qwen3.5-flash-2026-02-23` | Qwen3.5 9B Q4_K_M |
| 推理位置 | 阿里云 | 自己的 GPU 机器 |
| 本机需要模型权重 | 不需要，仅下载分词器 | 需要 GGUF 权重、分词器和 llama.cpp |
| 密钥 | 根目录 `.env` 中的 `QWEN_API_KEY` | 默认本地回环服务无密钥 |
| 配置 | `searchqa_api.json`、`hotpotqa_api.json` | 按本机路径生成 `*_local.json` |
| GPU 账本 | 不启用；服务商 GPU 占用未知 | 实际服务分配期间启用 |
| 说明 | 下方第 2–4 节 | [本地模型配置与运行](local_model/README.md) |

两条路线使用不同模型，不能合并实验结果或沿用另一条路线的速度、显存数据。原来的 `searchqa.json`、`hotpotqa.json` 记录了旧 Windows 环境，包含 `F:` 盘路径，不能直接用于当前 Mac。

## 2 配置 conda 环境

先进入仓库根目录。本机路径：

```bash
cd /Users/sirui/code/2026fall/NLP/AIAA4051_Project/AIAA4051_Project
```

本机环境已经装好，直接激活：

```bash
conda activate NLP_project
```

在新机器上首次安装：

```bash
conda env create -f environment.yml
conda activate NLP_project
```

如果已经有同名环境，补齐依赖即可：

```bash
conda activate NLP_project
python -m pip install -r requirements-api.txt
```

Python 版本为 3.11。API 客户端不需要 PyTorch；启动时提示 `PyTorch was not found` 不代表出错，分词器可以正常使用。已验证的依赖锁在 [api/requirements-lock.txt](api/requirements-lock.txt)。

## 3 配置并检查 Qwen API

本机根目录 `.env` 已有 `QWEN_API_KEY`，无需修改或 `source`。新机器需要自行在根目录 `.env` 设置这个变量；不要把真实密钥写进 JSON、脚本或 Git。已有进程环境变量优先于 `.env`，程序不打印密钥。

两份 `Baseline/configs/*_api.json` 已配置：

```text
backend = api
base_url = https://llm-ps7f58n2fo6d73fm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
model.id = qwen3.5-flash-2026-02-23
model.api_key_env = QWEN_API_KEY
enable_thinking = false
```

新工作空间应同时修改两份配置中的地址和模型 ID，并确认密钥有权限。正式实验开始后保持配置不变。

```bash
# 仅下载固定版本的两个分词器文件，不下载模型权重
python api/setup_tokenizer.py

# 只查询可见模型，不生成内容
python api/check.py --models

# 两次短请求，验证文字、JSON、模型 ID 和 usage
python api/check.py

# 可选：共四次请求，另验证初始技能和图变异
python api/check.py --operators
```

分词器在 API 模式下只作为技能/反馈长度代理；调用 token 数以服务端返回的 usage 为准。每次检查生成独立的 `api/runs/check-*/` 记录。

## 4 一条命令运行完整 Baseline

激活环境后，在仓库根目录执行：

```bash
# 顺序运行 SearchQA 和 HotpotQA
bash api/run_baseline.sh

# 或只运行一个数据集；用其中一条替换上面的命令
# bash api/run_baseline.sh searchqa
# bash api/run_baseline.sh hotpotqa
```

这三种写法任选一种。脚本为每个数据集依次完成：

1. 下载并准备固定数据，已有数据先校验再跳过。
2. pilot：固定 10 个 train ID，检查真实问答和技能生成。
3. shared_init：建立后续 B/C 共用的初始种群及 dev 评价。
4. N：在完整 dev 上评估无技能条件。
5. I：在完整 dev 上评估原始初始技能条件。
6. B：运行三轮优化，每轮两个新候选，按完整 dev 选优。

每轮两个父代使用相同的 20 题 train batch，失败抽样 k=3。QA 最多生成 1024 tokens，技能生成最多 3072 tokens，单个技能/反馈长度上限各 2048，最多两次尝试。

**查看进度可以另开终端，不会产生 API 调用：**

```bash
conda activate NLP_project
cd /Users/sirui/code/2026fall/NLP/AIAA4051_Project/AIAA4051_Project
python api/status.py
```

`incomplete_or_running` 表示这个阶段尚无完成文件；它不是进程存活检测。可用 `--run` 指定历史运行目录，或加 `--json` 获得结构化进度。控制台日志通常在阶段结束后集中输出，模型逐次请求会持续写入 `calls.jsonl` 和 `ledger.jsonl`。

## 5 输出位置与完成判断

每次启动使用新目录，不覆盖已有输出：

```text
Baseline/runs/api/<dataset>/<时间戳>-<进程号>/
├── pilot/                  pilot_summary.json 和逐题记录
├── shared_init/            B/C 共用初始化
├── N/                      无技能 dev 结果
├── I/                      原始技能 dev 结果
├── B/
│   ├── summary.json        状态、完成轮数、EM/F1、成本
│   ├── best_skill.md       最终选中的技能
│   ├── best_skill.json     技能结构
│   ├── best_skill.mmd      技能图
│   ├── history.json        各轮候选与幸存者
│   ├── evolution.mmd       进化关系图
│   ├── dev_analysis/       最终 dev 错误清单
│   ├── operators/          技能变异与失败尝试
│   ├── calls.jsonl         完整请求、响应和 usage，不含鉴权头
│   └── ledger.jsonl        调用和 token 账本
└── *.log                   各阶段控制台日志
```

正常完成时，N/I 的 `summary.json` 状态为 `static_complete`，B 为 `completed` 且 `completed_rounds=3`。候选可能因格式/图校验失败而被拒绝，拒绝情况在 `operators/`；三轮完成并不表示所有候选都有效。`budget_stopped:...` 表示达到上限，不是完整三轮。

每个运行目录预算为最多 600 次逻辑调用、700,000 生成 tokens、9 小时；它不是全项目统一金额上限。B 的逻辑成本包含共享初始化，实际计费统计应只累计 `physical_calls>0` 的账本记录，避免重复计算共享记录。失败请求没有 usage 时按预留上限保守计账，账本估算与云端账单可能有差异。

API 出错后，下一次请求至少等待配置中的 `api_retry_delay_seconds`（当前为 2 秒），不增加隐藏重试。某道 QA 的网络故障或未知 HTTP 错误在两次尝试后仍未恢复，程序保留诊断并停止。服务商明确返回 `data_inspection_failed` 时，按同一两次尝试上限执行；最终仍拒答则记为 `provider_refusal`、EM/F1 为 0，并继续下一题。拒答仍在分母中，单独计入 `provider_refusals`，不改写输入或绕过审核。非空目录不能直接续写；保留失败目录，用新输出目录重跑失败阶段。复用 `shared_init` 必须保持代码、配置、数据及分词器签名一致。

### 本机完整结果（2026-11-02）

[完整结果、成本和版本核对](api/runs/baseline-20261102/RESULTS.md) · [机器可读汇总](api/runs/baseline-20261102/summary.json)

| 数据集 | N EM | I EM | B EM | 完整运行目录 |
| --- | --- | --- | --- | --- |
| searchqa | 76.7% | 80.0% | 86.7% | [Baseline/runs/api/searchqa/20261102_191626-83154](Baseline/runs/api/searchqa/20261102_191626-83154/) |
| hotpotqa | 63.3% | 60.0% | 60.0% | [Baseline/runs/api/hotpotqa/20261102_191628-83182](Baseline/runs/api/hotpotqa/20261102_191628-83182/) |

两组均完成 pilot、共享初始化、N/I 和三轮 B。以上为各 30 题 dev 结果；C 和最终 test 尚未运行。本次两数据集并行、各自内部串行，延迟只代表这次运行的实际观察。

这些输出和历史诊断目录均被 Git 忽略，只保存在当前机器。正式汇总使用上表中的统一代码版本，不混入旧版本和中断运行。

## 6 数据与最终测试边界

- SearchQA：按固定上游 ID 清单准备 100 train / 30 dev / 50 test。
- HotpotQA：公开 distractor 数据准备 100 train / 30 dev，`test_pending=true`。课程测试材料和指定划分仍需接入，不能把 dev 当 test。
- 数据不随 Git 克隆。下载命令为 `python Baseline/download_data.py --dataset searchqa` 或 `hotpotqa`。
- `train` 优化的是技能文档，不是模型参数。`static` 和 `train` 的汇总均为 dev 成绩。
- C 从同一 `shared_init` 开始；N/I/B/C 全部定稿后才能 `freeze`、`test`。命令见 [Explore 说明](Explore/README.md)。

## 7 离线验证与目录说明

```bash
python -m unittest discover -s Baseline/tests -v
python Baseline/run.py demo --output Baseline/runs/demo-check
```

`demo` 使用合成数据和模拟模型，输出目录必须为空，不能作为真实实验成绩。

| 目录或文档 | 用途 |
| --- | --- |
| [Baseline](Baseline/README.md) | 公共实验流程与分阶段命令 |
| [Explore](Explore/README.md) | 对照反馈方法和后续 C/test 命令 |
| [api](api/README.md) | API 配置、检查、启动和进度查询 |
| [local_model](local_model/README.md) | 本地 GGUF 下载、服务和 GPU 账本 |
| [API 实施方案](Project5_API_Plan_zh.md) | 研究设计与 API 条件说明 |
| [原本地方案](Project5_Implementation_Plan_zh.md) | 原 Windows 本地部署与研究计划 |

模型、分词器缓存、真实数据、运行日志和 `.env` 均被 Git 忽略。上游代码固定于 `Baseline/UPSTREAM.json`，`vendor` 中的轻量划分 ID 是恢复数据所需配方。
