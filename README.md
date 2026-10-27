# AIAA4051 Project 5

课程项目：GraphSkillEvo 的课程规模复现，以及成功–失败对照反馈驱动的技能优化。

- `Baseline/`：N（无技能）、I（初始技能）、B（原版优化器），及数据适配、预算、评测、技能图和冻结流程。
- `Explore/`：C（成功–失败对照反馈），复用相同抽样、选择与资源上限。
- `local_model/`：本机 Qwen3.5-9B Q4_K_M 的下载、服务管理和 pilot 脚本。

## 仓库范围

本次提交包含源代码、prompts、测试、配置、依赖清单与运行说明。模型权重、CUDA 二进制、完整/准备后的数据、实验输出、调用日志、显存测量、备份和 Python 环境均不上传。

`Baseline/vendor/GraphSkillEvo/data/searchqa_id_split` 仅包含固定上游划分的轻量 ID 元信息，是恢复 SearchQA 数据所需的配方；不包含题目、上下文或答案。上游提交记录在 `Baseline/UPSTREAM.json`，源文件保留上游原貌。

## 安装与测试

在项目根目录运行：

```powershell
python -m pip install -r Baseline/requirements.txt
python -m unittest discover -s Baseline/tests -v
python Baseline/run.py demo --output Baseline/runs/demo
```

`demo` 使用合成数据和模拟响应验证流程，不是 benchmark 成绩。输出目录必须为空。

## 数据与真实模型

```powershell
python Baseline/download_data.py
```

下载脚本固定数据源版本和划分 ID。公开 HotpotQA 没有可供本实验评分的测试标签；正式测试仍需接入课程提供的划分，不能将 validation 改名成 test。

当前模型配置和 `local_model` 脚本保留了原作者 Windows 本机部署路径。其他机器需按实际目录调整模型/tokenizer、Conda 环境、项目路径及服务地址，再运行 pilot。仓库不附带模型或模型服务二进制。

详细步骤见 [Baseline 说明](Baseline/README.md)、[Explore 说明](Explore/README.md) 和 [本机模型服务说明](local_model/README.md)。模型与数据来源、版本及许可参见各自原始项目；课程说明和实施计划属于仓库原有材料。
