# 本机 Qwen3.5-9B 4-bit 服务

模型：`unsloth/Qwen3.5-9B-GGUF` 的 `Qwen3.5-9B-Q4_K_M.gguf`。
权重目录按用户指定为 `F:\Document\HKUST\2026fall\NLP\Qwen3.5_9B_4bit`，tokenizer 位于该目录的 `tokenizer/`。
精确 revision、文件大小和 SHA256 见权重目录中的 `deployment.json`。

Python 环境：`conda activate qwen3.5`，实际位于 `E:\Anaconda\envs\qwen3.5`。
GPU 服务使用项目内原生 CUDA 12.4 llama.cpp b11871；Python 环境负责 tokenizer、实验与部署脚本，无需额外安装 GPU 版 PyTorch。

## 启动和停止

在 `AIAA4051_Project` 项目根目录的 PowerShell 运行：

```powershell
conda activate qwen3.5
.\local_model\start.ps1
.\local_model\status.ps1
```

健康接口：`http://127.0.0.1:8000/health`；问答接口：`http://127.0.0.1:8000/v1/chat/completions`；模型 alias 为 `Qwen3.5-9B-Q4_K_M`。只监听回环地址，不向局域网开放。

固定设置：16384-token 上下文、单并发、全部可用层上 GPU、Flash Attention、关闭思考模式、禁止自动挪动上下文。模型输入过长时返回错误，不静默丢弃前文。SearchQA 与 HotpotQA 配置使用相同 checkpoint、量化和基本生成设置。

GPU 服务从启动直到停止的常驻时间计入 `Baseline/runs/gpu_accounting.jsonl`，每次真实模型调用前检查全项目 40 GPU 小时上限。停止分配时必须执行：

```powershell
.\local_model\stop.ps1
```

`stop.ps1` 会核对 PID、程序路径和模型路径，只停止此脚本管理的服务，然后关闭对应 GPU 账本会话。停止服务后启动脚本可再次加载模型。

## pilot

使用固定 SearchQA train 中预留的 10 个 ID，分别执行 no-skill、initial-skill、自动生成初始技能及一个自动变异候选，共约 40 次 QA，另加技能生成调用。只使用 train；不根据 dev/test 标签修改模型。

```powershell
$pilotOut = "Baseline/runs/searchqa/pilot-$(Get-Date -Format yyyyMMdd-HHmmss)"
python Baseline/run.py pilot --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --output $pilotOut
```

首次部署的测量记录由 `latest_pilot.json` 指向；结果包括 `pilot_summary.json`、逐题原始 prompt/响应、调用与 token 账本、`hardware_profile.json` 及 `measurements/*-gpu.csv`。这些是 train pilot 成绩，不是正式 test 成绩。

真实模型启动/请求检查保存在 `checks/service_validation.json`。模型产生的技能不会被人工补写；非法输出和失败尝试保留在日志中，重试服从共同上限。

## 重现下载和环境

```powershell
python local_model/download_model.py
python -m pip install -r local_model/requirements-lock.txt
```

分段下载先核对每段范围和长度，再对完整 GGUF 校验发布的 SHA256。断点片段是临时文件，在完整校验后可以清理；不影响已验证的最终权重。

## 来源

- [原始 Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
- [固定 Unsloth 量化版本](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5)
- [llama.cpp b11871 官方发布](https://github.com/ggml-org/llama.cpp/releases/tag/b11871)


## 本机后端的结构化输出设置

本机配置启用 `model.structured_operators=true`：只在技能初始化/变异请求中传入 JSON schema，约束 `skills` 或 `new_skill/notes` 的序列化格式。QA 仍输出 `<answer>` 标签，不添加 JSON 约束。此设置对 B/C 完全相同，原始图结构校验及重试上限不变。首轮未启用约束的 pilot 因模型生成无效 JSON 终止，失败响应、调用和 GPU 消耗全部保留。

本机 pilot 进一步统一冻结技能序列化/长度提示：最多六个简洁节点、三个不重复任务图，完整技能仍为 2048-token 上限；技能采样 temperature=0.6。原生成上限 3072、QA 上限 1024、两次尝试及图校验保持不变。该共同约束用于避免小模型在 JSON 输出时无限展开同类任务图，B/C 完全相同。
