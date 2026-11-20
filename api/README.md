# Qwen API 配置与运行

Windows 本机使用 conda `qwen3.5` 和队友部署的 `qwen3.5-9b`。此前使用阿里云的实验记录见 [API 实施方案](../Project5_API_Plan_zh.md)。

## 配置

- SearchQA 和 HotpotQA 的 API 配置使用 `http://10.4.162.57:19198/v1`，记录的服务端上下文窗口为 32768 tokens。
- 程序自动读取根目录 `.env` 中的 `PROJECT5_TEAM_QWEN_API_KEY`。同名进程环境变量优先。
- 分词器：`python api/setup_tokenizer.py`，下载约 13 MB 的分词器文件，不下载权重。
- 思考模式关闭。技能请求通过 JSON Schema 约束字段类型、初始化数组长度和变异字段。QA 输出答案标签，技能图和长度由本地代码校验。
- 每份 API 配置固定相同模型、生成参数和预算。分词器是长度代理，服务端 usage 是请求 token 依据。
- 当前 API 实验保留最多 4096 tokens 的文档上下文，优先保留完整文档。QA 最大输出为 2048 tokens。客户端按系统提示、问题、技能、上下文和结构化输出 schema 估算输入，再预留 512 tokens 覆盖聊天模板和分词差异。输入估算、输出上限和余量之和超过 32768 时停止发送。
- N/I/B/C 使用同一上下文截断规则。旧配置未设置 `qa_context_max_tokens` 时沿用原 6000 字符规则。

## 运行

Windows PowerShell 挂跑两个数据集可以直接执行以下命令。

```powershell
conda activate qwen3.5
python -u api/run_baseline.py
```

只跑一个数据集时使用 `python -u api/run_baseline.py hotpotqa` 或 `searchqa`。入口会建立新的时间戳目录，复制本次配置，并依次执行数据准备、pilot、共享初始化、N、I、B。每个阶段保存单独的日志，Python 进程失败或 B 提前结束都会停止。运行过程中可以在另一终端执行 `python api/status.py`。

token 预算变更后的实验需要从新的 pilot 和初始化开始。挂跑完成后，N/I/B 属于同一套新协议，后续 C 使用本次 shared_init。

如果校园网 API 被本机代理/TUN 路由影响，而绑定 WLAN 地址能正常连接，可只为当前实验进程设置本机 WLAN IPv4 地址（地址变化后需更新）：

```powershell
# 先用 Get-NetIPConfiguration 查看 WLAN 的当前 IPv4Address
$env:PROJECT5_API_SOURCE_IP = '10.4.144.134'
python api/check.py --models
python -u api/run_baseline.py hotpotqa
```

该选项仅用于私网 HTTP 模型接口，单次请求绑定指定网卡地址并绕过系统代理，不修改系统代理设置。运行目录的 `execution_environment.json` 会记录连接方式。不需要绑定时执行 `Remove-Item Env:PROJECT5_API_SOURCE_IP -ErrorAction SilentlyContinue`。

JSON 文件使用同目录的唯一临时文件原子替换；Windows 临时文件占用会进行最多 8 次替换尝试，持续失败时保留原文件与临时文件。对于三轮已全部完成、只在最终 `metrics.json` 替换时发生 WinError 5 的旧 B 运行，可使用 `python api/recover_final_save.py --run <运行目录>`。工具要求原始代码快照、全部观测、选择结果和成本账本校验通过，保留原摘要及恢复审计，不重新调用模型、不修改原始推理版本标识；其他中断不支持此恢复入口。

在仓库根目录选择需要的命令：

```bash
conda activate qwen3.5

# 查询模型，无生成调用
python api/check.py --models

# 文字与 JSON 检查，2 次短请求
python api/check.py

# 另检查技能初始化与图变异，共 4 次请求
python api/check.py --operators

# 完整 Baseline：下载数据、pilot、共享初始化、N、I、B
bash api/run_baseline.sh

# 只跑指定数据集时，用下面命令代替上一条
# bash api/run_baseline.sh searchqa
# bash api/run_baseline.sh hotpotqa
```

控制台会输出各阶段和结果目录。运行期间可在另一终端执行 `python api/status.py`；它只读取账本，不读取凭据、不调用模型。`--run <运行目录>` 查看历史运行，`--json` 输出结构化信息。`incomplete_or_running` 只表示没有完成文件，不能判断进程是否仍在运行。

## 输出与错误处理

- 短检查输出：`api/runs/check-*/`。
- 正式 Baseline 输出：`Baseline/runs/api/<dataset>/<时间戳>-<PID>/`。
- 每个运行目录独立计账，保留失败请求；已有非空目录不能覆盖或无校验续写。
- API 请求失败后，下一次请求至少等待 2 秒，仍遵守最多两次总尝试。网络或未知 HTTP 故障耗尽尝试后写出诊断并停止。明确的 `data_inspection_failed` 内容拒答在两次尝试后仍被拒绝时计零分、保留在分母中，并记录 `provider_refusals`；不改写输入或绕过审核。
- 401/403 检查工作空间密钥权限，404 检查模型 ID，429 检查账户限额；不会自动切换模型或伪造响应。
- `PyTorch was not found` 是 API/分词器环境的正常提示。

检查阶段的历史记录见 [VALIDATION.md](VALIDATION.md)。依赖锁在 [requirements-lock.txt](requirements-lock.txt)。`runtime/` 和 `runs/` 已加入 Git 忽略。

## 本次完整运行结果

2026-11-10 已补充 [Part 1/2 独立冻结测试与报告入口](PART12.md)。SearchQA 固定 50 题 N/I/B test 已完成；详细离线报告位于 `Baseline/runs/reports/part12_20261110.html`，包含两套数据的逐轮 SVG、全部已运行 dev/test 错误清单、人工复核和技能指令证据。HotpotQA 课程 test 缺失，预检明确停止，不以公开 dev 代替。新入口不运行 Explore C。

同日已完成 [Part 3 封存 C 对照](PART3.md)：两数据集 C 三轮运行、实际反馈审计、共同预算截点和 SearchQA C test。SearchQA test B/C 为 34/50、32/50；HotpotQA dev B/C 均 19/30。两套 C 均保留初始化最佳技能，当前结果不支持 C 优于 B。完整 HTML 位于 `Baseline/runs/reports/part3_20261110.html`，算法保持 test 前封存版本。

[2026-11-02 结果汇总](runs/baseline-20261102/RESULTS.md)包含两数据集 N/I/B 的 dev 指标、最终输出目录、成本及版本核对；[验证记录](VALIDATION.md)说明已修复问题和测试范围。
