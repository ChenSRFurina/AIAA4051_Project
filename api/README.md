# Qwen API 配置与运行

本机使用 conda `NLP_project` 和 `qwen3.5-flash-2026-02-23`，不部署模型、不需要本机推理显卡。环境安装、两种后端的区别及输出说明见 [主 README](../README.md)，研究设计见 [API 实施方案](../Project5_API_Plan_zh.md)。

## 配置

- 地址和模型：`Baseline/configs/searchqa_api.json`、`hotpotqa_api.json`。
- 密钥：根目录 `.env` 的 `QWEN_API_KEY`，自动加载，不打印；进程环境优先，不需 `source`。
- 分词器：`python api/setup_tokenizer.py`，下载约 13 MB 的分词器文件，不下载权重。
- 思考模式关闭，技能请求使用 JSON object，QA 输出答案标签；图和技能长度继续由本地代码校验。
- 每份 API 配置固定相同模型、生成参数和预算。分词器是长度代理，服务端 usage 是请求 token 依据。

## 运行

在仓库根目录选择需要的命令：

```bash
conda activate NLP_project

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

[2026-11-02 结果汇总](runs/baseline-20261102/RESULTS.md)包含两数据集 N/I/B 的 dev 指标、最终输出目录、成本及版本核对；[验证记录](VALIDATION.md)说明已修复问题和测试范围。
