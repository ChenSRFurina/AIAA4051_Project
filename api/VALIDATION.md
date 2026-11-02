# Qwen API 环境验证

验证日期：2026 年 11 月 2 日。环境为 macOS ARM64、conda `NLP_project`、Python 3.11.18；Python 包版本见 [requirements-lock.txt](requirements-lock.txt)。`pip check` 通过，未安装 PyTorch，未下载模型权重。

## 最新完整 Baseline

- 最终代码通过 38 项离线测试，包括已知服务端拒答、未知 HTTP 错误、失败后恢复、保密和初始化格式重试。
- SearchQA 与 HotpotQA 均完成 pilot、共享初始化、N/I 和三轮 B；最终六组 dev 每组 30 题，无未恢复的基础设施故障。
- [最终结果与成本](runs/baseline-20261102/RESULTS.md)；[含路径和协议指纹的 JSON](runs/baseline-20261102/summary.json)。
- 修复了请求失败后的等待和终止规则；明确的 data_inspection_failed 内容拒答记零分并保留分母，不改写题目或绕过审核。
- 修复了初始化 JSON 示例与单技能要求的冲突，明确工作流格式；第二次初始化尝试接收第一次校验错误，仍保持最多两次尝试。
- 真实生成失败、拒答、重试和旧运行全部保留。完整运行结果只使用最终统一版本，实际总账单还包含此前调试及中断消耗。
- 当前 Mac 使用 API 推理，未下载或启动本地模型权重。C、最终 test 和本机 GGUF 服务不在本次执行范围。

## 首次接入验证（历史记录）

- `python -m unittest discover -s Baseline/tests -v`：30 项全部通过，包括两个数据集的合成完整流程，以及 API 鉴权、保密、用量、格式和模型一致性回归检查。
- 两份 API 配置均能加载固定的本地分词器。分词器只作技能/反馈长度代理。
- 用户指定端点的模型列表查询成功，包含 `qwen3.5-flash-2026-02-23`。
- `python api/check.py --operators`：普通回答、JSON object、完整技能生成和图变异全部通过；生成结果通过现有图结构及长度校验。

成功记录：`api/runs/check-20261102T081618068654Z/summary.json`。四次请求，3,257 个输入 tokens、1,820 个生成 tokens，reasoning tokens 为 0。代码指纹为 `17ce46523ebda0c2a59349ae2b1d596e3c7a8fbc54ec3eb9c6ce211f7b2acc2f`。

第一次检查在技能初始化阶段因返回两个数组元素而失败，记录保留于 `api/runs/check-20261102T081349264413Z/`。API 模式的初始化格式说明已改为明确限定一个 Markdown 字符串，避免旧 JSON 闭合示例被输出成额外元素。程序拒绝多余元素，不删减或手工修补模型生成的技能；本地模式原提示保持不变。

两次检查合计七次模型调用，4,830 个输入 tokens、2,619 个生成 tokens。失败检查产生的用量包含在总数中。人民币金额以账户账单为准。原始调用和生成技能保存在上述 Git 忽略目录，不随代码提交。

## 首次接入检查的边界

本节七次接入检查当时使用合成提示。正式 SearchQA/HotpotQA Baseline 的结果见上方最新运行记录；HotpotQA 最终测试仍需课程材料。完整操作顺序见 [API 实施方案](../Project5_API_Plan_zh.md)。

密钥仅由运行程序加载，未打印、写入配置快照或修改 `.env`；测试使用假密钥，不读取实际凭据。
