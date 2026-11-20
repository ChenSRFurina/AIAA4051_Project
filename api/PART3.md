# Part 3：封存 C 的对照实验

本入口运行课程 Option A：修改技能优化器输入。模型权重、QA 提示、失败抽样、候选选择和预算沿用原 B。用户已确认先完成封存方法的正式对照，因此本次只增加运行、冻结、审计和报告工具，没有改动 `Explore/explore/` 的方法实现。

## 2026-11-10 完成结果

- 两套 C 均完成三轮，并通过输入、共享初始化、记忆与实际反馈审计。SearchQA dev 为 B 23/30、C 22/30；HotpotQA dev 为 B/C 均 19/30。
- SearchQA 固定 50 题 test 为 B 34/50、C 32/50（F1 77.80% / 71.69%）。C 相对 B 修复 1 题、退化 3 题；预声明的共同预算截点选中相同最终技能，因此 test 结果相同，复用观测不新增请求。
- 两套 C 均保留初始化最佳技能，未选出更好的演化技能。本次结果不支持 C 优于 B。
- 实际配对覆盖率为 SearchQA 2/18、HotpotQA 4/18 次所选失败曝光，分别仅涉及 1、2 道不同题目。三组独立配对均已复核；HotpotQA 两组主要是答案顺序或姓名粒度差异。
- 新增费用为 633 次物理请求、1,525,836 输入 tokens、108,838 生成 tokens。没有接口失败；SearchQA 有一次技能格式失败，固定重试成功且费用已计入。GPU 小时未知。
- 12 项离线检查通过。HTML 已检查筛选、26 个下载链接、20 个 SVG、手机布局及 JavaScript 错误；覆盖 220 条主比较/预算截点记录、37 条 C 错误和 2 条修复案例的复核。

最终报告：`Baseline/runs/reports/part3_20261110.html`。本地完整附件包为 `Baseline/runs/reports/part3_20261110_bundle.zip`。HotpotQA 课程 test 仍缺失，不能将公开 dev 当作最终测试。

## 实验边界

- SearchQA 使用 `20261107-125241-971323-38572` 的 B 和 `protocol-e6291efca3c7` 快照；HotpotQA 使用 `20261109-153042-871031-7652` 的 B 和 `protocol-067f9db47e18` 快照。
- C 从对应的共享初始化出发，不读取 B 分叉后产生的训练记忆。两个数据集各运行三轮，每轮两个候选。
- SearchQA 的 N/I/B test 已公开给研究者，但 C 方法和配置在此前封存。本次不得根据 test 修改方法；最终 C 技能和补充预算截点技能都在 C test 前冻结。
- HotpotQA 公开数据为 100 train / 30 dev / 0 test。课程 PDF 只有 `HotpotQA (provided)`，无数据路径或附件；因此目前只能完成开发集对照。
- 实际 API 权重 revision、精度和 GPU 使用量未知，报告不得据此声称跨日部署完全一致或推算 GPU 小时。

## 运行入口

在仓库根目录运行。以下路径对应本次真实实验，已有输出不能覆盖；复核结果无需重发模型请求。

```powershell
conda activate qwen3.5
python api/run_part3.py searchqa --source-ip 10.4.144.134
python api/finish_part3.py --source-ip 10.4.144.134
```

`finish_part3.py` 先等待 SearchQA C 完成，随后审计并冻结其技能，运行 HotpotQA C、审计，再执行 SearchQA test。各模型阶段串行执行。`--source-ip` 只在校园网 API 受本机代理/TUN 影响时使用；需用 `Get-NetIPConfiguration` 确认当前 WLAN 地址。此参数只影响进程连接方式，不改请求内容或系统路由。

状态记录在 `Explore/runs/api/part3-20261110/queue_status.json`。每个训练阶段成功必须有 `completed.json`；中断时保留输出和成本，写 `interrupted.json`。当前训练入口不支持直接覆盖或无校验续跑。

## 共同预算比较

主结果比较完整三轮、由 dev 选出的最佳技能，使用同一调用及生成 token 上限。补充结果按训练前声明的规则计算：

1. 共同调用上限取 B/C 最终逻辑调用数的较小值；共同生成上限取最终生成 token 数的较小值。
2. 每个方法取同时满足两项上限的最后一个完整轮次，初始化为轮 0，不纳入部分完成的轮次。
3. 比较该截点的 dev 最佳技能，报告剩余额度。该方法匹配共同上限，不意味着实际 token 消耗完全相同。
4. SearchQA 的截点技能在 C test 前冻结；内容相同的技能复用已有观测，保留来源，不记作新增请求或独立重复实验。

共享初始化在 B/C 各自的逻辑预算中计入一次，新增物理费用只计算实际请求。失败的算子尝试也计入费用。

## 审计与报告

```powershell
python api/report_part3.py --output Baseline/runs/reports/part3_20261110.html
python -m unittest discover -s api/tests -v
```

报告入口要求两套 C 审计及 SearchQA test 已完成。审计核对配置和代码签名、共同初始化、相同训练 batch、原失败抽样、记忆来源、上下文 hash、反馈长度及实际 mutation 提示中使用的反馈。

配对覆盖率以实际 mutation 父代所选失败的曝光次数为分母，同时列出唯一父代反馈数和实际尝试次数。反馈全文、压缩后数据包、每轮技能 SVG、候选差异、逐题 B/C 结果和预算均保存在 HTML 及同名 assets 目录。

可选人工复核文件为 `Explore/runs/api/part3-20261110/manual_review.json`：`pairs` 保存最多十条按固定顺序选择的训练配对备注；`cases` 保存 dataset、split、id、C 预测、`C_response_sha256`、category 和 note。报告会校验预测与原响应 SHA256；备注只说明证据，不改变评分或模型反馈。

`api/tests/test_part3.py` 验证共同预算的双约束、完整轮次、初始化费用与剩余额度。这些合成测试不是方法有效性的实验结果。
