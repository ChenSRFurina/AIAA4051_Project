# C2b：固定预算的 train/dev 验证

本版只修改优化器的训练反馈输入，代码在 `Explore/explore_v2/`。原 `Explore/explore/`、各数据集的归档协议、B/C 运行结果均保留。C2b 的方法哈希单独记录在 `policy.json` 和 `C2b/method.json`；`run_meta.json` 中的 `code_hash` 仍标识共享协议，不能单独代表 C2b 全部实现。

## 修改范围

- 诊断答案类型、答案 token 顺序、名称／粒度重叠、格式等可观察差异。缺乏证据时标记低置信度，不能从 EM=0 推断模型内部某一步失败。
- 从完整 train 可见上下文和响应中提取相关原文片段，保留文档标题、字符位置、原文 hash。问题词、引用短语、年份和 train 答案用于定位；这不是语义蕴含验证。最终 QA 输入不含 gold。
- 有界反馈保留所选失败、预测、训练参考、诊断和成功参照，按证据通道分配空间，不再统一保留原文前缀。无参照时仍提供失败诊断，不复制原 B 反馈文本，不虚构成功记录。
- 提醒优化器仅修改其算子允许的章节，并将修改写为可执行、条件明确、保留正确行为的通用规则。范围、gold 依赖、无变化等检查只做事后审计，不改变接受条件、重试或选择。

未修改：原 C 配对函数、Jaccard 阈值、题型分类、记忆容量／去重、随机失败 k=3、共享初始化、每轮批次、模型参数、QA 包装和评分、候选生成次数、重试上限、验证缓存、上游父代和生存选择。C2c 尚未实现。

每数据集预算仍为 600 次逻辑调用、700,000 生成 tokens、9 小时。共享初始化按原账本计费一次。3 轮，每轮两个父代各回答相同 20 道 train 题，产生两个候选，dev 固定 30 道。反馈与技能分别最多 2048 tokens。

## 本次运行

默认输出为 `Explore/runs/api/c2b-20261110/{searchqa,hotpotqa}/C2b`。使用既有 `.env` 凭据，不在命令、报告或 Git 中写密钥。私网 API 通过校园网 IPv4 绑定访问。

```powershell
conda activate qwen3.5
$env:PYTHONUTF8 = '1'
python -m unittest discover -s api/tests -p test_c2b.py -v
python api/check_c2b.py
python api/status_c2b.py

# 运行过的目录不能覆盖；重新运行时指定全新的目录。
python api/run_c2b.py searchqa --source-ip 10.4.144.134
python api/run_c2b.py hotpotqa --source-ip 10.4.144.134

# 两套完成后，离线审计和出报告（没有模型调用）。
python api/analyze_c2b.py all
# 图表依赖安装在独立目录，不改变训练环境中的包。
python -m pip install --target api/runtime/report-deps -r api/report-requirements.txt
python api/report_c2b.py
```

`--source-ip` 应为本机当前校园网 IPv4，不一定一直是示例地址。独立重跑可使用 `--output` 指定新目录；当前报告入口有意固定读取本次会话，防止无意替换正式比较来源。若要比较另一会话，应显式更新报告来源并使用新的报告文件名。

历史 C 的 train 离线重放仅验证新反馈构造，并在临时副本中改标条件名；这些轨迹绝不导入真实 C2b 记忆。真实运行的记忆仅接收自身 train 输出。每个数据集启动前保存方法源码副本和哈希，完成时再次检查。

## 比较与解释

主比较是各方法完整三轮后由原 dev 规则选择的最佳技能。次要分析将共同调用、生成 token 上限分别设为 B/C/C2b 最终逻辑用量的最小值，再各取同时满足两上限的最新完整轮次。剩余额度显式列出，不能说实现了精确相同实际费用。

train 表格汇总每轮父代批次观测，不是最终模型的全 train 评测。报告包含全部 dev 题、完整响应、配对修复与退化、候选差异、实际反馈、预算和技能图。原 C 的首轮摘录上限是 **160 token**，前一份方案将其写成字符，现予更正。

B/C 为历史对照；服务 revision 未公开。报告对相同第一轮输入的响应重复性做核查，但不能证明服务权重未变。此版仅一次 seed=42、每套 30 道参与选择的 dev，且同时改变了多个反馈环节，不能据此确认独立泛化收益或单个模块的因果贡献。本次不运行 test。后续新方法的正式验证需要冻结后未见的评测集。

## 已完成结果（2026-11-10）

两套 C2b 均完整完成三轮，方法源码在运行前后哈希一致。SearchQA 的 B / C / C2b dev 正确数为 23 / 22 / 22（各 30 题）；HotpotQA 均为 19/30。C2b 最终选择与 C 相同的共享初始化技能，没有取得最终 dev 提升。

本次新增 615 次物理调用、109,882 生成 tokens、1,509,898 输入 tokens。没有请求级失败。三个无效算子输出在原有重试次数内恢复；HotpotQA 另有四条候选 dev 回答达到 QA 输出上限且格式不合规，照原规则计分，未扩大预算或补跑。

完整报告位于 `Explore/runs/reports/C2b_results_20261110.html`，报告包为同目录的 `C2b_results_20261110_bundle.zip`。HTML 包含全部 60 道 dev 题的三方法响应、36 个候选的实际差异与定义图、C/C2b 实际反馈、所有最终错题复核、输出截断审计和共同预算截点比较。资产目录需与 HTML 一起保存；转交时使用 ZIP。

审计入口 `python api/analyze_c2b.py all` 会逐条重放 C2b 反馈，检查训练批次、选择、配对、来源位置、方法封存与账本。报告浏览器验证脚本为 `api/tests/check_c2b_report.cjs`，读取本机 Playwright 与 Edge（路径可通过环境变量覆盖）；验证记录及报告 SHA256 写入 `.validation.json`。这些步骤均为离线操作。
