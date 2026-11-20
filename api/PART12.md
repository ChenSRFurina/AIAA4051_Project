# Part 1/2：固定基线评测与报告

`part12.py` 独立冻结和评测 N/I/B，不运行技能优化器。原来的四组 `Baseline/run.py freeze` 接口保留不变。本入口用于先完成课程 Part 1/2，并明确不能把 N/I/B 的结果称为 Part 3 对照结果。

## 数据与冻结

- 必须已有同版本、完整完成的 N/I/B，以及训练时使用的完整数据 manifest。拒绝未完成运行、缺少 test、数据变化、技能变化、配置变化与代码快照变化。
- `--snapshot` 指向训练时的源码副本。默认查找 `api/runtime/protocol-<训练 code_hash 前 12 位>/`。它必须通过原 `tree_fingerprint` 校验。旧 SearchQA 与新 HotpotQA 使用各自快照，不修改历史 hash。
- 冻结记录包含 N/I/B 的运行证据、固定 test ID、数据和 tokenizer 签名、精确配置，以及本入口脚本 SHA256。
- Explore 的方法源代码与相关配置在看 test 前一并封存，但 C 不运行。后续同一 test 上的 C 对照只有在方法保持预先封存规格、test 不参与修改时才可作为正式比较；根据 test 调整的方法需要新的未见评估，或标为探索性分析。
- 每个 test 条件保存全量逐题响应、实际可见上下文、EM/F1、延迟、格式状态、调用和 token 账本。测试不写入训练记忆，也不重新选择技能。

## PowerShell 运行方式

在仓库根目录运行，输出路径必须是新的目录。下面是本次已经使用的路径；复核已有结果直接读文件，不要覆盖或重复评测。

```powershell
conda activate qwen3.5
python api/part12.py freeze `
  --run Baseline/runs/api/searchqa/20261107-125241-971323-38572 `
  --data Baseline/data/prepared/searchqa `
  --output Baseline/runs/part12/searchqa/frozen.json

python api/part12.py test `
  --frozen Baseline/runs/part12/searchqa/frozen.json `
  --output Baseline/runs/part12/searchqa/test `
  --source-ip 10.4.144.134
```

`--source-ip` 仅在本机代理/TUN 影响校园网 API 时使用；先用 `Get-NetIPConfiguration` 核对 WLAN 地址。它只改变该进程的连接方式，保持原 endpoint、请求体、评分和解码。既不修改系统路由，也不修改归档的模型代码。无需绑定时省略此参数。

运行中查看 `test/progress.json`；成功必须有 `test/results.json` 的 `status=completed`。异常写入 `incomplete.json`，保留已发生请求成本，不生成虚假的完整成绩。当前不支持无校验续跑。

## HotpotQA 的外部依赖

课程 PDF 第 20 页仅写 `HotpotQA (provided)`；第 19–23 页无数据 URL，PDF 无嵌入附件。Project 5 联系邮箱为 `jchen349@connect.hkust-gz.edu.cn`。当前公开 distractor 数据为 100 train / 30 dev / 0 test；不能临时把 dev 当成课程 test。

从课程平台或老师取得完整划分后，使用既有 `Baseline/run.py prepare` 在新目录准备课程数据，确认任务设定，并按新 manifest 重跑受影响的共同初始化和 N/I/B。之后可用同一 `part12.py` 入口评测。缺失数据的实际预检记录保存在 `Baseline/runs/part12/hotpotqa/preflight.json`。

## 生成报告

```powershell
python api/report_part12.py `
  --test-dir Baseline/runs/part12/searchqa/test `
  --test-review Baseline/runs/reports/part12_searchqa_test_review.json `
  --output Baseline/runs/reports/part12_20261110.html
```

`--test-review` 为人工复核结果，可省略；未复核错误明确标为未作语义归因。复核文件必须为 `review_status=verified`，每条记录带 condition、ID、预测及原始响应 SHA256。语义备注不改分。报告生成器重算并核对全部已报告响应的 EM/F1，与原始摘要一致才输出。

HTML 内嵌图与全部逐题文本，可离线浏览、搜索和筛选。配套 `_assets/` 提供 16 份独立技能 SVG、两套谱系图与进化曲线、完整错误 CSV/JSON、dev 指令诊断和逐轮真实差异。共享初始化缓存的观测保留 `observation_condition` 来源，报告按最终条件 B 展示，不把复用记录当作新请求。

`api/tests/test_part12.py` 用合成数据验证独立评测、输入篡改拒绝、缺失 test 拒绝、覆盖拒绝和无记忆更新。这些测试不是 benchmark 成绩。

```powershell
python -m unittest discover -s api/tests -v
```

本次增加的入口、报告与测试都在 `api/`，没有修改冻结的训练/评估核心、提示词或候选选择规则。
