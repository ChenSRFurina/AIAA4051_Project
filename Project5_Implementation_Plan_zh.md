# Project 5 实施方案

**课程题目：Evolving Graph-Structured Skills for LLM Agents**

**方案版本：v2，2026 年 10 月 27 日；本机真实 pilot 验证日期：2026 年 10 月 25 日。**

**研究题目：利用成功与失败对照改进图结构技能优化**

**英文工作标题：Success–Failure Contrastive Feedback for Graph-Structured Skill Optimization**

代码仓库：[AIAA4051_Project](https://github.com/ChenSRFurina/AIAA4051_Project)。本版根据已实现的 `Baseline/`、`Explore/`、本机模型服务和真实 pilot 记录更新，取代 v1 中“目录只有 PDF”“模型尚未确认”“程序待实现”的描述。源代码实现基点为提交 `6abf5d5fc0d115bdb5eb829339f5f5df3502fee3`；本文档后续修订以 Git 历史为准。

## 1 项目目标、研究边界与当前状态

研究假设保持不变：在相同模型调用与生成 token 上限下，为图结构技能优化器提供可比成功参照、失败偏差及应保留的有效行为，是否比只提供失败记录更容易修复错误、减少已有能力退化。

主实验选择课程 Part 3 的 **Option A：修改优化器输入**。B、C 共用失败抽样规则、训练 batch、父代与幸存者选择、验证评分、初始化和预算控制；C 只在父代的 `train_reflection_context` 中附加成功对照与局部反馈。优化结果必须由模型自动生成，不手写最终技能。

这里的“自我纠正”发生在 **train 经验驱动的技能迭代阶段**：相对于可比成功，识别有证据支持或明确标为假设的有害偏差，为下一版技能加入窄适用范围的检查，并保留有效行为。测试时只执行冻结技能，不读测试 gold，不按测试结果继续学习。

本版状态分为三类：

- **已实现并验证基本正确性：** Baseline 的 N/I/B 入口；Explore 的训练记忆、配对和对照反馈；共同预算、图校验、日志、失败清单、图导出和冻结入口；两数据集接口的合成流程；21 项正确性测试。
- **已完成的真实模型验证：** RTX 4070 Ti SUPER 上的 Qwen3.5-9B Q4_K_M 服务，以及固定 10 个 SearchQA train ID 的 pilot。它验证 N、I、自动生成初始技能和一个自动变异候选的执行链路。
- **尚未完成：** 正式共享初始化与完整 N/I/B/C 实验、B 的完整 dev 诊断、C 的真实模型对照实验、HotpotQA 真实 pilot 与课程版划分接入、两数据集正式 test、报告和演示材料。

因此，目前不能声称完成课程复现或证明 C 优于 B。正结果、无提升或性能下降都可以构成结论，但必须保留公平对照与完整证据。

## 2 课程协议与固定实现基础

课程要求以本目录 [课程项目说明](AIAA4051_26Fall_Final_Research_Projects.pdf) 的 Project 5、第 19–23 页为准。

每个数据集最多 100 个 train、30 个 dev、50 个 test；最多 3 轮进化，每轮最多评估 2 个新候选。必须比较 no-skill、initial-skill、原版优化器与冻结后的改进方法，报告 EM、F1、延迟、逐轮技能变化和完整失败清单。报告正文最多六页，演示按课程统一格式完成。

课程说明的硬件为 1 张 A6000、规划上限为 40 GPU 小时；**当前实际开发硬件是 RTX 4070 Ti SUPER 16GB**。本机实验沿用 40 GPU 小时的全项目分配账本，但不把本机测量标注成 A6000 实测。后续若迁移硬件，记录硬件与服务变化，并在统一条件下重新评测延迟。

上游固定为 [GraphSkillEvo 提交 `722a8a913b05dc9d1b4078f0bec604cde4dc7529`](https://github.com/ruisun7/GraphSkillEvo/tree/722a8a913b05dc9d1b4078f0bec604cde4dc7529)，源代码在 `Baseline/vendor/GraphSkillEvo/`。集成和共同修改记录见 [UPSTREAM.json](Baseline/UPSTREAM.json)。

当前复用的关键行为：

- `CourseTrainer` 继承原 `EvolutionarySkillTrainer`，复用原 `_sample_operator_inputs`、`_invoke_operator` 及父代反思输入接口。
- 失败样例由原 `_sample_failure_rows` 从 `hard=0` 的输出中按固定 seed 随机抽取，`k=3`，不加入难例排序、类别重采样或手工挑题。
- 父代排序、rank 权重、父代抽样、幸存者选择与最终选择调用原函数，保留它们不同的并列处理。父代排序看验证 EM/F1；幸存者以验证 EM 为主并保留原稳定顺序；最终 best 使用原 EM 优先规则。
- 种群大小为 2 时只调度 `mutation_other` 和 `mutation_graph`，不触发 crossover。前者通过原 `adopt_other` 保留父代图部分，后者通过原 `adopt_graph` 保留父代 Global Guidance。不能把这个课程规模运行称为完整四算子复现。
- SearchQA 为给定 question/context 的单轮 QA，保留原 6000 字符及 `[DOC]` 边界截断行为，并记录实际可见文本和 hash；不声称观察到逐节点内部执行顺序。

为课程规模和本机后端作出的修改是 **B/C 共同条件**，不是 C 的创新点：有限次初始化与候选生成、统一 token 上限、本地服务接入、HotpotQA 薄适配层、结构化技能输出和简洁长度提示。报告应同时披露这些共同修改和 C 的唯一反馈改动。

## 3 数据准备、划分与标签隔离

### 3.1 SearchQA：已准备 100/30/50

原始来源为 [`lucadiliello/searchqa`](https://huggingface.co/datasets/lucadiliello/searchqa/tree/c1a979068ba118d85467179b704031d113d689cc)，固定 revision `c1a979068ba118d85467179b704031d113d689cc`。

先按上游 GraphSkillEvo 的 train/val/test ID 清单恢复完整记录，再在各自 GraphSkillEvo 分区内按 ID 排序、以 seed=42 抽样，固定 100 train、30 dev、50 test。原始 HF split 来源保留在元数据中；实验分区以固定的 GraphSkillEvo 清单为准，不临时按答题结果重新划分。

本地准备数据位于 `Baseline/data/prepared/searchqa/`，包括 `train.json`、`dev.json`、`test.json`、对应的 `*_ids.json` 和 `manifest.json`。manifest 保存来源、文件 SHA256、样例内容 hash、数量和 pilot 的预留 train ID。程序拒绝重复/跨划分 ID、文件或 ID 清单篡改；记录规范化重复问题和词集合 Jaccard≥0.8 的跨划分近重复提示，不根据模型表现过滤或补抽。

### 3.2 HotpotQA：公开版仅作当前开发基础

已下载 [`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab) 的公开 distractor train/validation，固定 revision `1908d6afbbead072334abe2965f91bd2709910ab`，准备 100 train、30 dev。

**当前 `test.json` 和 `test_ids.json` 为空，数量为 0，manifest 中 `test_pending=true`、课程等价性为 `unconfirmed_public_distractor`。** 公开测试文件没有可用于本实验本地评分的 gold answer，而课程说明指定 HotpotQA provided。因此不把 validation 改名为 test，也不把公开开发集成绩包装成课程测试成绩。

拿到课程文件后核对版本、分区与任务设置，再用新的目录准备 `hotpotqa_course`。当前适配器处理已提供上下文的 QA，支持官方 `context=[[title,[sentences]]]` 及统一结构，保留文章标题、句子编号、supporting-facts 元数据。若课程版要求检索，必须接入相应检索环境并重算预算，不能直接当成当前单轮版本。

当前公开版可用于开发联调；正式 HotpotQA 条件必须在课程版确认后独立重新初始化、运行与冻结。数据 manifest 改变后不能直接复用旧运行作为正式对照。

### 3.3 各分区的职责

- **train：** 产生技能修改证据、成功/失败记忆和优化反馈；pilot 的 10 个唯一 ID 来自固定 train 内部，仍计入最多 100 题。
- **dev：** 完整验证候选、排名、父代与幸存者选择、最终技能选择，以及方法开发诊断。重复使用 dev 的偏差必须在报告披露；不将 dev 的具体题目、答案或响应装进 train 优化 batch。
- **test：** 四组、模型和设置全部冻结后才推理与评分；只做描述性错误分析，不调 prompt、不选模型、不更新记忆。

正常 QA prompt 只由基础指令、技能、question 和实际可见 context 构成。gold answer、HotpotQA supporting-facts 标签和带 `[EVALUATION RESULT]` 的评分日志不回送正常执行器。相同输出的复用不算独立新观测；重复实际调用和失败重试全部计入消耗。

## 4 已部署模型、后端与共同配置

### 4.1 模型与本机服务

当前执行器和优化器共用以下固定组合：

- 原模型：[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 量化模型：[`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5)，revision `3885219b6810b007914f3a7950a8d1b469d598a5`，文件 `Qwen3.5-9B-Q4_K_M.gguf`。
- 文件大小：5,680,522,464 bytes；SHA256：`03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`。已完成完整文件校验。
- 本机权重目录：`F:\Document\HKUST\2026fall\NLP\Qwen3.5_9B_4bit`；官方 tokenizer 位于其 `tokenizer/`，按上述原模型 revision 下载并在本地加载。
- Conda 环境：`qwen3.5`，Python 3.11.17；环境依赖锁见 [requirements-lock.txt](local_model/requirements-lock.txt)。
- GPU：RTX 4070 Ti SUPER，服务实测可见显存 16,376MiB。
- 服务：llama.cpp `b11871`，commit `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9`，Windows CUDA 12.4 原生二进制；其官方发布包已校验摘要。GPU 推理由 llama.cpp 执行，Python 环境用于 tokenizer 和实验脚本。
- 接口：`http://127.0.0.1:8000/v1`，alias `Qwen3.5-9B-Q4_K_M`，只监听本机。

服务使用 16384-token 上下文、单并发、GPU offload、Flash Attention，关闭思考模式和自动 context shift。请求过长时应报错，不静默挪掉前文。官方 tokenizer 与 GGUF 服务对样本文本的计数一致；这是一项接入检查，不等于穷尽验证全部文本的 tokenizer 行为。

启动/停止脚本见 [本机服务说明](local_model/README.md)。已完成 pilot 后服务主动停止，避免持续占用 GPU；后续正式运行前重新启动。

### 4.2 当前两数据集一致的限制

以 [SearchQA 配置](Baseline/configs/searchqa.json) 和 [HotpotQA 配置](Baseline/configs/hotpotqa.json) 为准：

- 固定 seed=42；种群大小 2；最多 3 轮；每轮每个父代使用相同的 20 个 train 样例；每轮最多 2 个新候选；失败反馈 k=3。
- QA 最大生成 1024 tokens，temperature=0；技能生成最大 3072 tokens，temperature=0.6；共同 top-p=1；并发数 1；请求超时 300 秒。
- 每份技能最多 2048 tokens；反思/对照反馈区最多 2048 tokens；初始化、候选及 QA 的重试服从最多 2 次总尝试限制，不存在预算外的隐藏无限重采样。
- 每个 runner 当前调用上限 600、生成 token 总上限 700000、运行 wall time 上限 9 小时；优化 run 包括导入的共享初始化逻辑成本。它们是一次运行的上限，不是全项目总量，也不意味着会实际耗满。
- `structured_operators=true`：只对初始化和 mutation 的响应施加 JSON schema，约束 `skills` 或 `new_skill/notes` 的序列化；不对 QA 施加 JSON 约束。

本机后端同时为技能生成附加共同的序列化/长度提示：**最多六个简洁节点、三个不重复 task graphs**，保持原 2048-token 技能上限、图结构检查和原作用域采用规则。这里的节点/图数量是当前提示合同，不是新增的独立计数校验或人工 veto；实际硬校验仍由原图结构验证和 token 上限执行。

这些设置是在 train pilot 诊断阶段确定的，两组必须一致。它们不构成 C 相对于 B 的额外干预。后端是否完全确定性仍应按实际重复运行判断，不能仅凭 temperature=0 宣称绝对可复现。

## 5 真实 pilot 的结果与修复记录

已完成的成功运行是 `Baseline/runs/searchqa/pilot-qwen35-q4-20261025-163349/`，不是完整三轮 B，也不是 C 的真实实验。

该 pilot 固定 10 个唯一 train ID，分别执行 N、I、模型自动生成的初始技能和一个自动变异候选，共 **40 次 QA + 2 次技能生成 = 42 次真实模型调用**。全部过程不使用 dev/test 作为优化证据。

成功运行记录：

- wall time 约 58.765 秒；整卡显存峰值 7518MiB，约 7.34GiB，包含桌面等进程占用。
- 输入 tokens 68245，生成 tokens 4404；当前关闭思考模式，报告 reasoning tokens 为 0。
- 接口错误、答案格式失败、生成达到长度上限的次数均为 0；初始技能与变异候选通过 JSON、图结构和技能长度检查。
- 当前 21 项正确性测试通过，上传前从仅含暂存源代码的隔离副本运行也通过。

此前的两次失败 pilot 均未删除：第一次完成 20 次 QA 后，技能生成返回无效 JSON；启用 schema 后，第二次因重复扩展相似任务图而达到生成上限。随后统一收紧输出长度提示并把技能 temperature 从 0.2 调整到 0.6，才完成上述成功运行。未手工修补最终技能，也未为某一实验组放宽上限。

包括诊断和两次失败运行在内，部署验证合计记录 **89 次模型调用**；当时 GPU 服务分配账本累计约 **0.221655 GPU 小时**，已关闭该会话。这个数含加载、调试和服务常驻时间，不能与成功 pilot 的 58.765 秒 wall time 混为一谈。

本地证据为 `local_model/deployment_validation.json`、`checks/`、`latest_pilot.json` 及成功 pilot 的 `pilot_summary.json`、`hardware_profile.json`、逐题输出和技能图。这些运行产物不随代码上传 GitHub。

pilot 只证明当前模型在本机可以执行这条链路，不证明 B/C 有收益，也不能把这 10 题 train 指标写成正式测试成绩。HotpotQA 的真实模型 pilot 尚未完成。

## 6 四组正式条件与公平性

每个数据集独立建立四个条件：

1. **N：No skill。** 仅基础 QA 指令，不附加技能。
2. **I：Initial skill。** 使用上游原始 SearchQA 初始文档。该文档主要是图执行说明和空节点/任务图模板；当前 wrapper 对 HotpotQA 也使用这个同一通用模板，不声称是上游已提供的 HotpotQA 专用技能。
3. **B：课程规模原版 GraphSkillEvo 优化器。** 在共同包装层上使用失败反思输入。
4. **C：对照反馈优化器。** 在同一包装层上使用成功/失败对照反馈。

正式共享初始化由 `initialize` 生成，包含原模板和模型生成的另一个个体，并在完整 dev 上评分。I 始终指原模板，不把 pilot 中模型生成的初始个体或上游已进化 `best_skill.md` 偷换成 I。

B/C 从同一正式初始化产物分叉，共享初始化种群和相应 dev 缓存的逻辑成本；C 不读取 B 分叉后生成的训练成功输出。pilot 产物不直接充当正式共享初始化，不作为独立重复实验。

两组固定相同模型、量化、服务、解码、轮数、候选数、batch 规则、dev、重试与预算上限。N/I 不需要人为消耗同样的优化预算。相同上限不等于相同实际消耗，正式结果必须同时报告实际调用数、输入/生成 tokens、时延和 GPU 分配。

当前代码严格检查初始化的代码、配置、数据、tokenizer 和 demo 签名。正式初始化前应完成实现与基本配置定稿；若 B 开发诊断后又修改代码或这些配置，需重建共同初始化并重跑受影响的正式条件，不能把跨版本产物拼成一组公平对照。

## 7 Explore：成功–失败对照反馈的具体实现

### 7.1 与 JitRL 的关系

借鉴 [JitRL 第 4 节](https://arxiv.org/html/2601.18510v1#S4) 的经验记忆和局部反馈思路。JitRL 估计动作优势并修改动作 logits；当前方案不实现那条数值更新，也不宣称继承其理论保证或提出新的 RL 算法。

研究贡献候选是：**可比成功参照能否改善图结构技能修改的证据，减少错误归因与能力退化。** 因此 C 的修改集中于反馈材料与保留要求，而非奖励函数、选择评分或推理动作重排。

### 7.2 训练记忆与更新时序

`Explore/explore/memory.py` 只接受本方法、同一数据集、固定 train ID 的真实输出；拒绝 dev/test、其他方法分叉后的记录，不把缓存或基础设施错误响应记为有效经验。

保留 ID、round、skill hash、可见 context hash、question、原始响应、抽取答案、EM/F1、generation config 和 observation ID。每题最多保存两份不同成功和两份不同失败，以响应和可见上下文去重；新一轮重复真实输出只更新来源，不当作新的独立模式。原始调用和全部输出仍在未压缩日志中。

一轮中先让两个父代执行同一 batch，再一次性观察全部输出，之后分别构造反馈。这样第一个父代可以引用第二个父代正常产生的同轮成功，不因执行顺序造成配对不对称。不额外调用成功采样器或大型评审模型。

### 7.3 保留原失败抽样，再查找成功参照

只处理原抽样器已经选中的失败，不按配对是否容易成功更换失败 ID。

1. **same_question：** 相同题目 ID、相同可见 context hash 的 EM=1 输出优先；先当前轮、再最近历史，以固定 ID/技能/observation 顺序打破并列。忽略未来轮记录。
2. **similar_task：** 无同题成功时，仅从相同数据集、其他题目 ID、相同粗任务类型的成功中筛选问题词集合 Jaccard≥0.2 的参照。先当前轮，再相似度和固定来源顺序。同类参照只支持抽象策略比较，不支持确定的因果替换结论。
3. **unpaired：** 没有可比成功时保留原失败，不编造成功轨迹，不把标准答案冒充模型已经完成的解法。所有选中失败均无配对时，C 原样回退到 B 的有界反馈行为。

粗任务类型由固定的 **问题文本规则** 决定，包含 yes/no、比较、计数/日期、关系链、实体属性和 unknown；不读取 HotpotQA gold type 或测试表现来分类。

### 7.4 可观察检查与不确定性

目前程序确定性检查答案标签格式，以及成功响应中的显式引文片段是否出现在该响应实际可见的 context 中。`reference_has_verifiable_quote=true` 只说明找到字面引文，不说明引文在语义上支持答案；无法核验时标为 `answer_only_or_unverified`。

实体混淆、关系方向、遗漏约束、无依据推断和对应技能节点的归因，交由同一次技能生成调用依据材料提出，并始终作为需要核对的诊断假设。SearchQA 单轮输出不是逐节点执行日志；EM=1 也不证明所有推理都正确。没有对应节点时可以提出缺少规则，不虚构曾经观测到该节点执行。

反馈指令要求识别适用条件、可支持或疑似有害的偏差、应保留的有效行为，再加入窄范围检查；禁止把所有词面差异当成错误，禁止复制题目实体、答案或完整解例进最终技能。最多抽查 10 对 train 记录以核对材料与诊断规则，不手改最终技能。

### 7.5 审计包与真正送入优化器的反馈

审计包字段与现实现一致，下面仅为接口示意，不是实验结果：

```json
{
  "failure_id": "train_id",
  "pair_type": "same_question",
  "similarity": 1.0,
  "task_type": "entity_attribute",
  "failure": {
    "example_id": "train_id",
    "skill_hash": "...",
    "visible_context_hash": "...",
    "question": "...",
    "response_excerpt": "...",
    "predicted_answer": "...",
    "gold_answers": ["..."],
    "em": 0
  },
  "success_reference": {
    "example_id": "train_id",
    "skill_hash": "...",
    "response_excerpt": "...",
    "predicted_answer": "...",
    "em": 1
  },
  "observable_checks": {
    "answer_format_ok": true,
    "reference_has_verifiable_quote": false,
    "reference_evidence_status": "answer_only_or_unverified",
    "diagnosis_status": "hypothesis_to_check",
    "node_execution_observed": false
  }
}
```

原始响应完整保存在运行日志/记忆中；审计包提供来源元数据和初步摘要。实际优化器输入进一步按统一 2048-token 上限缩短字段，并记录 `actual_packets` 与 `actual_feedback`。优先保留全部所选失败 ID 及其成功参照，使用逐字段缩短和短版指令，不直接切断已序列化 JSON。若最小材料仍放不下则报错，不能悄悄删掉失败或截掉成功一侧。

正式分析报告同题配对率、同类参照率、无配对率，并以真正被选择为 mutation 父代的反馈为主要统计对象。没有失败的父代不人为制造样例分母。

## 8 已实现的代码组织与证据输出

当前目录组织如下，`data/`、`runs/`、运行库和权重不包含在 GitHub 源码发布中：

```text
AIAA4051_Project/
  Project5_Implementation_Plan_zh.md
  Baseline/
    baseline/             # 数据、QA、原版反馈、预算、进化包装、分析与冻结
    configs/              # SearchQA / HotpotQA 的共同模型和资源配置
    tests/                # 协议与端到端合成测试
    vendor/GraphSkillEvo/  # 固定上游源代码及必要 SearchQA ID 元信息
    download_data.py
    run.py
    data/                 # 本地原始数据、准备数据、固定 IDs 与 manifest
    runs/                 # 本地初始化、N/I/B、pilot 与最终评估
  Explore/
    explore/              # memory.py / pairing.py / feedback.py
    run.py
    runs/                 # 本地 C、反馈设计冻结文件及训练记忆
  local_model/
    start.ps1 / stop.ps1 / status.ps1
    download_model.py / run_pilot_measured.py
    requirements-lock.txt
    runtime/              # 本地 llama.cpp / CUDA 二进制
    logs/ / checks/ / measurements/
```

每个正式 run 保存实际 config、模型与 tokenizer 签名、prompt/原始响应、逐题指标、train batch IDs、候选/失败尝试、父代和幸存者记录、最终技能、dev 全量失败及 `.json/.mmd` 技能图。预算事件能追踪到对应调用；缓存复用保存来源 observation，不伪装成新观测。

图导出来自实际 `Node Lists`、`Use when` 和 workflow 顺序。节点内自然语言条件保留在结构化文本中，不推断不存在的内部执行轨迹或虚构分支。尚未完成正式三轮演化，因此不能把 pilot 的三份技能图当成完整 B/C 演化结果。

GitHub 发布源代码、prompts、配置、测试、依赖锁和说明，保留恢复 SearchQA 所需的轻量上游 ID 配方；排除模型、实际数据正文、CUDA 二进制、运行/测量日志、缓存与备份。正式提交课程实验材料时，另按要求提供冻结的实际数据 ID 和结果，不能因 Git 忽略而省略验收证据。

## 9 接下来正式执行的顺序

以下命令在项目根目录运行，所有输出目录须为空；重复运行应使用新的明确目录，不覆盖旧账本。

### 9.1 启动服务并确认代码定稿

```powershell
conda activate qwen3.5
.\local_model\start.ps1
.\local_model\status.ps1
python -m unittest discover -s Baseline/tests -v
```

服务脚本自动为实际分配开启 GPU 会话，不要再手工叠加一个重复会话。当前 SearchQA pilot 已通过，无须为“填预算”反复执行。若改动模型/后端/输出限制，使用新 pilot 目录重新验证，再决定是否共同调整正式配置。

### 9.2 SearchQA：共享初始化、N/I/B 与完整 dev 诊断

```powershell
$cfg = "Baseline/configs/searchqa.json"
$data = "Baseline/data/prepared/searchqa"
$out = "Baseline/runs/searchqa"

python Baseline/run.py initialize --config $cfg --data $data --output "$out/shared_init"
python Baseline/run.py static --condition N --config $cfg --data $data --output "$out/N"
python Baseline/run.py static --condition I --config $cfg --data $data --output "$out/I"
python Baseline/run.py train --config $cfg --data $data --initialization "$out/shared_init" --output "$out/B"
```

检查 B 的 `dev_analysis/failures.json`、`failures.csv` 和完整输出，分析全部 EM=0 样例，区分观察与原因假设。必要的实现调整在正式比较前完成；若改变当前签名，按第 6 节重新建立同版本产物。

### 9.3 冻结反馈设计并运行 C

```powershell
python Explore/run.py freeze-design --config $cfg --baseline-run "$out/B" --output Explore/runs/searchqa/feedback_design.json
python Explore/run.py train --config $cfg --data $data --initialization "$out/shared_init" --design Explore/runs/searchqa/feedback_design.json --output Explore/runs/searchqa/C
```

`freeze-design` 保存 B 全量 dev 失败证据 hash、配对规则、反馈指令和代码签名，是实现规格的冻结记录，不代表人工诊断自动完成。C 训练前检查记录一致；不能在看过 test 后修改。

### 9.4 四组同时冻结后才评测 test

```powershell
python Baseline/run.py freeze --N "$out/N" --I "$out/I" --B "$out/B" --C Explore/runs/searchqa/C --data $data --output "$out/frozen.json"
python Baseline/run.py test --frozen "$out/frozen.json" --output "$out/test"
```

冻结检查 N/I/B/C 的模型、配置、代码、tokenizer、数据和 B/C 共同初始化一致，并对技能与运行证据计算 hash。冻结后改动代码、技能、数据或相关证据会拒绝评测。test 只读取技能，不加载或更新训练记忆。

### 9.5 HotpotQA 与结束服务

拿到课程文件后先建立新数据目录：

```powershell
python Baseline/run.py prepare --dataset hotpotqa --train COURSE_TRAIN.json --dev COURSE_DEV.json --test COURSE_TEST.json --output Baseline/data/prepared/hotpotqa_course
```

确认任务设定后，用 `Baseline/configs/hotpotqa.json`、新数据目录和独立的 HotpotQA 输出目录执行同样的 pilot、初始化、N/I/B、诊断、C 和冻结测试步骤。当前公开 HotpotQA 的 test_pending 会阻止正式冻结评测，不绕过这项检查。

实际停止 GPU 服务分配时执行：

```powershell
.\local_model\stop.ps1
```

## 10 预算匹配、缓存与停止条件

模型请求统一经过预算包装器，下一次调用前检查调用额度、预留完整最大生成 token 额度，并检查 run wall time 和全项目 GPU 账本。失败请求/无法取得 usage 的请求按预留上限保守记账；有效 usage 保存原始值。reasoning tokens 是 completion 的子集，单列但不重复相加。

共享初始化的物理调用只在实际运行账本中发生一次，B/C 各导入相同逻辑使用成本并标记 `physical_calls=0`。方法内确定性 dev 输出缓存保留真实来源，不作为独立重复。train 不复用缓存充当新观测，也不删除重复真实调用的成本。

最多两次生成仍失败时，该候选失败，不无限重采样父代。达到预算时停止，保留按 dev 选出的技能；尚未完成 dev 的候选不参与选择，已完整验证的候选仍可按原规则竞争。报告停止原因、实际完成轮数和有效候选数。

按两个数据集都达到目标规模、两个不同初始化个体、三轮、每轮两个父代和两个新候选粗估：每个数据集 B/C 合计最多约 240 次 train QA 和 420 次唯一 dev QA；加上当前 `static` 实现中 N/I 各自完整 dev 的 60 次 QA，以及四组各 50 题的 200 次 test QA，约为 **920 次 QA/数据集、1840 次 QA/两数据集**，另加初始化、技能生成、pilot、失败重试和可选诊断。该估算取决于缓存命中和完整轮数，不是实测任务数或 GPU 小时承诺。

全项目实际 GPU 时间以服务分配账本为准，包含加载、常驻、失败运行和诊断；一张卡并发不重复加成多张卡。当前 9 小时、600 calls 和 700000 generated tokens 是 runner 上限；最终 `test` 为各条件独立记账，不能把这些值理解成整个项目的一个共享调用包。

如果 B/C 实际开销明显不同，补充在共同累计预算截点、使用当时 dev 选定技能的比较。当前代码已保存每轮累计成本和技能历史；共同截点的补充评估尚未完成，不声称已有自动预算对齐主结果。不能使用 test 表现选择截点。

不为耗满预算做无意义调用。正式开始后如预算不足，先放弃可选消融和数值惩罚扩展；任何共同缩减轮数/batch 的决定须在正式比较前确定，不能只给 C 优惠。

## 11 指标、完整失败清单与能力退化

正式结果按数据集、条件分别报告样本数、正确计数、EM、F1、单题均值/P50/P95 延迟、输入与生成 tokens、实际调用数和 GPU 分配时间。无结果的正式条件保留“待测”，不填 pilot 或预期数值。

- **SearchQA：** 原答案抽取优先取最后一个 `<answer>...</answer>`，缺失时使用上游末行 fallback；使用规范化 EM 和多参考答案最大 token F1。substring match 仅作诊断。
- **HotpotQA：** 当前实现为 answer-only EM/F1，并处理 yes/no/noanswer 的官方特殊 F1 规则；不宣称已实现 supporting-facts 或 joint 指标。课程若指定不同 scorer，先核对并共同冻结。
- **基础设施和格式失败：** 保留在总分母中，单列数量；不能只汇报成功请求或删除超时题。
- **配对变化：** 在相同固定题上统计 B 错 C 对、B 对 C 错、两者都对、两者都错；修复率为修复数/B 失败数，退化率为退化数/B 成功数，零分母为 NA。
- **不确定性：** 50 题时一题对应 2 个百分点 EM。可使用题目级 paired bootstrap，但不能把它解释为不同训练随机种子之间的稳定性。

B 的全部 dev 失败，以及冻结后各方法全部 test 失败，都要提供 ID、问题、参考/预测、EM/F1、可见证据、主类别和诊断解释。自动分析目前只明确归类基础设施错误和答案格式问题；其余默认“未知/证据不足”。实体混淆、关系链错误、比较/约束遗漏、证据缺失、无依据推断、过强技能约束等需要依据完整材料复核，不声称已有可靠自动因果分类。

每题只计一个主标签，可附次标签；将事实观察、原因假设和相关技能节点分开。人工抽查与研究分析不等于人工修改最终技能。测试错误只用于最终分析，不反向修订配对、prompt 或技能。

## 12 里程碑、验收与报告

接下来的依赖顺序是：

1. 复核当前共同实现/模型配置，完成正式 SearchQA 共享初始化与 N/I/B。
2. 整理 B 的全量 dev 失败，核对对照材料质量，冻结反馈规格。
3. 从同一初始化运行 C，检查原失败抽样一致、记忆来源正确、实际输入满足反馈上限。
4. 取得并确认课程 HotpotQA，完成真实 pilot，再独立完成四组正式条件。
5. 四组同版本冻结后分别评测两套固定 test，生成全量错误、修复/退化与成本结果。
6. 完成最多六页正文、附录和课程演示材料，另行打包未上传 GitHub 的固定 ID、图、原始结果和 GPU 账本。

最终验收清单：

- [x] 上游固定版本、Baseline/Explore 包装层、本机模型服务和共同预算实现。
- [x] SearchQA 已固定 100/30/50；公开 HotpotQA 已固定 100/30 并明确 test_pending。
- [x] 两数据集接口的合成流程与 21 项正确性测试通过。
- [x] RTX 4070 Ti SUPER 上真实 SearchQA 10 题 pilot 通过，失败调试记录和消耗保留。
- [x] 代码、配置和文档已发布 GitHub，模型、数据正文与运行产物排除。
- [ ] 正式共享初始化、N/I/B 全轮运行及完整 B dev 诊断。
- [ ] C 的真实对照运行、配对覆盖率和人工证据抽查。
- [ ] HotpotQA 课程版核对、真实 pilot 与独立四组实验。
- [ ] 两数据集冻结后的正式 test、完整错误和能力修复/退化分析。
- [ ] 公平预算/共同截点分析、实际 GPU 统计、六页报告和演示材料。

报告建议包含背景与研究问题、共同实现与数据协议、对照反馈方法、正式结果、失败/修复/退化案例及局限。必须披露本机硬件、4-bit 量化、非思考模式、共同 JSON/长度约束、反复使用 dev 和单一训练 seed 的限制。不预先承诺论文级新颖性或普遍提升。

## 13 单列的后续扩展：动作级数值惩罚

当前 **未实现** 直接修改动作分数或 logits 的版本。若主项目完成后仍有预算，再单列研究基于配对经验的有限候选动作重排；不混入 Option A 主实验。

一个待研究的有界形式为：

```text
penalty(s, a) = clip(c(s, a) * d_hat(s, a), 0, 1)
new_score(s, a) = base_score(s, a) - lambda * penalty(s, a)
```

它需要额外定义可观察状态、有限动作、配对证据、置信度、base_score 的来源、lambda 的开发集选择和候选生成预算。没有可比配对、只有词面差异或无法定位证据时 penalty=0；不能把整条轨迹的最终失败广播为全部动作的惩罚。

SearchQA 当前接口没有天然的动作级 logits。候选动作概率、整段序列评分与显式打分也不能混为同一种 logits。没有无惩罚、整段统一惩罚、局部对照惩罚的独立公平比较前，不宣称实现 JitRL 更新或获得其理论保证。预算不足时不开展该扩展。

## 14 复现入口与参考资料

项目入口和当前代码说明：

- [仓库首页](README.md)、[Baseline 说明](Baseline/README.md)、[Explore 说明](Explore/README.md)、[本机服务说明](local_model/README.md)。
- [固定上游与共同修改](Baseline/UPSTREAM.json)、[两数据集配置目录](Baseline/configs)、[正确性测试](Baseline/tests/test_protocol.py)。
- 本地运行证据路径已在第 5 节列出；它们有意不作为 GitHub 文件链接，以免误导远端读者以为仓库包含模型或实验日志。

方法和数据来源：

- [课程项目说明](AIAA4051_26Fall_Final_Research_Projects.pdf)，Project 5，第 19–23 页。
- [GraphSkillEvo 论文](https://arxiv.org/abs/2609.21749) 与 [固定官方代码](https://github.com/ruisun7/GraphSkillEvo/tree/722a8a913b05dc9d1b4078f0bec604cde4dc7529)。
- [GraphSkillEvo 数据准备说明](https://github.com/ruisun7/GraphSkillEvo/blob/722a8a913b05dc9d1b4078f0bec604cde4dc7529/data/README.md)。
- [JitRL](https://arxiv.org/abs/2601.18510)，经验记忆/动作优势用于研究启发，不等同于当前反馈方法。
- [固定 SearchQA 数据源](https://huggingface.co/datasets/lucadiliello/searchqa/tree/c1a979068ba118d85467179b704031d113d689cc)、[HotpotQA 官方说明](https://hotpotqa.github.io/) 与 [固定 HF 镜像](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)。
- [Qwen3.5-9B 原模型](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)、[固定 GGUF 量化版本](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5)、[llama.cpp b11871](https://github.com/ggml-org/llama.cpp/releases/tag/b11871)。
