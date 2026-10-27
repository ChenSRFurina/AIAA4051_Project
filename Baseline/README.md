# Project 5：固定基线与实验公共设施

实现对应 `Project5_Implementation_Plan_zh.md` 第 1–12、14 节及课程 PDF Project 5。
`Baseline` 负责 N（无技能）、I（未进化初始技能）、B（课程规模 GraphSkillEvo）；相邻 `Explore` 负责 C（成功–失败对照反馈）。
代码已通过合成样例验证；目前没有真实模型实验成绩。

## 当前数据与模型状态

- SearchQA 完整原始 parquet 已下载，按固定上游 GraphSkillEvo 划分恢复源样例，再以 seed=42 固定 100 train / 30 dev / 50 test。ID、来源版本、文件 SHA256 均在 `data/prepared/searchqa/manifest.json`。
- HotpotQA 公开 distractor 的完整 train / validation parquet 已下载，固定 100 train / 30 dev。其公开测试文件不提供答案，PDF 又指定课程提供版，因此 `test.json` 为 `[]`，manifest 明确标记 `test_pending=true`、课程等价性未确认。没有把 validation 改名成 test。
- 正式模型服务未配置；两个 `configs/*.json` 中的 `SET_*` 必须根据实际服务填写。执行器与优化器共用同一个 model ID、revision、tokenizer 和精度。
- 下载约 685 MB，原始数据默认被 git 忽略；固定的 100/30/50 清单和准备后的少量样例可以版本化。

所有命令在 **AIAA4051_Project 项目根目录** 执行。需要 Python >=3.10；本次实际验证环境见 `VALIDATION.md`。

## 安装、下载与离线验证

建议在专用 Python 环境中安装依赖：

```powershell
python -m pip install -r Baseline/requirements.txt
python Baseline/download_data.py
python -m unittest discover -s Baseline/tests -v
python Baseline/run.py demo --output Baseline/runs/demo
```

下载脚本固定 HF 数据版本；已有准备数据会校验后跳过，不重复下载。`demo` 使用合成上下文和确定性模拟响应，在两个数据集接口上跑完共享初始化、N/I/B/C、冻结和最终评测；输出 `SYNTHETIC_ONLY.json`，**不能用作论文成绩**。再次运行要换输出目录。

## 正式模型与 pilot

在 `Baseline/configs/searchqa.json` 与 `hotpotqa.json` 填写实际 model ID、精确 revision、本地 tokenizer 目录、精度、服务版本、兼容接口地址；两个配置使用同一模型。tokenizer 通过 `local_files_only=True` 加载，模型请求走 `/v1/chat/completions`，使用 `max_tokens`、`seed`、`temperature`、`top_p`。服务必须返回包含 reasoning 的 completion usage。额外模板参数写入 `extra_body`，不得覆盖固定的 model、messages 或生成上限。

本实现不自动下载模型权重、不假定某个模型名称存在，也不在没有 GPU 服务时伪造实测结果。请先确认服务支持该检查点和固定参数。真实调用使用全项目共享的 GPU 分配账本：

```powershell
python Baseline/run.py gpu start --ledger Baseline/runs/gpu_accounting.jsonl --session a6000-service-1
python Baseline/run.py pilot --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --output Baseline/runs/searchqa/pilot
python Baseline/run.py pilot --config Baseline/configs/hotpotqa.json --data Baseline/data/prepared/hotpotqa --output Baseline/runs/hotpotqa/pilot
python Baseline/run.py gpu status --ledger Baseline/runs/gpu_accounting.jsonl
```

GPU ledger 应与实际 GPU 服务分配同时 start/stop，包括空闲常驻、失败运行和 pilot。每次真实请求前检查累计分配是否达到 40 小时；该检查不负责停止外部模型服务器。结束实际服务分配后执行 `gpu stop --ledger ... --session a6000-service-1`。不要按并发请求重复计算同一张卡，不把 HTTP latency 当成 GPU 小时。未接入实际服务的 demo 不计为 GPU 实测。

pilot 只用固定 train 内的前 10 个保留 ID，运行 N/I、生成初始个体和一个候选。检查 `calls.jsonl` 的原始请求、响应、usage、finish_reason、截断及失败率，再统一调整两种方法的上限；正式实验开始后不单独放宽 C 的预算。

## SearchQA 四组实验

先固定配置。所有输出目录必须为空，避免覆盖运行或漏算重试。原始初始技能是上游的空节点模板，I 如实使用它；不能拿 `vendor/.../ckpt` 中已进化技能替代。

```powershell
python Baseline/run.py initialize --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --output Baseline/runs/searchqa/shared_init
python Baseline/run.py static --condition N --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --output Baseline/runs/searchqa/N
python Baseline/run.py static --condition I --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --output Baseline/runs/searchqa/I
python Baseline/run.py train --config Baseline/configs/searchqa.json --data Baseline/data/prepared/searchqa --initialization Baseline/runs/searchqa/shared_init --output Baseline/runs/searchqa/B
```

先检查 B 的 `dev_analysis/failures.json`、`failures.csv` 和全部原始输出，再冻结反馈设计并启动 C（详见 `Explore/README.md`）。四组完成后：

```powershell
python Baseline/run.py freeze --N Baseline/runs/searchqa/N --I Baseline/runs/searchqa/I --B Baseline/runs/searchqa/B --C Explore/runs/searchqa/C --data Baseline/data/prepared/searchqa --output Baseline/runs/searchqa/frozen.json
python Baseline/run.py test --frozen Baseline/runs/searchqa/frozen.json --output Baseline/runs/searchqa/test
```

冻结文件包含代码、实际 prompts、模型配置、tokenizer、数据、技能及运行证据 hash。修改任一冻结项后会拒绝测试。测试只读取冻结技能；不读取/更新训练记忆，不将评分日志送回执行器，不根据 test 结果选技能。

## HotpotQA 课程版接入

公开数据目前可以做 pilot/train/dev，正式 test 被阻止。拿到课程文件后先确认 distractor / retrieval 设置；此适配器仅支持**已提供上下文的单轮 QA**。需要检索的课程版本必须另接检索接口，不能直接使用当前适配器声称等价复现。

```powershell
python Baseline/run.py prepare --dataset hotpotqa --train COURSE_TRAIN.json --dev COURSE_DEV.json --test COURSE_TEST.json --output Baseline/data/prepared/hotpotqa_course
```

支持官方 HotpotQA `_id, question, answer, context=[[title,[sentences]]], supporting_facts`，以及统一的 `id, question, context, answers, metadata` JSON/JSONL。保留文章标题、句子编号和 supporting-facts 标签；后两者的标签部分仅供评分/诊断。按课程原划分排序、seed=42 抽样，不跨划分补齐。接入后必须使用新数据重新初始化、运行并冻结 N/I/B/C。

## 保留的上游行为与共同修正

上游源码原样归档在 `vendor/GraphSkillEvo`，固定提交见 `UPSTREAM.json`。

- `CourseTrainer` 继承原 `EvolutionarySkillTrainer`，通过原 `_sample_operator_inputs`、`_invoke_operator` 接入 `train_reflection_context`；直接调用原父代权重、排序、幸存者函数，保留其不同的并列处理。
- population=2，只调度 `mutation_other` 和 `mutation_graph`；不宣称覆盖 crossover。使用原 prompts、JSON 解析和 `adopt_other/adopt_graph` 限制修改范围。
- B/C 使用相同的原 `_sample_failure_rows`，k=3，seed=`42+round*10000`。每轮两个父代使用相同 20 题训练 batch；新候选评估完整固定 dev。
- 为满足课程资源约束，外层循环共同限制为每候选最多 2 次**总生成尝试**；两次都非法则该候选失败，不无限重采样父代。初始化也有共同总尝试上限。所有失败调用均计费。
- QA、反馈与技能长度、上下文 6000 字符截断规则对四组一致；保留上游截断代码的边界细节。B 的反思文本使用原格式化函数并均分统一反馈额度。C 只替换这部分反馈内容。
- 相同技能的确定性 dev 缓存有明确来源，不作为新观测；不缓存训练输出充当新样本。共享初始化向 B/C 各记逻辑成本，物理调用只在初始化原账本中计算一次。
- 下一次调用前预留完整最大生成额度；网络失败/usage 缺失按上限保守记账。reasoning 是 completion 的子集，单列但不重复累加。

## 输出与局限

每个 run 保存 `config.json`、`run_meta.json`、`calls.jsonl`、`ledger.jsonl`、逐题 prompt/输出、训练 batch ID、候选及失败尝试、逐轮选择记录、`best_skill.md`、技能 `.json/.mmd`、`evolution.mmd`、完整 dev 失败清单。最终 test 保存 N/I/B/C 的 EM、F1、均值/P50/P95 延迟、成本、全量错误和 B/C 修复/退化计数、题目级 paired bootstrap。

自动诊断只对请求错误/答案格式作明确分类；其余默认“证据不足”。实体混淆、关系链错误、技能节点归因等需要依据完整记录复核，不把模型解释当作内部执行日志。图导出逐字记录 `Use when`、节点和 workflow；节点内部自然语言条件保留为文本，不自动虚构分支。

当前按同样上限和轮数比较并保存每轮累计成本；如真实 B/C 消耗差异明显，需要按共同预算截点补充评估。服务器是否真正确定性、A6000 内存/吞吐、真实模型端到端兼容性、六页报告与演示材料仍需实际实验完成。没有写入预期结果。

## 来源

- [GraphSkillEvo 官方源码](https://github.com/ruisun7/GraphSkillEvo/tree/722a8a913b05dc9d1b4078f0bec604cde4dc7529)
- [SearchQA 数据源](https://huggingface.co/datasets/lucadiliello/searchqa/tree/c1a979068ba118d85467179b704031d113d689cc)
- [HotpotQA 官方说明及数据许可](https://hotpotqa.github.io/) 与 [固定 HF 镜像](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)


## 本机后端的结构化输出设置

本机配置启用 `model.structured_operators=true`：只在技能初始化/变异请求中传入 JSON schema，约束 `skills` 或 `new_skill/notes` 的序列化格式。QA 仍输出 `<answer>` 标签，不添加 JSON 约束。此设置对 B/C 完全相同，原始图结构校验及重试上限不变。首轮未启用约束的 pilot 因模型生成无效 JSON 终止，失败响应、调用和 GPU 消耗全部保留。

本机 pilot 进一步统一冻结技能序列化/长度提示：最多六个简洁节点、三个不重复任务图，完整技能仍为 2048-token 上限；技能采样 temperature=0.6。原生成上限 3072、QA 上限 1024、两次尝试及图校验保持不变。该共同约束用于避免小模型在 JSON 输出时无限展开同类任务图，B/C 完全相同。
