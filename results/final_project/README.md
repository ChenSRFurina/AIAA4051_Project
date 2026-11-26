# Project 5 最终提交与复现指南

从解压后的 `results/final_project/index.html` 进入。查看已有结果只需浏览器。论文和证据附录对应课程原 PDF 的 Project 5 第 20–23 页；HotpotQA 的抽样与上下文要求采用课程 11 月 2 日的更新。

## 阅读与演示

- [中文总览](index.html)汇总两套固定测试集的六种条件及共同预算对照。
- [六页英文论文](report.pdf)覆盖研究问题、方法、选择协议、测试、失败分析与结论。可另开[网页版](report.html)。
- [演示](demo.html)支持按钮和左右方向键。[证据附录](appendix.html)可切换数据集、方法和候选，显示每一轮的流程图、分流条件、执行次序和父子代答题变化，也可按原题 ID、关键词和类别查找错题。
- [对照表](comparison.csv)、[逐题评测](predictions.json)、[原始文本与错题诊断](failure_inventory.json)、[错题 CSV](failure_inventory.csv)、[完整数值与预算](summary.json)可独立复算。每组测试预测的 `test_source` 来源见汇总，各方法共享观测的情形在论文第 4 页说明。
- [固定 ID](fixed_ids.json)、[课程原名三份 ID 列表](../hotpotqa_course/split_ids.json)、[抽样版本与校验](../hotpotqa_course/split_manifest.json)、[排除的 30 个 ID](../hotpotqa_course/excluded_old_dev_ids.json)、[协议与配置](protocols.json)、[GPU 台账](gpu_log.json)用于审计。
- [技能全文与候选表现](skill_graphs.json)、`graphs/` 中的 SVG、[每次变异的真实输入](operator_inputs.json)、[演示所用原始证据](demo_evidence.json)保留实际采用的文字。[实际请求示例](request_examples.json)给出每套数据的初始化、QA 和两种变异调用，含各自系统提示与运行参数。QA 系统提示见 [rollout_system.md](../../Baseline/vendor/GraphSkillEvo/graphskillevo/envs/searchqa/prompts/rollout_system.md)，初始化与变异模板见[上游提示目录](../../Baseline/vendor/GraphSkillEvo/graphskillevo/prompts/)。

### 四分钟现场讲解

1. **0:00–0:35** 打开演示第 1 步。展示 `songwriters` 对 `songwriter` 的真实训练失误，解释为何语义相近仍会失去 EM。
2. **0:35–1:25** 翻至第 2、3 步，展开完整反馈。成功参照回答了另一道题，答案是 `genetics`。随后展示子代实际加入的单复数规则，指出推理时无 gold、评分会忽略大小写。
3. **1:25–2:15** 第 4 步对照三道 dev 题。子代修复 `operas → opera`、`2 → two`，同时将正确的 `grandfather` 改成 `father`。这些观察支持答案形式假设，不能确定是哪一句指令造成变化。
4. **2:15–3:05** 第 5 步讲测试。原版 C 在 HotpotQA 净增 0 题，在 SearchQA 净减 2 题。HotpotQA 的共同预算截点选中同一技能，因此两组引用同一测试观测。C2c 的两套数据方向不同。
5. **3:05–4:00** 切到附录，展示 HotpotQA → B → `g0002_i0001` 的两条路线。切换到 HotpotQA dev，搜索 `Millettia`，展开响应与文章摘录，说明为何记录参考疑点并保留原评分。

English speaking cues: “The failure concerns answer form. The retrieved success demonstrates a short answer to a different question. Here is what the optimizer actually saw. Two validation predictions improved and one regressed. Under matched ceilings the HotpotQA checkpoints produced identical prompts. The effect did not transfer consistently across the two evaluation sets.”

## 不调用模型的验证

在解压后的仓库根目录运行。以下两个命令只依赖 Python 3.10+ 标准库。

```powershell
python api/verify_final_project.py
python api/render_final_project.py
```

第一个命令独立检查两套划分、六组逐题输出的顺序和指标、预算与技能哈希、448 条错题、候选图文件和本地页面链接。第二个从已发布证据重建 HTML、SVG、比较表。更新 PDF 时，可用 Chrome/Edge 打开 `report.html` 后打印为 PDF，纸张选 A4，使用 CSS 页面尺寸、背景图形，关闭浏览器页眉页脚。提交包已经包含经过逐页检查的 PDF。

## 重新运行模型实验

以下命令适用于 Windows PowerShell。完整复跑会产生新的模型调用。原实验使用单张 RTX 4070 Ti SUPER、llama.cpp b11871、16,384 上下文、固定 GGUF SHA-256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`。权重和推理二进制的获取方式见[部署说明](../../local_model/README.md)。将 b11871 服务器放在 `local_model/runtime/b11871/`，可用启动参数覆盖路径。

```powershell
conda activate qwen3.5
python -m pip install -r Baseline/requirements.txt
python api/setup_tokenizer.py
python Baseline/download_data.py --dataset searchqa
python Baseline/prepare_hotpotqa_course.py
```

准备好权重和数据后，在同一 PowerShell 中执行。此处给出的默认权重路径对应原实验机；其他机器替换 `$weights`，也可在启动命令指定 `-ServerExe`。

```powershell
$run = Get-Date -Format 'yyyyMMdd-HHmmss'
$weights = 'F:\Document\HKUST\2026fall\NLP\Qwen3.5_9B_4bit\Qwen3.5-9B-Q4_K_M.gguf'
$py = (Get-Command python -CommandType Application | Select-Object -First 1).Source
.\local_model\start.ps1 -PythonExe $py -ModelFile $weights
try {
    python api/run_hotpotqa_course.py --session "Explore/runs/local/replay-hotpot-$run"
    if ($LASTEXITCODE -ne 0) { throw 'HotpotQA run failed' }
    python api/finish_local_searchqa.py --fresh --output "Explore/runs/local/replay-search-$run"
    if ($LASTEXITCODE -ne 0) { throw 'SearchQA run failed' }
} finally {
    .\local_model\stop.ps1
}
```

HotpotQA 读取固定配置并复用同一初始种群；SearchQA 从仓库内固定协议快照启动。最终结果分别保存在新目录里的 `test/results.json` 和 `completed.json`。验证冻结技能后才访问 test，测试题不进入优化器。一次独立复跑可能因采样、后端或硬件而产生不同候选，必须连同自己的原始输出另行分析。

`api/build_final_evidence.py` 供保有原实验完整本机运行目录者从原始文件重新组装本次证据。逐题人工可读标签绑定已记录的响应哈希，新的推理需要重新复核。离线复算请用上面的 `verify_final_project.py`。

## 结论边界

- 这是单次种子的运行。区间对题目重采样，未估计多训练种子的方差。SearchQA 测试 ID 已用于早期 API 实验，后续方法比较按已公开划分的复现解读。
- HotpotQA 项目 test 是从官方 distractor dev 抽出的 1,000 个 ID。先前看过的 30 道 dev 已排除。
- 本机全部已记录服务时段共 7.067 GPU 小时，档案中含 pilot 与本机正式运行。早期队友 API 的实际 GPU 分配未知，无法据此认证全队总工时或换算为 A6000 小时。
- 方法选择为 Part 3 Option A。课程可选的跨数据集迁移与联合技能训练未纳入本研究。两套数据上的 C2b、C2c 是反馈方法的补充对照。
