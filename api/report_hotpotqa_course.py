"""Verify the six frozen evaluations and build a standalone Chinese HTML report."""
import argparse
import html
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from run_hotpotqa_course import ARMS, CONFIG, DATA, RESULTS, ROOT, SESSION, policy
from baseline.analysis import compare
from baseline.config import load_config
from baseline.data import load_data
from baseline.io import digest, read_json, read_jsonl, write_json
from baseline.protocol import artifact_hashes
from part3_analysis import checked_rows, checkpoint_data, choose_checkpoint

METHODS = {"N": "无技能", "I": "上游初始技能", "B": "原版失败反馈",
           "C": "成功与失败配对", "C2b": "配对、可观察诊断与证据片段",
           "C2c": "C2b 加问题所求类型匹配"}
OPT = ("B", "C", "C2b", "C2c")
COST = ("physical_calls", "prompt_tokens", "generated_tokens", "reasoning_tokens")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def actual_cost(path):
    events = [e for e in read_jsonl(path / "ledger.jsonl") if e.get("physical_calls")]
    return {k: sum(e.get(k, 0) for e in events) for k in COST} | {
        "failed_requests": sum(e["status"] != "ok" for e in events)}


def read_gpu(session_names, config):
    events = read_jsonl(config["gpu_ledger_path"])
    spans = []
    for name in session_names:
        pair = [e for e in events if e["session"] == name]
        require([e["action"] for e in pair] == ["start", "stop"], f"Unclosed GPU session {name}")
        start, end = (datetime.fromisoformat(e["utc"]) for e in pair)
        spans.append({"session": name, "start": start.isoformat(), "end": end.isoformat(),
                      "hours": (end - start).total_seconds() / 3600 * pair[0]["gpu_count"]})
    return spans


def pairing(run):
    counts = Counter()
    refs, failed = set(), set()
    for h in read_json(run / "history.json"):
        for op in h["operator_records"]:
            parent = op["parent_ids"][0]
            path = run / "evaluations" / f"g{h['generation']}" / parent / "train/feedback.json"
            row = read_json(path)
            counts["selected"] += len(row["selected_failure_ids"])
            # Count one exposure per scheduled mutation, not once for each retry.
            packed = row["actual_packets"]
            require(not packed or [p["failure_id"] for p in packed] == row["selected_failure_ids"],
                    "Packed feedback changed the selected failure IDs")
            if not packed:
                require(all(p["pair_type"] == "unpaired" for p in row["packets"]), "Missing success reference")
                counts["unpaired"] += len(row["selected_failure_ids"])
            for packet in packed:
                reference = packet.get("success_reference") or packet.get("success")
                if reference:
                    counts[packet["pair_type"]] += 1
                    failed.add(packet["failure_id"])
                    refs.add(reference.get("id", reference.get("example_id")))
                else:
                    counts["unpaired"] += 1
    return dict(counts) | {"unique_failure": len(failed), "unique_reference": len(refs - {None})}


def audit(session, gpu_sessions):
    config = load_config(CONFIG)
    splits, manifest = load_data(DATA, "hotpotqa")
    require(policy(config, manifest) == read_json(session / "policy.json"), "Pre-registered protocol changed")
    ids = read_json(RESULTS / "split_ids.json")
    known = set(read_json(RESULTS / "excluded_old_dev_ids.json"))
    for split, key in (("train", "train_ids"), ("dev", "val_ids"), ("test", "test_ids")):
        require(ids[key] == [r["id"] for r in splits[split]], "Published ID mismatch")
    require(not known.intersection(ids["test_ids"]) and len(known) == 30, "Test overlaps old dev")
    freeze = read_json(session / "frozen.json")
    require(freeze["freeze_id"] == digest({k: v for k, v in freeze.items() if k != "freeze_id"}), "Invalid freeze")
    require(freeze["signature"]["data_hash"] == digest(manifest) and freeze["test_ids_hash"] == digest(ids["test_ids"]), "Split mismatch")
    test_report = read_json(session / "test/results.json")
    require(freeze["freeze_id"] == test_report["freeze_id"] and set(freeze["conditions"]) == set(ARMS), "Incomplete test")
    report = {"freeze_id": freeze["freeze_id"], "run_path": Path(os.path.relpath(session.resolve(), RESULTS)).as_posix(),
              "data": manifest, "config": config,
              "preflight": read_json(session / "preflight.json"), "gpu": read_gpu(gpu_sessions, config),
              "policy": read_json(session / "policy.json"), "arms": {}, "comparison": {}, "matched": {}}
    cases = {split: {r["id"]: {"id": r["id"], "question": r["question"], "gold": r["answers"],
                                "level": r["metadata"]["level"], "predictions": {}} for r in splits[split]}
             for split in ("dev", "test")}
    all_rows = {"dev": {}, "test": {}}
    physical = [actual_cost(session / "shared_init")]
    checkpoints = {}
    for arm in ARMS:
        run, test = session / arm, session / "test" / arm
        summary = read_json(run / "summary.json")
        frozen = freeze["conditions"][arm]
        require(artifact_hashes(run) == frozen["artifacts"], f"{arm} frozen evidence changed")
        skill = (run / "best_skill.md").read_text(encoding="utf-8")
        require(digest(skill) == frozen["skill_hash"] and not summary["demo"], "Unexpected skill or model")
        stats = {"method": METHODS[arm], "best_id": summary["best"]["id"], "skill_hash": digest(skill),
                 "skill": skill, "development_cost": summary["cost"], "test_cost": actual_cost(test)}
        for split, file in (("dev", run / "final_dev_outputs.json"), ("test", test / "outputs.json")):
            rows = read_json(file)
            require([r["id"] for r in rows] == list(cases[split]), f"{arm} {split} incomplete or out of order")
            metrics = checked_rows(rows, "hotpotqa")
            ledger_stats = summary["dev"] if split == "dev" else test_report["conditions"][arm]
            require(all(ledger_stats[k] == v for k, v in metrics.items()), "Score recomputation mismatch")
            by_level = {}
            for row, item in zip(rows, splits[split]):
                require(not row["context_truncated"] and row["visible_context"] == item["context"]
                        and row["gold_answers"] == item["answers"] and row["skill_hash"] == digest(skill), "Input mismatch")
                level = item["metadata"]["level"]
                by_level.setdefault(level, []).append(row)
                cases[split][row["id"]]["predictions"][arm] = {
                    "answer": row["predicted_answer"], "em": row["em"], "f1": row["f1"]}
            stats[split] = metrics
            stats[split + "_by_level"] = {level: {"n": len(group), "correct": sum(r["hard"] for r in group),
                "f1": sum(r["f1"] for r in group) / len(group)} for level, group in by_level.items()}
            all_rows[split][arm] = rows
        physical.extend([actual_cost(run), stats["test_cost"]])
        if arm in OPT:
            points, _, _ = checkpoint_data(run, session / "shared_init")
            checkpoints[arm] = points
            stats["curve"] = [{"round": p["round"], "correct": round(p["em"] * 150)} for p in points]
            stats["candidates"] = []
            for h in read_json(run / "history.json"):
                children = {c["id"]: c for c in h["candidates"]}
                for op in h["operator_records"]:
                    child = children.get(op["child_id"])
                    stats["candidates"].append({"round": h["generation"], "id": op["child_id"], "parent": op["parent_ids"],
                        "accepted": op["accepted"], "survived": op["child_id"] in h["survivor_ids"],
                        "correct": round(child["val_hard"] * 150) if child else None, "attempts": len(op["attempts"])})
        if arm in ("C", "C2b", "C2c"):
            stats["pairing"] = pairing(run)
        report["arms"][arm] = stats
    report["physical_total"] = {k: sum(e[k] for e in physical) for k in physical[0]}
    calls = min(report["arms"][a]["development_cost"]["calls"] for a in OPT)
    tokens = min(report["arms"][a]["development_cost"]["generated_tokens"] for a in OPT)
    report["matched"] = {"calls": calls, "tokens": tokens,
                         "arms": {arm: choose_checkpoint(checkpoints[arm], calls, tokens) for arm in OPT}}
    for split in ("dev", "test"):
        report["comparison"][split] = {}
        for arm in ("C", "C2b", "C2c"):
            delta = compare(all_rows[split]["B"], all_rows[split][arm])
            report["comparison"][split][arm] = {k: v for k, v in delta.items() if k != "transitions"}
    report["additional_test_comparisons"] = {}
    for base in ("N", "C2b"):
        delta = compare(all_rows["test"][base], all_rows["test"]["C2c"])
        report["additional_test_comparisons"][base] = {k: v for k, v in delta.items() if k != "transitions"}
    return report, {key: list(rows.values()) for key, rows in cases.items()}


def table(headers, rows):
    def cell(v):
        return html.escape(str(v))
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{cell(h)}</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(v)}</td>' for v in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def page(report, cases):
    pct = lambda n: '—' if n is None else f'{n:.2%}'
    arms = report["arms"]
    pair = report["comparison"]["test"]["C"]
    low, high = pair["paired_bootstrap_95"]
    direction = ("两端均高于零" if low > 0 else "两端均低于零" if high < 0 else "涵盖零")
    data = report["data"]
    p = report["preflight"]
    total = report["physical_total"]
    gpu = sum(s["hours"] for s in report["gpu"])
    run_link = html.escape(report["run_path"], quote=True)
    top_correct = max(m["test"]["correct"] for m in arms.values())
    top_names = "、".join(a for a,m in arms.items() if m["test"]["correct"] == top_correct)
    no_skill_gap = arms["B"]["test"]["correct"] - arms["N"]["test"]["correct"]
    overview = table(["组别", "优化器输入", "验证正确 / 150", "验证 F1", "测试正确 / 1,000", "测试 EM", "测试 F1", "测试 P50 / P95 秒"],
        [[a, METHODS[a], m["dev"]["correct"], pct(m["dev"]["f1"]), m["test"]["correct"], pct(m["test"]["em"]),
          pct(m["test"]["f1"]), f'{m["test"]["p50_latency_seconds"]:.2f} / {m["test"]["p95_latency_seconds"]:.2f}'] for a, m in arms.items()])
    pairs = table(["集合", "比较", "修复", "退化", "同对", "同错", "修复率", "退化率", "ΔEM（百分点）", "配对 95% 区间（百分点）"],
        [[split, a+" 相对 B", *[c["counts"][k] for k in ("repaired", "regressed", "both_correct", "both_wrong")],
          pct(c["repair_rate"]), pct(c["regression_rate"]), f'{100*c["em_difference_C_minus_B"]:+.2f}',
          ' 至 '.join(f'{100*v:+.2f}' for v in c["paired_bootstrap_95"])] for split, comparisons in report["comparison"].items() for a,c in comparisons.items()])
    curve = table(["组别", "初始化", "第 1 轮", "第 2 轮", "第 3 轮", "最终技能", "有效候选 / 6"],
        [[a, *[p["correct"] for p in arms[a]["curve"]], arms[a]["best_id"], sum(c["accepted"] for c in arms[a]["candidates"])] for a in OPT])
    levels = table(["组别","easy 正确 / 50","medium 正确 / 50","hard 正确 / 50"],
        [[a,*[m["dev_by_level"][level]["correct"] for level in ("easy","medium","hard")]] for a,m in arms.items()])
    costs = table(["组别","开发逻辑调用","开发物理调用","开发逻辑生成 tokens","测试物理调用","测试输入 tokens","测试生成 tokens","测试平均秒","测试格式异常"],
        [[a,m["development_cost"]["calls"],m["development_cost"]["physical_calls"],f'{m["development_cost"]["generated_tokens"]:,}',
          m["test_cost"]["physical_calls"],f'{m["test_cost"]["prompt_tokens"]:,}',f'{m["test_cost"]["generated_tokens"]:,}',
          f'{m["test"]["mean_latency_seconds"]:.2f}',m["test"]["format_failures"]] for a,m in arms.items()])
    coverage = table(["组别","送入变异的失败例次数","同题配对","跨题配对","无参照","配对覆盖率","不同已配对失败题","不同成功参照"],
        [[a,c["selected"],c.get("same_question",0),c.get("similar_task",0),c.get("unpaired",0),
          pct((c.get("same_question",0)+c.get("similar_task",0))/c["selected"] if c["selected"] else None),
          c["unique_failure"],c["unique_reference"]] for a in ("C","C2b","C2c") for c in [arms[a]["pairing"]]])
    observed = []
    for arm in ("C", "C2b", "C2c"):
        same = sum(r["predictions"][arm]["answer"] == r["predictions"]["B"]["answer"] for r in cases["test"])
        common = arms[arm]["skill_hash"] == arms["B"]["skill_hash"]
        observed.append([arm, "相同" if common else "不同", same, 1000 - same,
                         f'{100*(arms[arm]["test"]["f1"]-arms["B"]["test"]["f1"]):+.2f}'])
    agreement = table(["相对 B", "最终技能文本", "预测字符串相同", "预测字符串不同", "ΔF1（百分点）"], observed)
    ext = report["comparison"]["test"]["C2c"]
    raw = report["additional_test_comparisons"]["N"]
    interval = lambda c: " 至 ".join(f"{v*100:+.1f}" for v in c["paired_bootstrap_95"])
    extra = table(["测试对照", "修复", "退化", "ΔEM（百分点）", "配对 95% 区间（百分点）"],
                  [["C2c 相对 "+a, c["counts"]["repaired"], c["counts"]["regressed"],
                    f'{100*c["em_difference_C_minus_B"]:+.2f}', interval(c)]
                   for a,c in report["additional_test_comparisons"].items()])
    examples = []
    for method in ("C", "C2c"):
        for label, before, after in (("修复", 0, 1), ("退化", 1, 0)):
            for case in cases["test"]:
                b, c = (case["predictions"][a] for a in ("B", method))
                if b["em"] == before and c["em"] == after:
                    examples.append([method+" "+label, case["id"], case["question"], " / ".join(case["gold"]), b["answer"], c["answer"]])
                    break
    examples_html = ('<h3>成对预测实例</h3><p>每类展示按固定测试顺序遇到的第一题。完整清单可在逐题视图筛选。</p>'
                     + table(["变化","原始 ID","问题","参考答案","B","对照方法"], examples)) if examples else ""
    matched=report["matched"]
    ceiling = table(["组别","共同上限内最后完整轮","验证正确 / 150","累计调用","累计生成 tokens"],
        [[a,m["round"],round(m["em"]*150),m["cost"]["calls"],f'{m["cost"]["generated_tokens"]:,}'] for a,m in matched["arms"].items()])
    skills=''.join('<details><summary>'+html.escape(a+' · '+arms[a]["best_id"])+'</summary><pre>'+
                   html.escape(arms[a]["skill"] or '无技能')+'</pre></details>' for a in ARMS)
    details=''.join('<details><summary>'+a+' · 六个候选</summary>'+table(["轮次","候选","父代","生成尝试","通过校验","进入幸存组","验证正确 / 150"],
        [[c["round"],c["id"],', '.join(c["parent"]),c["attempts"],"是" if c["accepted"] else "否",
          "是" if c["survived"] else "否",c["correct"] if c["correct"] is not None else "—"] for c in arms[a]["candidates"]])+'</details>' for a in OPT)
    evidence = json.dumps(cases,ensure_ascii=False,separators=(',',':')).replace('<',r'\u003c')
    css = """*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f8f7f3;color:#182a34;font:16px/1.65 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1220px;padding:40px clamp(16px,4vw,62px) 80px;margin:auto}h1,h2,h3{line-height:1.28;color:#102a35}h1{font-size:clamp(29px,4vw,46px);margin:9px 0 18px}h2{margin:38px 0 12px;font-size:24px}h3{font-size:17px}a{color:#066a60}nav,.links{display:flex;gap:18px;flex-wrap:wrap}nav{padding:15px 0;border-block:1px solid #cbd3cf}p{max-width:92ch}.lead{font-size:19px;border-left:4px solid #058172;padding-left:16px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;background:white;font-size:14px}th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid #d9e0dc}th{background:#eaf1ef}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}details{padding:8px 0;border-bottom:1px solid #d2d8d4}summary{cursor:pointer}label,button,select,input{font:inherit}label{display:inline-block;margin:3px 10px 4px 0}select,input,button{padding:5px 8px;border:1px solid #9daea9;border-radius:3px;background:white}button{cursor:pointer}small{color:#49605e}code{overflow-wrap:anywhere}.tools{margin:10px 0}.case{border-top:1px solid #cbd7d2;padding:12px 0}.case h3{margin:0}.tag{font-size:13px;color:#3b5b57}.pager{display:flex;align-items:center;gap:12px}@media print{nav,.tools,.pager{display:none}main{max-width:none;padding:0}details{break-inside:avoid}}"""
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>课程 HotpotQA · Baseline 与 Explore 对照</title><style>{css}</style><main>
<header><small>AIAA4051 · Project 5 · 本机 Qwen3.5-9B Q4_K_M</small><h1>HotpotQA 课程划分上的技能优化对照</h1>
<p class="lead">在相同的 1,000 道冻结测试题上，B 答对 {arms['B']['test']['correct']} 道，原版 C 答对 {arms['C']['test']['correct']} 道。C 修复 {pair['counts']['repaired']} 道 B 的错题，同时让 {pair['counts']['regressed']} 道原本正确的题变错。ΔEM 为 {pair['em_difference_C_minus_B']*100:+.2f} 个百分点，配对区间{direction}。</p></header>
<nav><a href="#scores">主结果</a><a href="#training">训练与反馈</a><a href="#cost">预算与延迟</a><a href="#questions">逐题对照</a><a href="#reproduce">数据与复现</a></nav>
<section id="scores"><h2>验证与冻结测试</h2>{overview}<p>本次单种子测试中，{top_names} 得到最高 EM，答对 {top_correct} 道。B 相对无技能 N 的正确数变化为 {no_skill_gap:+d} 道。验证集参与技能选择，三个难度等量。测试题全部是官方 distractor 开发集的 hard 题；划分时排除了此前实验看过的 30 道开发题。EM 为归一化精确匹配，F1 衡量答案词元重合，yes/no/noanswer 使用 HotpotQA 的特殊评分规则。</p><h3>同一道题上，哪些收益被退化抵消</h3>{pairs}
<p>修复率以 B 的错题为分母，退化率以 B 的正确题为分母。区间使用 2,000 次题目级配对 bootstrap，随机种子 42。训练只用一个种子，区间仅描述此次评测题目抽样的不确定性。预先指定的主比较是 B 对原版 C，C2b、C2c 为扩展比较。</p>
<h3>C2c 的增益取决于比较对象</h3><p>C2c 相对 B 净变化 {ext["counts"]["repaired"]-ext["counts"]["regressed"]:+d} 题；相对无技能 N 净变化 {raw["counts"]["repaired"]-raw["counts"]["regressed"]:+d} 题。与 N 的差值区间为 {interval(raw)} 个百分点。扩展比较均为同一训练种子上的描述性分析，区间未作多重比较校正。</p>{extra}
<h3>最终技能与预测是否相同</h3>{agreement}{examples_html}<h3>验证集分难度表现</h3>{levels}</section>
<section id="training"><h2>三轮训练如何选出最终技能</h2>{curve}<p>每轮使用同一批 100 道训练题评估两个父代。每种方法完整覆盖 300 个不同训练 ID，三轮无重复。每轮尝试两个子代，失败样本抽样 k=3，每个算子最多尝试两次。候选在全部 150 道验证题上评分，按固定上游规则保留两个个体。相同技能的确定性验证结果按原协议复用。</p>{details}
<h3>进入优化器的成功参照</h3>{coverage}<p>同一父代若被两个变异算子使用，其反馈计两次曝光。无参照时，C 可回退到原版失败反馈，C2b 和 C2c 仍给出训练失败诊断。配对只访问本方法观察到的训练样本。覆盖率反映参照出现的频率，单靠此指标无法确定分数差异的原因。</p><h3>最终技能全文</h3>{skills}</section>
<section id="cost"><h2>调用成本与速度</h2>{costs}<p>共享初始化在每个优化组的逻辑预算中计入，在上述分组物理调用中单列。全流程含一次共享初始化，共 {total['physical_calls']:,} 次实际请求，{total['prompt_tokens']:,} 输入和 {total['generated_tokens']:,} 生成 tokens，失败请求 {total['failed_requests']} 次。独占模型服务按实际启动到关闭累计 {gpu:.3f} GPU·h。延迟为单题串行客户端观测值。</p>
<h3>按共同开发预算回看验证集</h3><p>在四个优化组各自实际消耗中，取最小调用数 {matched['calls']:,} 和最小生成 token 数 {matched['tokens']:,} 作为共同上限，回看同时满足两项上限的最后一个完整轮次。</p>{ceiling}</section>
<section id="questions"><h2>逐题核对</h2><p>可按方法查看全部答案、相对 B 的修复或退化，以及该方法的全部错题。切换选项只筛选已保存的结果。</p>
<div class="tools"><label>集合 <select id="split"><option value="test">Test · 1,000</option><option value="dev">Validation · 150</option></select></label><label>方法 <select id="arm"><option>C</option><option>C2b</option><option>C2c</option><option>B</option><option>I</option><option>N</option></select></label><label>筛选 <select id="filter"><option value="all">全部</option><option value="fix">修复 B 错题</option><option value="loss">相对 B 退化</option><option value="wrong">该方法答错</option></select></label><label>搜索 <input id="search" placeholder="题目或原始 ID"></label></div>
<div class="pager"><button id="prev">上一页</button><span id="count"></span><button id="next">下一页</button></div><div id="list"></div></section>
<section id="reproduce"><h2>数据、冻结与复现</h2><p>使用 HotpotQA v1.1 train 和 v1 distractor dev 的 Hugging Face 镜像，版本 <code>{data['source_revision']}</code>。从 {data['source_counts']['train']:,} 道官方训练题中依 easy、medium、hard 顺序，对每层按原始 ID 排序后用 Python Random(42) 无放回抽 150 道，前 100 归 train，后 50 归 validation。官方 dev 共 {data['source_counts']['validation']:,} 道，剔除旧 30 题，在余下 7,375 题中无放回抽 1,000 道作为课程 test。三个 ID 集合两两交集均为零。</p>
<p>每次推理提供所有标题和段落，保留原顺序；模型只收到技能、上下文和问题。推理温度 0，单题最多 2,048 生成 tokens，外部检索关闭。最长无技能输入为 {max(p['max_unskilled_input_tokens'].values()):,} tokens，预留最长技能、回复和边界后为 {p['worst_case_with_skill_completion_and_reserve']:,}，低于服务窗口 {p['context_window']:,}。六组同时冻结，冻结摘要 <code>{report['freeze_id']}</code>。开发与测试预算各组独立，上限均为 2,000 次调用、2,000,000 生成 tokens、12 小时。</p>
<div class="links"><a href="split_ids.json">三份原始 ID</a><a href="excluded_old_dev_ids.json">排除的 30 题</a><a href="split_manifest.json">采样与来源校验</a><a href="summary.json">指标、模型与方法签名</a><a href="predictions.json">完整逐题预测</a><a href="{run_link}/frozen.json">本机冻结清单</a><a href="../../Baseline/prepare_hotpotqa_course.py">划分脚本</a><a href="../../api/run_hotpotqa_course.py">实验入口</a></div></section>
<script type="application/json" id="cases">{evidence}</script>"""
    js = """const cases=JSON.parse(document.querySelector('#cases').textContent), el=id=>document.getElementById(id);let page=0;function node(tag,text,cls=''){const n=document.createElement(tag);n.textContent=text;n.className=cls;return n}function draw(){let a=el('arm').value,f=el('filter').value,q=el('search').value.trim().toLowerCase();let rows=cases[el('split').value].filter(r=>{const p=r.predictions[a],b=r.predictions.B;return (!q||(r.id+' '+r.question).toLowerCase().includes(q))&&(f==='all'||(f==='wrong'&&!p.em)||(f==='fix'&&p.em&&!b.em)||(f==='loss'&&!p.em&&b.em))});const size=8;page=Math.max(0,Math.min(page,Math.ceil(rows.length/size)-1));el('count').textContent=rows.length?'共 '+rows.length+' 题 · 第 '+(page+1)+' / '+Math.ceil(rows.length/size)+' 页':'0 道符合条件的题';el('prev').disabled=page===0;el('next').disabled=(page+1)*size>=rows.length;el('list').replaceChildren();for(const r of rows.slice(page*size,(page+1)*size)){let d=node('article','','case');d.append(node('span',r.level+' · '+r.id,'tag'),node('h3',r.question),node('div','参考答案　'+r.gold.join(' / ')));for(const k of [...new Set(['B',a])]){let p=r.predictions[k];d.append(node('div',k+'　'+p.answer+'　· EM '+p.em+'　F1 '+p.f1.toFixed(3)))}el('list').append(d)}}for(const id of ['arm','filter','split','search'])el(id).addEventListener('input',()=>{page=0;draw()});el('prev').onclick=()=>{page--;draw()};el('next').onclick=()=>{page++;draw()};draw();"""
    return body+'<script>'+js+'</script></main></html>'


def progress(session):
    if (RESULTS / "summary.json").exists() and (RESULTS / "index.html").exists():
        return
    entries = []
    names = ("shared_init", *ARMS)
    for name in names:
        folder = session / name
        summary = read_json(folder / "summary.json") if (folder / "summary.json").exists() else {}
        ledger = folder / "ledger.jsonl"
        events = read_jsonl(ledger) if ledger.exists() else []
        test_dir = session / "test" / name / "predictions"
        count = sum(1 for _ in test_dir.glob("*/result.json")) if test_dir.exists() else 0
        status = summary.get("status", "运行中" if folder.exists() else "等待")
        validation = summary.get("dev", {})
        entries.append([name, status, sum(e.get("physical_calls", 0) for e in events),
                        f'{validation["correct"]} / {validation["n"]}' if validation else "—",
                        f"{count} / 1,000" if name in ARMS else "—"])
    meta = read_json(RESULTS / "split_manifest.json")
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="refresh" content="60"><title>课程 HotpotQA · 实验进度</title><style>body{{max-width:900px;margin:6% auto;padding:0 18px;font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#173338;background:#f8f7f3}}h1{{line-height:1.25}}a{{color:#076b60}}.scroll{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{text-align:left;padding:8px 12px;border-bottom:1px solid #d7deda}}small{{color:#516865}}</style>
    <main><small>AIAA4051 · Project 5 · {now}</small><h1>HotpotQA 课程实验进度</h1>
    <p>已固定 {meta['counts']['train']} 道训练题、{meta['counts']['dev']} 道验证题和 {meta['counts']['test']:,} 道测试题。测试池已排除旧开发集的 30 个 ID。各方法共用完整上下文，所有技能冻结后开始测试。</p>
    {table(["阶段", "运行状态", "开发实调次数", "当前验证正确数", "已写入测试预测"], entries)}
    <p>页面每分钟刷新。六组测试全部结束并完成逐题复算后，此处将显示最终 HTML 对比报告。</p>
    <p><a href="split_ids.json">固定题目 ID</a> · <a href="excluded_old_dev_ids.json">旧 30 题</a> · <a href="split_manifest.json">来源与分层</a></p></main></html>"""
    temporary = RESULTS / "progress.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(RESULTS / "index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, default=SESSION)
    ap.add_argument("--gpu-session", action="append")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if args.progress:
        progress(args.session)
        raise SystemExit(0)
    if not args.gpu_session:
        ap.error("Final reporting requires the measured GPU service session")
    data, cases = audit(args.session, args.gpu_session)
    write_json(RESULTS / "summary.json", data)
    write_json(RESULTS / "predictions.json", cases)
    (RESULTS / "index.html").write_text(page(data, cases), encoding="utf-8")
    print(f"Verified and wrote {RESULTS / 'index.html'}", flush=True)

