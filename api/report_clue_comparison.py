"""Summarize three completed local SearchQA runs and verify their published metrics."""
import argparse
import difflib
import html
import shutil
from datetime import datetime
from pathlib import Path

from check_clue_retrieval import replay
from part3_analysis import checked_rows, checkpoint_data, choose_checkpoint
from run_clue_comparison import ROOT, seal
from baseline.analysis import compare
from baseline.budget import TokenCounter
from baseline.config import load_config
from baseline.io import read_json, read_jsonl, write_json


ARMS = ("B", "C2b", "C2c")


def paired(left, right):
    detail = compare(left, right)
    return {"counts": detail["counts"], "em_delta": detail["em_difference_C_minus_B"],
            "bootstrap95": detail["paired_bootstrap_95"],
            "repair_rate": detail["repair_rate"], "regression_rate": detail["regression_rate"],
            "changes": [{"id": r["B"]["id"], "question": r["B"]["question"],
                         "gold": r["B"]["gold_answers"], "before": r["B"]["predicted_answer"],
                         "after": r["C"]["predicted_answer"], "transition": r["transition"]}
                        for r in detail["transitions"] if r["transition"] in ("repaired","regressed")]}


def allocation(ledger, sessions):
    entries = read_jsonl(ledger)
    result = []
    for session in sessions:
        events = [e for e in entries if e["session"] == session]
        assert [e["action"] for e in events] == ["start","stop"]
        start, stop = (datetime.fromisoformat(e["utc"]) for e in events)
        result.append({"session":session,"start_utc":events[0]["utc"],"stop_utc":events[1]["utc"],
                       "gpu_hours":(stop-start).total_seconds() / 3600 * events[0]["gpu_count"]})
    return result


def analyze(session, first_gpu_session):
    init = session / "shared_init"
    protocol = read_json(session / "policy.json")
    cfg = load_config(session / "config.json")
    assert protocol == seal(cfg, session)
    ids = read_json(ROOT / "Baseline/data/prepared/searchqa/dev_ids.json")
    init_summary = read_json(init / "summary.json")
    counter = TokenCounter(cfg["model"]["tokenizer_path"])
    final_rows, points, checkpoints, methods, ledgers = {}, {}, {}, {}, {}
    for arm in ARMS:
        path = session / arm
        summary = read_json(path / "summary.json")
        assert summary["status"] == "completed" and summary["completed_rounds"] == 3 and not summary["demo"]
        for key in ("config_hash","data_hash","code_hash","tokenizer_hash","upstream_commit"):
            assert summary[key] == init_summary[key]
        rows = read_json(path / "final_dev_outputs.json")
        assert set(r["id"] for r in rows) == set(ids) and len(rows) == len(ids) == 30
        assert checked_rows(rows, "searchqa") == summary["dev"]
        assert read_json(path / "shared_initialization.json") == read_json(session / "B/shared_initialization.json")
        ledger = read_jsonl(path / "ledger.jsonl")
        assert {e["phase"] for e in ledger} <= {"initialize", "train", "dev", "optimize"}
        assert {k:sum(e[k] for e in ledger) for k in summary["cost"]} == summary["cost"]
        assert summary["cost"]["calls"] <= cfg["budget"]["max_calls"] and summary["cost"]["generated_tokens"] <= cfg["budget"]["max_generated_tokens"]
        ledgers[arm] = [e for e in ledger if e["physical_calls"]]
        p, records, by_id = checkpoint_data(path, init)
        points[arm], checkpoints[arm], final_rows[arm] = p, by_id, rows
        assert p[-1]["best_id"] == summary["best"]["id"]
        slots = []
        for h in read_json(path / "history.json"):
            g = h["generation"]
            assert read_json(path / f"train_batches/{g}.json") == read_json(session / f"B/train_batches/{g}.json")
            assert len(h["operator_records"]) == 2
            for op in h["operator_records"]:
                child = records.get(op["child_id"])
                assert 1 <= len(op["attempts"]) <= cfg["max_attempts"]
                parent_id = op["parent_ids"][0]
                parent_text = records[parent_id]["content"]
                slots.append({"round":g, "operator":op["operator"],"id":op["child_id"],"parent":op["parent_ids"][0],
                              "accepted":op["accepted"],"attempts":len(op["attempts"]),
                              "errors":[a.get("error","invalid output") for a in op["attempts"] if a["status"] != "ok"],
                              "correct":round(child["val_hard"]*len(ids)) if child else None,
                              "f1":child["val_soft"] if child else None, "retained":op["child_id"] in h["survivor_ids"],
                              "parent_changes":paired(by_id[parent_id],by_id[child["id"]]) if child else None,
                              "skill_diff":list(difflib.unified_diff(parent_text.splitlines(),child["content"].splitlines(),
                                  fromfile=parent_id,tofile=child["id"],lineterm="")) if child else []})
        methods[arm] = {"dev":summary["dev"],"best":summary["best"]["id"],"best_hash":p[-1]["skill_hash"],
                        "cost":summary["cost"],"physical_cost":{k:sum(e[k] for e in ledgers[arm]) for k in ("physical_calls","generated_tokens","prompt_tokens")},
                        "errors":sum(e["status"] != "ok" for e in ledgers[arm]),"curve":p,"candidates":slots,
                        "cached_dev_rows":sum(len(read_json(f)["source_observation_ids"])
                                              for f in (path / "cache_events").glob("*.json"))}
        if arm != "B":
            methods[arm]["retrieval"] = replay(path, arm, counter)
    caps = {key:min(m["cost"][key] for m in methods.values()) for key in ("calls","generated_tokens")}
    matched = {arm:choose_checkpoint(points[arm], caps["calls"], caps["generated_tokens"]) for arm in ARMS}
    ids_final = [{r["id"]:r for r in final_rows[a]} for a in ARMS]
    cases = [{"id":key,"question":ids_final[0][key]["question"],"gold":ids_final[0][key]["gold_answers"],
              "outputs":{a:{k:rows[key][k] for k in ("predicted_answer","em","f1","format_ok")} for a,rows in zip(ARMS,ids_final)}}
             for key in ids]
    state = read_json(session / "baseline_comparator.json")
    sessions = [first_gpu_session,(session / "B_gpu_session.txt").read_text(encoding="utf-8-sig").strip()]
    gpus = allocation(cfg["gpu_ledger_path"],sessions)
    all_physical = read_jsonl(init / "ledger.jsonl") + [e for arm in ARMS for e in ledgers[arm]]
    refs = {arm:paired(final_rows[arm],final_rows["C2c"]) for arm in ("B","C2b")}
    matched_refs = {arm:paired(checkpoints[arm][matched[arm]["best_id"]],checkpoints["C2c"][matched["C2c"]["best_id"]]) for arm in ("B","C2b")}
    return {"scope":"Local SearchQA selection dev, seed 42, exploratory","model":cfg["model"]["id"],
            "precision":cfg["model"]["precision"],"design":protocol["design"],"supplementary_B":state,
            "signature":{k:init_summary[k] for k in ("config_hash","data_hash","code_hash","tokenizer_hash")},
            "method_files":protocol["files"],"methods":methods,"comparisons":refs,"common_cap":caps,
            "matched":matched,"matched_comparisons":matched_refs,"gpu":gpus,"test_evaluations":0,"cases":cases,
            "physical_cost":{k:sum(e[k] for e in all_physical) for k in ("physical_calls","prompt_tokens","generated_tokens")},
            "historical_replay":read_json(session / "historical_train_replay.json")}


def table(columns, rows):
    def cell(x):
        return html.escape(str(x))
    head = "".join(f"<th>{cell(c)}</th>" for c in columns)
    body = "".join("<tr>"+"".join(f"<td>{cell(v)}</td>" for v in row)+"</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def page(r):
    methods = r["methods"]
    score = table(["方法","最终技能","dev 正确","EM","F1","逻辑调用¹","逻辑生成 tokens¹"],[
        [a,m["best"],f'{m["dev"]["correct"]}/30',f'{m["dev"]["em"]:.1%}',f'{m["dev"]["f1"]:.2%}',
         m["cost"]["calls"],f'{m["cost"]["generated_tokens"]:,}'] for a,m in methods.items()])
    curves = table(["方法","初始化","第 1 轮","第 2 轮","第 3 轮","六个新候选的正确数"],[
        [a,*[round(p["em"]*30) for p in m["curve"]],", ".join(str(s["correct"]) if s["accepted"] else "未通过" for s in m["candidates"])]
        for a,m in methods.items()])
    coverage = []
    for name, key in [("C2b","original_reference"),("C2c","expanded_reference")]:
        re = methods[name]["retrieval"]
        coverage.append([name,re["counts"]["selected"],sum(e[key] is not None for e in re["exposures"]),
                         re["distinct_failed_questions_paired"][key],re["distinct_references"][key]])
    details = table(["题目","参考","B 预测","C2b 预测","C2c 预测","EM（B / C2b / C2c）"],[
        [c["question"]," / ".join(c["gold"]),*[c["outputs"][a]["predicted_answer"] for a in ARMS],
         " / ".join(str(int(c["outputs"][a]["em"])) for a in ARMS)] for c in r["cases"]])
    paired_table = table(["C2c 相对","修复","退化","同对","同错","ΔEM（百分点）","配对 bootstrap 95% 区间"],[
        [name,*[v["counts"][k] for k in ("repaired","regressed","both_correct","both_wrong")],f'{v["em_delta"]*100:+.2f}',
         " 至 ".join(f"{x*100:+.2f}" for x in v["bootstrap95"])] for name,v in r["comparisons"].items()])
    matches = table(["方法","共同上限内最后完整轮","dev 正确","剩余调用","剩余生成 tokens"],[
        [a,p["round"],round(p["em"]*30),p["unused_calls"],p["unused_generated_tokens"]] for a,p in r["matched"].items()])
    samples = {(e["failure"],e["expanded_reference"]): e for e in methods["C2c"]["retrieval"]["exposures"] if e["rule"].startswith("target_kind:")}
    samples_table = table(["失败训练题","参考成功训练题","匹配类别"],[[e["question"],e["success_question"],e["rule"].split(":")[-1]] for e in samples.values()]) if samples else "<p>此次实际算子输入未用到新增规则的参照。</p>"
    mutation_details = []
    for arm,m in methods.items():
        for s in m["candidates"]:
            tt = s["parent_changes"]
            annotation = (f'修复 {tt["counts"]["repaired"]}，退化 {tt["counts"]["regressed"]}' if tt else "候选未通过")
            code = html.escape("\n".join(s["skill_diff"]) or ("采用后与父代相同" if s["accepted"] else "未采用"))
            transitions = table(["变动","题目","参考答案","父代预测","候选预测"],[
                [x["transition"],x["question"]," / ".join(x["gold"]),x["before"],x["after"]] for x in tt["changes"]
            ]) if tt and tt["changes"] else ""
            mutation_details.append(f'<details><summary>{arm} · {s["id"]} · {annotation}</summary>{transitions}<pre>{code}</pre></details>')
    gpu = sum(e["gpu_hours"] for e in r["gpu"])
    cost = r["physical_cost"]
    numbers = "，".join(f'{a} {m["dev"]["correct"]}/30' for a,m in methods.items())
    skill_links = " · ".join(f'<a href="skills/{a}.md">{a} 最终技能</a>' for a in ARMS)
    cache_table = table(["阶段","新调用","新生成 tokens","复用的候选 dev 行"], [
        [a,m["physical_cost"]["physical_calls"],m["physical_cost"]["generated_tokens"],m["cached_dev_rows"]]
        for a,m in methods.items()])
    diff = r["comparisons"]["C2b"]["counts"]
    history = r["historical_replay"]["datasets"]
    replay_table = table(["历史 C2b 训练轨迹","失败曝光","原检索配对","新检索配对"],[
        [ds,v["counts"]["selected"],v["counts"]["original_pairs"],v["counts"]["expanded_pairs"]] for ds,v in history.items()])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SearchQA 本机成功参照检索实验</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#faf9f6;color:#20362e;font:16px/1.7 "Microsoft YaHei",system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:30px 20px 56px}}h1{{font-size:clamp(25px,5vw,38px);line-height:1.2}}h2{{margin:36px 0 8px;font-size:22px}}header{{border-bottom:3px solid #48795f;padding-bottom:16px}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px 13px;border-bottom:1px solid #ced8cf;text-align:left}}th{{background:#e2ebe3;white-space:nowrap}}a{{color:#235b47}}summary{{cursor:pointer}}small{{color:#48574d}}</style><main>
<header><small>2026-11-23 · SearchQA · RTX 4070 Ti SUPER · Qwen3.5-9B-Q4_K_M</small>
<h1>给陈述式线索检索成功参照，能多答对几题？</h1><p>{numbers}。C2c 相对同机 C2b 修复 {diff['repaired']} 题，退化 {diff['regressed']} 题。</p></header>
<h2>三轮后的最终结果</h2>{score}<p><small>¹ 每组逻辑用量含一次共享初始化。每组上限为 600 次调用和 700,000 个生成 tokens。</small></p>{paired_table}
<h2>逐轮选中的技能</h2>{curves}<p>每轮用相同的 20 道 train、两个父代和两个候选。模型按原算子生成技能，固定 30 道 dev 按 EM 选优。表中正确数均以 30 为分母。</p>
<details><summary>查看各候选相对父代的技能修改及修复、退化数</summary>{''.join(mutation_details)}</details>
<h2>实际进入优化器的成功参照</h2>{table(["方法","失败曝光","有成功参照","不同被配对失败题","不同成功参照题"],coverage)}
<p>C2c 优先沿用 C2b 已找到的参照。遇到无参照的失败，才从当时的本组训练记忆中检索同一答案类别的正确记录，同一份反馈里优先使用不同题。一个父代被两个算子选中时，反馈曝光计两次。</p>
<details><summary>查看新增规则实际配到的不同题对</summary>{samples_table}</details>
<h2>同一批历史训练记录的离线复核</h2>{replay_table}<p>此处固定了失败题和成功记忆。计数只衡量检索覆盖。下游成绩来自上方本机 SearchQA 实验。</p>
<h2>成本与比较边界</h2><p>三组及共享初始化合计 {cost["physical_calls"]:,} 次物理调用，{cost["prompt_tokens"]:,} 输入 tokens、{cost["generated_tokens"]:,} 生成 tokens。GPU 分配账本记录 {gpu:.3f} GPU·h，包括模型加载与空闲。</p>
{cache_table}<p><small>每个方法只对新技能执行 dev 调用，内容完全相同的技能复用已记账观测。共享初始化的新调用单独计入总数。</small></p>
<p>共同逻辑上限取各组最终用量的较小值，分别为 {r['common_cap']['calls']} 次调用与 {r['common_cap']['generated_tokens']:,} 生成 tokens。按两项上限内最后一个完整轮次比较。</p>{matches}
<p>本次使用本机 4-bit 模型、单一 seed 和参与选择的 30 道 dev。历史托管模型的精度和 revision 未披露，应在各自后端内解读效果。HotpotQA 的新数字仅来自离线训练记录，本次 test 调用为 0。</p>
<h2>逐题结果与运行记录</h2><details><summary>展开 30 道 dev 预测</summary>{details}</details><p><a href="summary.json">完整统计与 GPU 记录</a> · <a href="design.json">运行设计</a> · {skill_links}</p></main></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-gpu-session", required=True)
    args = parser.parse_args()
    r = analyze(args.session, args.first_gpu_session)
    write_json(args.output / "summary.json", r)
    write_json(args.output / "design.json", read_json(args.session / "policy.json"))
    (args.output / "skills").mkdir(exist_ok=True)
    for arm in ARMS:
        shutil.copyfile(args.session / arm / "best_skill.md", args.output / "skills" / (arm + ".md"))
    (args.output / "index.html").write_text(page(r), encoding="utf-8")
    print({a:r["methods"][a]["dev"]["correct"] for a in ARMS},r["physical_cost"])
