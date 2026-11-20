"""Build a self-contained, audited HTML report from two completed baseline runs."""
import argparse
from collections import Counter
from datetime import datetime
from html import escape as esc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Baseline"))
from baseline.analysis import compare
from baseline.evaluation import aggregate, score
from baseline.io import read_json as rd, read_jsonl as jl, digest

STAGES = ("pilot", "shared_init", "N", "I", "B")
LABELS = {"N": "无技能", "I": "原始技能", "B": "GraphSkillEvo 优化技能"}


def pct(x):
    return f"{x * 100:.2f}%"


def percentile(values, fraction):
    values=sorted(values)
    position=(len(values)-1)*fraction
    lower=int(position);upper=min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def load_run(run, dataset):
    run = Path(run).resolve()
    config = rd(run / "B/config.json")
    assert config["dataset"] == dataset
    assert config.get("qa_context_max_tokens") == 4096 and config["qa_max_tokens"] == 2048, "Report requires the latest token protocol"
    summaries = {c: rd(run / c / "summary.json") for c in ("N", "I", "B")}
    rows = {c: rd(run / c / "final_dev_outputs.json") for c in summaries}
    for c, s in summaries.items():
        assert s["status"] == ("completed" if c == "B" else "static_complete"), (run, c, s["status"])
        assert aggregate(rows[c]) == s["dev"]
        assert s["dev"]["infrastructure_errors"] == 0
        for row in rows[c]:
            expected = score(row["raw_response"], row["gold_answers"], dataset)
            assert expected["em"] == row["em"] and expected["f1"] == row["f1"]
    for key in ("code_hash", "config_hash", "data_hash", "tokenizer_hash"):
        assert len({s[key] for s in summaries.values()}) == 1
    history = rd(run / "B/history.json")
    assert len(history) == summaries["B"]["completed_rounds"] == config["rounds"]
    assert (run / "B/best_skill.md").read_text(encoding="utf-8") == summaries["B"]["best"]["content"]
    initial = rd(run / "shared_init/population.json")
    curve = [max(p["val_hard"] for p in initial)] + [next(p["val_hard"] for p in h["parent_population"] + h["candidates"] if p["id"] == h["best_id"]) for h in history]
    stages, calls = [], []
    for stage in STAGES:
        events = jl(run / stage / "ledger.jsonl")
        physical = [e for e in events if e.get("physical_calls", 0)]
        record = {"stage": stage, "logical": sum(e.get("calls", 0) for e in events),
                  "physical": sum(e.get("physical_calls", 0) for e in events),
                  "prompt": sum(e.get("prompt_tokens", 0) for e in physical),
                  "generated": sum(e.get("generated_tokens", 0) for e in physical),
                  "seconds": sum(e.get("wall_seconds", 0) for e in physical),
                  "errors": sum(e.get("status") == "error" for e in physical)}
        stages.append(record)
        calls.extend(dict(c, folder=stage) for c in jl(run / stage / "calls.jsonl"))
    return dict(run=run, dataset=dataset, config=config, summaries=summaries, rows=rows, history=history,
                initial=initial, curve=curve, stages=stages, calls=calls,
                pairs={c: compare(rows[c], rows["B"]) for c in ("N", "I")})


CSS = """
:root{--ink:#18342f;--muted:#60746d;--line:#dbe3dd;--bg:#f5f7f2;--accent:#116c57;--amber:#935221}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}a{color:var(--accent);text-underline-offset:3px}aside{position:fixed;inset:0 auto 0 0;width:220px;padding:35px 26px;border-right:1px solid var(--line);background:#edf2eb}aside b{font-size:20px}nav{display:grid;gap:16px;margin-top:35px}nav a{text-decoration:none;font-size:13px}main{max-width:1400px;margin-left:220px;padding:48px 54px}h1{font-size:40px;line-height:1.3;letter-spacing:-1px;max-width:850px;margin:18px 0}h2{font-size:25px;margin:0 0 18px}h3{font-size:18px;margin:12px 0}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--accent)}.lede{font-size:18px;max-width:850px}.muted,small{color:var(--muted)}small{display:block;font-size:12px}.badge{display:inline-block;border:1px solid #bdcfc1;padding:4px 10px;border-radius:20px;font-size:12px;margin:4px 6px 0 0}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin:30px 0}.kpi{padding:20px 0;border-top:3px solid var(--accent)}.kpi strong{display:block;font-size:39px;font-weight:600}.kpi span{font-size:13px}section{margin-top:52px;scroll-margin-top:20px}.panel{background:white;border:1px solid var(--line);border-radius:8px;padding:24px;margin:18px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.two>.panel{min-width:0;margin:0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}th{color:var(--muted);font-weight:500}td:first-child{font-weight:600}.note{border-left:3px solid var(--accent);padding:12px 18px;background:#edf4ef;margin:18px 0}.warn{border-left-color:var(--amber);background:#f7efe4}.negative{color:#a23f2b}.positive{color:var(--accent)}.source{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.barrow{display:grid;grid-template-columns:26px 1fr 72px;gap:12px;align-items:center;margin:16px 0;font-size:13px}.bartrack{height:22px;background:#edf1ec}.barfill{height:100%;background:#9cae9f}.barrow:last-of-type .barfill{background:var(--accent)}.axis{font-size:12px;fill:var(--muted)}.value{font-size:13px;fill:var(--accent)}svg{display:block;width:100%;height:auto}svg polyline{fill:none;stroke:var(--accent);stroke-width:3}.filters{display:flex;flex-wrap:wrap;gap:10px}select,input,button{font:inherit;padding:9px 12px;border:1px solid #c5d2c7;border-radius:5px;background:white;color:var(--ink)}input{flex:1;min-width:180px}button{cursor:pointer}.case{border-top:1px solid var(--line);padding:22px 0}.casehead{display:flex;justify-content:space-between;gap:14px;font-size:12px;color:var(--muted)}.question{font-size:16px;font-weight:600;margin:9px 0}.answers{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.answer{background:#f0f5ef;padding:10px 14px;border-radius:4px;overflow-wrap:anywhere}.bad{background:#faf0e9}.mono,code,pre{font-family:Consolas,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:17px;background:#f1f4f0;max-height:450px;overflow:auto}details{margin:12px 0}summary{cursor:pointer;color:var(--accent)}.hash{word-break:break-all;font-size:11px}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin:24px 0}.empty{padding:25px;color:var(--muted)}.foot{margin:45px 0 0;padding-top:20px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}ul{padding-left:22px}li{margin:8px 0}.candidate{font-size:12px} @media(max-width:1050px){aside{display:none}main{margin:0;padding:30px}}@media(max-width:650px){main{padding:22px 16px}h1{font-size:30px}.two,.kpis,.answers{grid-template-columns:1fr}.kpis{gap:5px}.panel{padding:17px}.toolbar{align-items:flex-start}th,td{padding:10px}.casehead{display:block}}@media print{aside,.toolbar,.filters{display:none}main{margin:0;padding:0;max-width:none}body{background:white;font-size:11px}.panel{break-inside:avoid}h1{font-size:28px}section{margin-top:25px}pre{max-height:none}.case{break-inside:avoid}}
"""


def evolution_chart(values, name):
    points = [(70 + i * 135, 210 - v * 160) for i, v in enumerate(values)]
    grid = ''.join(f'<line x1="48" x2="520" y1="{210-v*160}" y2="{210-v*160}" stroke="#dbe3dd"/><text x="12" y="{215-v*160}" class="axis">{int(v*100)}</text>' for v in (0, .5, 1))
    marks = ''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="#116c57"/><text x="{x}" y="{y-13}" text-anchor="middle" class="value">{v*100:.1f}</text><text x="{x}" y="238" text-anchor="middle" class="axis">{label}</text>' for (x,y),v,label in zip(points,values,("初始化","第 1 轮","第 2 轮","第 3 轮")))
    return f'<svg viewBox="0 0 550 275" role="img" aria-label="{name} 每轮最佳 dev EM"><text x="12" y="20" class="axis">dev EM（%）</text>{grid}<polyline points="{" ".join(f"{x},{y}" for x,y in points)}"/>{marks}<text x="280" y="267" text-anchor="middle" class="axis">优化阶段</text></svg>'


def build(search, hotpot, output, checks=(), review=None, context_audit=None):
    output = Path(output).resolve()
    data = [load_run(search,"searchqa"), load_run(hotpot,"hotpotqa")]
    context_html = ''
    if context_audit:
        context_stats = rd(context_audit)
        context_html = '<div class="panel"><h3>文档上下文预算核对</h3>'+table(
            ["数据集", "split", "题数", "完整文档最大 tokens", "旧规则截断题数", "新规则截断题数"],
            [[dataset, split, str(s["n"]), str(s["max_full_context_tokens"]), str(s["old_6000_char_truncated"]), str(s["new_4096_token_truncated"])]
             for dataset, splits in context_stats.items() for split, s in splits.items()])+ '<small>使用固定本地 tokenizer 对实际 train/dev 文档重新计数，不包括问题、系统提示或技能。完整请求仍须预留输出与聊天模板余量。未读取 test 内容。</small></div>'
    check_rows = []
    for folder in checks:
        check = rd(Path(folder) / "summary.json")
        check_rows.append([esc(Path(folder).name), esc(check["status"]), esc(check.get("error_type", "—")),
                           str(check["cost"]["physical_calls"]), f'{check["cost"]["prompt_tokens"]:,}', f'{check["cost"]["generated_tokens"]:,}'])
    checks_html = ('<div class="panel"><h3>额外 API 检查成本（不计入 Baseline 主表）</h3>' +
                   table(["检查目录", "状态", "本地校验异常", "物理请求", "输入 tokens", "生成 tokens"], check_rows) +
                   '<p>请求成功与图校验通过是两个不同条件。此次合成检查的图变异生成了未引用节点，被本地校验拒绝；这次失败保留在检查记录中。后续正式 pilot 与全部实验阶段的实际结果独立报告。</p></div>') if check_rows else ''
    def link(path,label):
        assert Path(path).exists(), path
        return f'<a href="{esc(os.path.relpath(path,output.parent).replace(chr(92),chr(47)),quote=True)}">{esc(label)}</a>'
    old_paths = {"searchqa": ROOT / "Baseline/runs/api/searchqa/20261106-114330-583", "hotpotqa": ROOT / "Baseline/runs/api/hotpotqa/20261106-172212-932"}
    metrics, quality, costs, pairs, versions, old_comp, explorers, rounds, skills, sources = [], [], [], [], [], [], [], [], [], []
    operator_issues = []
    token_distribution = []
    bar_panels, curve_panels, findings = [], [], []
    totals = {k: sum(s[k] for d in data for s in d["stages"]) for k in ("physical","prompt","generated","seconds","errors")}
    for d in data:
        name = "SearchQA" if d["dataset"] == "searchqa" else "HotpotQA"
        run, summary = d["run"], d["summaries"]
        bm, nm, im = (summary[c]["dev"] for c in ("B", "N", "I"))
        gain = (d["curve"][-1] - d["curve"][0]) * 100
        findings.append(f'<p><b>{name}</b>：B 答对 {bm["correct"]}/{bm["n"]}，相对 N 的正确题数净变化 {bm["correct"]-nm["correct"]:+d}，相对 I 为 {bm["correct"]-im["correct"]:+d}；三轮优化相对初始化最佳 EM 的额外变化为 {gain:+.2f} pp。'+
                        ('本次优化没有超过初始化最佳，不能把 B 相对 N 的全部增益归为三轮进化的贡献。' if gain == 0 else '该变化发生在参与候选选择的 dev 集上，仍需独立测试确认。')+'</p>')
        if nm["correct"] == im["correct"]:
            nc,ic=(next(s for s in d["stages"] if s["stage"]==c) for c in ("N","I"))
            findings.append(f'<p>{name} 的 N 与 I EM 相同；两组 dev 实际生成 tokens 分别为 {nc["generated"]:,} 与 {ic["generated"]:,}。加入固定技能并不自动带来准确率收益，输出成本需要一并比较。</p>')
        for relative in ("pilot/initialization_errors.json", "pilot/mutation_errors.json", "shared_init/initialization_errors.json"):
            path=run/relative
            if path.exists():
                for error in rd(path):
                    operator_issues.append([name,esc(relative),str(error.get("attempt", "—")),esc(error.get("error","unknown")),"阶段最终完成"])
        old_rows = {c:rd(old_paths[d["dataset"]]/c/"final_dev_outputs.json") for c in summary}
        old_by_id = {r["id"]:r for r in old_rows["B"]}
        current_by_condition = {c:{r["id"]:r for r in d["rows"][c]} for c in summary}
        bar = ''
        for c in summary:
            m=summary[c]["dev"]
            metrics.append([name,c+" · "+LABELS[c],f'{m["correct"]}/{m["n"]}',pct(m["em"]),pct(m["f1"]),"完成"])
            length=sum(any(a.get("usage",{}).get("finish_reason")=="length" for a in r["attempts"]) for r in d["rows"][c])
            quality.append([name,c,str(sum(r["context_truncated"] for r in d["rows"][c])),str(length),str(m["format_failures"]),str(m["infrastructure_errors"]),str(m["provider_refusals"]),f'{m["mean_latency_seconds"]:.2f}',f'{m["p95_latency_seconds"]:.2f}'])
            bar += f'<div class="barrow"><b>{c}</b><div class="bartrack"><div class="barfill" style="width:{m["em"]*100}%"></div></div><span>{pct(m["em"])}</span></div>'
            oldm=aggregate(old_rows[c]); by_old={r["id"]:r for r in old_rows[c]}
            assert set(by_old)==set(current_by_condition[c])
            assert all(r["gold_answers"]==by_old[r["id"]]["gold_answers"] and r["question"]==by_old[r["id"]]["question"] for r in d["rows"][c])
            changed_context=sum(r["visible_context_hash"]!=by_old[r["id"]]["visible_context_hash"] for r in d["rows"][c])
            repaired=sum(r["hard"] and not by_old[r["id"]]["hard"] for r in d["rows"][c])
            regressed=sum(not r["hard"] and by_old[r["id"]]["hard"] for r in d["rows"][c])
            old_comp.append([name,c,pct(oldm["em"]),pct(m["em"]),f'{(m["em"]-oldm["em"])*100:+.2f} pp',f'{repaired} / {regressed}',str(changed_context)])
        bar_panels.append(f'<article class="panel"><h3>{name} · dev EM</h3>{bar}<small>横向条长：EM，0–100%；纵向：N / I / B。来源：本次各条件最终 dev 输出，n=30。</small></article>')
        curve_panels.append(f'<article class="panel"><h3>{name} · 保留最佳技能</h3>{evolution_chart(d["curve"],name)}<small>来源：共享初始化与 B/history.json。dev 同时用于候选选择。</small></article>')
        for c,p in d["pairs"].items():
            counts=p["counts"];ci=p["paired_bootstrap_95"]
            pairs.append([name,c+" → B",str(counts["repaired"]),str(counts["regressed"]),f'{p["em_difference_C_minus_B"]*100:+.2f} pp',f'[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] pp'])
        for s in d["stages"]:
            costs.append([name,s["stage"],str(s["logical"]),str(s["physical"]),f'{s["prompt"]:,}',f'{s["generated"]:,}',f'{s["seconds"]/60:.2f}',str(s["errors"])])
        for kind in ("qa", "skill"):
            calls=[c for c in d["calls"] if (c["stage"]=="qa")==(kind=="qa") and c["accounting"].get("status")=="ok" and c["accounting"].get("usage_status")=="server_reported"]
            if calls:
                inputs=[c["accounting"]["prompt_tokens"] for c in calls]
                outputs=[c["accounting"]["generated_tokens"] for c in calls]
                token_distribution.append([name,"QA" if kind=="qa" else "技能生成",str(len(calls)),f'{percentile(inputs,.5):.1f}',str(max(inputs)),f'{percentile(outputs,.5):.1f}',f'{percentile(outputs,.95):.1f}',str(max(outputs)),str(sum(c.get("finish_reason")=="length" for c in calls))])
        for h in d["history"]:
            for record in h["operator_records"]:
                for attempt_no, attempt in enumerate(record["attempts"], 1):
                    if attempt["status"] != "ok":
                        operator_issues.append([name,esc(record["child_id"]),str(attempt_no),esc(attempt.get("error","unknown")),"最终接受" if record["accepted"] else "拒绝候选"])
            for c in h["candidates"]:
                record=next(r for r in h["operator_records"] if r["child_id"]==c["id"])
                rounds.append([name,str(h["generation"]),esc(c["id"]),esc(c["origin"]),pct(c["val_hard"]),pct(c["val_soft"]),str(len(record["attempts"])),"最佳" if h["best_id"]==c["id"] else "—"])
        best=summary["B"]["best"]
        graph=rd(run/"B/best_skill.json")
        route_html=''.join('<details><summary>'+esc(route["task_type"])+'</summary><small>Use when: '+esc(route["use_when"])+
                           '</small><p>'+ ' → '.join('<code>'+esc(node)+'</code>' for node in route["sequence"])+ '</p></details>' for route in graph["routes"])
        graph_note=f'<p>结构包含 {len(graph["nodes"])} 个定义节点、{len(graph["routes"])} 条任务流程。</p>'
        if len(graph["nodes"])>6 or len(graph["routes"])>3:
            graph_note+='<p class="note warn">生成内容超过提示词建议的 6 节点 / 3 任务图数量。当前硬校验检查引用、格式与技能 token 上限，没有强制该数量建议；这里如实保留运行结果。</p>'
        skills.append('<article class="panel"><h3>'+name+' · 任务流程</h3>'+graph_note+route_html+'</article>')
        skills.append(f'<article class="panel"><h3>{name} · {esc(best["id"])}</h3><p>来源 {esc(best["origin"])}；初始化最佳 EM {pct(d["curve"][0])} → 最终 {pct(d["curve"][-1])}，三轮额外提升 {(d["curve"][-1]-d["curve"][0])*100:+.2f} pp。</p><details><summary>展开最佳技能全文</summary><pre>{esc(best["content"])}</pre></details></article>')
        for r in d["rows"]["B"]:
            n=current_by_condition["N"][r["id"]];i=current_by_condition["I"][r["id"]];old=old_by_id[r["id"]]
            state="repaired" if not n["hard"] and r["hard"] else "regressed" if n["hard"] and not r["hard"] else "both_correct" if r["hard"] else "both_wrong"
            explorers.append({"dataset":name,"id":r["id"],"question":r["question"],"gold":r["gold_answers"],"context":r["visible_context"],"state":state,
                "N":n["predicted_answer"],"I":i["predicted_answer"],"B":r["predicted_answer"],"N_em":n["em"],"I_em":i["em"],"em":r["em"],"f1":r["f1"],
                "raw":r["raw_response"],"old":old["predicted_answer"],"old_em":old["em"],"old_context_changed":old["visible_context_hash"]!=r["visible_context_hash"],
                "format_ok":r["format_ok"],"truncated":r["context_truncated"]})
        versions.append([name,esc(run.name),'<span class="hash">'+summary["B"]["code_hash"]+'</span>','<span class="hash">'+summary["B"]["config_hash"]+'</span>'])
        sources.append(f'<li><b>{name}</b>：'+ ' · '.join(link(run/c/"summary.json",c+" summary") for c in summary)+' · '+link(run/"B/history.json","演化历史")+' · '+link(run/"B/final_dev_outputs.json","逐题输出")+' · '+link(run/"B/best_skill.md","最终技能")+' · '+link(run/"B/config.json","配置")+'</li>')
    recovery=data[0]["run"]/"B/recovery/audit.json"
    recovery_note=('已逐项核对三轮训练批次、全部候选 dev 观测、选择结果、最佳技能与成本账本，再恢复最终摘要；新增模型调用为 0。原推理代码 hash 保留，恢复代码 hash 单独记录。'+link(recovery,"恢复审计")) if recovery.exists() else 'SearchQA 来自已完成运行。'
    new_cost_note=''
    if recovery.exists():
        hot_calls=sum(s["physical"] for s in data[1]["stages"])
        check_calls=sum(rd(Path(p)/"summary.json")["cost"]["physical_calls"] for p in checks)
        new_cost_note=f'<div class="note">本次接手新增模型生成请求为 HotpotQA {hot_calls} 次＋额外 API 检查 {check_calls} 次，合计 {hot_calls+check_calls} 次。SearchQA 恢复收尾新增 0 次。上方两套 Baseline 合计包含 11 月 7 日已经完成的 SearchQA 推理，不代表这些请求都在今天重新执行。</div>'
    length_groups=Counter(f'{d["dataset"]}/{c["folder"]}/{c["stage"]}/{c.get("phase", "?")}' for d in data for c in d["calls"] if c.get("finish_reason")=="length")
    length_table=table(["达到输出上限的请求分组","次数"],[[esc(k),str(v)] for k,v in length_groups.items()]) if length_groups else '<p>本次两套运行的调用记录没有 length 结束。</p>'
    operator_html = ('<div class="panel"><h3>初始化与技能变异校验记录</h3>' +
                    (table(["数据集","阶段 / 候选","尝试","错误","最终处理"],operator_issues) if operator_issues else
                     '<p>两套 Baseline 的初始化与技能变异未记录校验失败。额外 API 检查与正式实验分别计账。</p>')+'</div>')
    token_html='<div class="panel"><h3>实际 token 分布与生成上限</h3>'+table(["数据集","请求类型","成功请求数","输入 p50","输入最大值","生成 p50","生成 p95","生成最大值","length 次数"],token_distribution)+'<small>单位均为 tokens；基于服务端报告的成功请求，覆盖 pilot、初始化与正式实验；分位数采用线性插值。QA 生成上限 2048、技能生成上限 3072；到达上限的请求仍保留在评分与成本中。</small></div>'
    extra={"generated_at":datetime.now().astimezone().isoformat(),"totals":totals,"runs":{d["dataset"]:str(d["run"]) for d in data},"summaries":{d["dataset"]:d["summaries"] for d in data}}
    extra["check_runs"] = [str(Path(p).resolve()) for p in checks]
    extra["context_audit"] = str(Path(context_audit).resolve()) if context_audit else None
    review_html = ''
    if review:
        reviewed = rd(review)
        assert reviewed.get("review_status") == "verified", "Manual review must be checked against the final run"
        for d in data:
            assert reviewed["runs"][d["dataset"]] == d["run"].name
        counts = Counter()
        for item in reviewed["cases"]:
            row = next(r for r in explorers if r["id"] == item["id"] and r["dataset"].lower() == item["dataset"])
            assert not row["em"] and row["B"] == item["predicted_answer"]
            if "response_sha256" in item:
                assert digest(row["raw"]) == item["response_sha256"], "Manual review is stale"
            row["review"] = item["note"]
            row["category"] = item["category"]
            counts[(row["dataset"],item["category"])] += 1
        assert len(reviewed["cases"]) == sum(not r["em"] for r in explorers), "Review must cover all B errors"
        review_html = '<div class="panel"><h3>人工复核：错误输出呈现的主要现象</h3>'+table(["数据集","观察类别","题数"],[[a,esc(b),str(n)] for (a,b),n in counts.items()])+'<p>'+esc(reviewed["summary"])+'</p><small>每道 B 错题归入一个主要观察类别。分类基于输入与输出证据，不能直接证明内部推理失败原因；不修改原始评分。</small></div>'
        extra["manual_review"] = str(Path(review).resolve())
    report_data=json.dumps(explorers,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    sb,hb=(d["summaries"]["B"]["dev"] for d in data)
    body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project 5 · 两数据集 Baseline 完整结果</title><style>{CSS}</style></head><body>
<aside><div class="eyebrow">AIAA4051 / FALL 2026</div><br><b>Baseline<br>完整实验报告</b><nav><a href="#results">01　最终结果</a><a href="#evolution">02　技能演化</a><a href="#versions">03　新旧版本对照</a><a href="#quality">04　质量与成本</a><a href="#cases">05　逐题错误分析</a><a href="#recovery">06　修复与运行记录</a><a href="#protocol">07　协议与证据</a></nav><p class="source" style="margin-top:40px">qwen3.5-9b<br>seed 42 · 3 轮<br>4096 context / 2048 output</p></aside>
<main><header><div class="eyebrow">PROJECT 5 / BASELINE EXPERIMENTS</div><h1>SearchQA 与 HotpotQA<br>最新 Baseline 已完成</h1><p class="lede">两套数据均完成 pilot、共享初始化、N / I 和 B 三轮优化。所有最终指标均从保存的逐题响应重新评分核对。</p><div><span class="badge">N / I / B 全部完成</span><span class="badge">每数据集 30 道 dev</span><span class="badge">不含 Explore C / 最终 test</span></div></header>
<div class="toolbar"><small>生成时间：{datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")} · 离线报告</small><button id="print">打印 / 保存 PDF</button></div>
<div class="kpis"><div class="kpi"><span>SearchQA · B 最终 EM</span><strong>{pct(sb["em"])}</strong><small>{sb["correct"]}/30 正确 · F1 {pct(sb["f1"])}</small></div><div class="kpi"><span>HotpotQA · B 最终 EM</span><strong>{pct(hb["em"])}</strong><small>{hb["correct"]}/30 正确 · F1 {pct(hb["f1"])}</small></div><div class="kpi"><span>两套 Baseline 实际请求</span><strong>{totals["physical"]:,}</strong><small>{totals["prompt"]+totals["generated"]:,} 输入＋生成 tokens</small></div></div>
<section id="results"><h2>01　最终结果与同题增益</h2><div class="note">{''.join(findings)}</div><div class="two">{''.join(bar_panels)}</div><div class="panel">{table(["数据集","条件","正确数","EM","F1","状态"],metrics)}</div><p>N 为无技能问答，I 使用固定的上游初始模板（Node Lists 与 Task Graphs 为空），B 为基于训练失败反馈优化得到的技能。HotpotQA 此处报告答案 EM/F1，不含 supporting facts 或 joint 指标。</p>
<div class="panel"><h3>同一批输入下，B 纠正了多少题</h3>{table(["数据集","比较","纠正题数","退步题数","EM 差","配对 bootstrap 95% 区间"],pairs)}<small>2000 次题目重采样，seed 42；区间仅反映这 30 道题的重采样变化，不覆盖训练随机性。dev 同时用于选择技能，因此不能替代独立 test 结论。</small></div></section>
<section id="evolution"><h2>02　三轮技能演化</h2><div class="two">{''.join(curve_panels)}</div><div class="panel">{table(["数据集","轮数","候选 ID","算子","dev EM","dev F1","尝试数","当轮最佳"],rounds)}</div>{''.join(skills)}<p>每轮对两个父代使用相同 20 道训练题；由 mutation_other 和 mutation_graph 各生成一个候选。候选先经图结构与 token 上限校验，再按 dev EM 选保留种群。图格式正确不等于技能有效。</p></section>
<section id="versions"><h2>03　与上一版结果对照</h2>{context_html}<div class="panel">{table(["数据集","条件","旧 EM","新 EM","变化","旧→新纠正 / 退步","可见上下文改变题数"],old_comp)}</div><div class="note warn">历史对照使用 11 月 6 日已完成运行：旧规则最多 6000 字符上下文、1024-token QA 输出；新规则为 4096-token 上下文、2048-token QA 输出。初始化、优化技能、运行时间及服务状态也可能不同，因此这不是只改变一个变量的消融实验。SearchQA 的可见上下文未改变；HotpotQA 的上下文改变题数见上表。</div></section>
<section id="quality"><h2>04　输出质量与真实调用成本</h2><div class="panel"><h3>最终保留的 dev 观测</h3>{table(["数据集","条件","上下文截断","输出到上限","标签缺失","未解决请求错误","服务拒答","平均延迟 / s","p95 / s"],quality)}<small>每行 n=30。B 的最终输出可以复用早期已评估候选，延迟属于原始请求观测，并非结束时重新测量。</small></div><div class="panel"><h3>所有阶段的物理调用</h3>{table(["数据集","阶段","逻辑调用","物理调用","输入 tokens","生成 tokens","请求时长 / min","失败请求"],costs)}</div><p>合计输入 <b>{totals["prompt"]:,}</b> tokens，生成 <b>{totals["generated"]:,}</b> tokens，请求耗时之和 <b>{totals["seconds"]/3600:.2f} 小时</b>，失败请求 <b>{totals["errors"]}</b> 次。B 逻辑成本包含共享初始化，物理汇总仅计 physical_calls &gt; 0，避免重复计费。此汇总包括两套 Baseline 的 pilot，不包括单独的 API 连通性检查、历史运行或离线修复。</p><p>服务端 usage 为 token 统计依据。请求时长之和不等于 GPU 占用时间，也不含离线等待/修复；托管服务 GPU 小时未知。未提供单价，因此不估算人民币费用。</p><details><summary>全流程达到输出上限的请求</summary>{length_table}</details>{token_html}{operator_html}{checks_html}{new_cost_note}</section>
<section id="cases"><h2>05　60 道题的逐题核对</h2>{review_html}<p>默认显示 B 未完全答对的题。可查看 N→B 的纠正/退步、历史 B 回答以及当前可见证据。下面展示的是输出层面的观察，不把 EM=0 自动归因于某个技能节点。</p><div class="panel"><div class="filters"><select id="dataset" aria-label="数据集"><option value="all">两个数据集</option><option value="SearchQA">SearchQA</option><option value="HotpotQA">HotpotQA</option></select><select id="state" aria-label="题目状态"><option value="failed">B 的全部错题</option><option value="repaired">N → B 纠正</option><option value="regressed">N → B 退步</option><option value="old_regressed">旧 B → 新 B 退步</option><option value="all">全部题目</option></select><input id="search" type="search" placeholder="搜索题目、答案或 ID" aria-label="搜索题目"></div><p id="count" class="muted" aria-live="polite"></p><div id="case-list"></div></div></section>
<section id="recovery"><h2>06　本次修复与完成依据</h2><div class="panel"><h3>SearchQA：恢复最终落盘</h3><p>此前最后一次 metrics.json 原子替换触发 Windows WinError 5，导致三轮已结束但摘要仍为 running。{recovery_note}</p><h3>文件写入：有限重试，保留证据</h3><p>改为同目录唯一临时文件；遇到 PermissionError 最多尝试 8 次原子替换。持续失败时保留旧结果与完整临时文件，不把部分写入当成成功。</p><h3>网络：为本次模型请求绑定校园网接口</h3><p>本机普通路径和去代理路径曾超时，绑定 WLAN 地址后 /health 与模型查询返回成功。通过 PROJECT5_API_SOURCE_IP 为实验进程选择接口，不修改系统代理或模型请求参数。</p><p>修复后的离线测试共 50 项通过，覆盖文件锁重试和接口绑定。HotpotQA 以新的代码指纹独立运行，所有 N/I/B 条件内部版本保持一致。</p></div></section>
<section id="protocol"><h2>07　实验协议、边界与原始证据</h2><div class="two"><article class="panel"><h3>固定设置</h3><ul><li>模型 qwen3.5-9b，思考模式关闭，窗口 32768 tokens。</li><li>文档上下文 4096 tokens；QA 输出 2048；安全余量 512。</li><li>技能输出最多 3072 tokens，技能正文最多 2048。</li><li>seed 42，种群 2，三轮，每轮训练批次 20。</li><li>QA temperature 0，技能生成 temperature 0.6；并发 1。</li><li>每阶段最多 600 次逻辑调用、700000 生成 tokens、9 小时；每操作最多 2 次总尝试。</li></ul></article><article class="panel"><h3>结论适用范围</h3><ul><li>SearchQA：100 train / 30 dev / 50 test；本报告未运行 test。</li><li>HotpotQA：公开 distractor 数据 100 train / 30 dev；课程 test 文件仍未提供。</li><li>只完成固定 Baseline N/I/B；Explore C 不在本次补跑范围，不能据此声称自我纠正方案胜过 B。</li><li>单次 seed、30 道 dev，dev 参与技能选择；准确率提升需独立测试集和多 seed 验证。</li><li>服务权重 revision 与量化精度尚未披露，无法保证跨日期部署完全相同。</li></ul></article></div><div class="panel">{table(["数据集","运行目录","推理代码 SHA256","配置 SHA256"],versions)}</div><p>SearchQA 与 HotpotQA 的代码指纹不同来自本次 I/O、连接方式及相应测试修复；评分、训练抽样、提示词和技能选择代码未修改。SearchQA 保留原始推理 hash，恢复审计另列当前 hash。</p><ul>{''.join(sources)}</ul><p class="source">原始文件链接依赖本机目录；表格、图和全部逐题内容已内嵌，可离线打开。报告生成器同时输出 audit.json，记录来源与聚合总数。</p></section><footer class="foot">AIAA4051 Project 5 · 证据驱动的 Baseline 结果快照 · {esc(extra["generated_at"])}</footer></main>
<script id="report-data" type="application/json">{report_data}</script><script>
const rows=JSON.parse(document.getElementById('report-data').textContent);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const ds=document.getElementById('dataset'),st=document.getElementById('state'),search=document.getElementById('search');
function render(){{const q=search.value.trim().toLowerCase();const visible=rows.filter(r=>(ds.value==='all'||r.dataset===ds.value)&&(st.value==='all'||(st.value==='failed'?!r.em:st.value==='old_regressed'?r.old_em&&!r.em:r.state===st.value))&&(!q||[r.id,r.question,r.N,r.I,r.B,r.gold.join(' ')].join(' ').toLowerCase().includes(q)));document.getElementById('count').textContent='显示 '+visible.length+' / 60 题';document.getElementById('case-list').innerHTML=visible.map(r=>`<article class="case"><div class="casehead"><span>${{esc(r.dataset)}} · ${{{{repaired:'N 错 → B 对',regressed:'N 对 → B 错',both_correct:'N / B 均对',both_wrong:'N / B 均未完全匹配'}}[r.state]}}</span><span>B EM ${{r.em}} · F1 ${{(r.f1*100).toFixed(1)}}%</span></div><div class="question">${{esc(r.question)}}</div><div class="answers">${{['N','I','B'].map(c=>`<div class="answer ${{(c==='B'?r.em:r[c+'_em'])?'':'bad'}}"><small>${{c}}</small>${{esc(r[c])}}</div>`).join('')}}</div><p>参考答案：${{esc(r.gold.join(' / '))}}</p><small>历史 B：${{esc(r.old)}}（EM ${{r.old_em}}）；可见上下文${{r.old_context_changed?'有变化':'相同'}}。当前答案标签${{r.format_ok?'完整':'缺失'}}，上下文${{r.truncated?'已截断':'完整'}}。</small>${{r.review?`<p class="note"><b>${{esc(r.category)}}</b>：${{esc(r.review)}}</p>`:''}}<details><summary>原始响应与输入证据</summary><p class="mono">${{esc(r.id)}}</p><h4>B 原始响应</h4><pre>${{esc(r.raw)}}</pre><h4>当前可见上下文</h4><pre>${{esc(r.context)}}</pre></details></article>`).join('')||'<p class="empty">此筛选条件下没有题目。</p>';}}
ds.addEventListener('change',render);st.addEventListener('change',render);search.addEventListener('input',render);document.getElementById('print').addEventListener('click',()=>window.print());render();
</script></body></html>'''
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(body,encoding="utf-8")
    output.with_suffix('.audit.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {"output":str(output),"bytes":output.stat().st_size,"totals":totals,"failed_cases":sum(not r["em"] for r in explorers)}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--searchqa',type=Path,required=True)
    parser.add_argument('--hotpotqa',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checks',type=Path,nargs='*',default=[])
    parser.add_argument('--review',type=Path)
    parser.add_argument('--context-audit',type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.searchqa,args.hotpotqa,args.output,args.checks,args.review,args.context_audit),ensure_ascii=False,indent=2))
