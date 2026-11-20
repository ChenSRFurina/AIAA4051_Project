"""Build an offline course Parts 1/2 report and standalone skill SVGs.

Read-only analysis: no model calls, no changes to frozen evidence or scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import difflib
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Baseline"))
from baseline.evaluation import aggregate, score
from baseline.analysis import compare
from baseline.io import digest

RUNS = {
    "searchqa": ROOT / "Baseline/runs/api/searchqa/20261107-125241-971323-38572",
    "hotpotqa": ROOT / "Baseline/runs/api/hotpotqa/20261109-153042-871031-7652",
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def esc(v):
    return html.escape(str(v), quote=True)


def pct(v):
    return f"{v * 100:.2f}%"


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + str(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def svg_text(x, y, value, width=38, size=13, color="#203047", weight=400):
    lines = textwrap.wrap(str(value), width=width, break_long_words=True) or [""]
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">' + ''.join(f'<tspan x="{x}" dy="{0 if i == 0 else size + 5}">{esc(line)}</tspan>' for i, line in enumerate(lines)) + '</text>', len(lines) * (size + 5)


def skill_svg(graph, title):
    """Literal routing and ordered nodes; render only explicit named return edges."""
    routes = graph['routes']
    if not routes:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 160" role="img"><rect width="720" height="160" fill="#f7f9fc"/><text x="28" y="46" font-family="sans-serif" font-size="20">' + esc(title) + '</text><rect x="28" y="70" width="660" height="60" rx="8" fill="#e8edf3"/><text x="48" y="106" font-family="sans-serif" font-size="16">Empty upstream template: no nodes or task routes</text></svg>'
    width = max(760, len(routes) * 360 + 40)
    body = []
    explicit = []
    max_nodes = max(len(r['sequence']) for r in routes)
    route_height = max(125, max(len(textwrap.wrap(r['use_when'], 37)) * 17 + 65 for r in routes))
    base_y = 150 + route_height + 35
    end_y = base_y + max_nodes * 76
    for name, instructions in graph['nodes'].items():
        for line in instructions.splitlines():
            if re.search(r'\b(if|unless|otherwise)\b', line, re.I):
                explicit.append(name + ': ' + line.lstrip('- '))
    foot = []
    foot_y = end_y + 30
    for line in dict.fromkeys(explicit):
        txt, height = svg_text(30, foot_y, line, width=max(60, (width - 60) // 7), size=12)
        foot.append(txt)
        foot_y += height + 12
    height = max(end_y + 60, foot_y + 25)
    body.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    body.append(svg_text(28, 32, title, width=110, size=20, weight=600)[0])
    body.append(f'<rect x="{width/2-130}" y="58" width="260" height="45" rx="7" fill="#173a59"/>')
    body.append(f'<text x="{width/2}" y="86" text-anchor="middle" fill="white" font-size="16">Task-type routing</text>')
    for i, route in enumerate(routes):
        x, center = 30 + i * 360, 185 + i * 360
        body.append(f'<path d="M {width/2} 103 V 124 H {center} V 148" fill="none" stroke="#5e7288" marker-end="url(#arr)"/>')
        body.append(f'<rect x="{x}" y="150" width="310" height="{route_height}" rx="7" fill="#eaf2f9" stroke="#c9d9e7"/>')
        title_text, th = svg_text(x + 13, 173, route['task_type'], 30, 14, weight=600)
        body.append(title_text)
        body.append(svg_text(x + 13, 173 + th + 4, 'Use when: ' + route['use_when'], 38, 12)[0])
        for j, name in enumerate(route['sequence']):
            y = base_y + j * 76
            prev = 150 + route_height if j == 0 else y - 18
            body.append(f'<path d="M {center} {prev} V {y-3}" fill="none" stroke="#5e7288" marker-end="url(#arr)"/>')
            body.append(f'<rect x="{x+20}" y="{y}" width="270" height="58" rx="7" fill="#f5f7fa" stroke="#cbd5df"/>')
            body.append(svg_text(x + 32, y + 22, f'{j+1}. {name}', 29, 13, weight=500)[0])
            instructions = graph['nodes'][name]
            for line in instructions.splitlines():
                if not re.search(r'\b(return to|go back to)\b', line, re.I):
                    continue
                targets = re.findall(r'`([^`]+)`', line)
                for k, target in enumerate(targets):
                    if target in route['sequence'][:j]:
                        dest = base_y + route['sequence'].index(target) * 76 + 29
                        lane = x + 7 + k * 5
                        body.append(f'<path d="M {x+20} {y+40+k*6} H {lane} V {dest} H {x+16}" fill="none" stroke="#8a597d" stroke-dasharray="5 3" marker-end="url(#arr)"/><title>{esc(line)}</title>')
    if explicit:
        body.append(svg_text(30, end_y + 8, 'Explicit conditions / named return edges (dashed):', max(80, width // 8), 13, weight=600)[0])
    marker = 'arr-' + hashlib.sha256(title.encode()).hexdigest()[:12]
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" font-family="Arial, sans-serif"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#5e7288"/></marker></defs>' + ''.join(body + foot) + '</svg>'
    return svg.replace('id="arr"', f'id="{marker}"').replace('url(#arr)', f'url(#{marker})')


def curve_svg(ds, curve):
    points = ' '.join(f'{70+i*170},{230-v*180}' for i, v in enumerate(curve))
    lines = ''.join(f'<line x1="60" y1="{230-v*180}" x2="610" y2="{230-v*180}" stroke="#e2e8ee"/><text x="13" y="{234-v*180}" font-size="12">{v*100:.0f}%</text>' for v in [0,.25,.5,.75,1])
    labels = ''.join(f'<circle cx="{70+i*170}" cy="{230-v*180}" r="5" fill="#287291"/><text x="{70+i*170}" y="{216-v*180}" text-anchor="middle" font-size="13">{pct(v)}</text><text x="{70+i*170}" y="255" text-anchor="middle" font-size="12">{"Init" if i==0 else "Round "+str(i)}</text>' for i, v in enumerate(curve))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 292" role="img" aria-label="{ds} best dev EM" font-family="Arial, sans-serif"><text x="60" y="22" font-size="17">{ds} · best retained dev EM</text>{lines}<polyline points="{points}" fill="none" stroke="#287291" stroke-width="3"/>{labels}<text x="280" y="282" font-size="12">Evolution round</text></svg>'


def lineage_svg(ds, init, history):
    records = {r['id']: r for r in init}
    for h in history:
        records.update({r['id']: r for r in h['candidates']})
    positions = {key: (35 + int(key[1:5]) * 260, 75 + int(key[-4:]) * 150) for key in records}
    bits = ['<rect width="1080" height="380" fill="white"/>', '<text x="30" y="30" font-size="20">' + esc(ds) + ' · candidate lineage / dev EM</text>']
    for key, r in records.items():
        x, y = positions[key]
        for parent in r.get('parent_ids', []):
            px, py = positions[parent]
            bits.append(f'<path d="M {px+210} {py+43} L {x} {y+43}" stroke="#8a9bad" fill="none"/>')
    best = history[-1]['best_id']
    for key, r in records.items():
        x, y = positions[key]
        bits.append(f'<rect x="{x}" y="{y}" width="210" height="86" rx="8" fill="{"#e0f0ed" if key == best else "#f1f4f7"}" stroke="#b7c7d5"/>' + svg_text(x+12,y+24,key,30,14,weight=600)[0] + svg_text(x+12,y+48,f"EM {pct(r['val_hard'])}",30,14)[0] + svg_text(x+12,y+70,'SELECTED FINAL' if key==best else r['origin'],30,11)[0])
    bits.append('<text x="30" y="353" font-size="13">Edges = recorded parent IDs; color = final selected skill. All eight initial/candidate skills are included.</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 380" font-family="Arial, sans-serif" role="img">' + ''.join(bits) + '</svg>'


DEV_LINKS = {
    'f0c26f4f2e9f4d1d84b4f13cb201cbeb': (["Parse Question", "Verify Evidence"], "并列语义约束未同时满足；技能只有泛化的意图解析，没有明确要求逐一核对题目中的两个定义。"),
    'f03757766e034c539bf2595428c03626': (["Parse Question", "Formulate Answer"], "双关题的两个对应词未完整输出。简短答案要求不能代替覆盖全部子问题的检查。"),
    'e3b96fc1b99247ebada09a84fa5fd205': (["Parse Question"], "输入缺少足够的答案意图信息；先回查题目类别，不能把参考与预测不同直接归因于解析节点。"),
    '1a78a8cf476c4a0dbf551558da1d879b': (["Formulate Answer"], "已识别目标名词，但修饰语粒度与参考不一致；简短和精确两个原则没有给出可操作的消歧规则。"),
    'c7bd32aee95f485d85d0b6749d542c51': (["Parse Question", "Formulate Answer"], "最终输出 True，与题目要求的人物姓名类型不符。输出可见的检查失效，不证明内部节点未执行。"),
    'c9f3a971abe64853aca3e0bb1fea6962': (["Formulate Answer"], "人物实体与参考接近，额外头衔触发 EM=0；没有证据证明人物关系推理错误。"),
    '982e414b371248788c72732975949957': (["Verify Evidence", "Wrap Answer"], "响应已经识别 Pastor Aleman 的对应关系，却仍按缺少逐字直接证据拒答；严格证据规则可能被过度应用。"),
    '5a733b835542991f9a20c6b2': (["Parse Goal", "Verify Against Context"], "电影与童话的答案类型存在标注疑点；节点诊断保持未定，不把该题作为确定的推理失败证据。"),
    '5a793bc95542994bb9456fc3': (["Synthesize Final Answer"], "年份比较正确，常用名与正文全名不同；抽取原文的要求没有指定应采用标题名还是正文全名。"),
    '5add3fe05542992c1e3a2580': (["Retrieve Relevant Passages", "Extract Facts"], "响应称无其他相关歌曲，但另一篇独立文档有候选；覆盖检索不足是可见现象，年份口径另需复核。"),
    '5a776b82554299373536026f': (["Synthesize Final Answer"], "关系链已完成，货币名称多了国家修饰语；这是答案边界问题而非币种识别错误。"),
    '5adf62d05542992d7e9f932d': (["Parse Goal", "Infer Intermediate Conclusion"], "政治人物与乐队成员同名，响应混用两者；与技能要求核对桥接实体一致性存在可观察的不一致。"),
    '5ac2cdaa554299657fa29070': (["Parse Goal", "Verify Against Context"], "where 的地理/气候含义及参考适用范围不清，暂不把差异归为已确认的技能错误。"),
    '5ae788fa55429952e35ea964': (["Infer Intermediate Conclusion", "Verify Against Context"], "专辑→乐队→现场特色的两段证据可串联，但响应要求直接描述整条关系；跨文档组合与逐字核验存在张力。"),
    '5ab236645542993be8fa98c9': (["Synthesize Final Answer"], "数量 two 正确，参考包含奖项全短语；按问题给最短答案仍可能不满足该参考的粒度。"),
    '5a81e8ea5542995ce29dcc73': (["Synthesize Final Answer"], "已找到演员，正文全名与标题常用名不同；没有内部执行轨迹证明是推理链错误。"),
    '5adf34d35542993a75d26425': (["Synthesize Final Answer"], "已定位同一人物，参考带正式名和昵称；这里需要名称形式分析，不能改写原始 EM。"),
    '5abe87cb5542993f32c2a156': (["Parse Goal", "Infer Intermediate Conclusion", "Synthesize Final Answer"], "输出资助机构这个中间实体，没有继续提取彩票类型；应核对是否回答了 Parse Goal 中的最终属性。"),
}


def validate_rows(rows, ds, expected=None):
    assert rows and len({r['id'] for r in rows}) == len(rows)
    if expected is not None:
        assert [r['id'] for r in rows] == expected
    for r in rows:
        s = score(r['raw_response'], r['gold_answers'], ds)
        assert not r['error'], 'Incomplete infrastructure result'
        assert abs(r['em'] - s['em']) < 1e-9 and abs(r['f1'] - s['f1']) < 1e-9
        assert digest(r['visible_context']) == r['visible_context_hash']


def pair(left, right):
    x = compare(left, right)
    return {"counts": x['counts'], "delta": x['em_difference_C_minus_B'], "ci": x['paired_bootstrap_95'],
            "changed_ids": [{"id": t['id'], "transition": t['transition']} for t in x['transitions'] if t['transition'] in ['repaired','regressed']]}


def build(output, test_dir=None, test_review=None):
    output = Path(output).resolve()
    assets = output.parent / (output.stem + '_assets')
    assets.mkdir(parents=True, exist_ok=True)
    review = read(ROOT / 'Baseline/runs/reports/baseline_review_20261109.json')
    review_map = {(r['dataset'],r['id']):r for r in review['cases']}
    test_review_map = {}
    if test_review:
        tr = read(test_review)
        assert tr['review_status'] == 'verified'
        test_review_map = {(r['condition'],r['id']):r for r in tr['cases']}
    all_cases, summaries, evolutions, source_hashes = [], {}, {}, {}
    sections, metrics, transitions, cost_rows = [], [], [], []
    diagnosis_export, test_diagnoses, evolution_export = [], [], []
    total_dev_cost = Counter()
    for ds, run in RUNS.items():
        summaries[ds] = {c:read(run/c/'summary.json') for c in ['N','I','B']}
        data = read(ROOT/'Baseline/data/prepared'/ds/'manifest.json')
        ids = read(ROOT/'Baseline/data/prepared'/ds/'dev_ids.json')
        all_dev = {c:read(run/c/'final_dev_outputs.json') for c in ['N','I','B']}
        for c, rows in all_dev.items():
            validate_rows(rows,ds,ids)
            assert aggregate(rows) == summaries[ds][c]['dev']
            assert summaries[ds][c]['status'] in ['completed','static_complete']
            m = aggregate(rows)
            metrics.append([ds,'dev',c,m['n'],m['correct'],pct(m['em']),pct(m['f1']),f"{m['mean_latency_seconds']:.2f}",f"{m['p50_latency_seconds']:.2f}",f"{m['p95_latency_seconds']:.2f}",m['format_failures']])
        for l,r in [('N','I'),('N','B'),('I','B')]:
            pp=pair(all_dev[l],all_dev[r]); transitions.append([ds,'dev',l+' → '+r,pp['counts']['repaired'],pp['counts']['regressed'],pp['counts']['both_correct'],pp['counts']['both_wrong'],f"{pp['delta']*100:+.2f}"])
        best_graph = read(run/'B/best_skill.json')
        best_skill = (run/'B/best_skill.md').read_text(encoding='utf-8')
        for c, rows in all_dev.items():
            for row in rows:
                # A retained initialization skill reuses its dev observations.
                # Display the evaluated condition while preserving provenance.
                rr = dict(row, condition=c, observation_condition=row['condition'], manual=None, skill_diagnosis=None)
                if c=='B' and row['em']==0:
                    reviewed=review_map[(ds,row['id'])]
                    assert reviewed['predicted_answer']==row['predicted_answer']
                    assert reviewed['response_sha256']==hashlib.sha256(row['raw_response'].encode()).hexdigest()
                    nodes,note=DEV_LINKS[row['id']]
                    quotes={n:best_graph['nodes'][n] for n in nodes}
                    assert all(q in best_skill for q in quotes.values())
                    rr['manual']=reviewed
                    rr['skill_diagnosis']={"nodes":nodes,"instruction_quotes":quotes,"note":note,"status":"instruction_output_comparison_not_observed_execution","skill_hash":digest(best_skill)}
                    diagnosis_export.append(rr)
                all_cases.append(rr)
        init=read(run/'shared_init/population.json')
        hist=read(run/'B/history.json')
        assert len(hist)==3
        curve=[max(x['val_hard'] for x in init)]
        records={x['id']:x for x in init}
        rows_by_id={}
        cache=read(run/'shared_init/validation_cache.json')
        for item in init:
            rows_by_id[item['id']]=cache[digest(item['content'])[:16]]
        for h in hist:
            records.update({r['id']:r for r in h['candidates']})
            curve.append(records[h['best_id']]['val_hard'])
            for child in h['candidates']:
                path=run/'B/evaluations'/('g'+str(h['generation']))/child['id']/'dev/outputs.jsonl'
                rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
                validate_rows(rows,ds,ids)
                rows_by_id[child['id']]=rows
        lineage=lineage_svg(ds,init,hist)
        (assets/f'{ds}-lineage.svg').write_text(lineage,encoding='utf-8')
        curve_img=curve_svg(ds,curve)
        (assets/f'{ds}-curve.svg').write_text(curve_img,encoding='utf-8')
        gallery=[]
        for key,item in records.items():
            graph=read(run/'B/skills'/(key+'.json'))
            assert digest(item['content'])==graph['hash']
            svg=skill_svg(graph,f'{ds} / {key} / dev EM {pct(item["val_hard"])}')
            (assets/f'{ds}-{key}.svg').write_text(svg,encoding='utf-8')
            gallery.append(f'<details class="skill"><summary>{key} · {esc(item["origin"])} · EM {pct(item["val_hard"])} · {len(graph["nodes"])} nodes / {len(graph["routes"])} routes</summary><div class="diagram">{svg}</div><a download href="{esc(assets.name)}/{ds}-{key}.svg">下载 SVG</a><details><summary>技能全文与节点内条件</summary><pre>{esc(item["content"])}</pre></details></details>')
        round_cards=[]
        for h in hist:
            cards=[]
            for child in h['candidates']:
                parent=records[child['parent_ids'][0]]
                pp=pair(rows_by_id[parent['id']],rows_by_id[child['id']])
                diff='\n'.join(difflib.unified_diff(parent['content'].splitlines(),child['content'].splitlines(),fromfile=parent['id'],tofile=child['id'],lineterm=''))
                a=read(run/'B/skills'/(parent['id']+'.json')); b=read(run/'B/skills'/(child['id']+'.json'))
                changed_nodes=[n for n in sorted(set(a['nodes'])|set(b['nodes'])) if a['nodes'].get(n)!=b['nodes'].get(n)]
                actual={"dataset":ds,"round":h['generation'],"parent":parent['id'],"child":child['id'],"operator":child['origin'],"parent_em":parent['val_hard'],"child_em":child['val_hard'],"changed_nodes":changed_nodes,"routes_changed":a['routes']!=b['routes'],"paired":pp,"diff":diff}
                evolution_export.append(actual)
                effect='验证集净改善' if pp['delta']>1e-9 else '验证集净下降' if pp['delta'] < -1e-9 else '验证集 EM 持平'
                changes='；'.join([('节点变更：'+', '.join(changed_nodes)) if changed_nodes else '节点正文未变', '路由有变更' if actual['routes_changed'] else '路由未变'])
                examples=[]
                before={r['id']:r for r in rows_by_id[parent['id']]}; after={r['id']:r for r in rows_by_id[child['id']]}
                for t in pp['changed_ids']:
                    old,new=before[t['id']],after[t['id']]
                    examples.append([esc(t['id']), '修复' if t['transition']=='repaired' else '退化',esc(new['question']),esc(old['predicted_answer']),esc(new['predicted_answer']),esc(' / '.join(new['gold_answers']))])
                cards.append(f'<article class="change"><h4>{child["id"]} · {esc(child["origin"])}</h4><p>{parent["id"]} → {child["id"]}：{pct(parent["val_hard"])} → {pct(child["val_hard"])} · <strong>{effect}</strong></p><p>{esc(changes)}。修复 {pp["counts"]["repaired"]} 题，退化 {pp["counts"]["regressed"]} 题。</p><details><summary>实际采用的技能差异</summary><pre>{esc(diff or "No content change")}</pre></details>' + (f'<details><summary>全部 {len(examples)} 道正确性变化题</summary>{table(["ID","变化","问题","父代回答","候选回答","参考"],examples)}</details>' if examples else '<p class="muted">此候选没有 EM 正确性变化题。</p>') + '</article>')
            round_cards.append(f'<div class="round"><h3>Round {h["generation"]} · 保留最佳 {h["best_id"]}</h3>'+''.join(cards)+'</div>')
        sections.append(f'<article class="dataset"><h3>{ds}</h3><div class="twocol"><div>{curve_img}</div><div class="finding"><p class="eyebrow">优化额外收益</p><p class="big">{(curve[-1]-curve[0])*100:+.2f} pp</p><p>相对共同初始化中的最佳技能；并非相对无技能组。</p><p>{"三轮保留的最佳技能仍是初始化版本，不能把全部 N→B 提升归功于进化。" if ds=="hotpotqa" else "三轮后比最佳初始化多答对 1 题。dev 同时用于选技能，须与独立 test 分开解读。"}</p></div></div><div class="diagram">{lineage}</div><p class="muted">以下差异来自实际保存的技能正文；优化器 notes 只是生成时的自述，不代替真实变更。</p>'+''.join(round_cards)+'<details class="gallery"><summary>初始化与全部候选的 8 份流程图</summary>'+''.join(gallery)+'</details></article>')
        for stage in ['pilot','shared_init','N','I','B']:
            p=run/stage/'ledger.jsonl'
            events=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
            totals={k:sum(r.get(k,0) for r in events if r.get('physical_calls',0)>0) for k in ['physical_calls','prompt_tokens','generated_tokens']}
            total_dev_cost.update(totals)
            cost_rows.append([ds,'开发 / '+stage,totals['physical_calls'],f"{totals['prompt_tokens']:,}",f"{totals['generated_tokens']:,}"])
        source_hashes[ds]={"run":str(run),"code_hash":summaries[ds]['B']['code_hash'],"config_hash":summaries[ds]['B']['config_hash'],"data_hash":summaries[ds]['B']['data_hash'],"manifest_counts":data['counts']}
    test_status='SearchQA test 尚未完成'
    test_finding=''
    test_summary=None
    if test_dir:
        test_dir=Path(test_dir)
        test_summary=read(test_dir/'results.json')
        assert test_summary['status']=='completed' and test_summary['demo'] is False and test_summary['scope']=='part1_part2_baseline_only'
        frozen=read(test_summary['frozen_manifest'])
        assert test_summary['freeze_id']==frozen['freeze_id']
        assert frozen['freeze_id']==digest({k:v for k,v in frozen.items() if k!='freeze_id'})
        test_rows={c:read(test_dir/c/'outputs.json') for c in ['N','I','B']}
        if test_review:
            assert set(test_review_map)=={(c,r['id']) for c,rs in test_rows.items() for r in rs if r['em']==0}, 'Review must cover exactly the complete test failure inventory'
        test_graph=read(RUNS['searchqa']/'B/best_skill.json')
        for c,rows in test_rows.items():
            validate_rows(rows,'searchqa',frozen['test_ids'])
            assert all(r['split']=='test' and r['condition']==c and r['skill_hash']==frozen['conditions'][c]['skill_hash'] for r in rows)
            m=aggregate(rows)
            assert all(test_summary['conditions'][c][k]==v for k,v in m.items())
            metrics.append(['searchqa','test',c,m['n'],m['correct'],pct(m['em']),pct(m['f1']),f"{m['mean_latency_seconds']:.2f}",f"{m['p50_latency_seconds']:.2f}",f"{m['p95_latency_seconds']:.2f}",m['format_failures']])
            costs=test_summary['conditions'][c]['cost'];cost_rows.append(['searchqa','test / '+c,costs['physical_calls'],f"{costs['prompt_tokens']:,}",f"{costs['generated_tokens']:,}"])
            for row in rows:
                rr=dict(row,condition=c,observation_condition=row['condition'],manual=None,skill_diagnosis=None)
                if (c,row['id']) in test_review_map:
                    rev=test_review_map[(c,row['id'])]
                    assert rev['predicted_answer']==row['predicted_answer'] and rev['response_sha256']==hashlib.sha256(row['raw_response'].encode()).hexdigest()
                    rr['manual']=rev
                    if c=='B' and rev.get('skill_nodes'):
                        rr['skill_diagnosis']={'nodes':rev['skill_nodes'],'instruction_quotes':{n:test_graph['nodes'][n] for n in rev['skill_nodes']},'note':rev['note'],'status':'post_freeze_descriptive_only_not_optimizer_feedback','skill_hash':row['skill_hash']}
                        test_diagnoses.append(rr)
                all_cases.append(rr)
        for l,r in [('N','I'),('N','B'),('I','B')]:
            pp=pair(test_rows[l],test_rows[r]);transitions.append(['searchqa','test',l+' → '+r,pp['counts']['repaired'],pp['counts']['regressed'],pp['counts']['both_correct'],pp['counts']['both_wrong'],f"{pp['delta']*100:+.2f}"])
        test_status=f'SearchQA：固定 {len(frozen["test_ids"])} 题 × N/I/B 已完成'
        sm=test_summary['conditions'];nb=pair(test_rows['N'],test_rows['B']);ib=pair(test_rows['I'],test_rows['B'])
        test_finding=f'<div class="notice"><strong>独立 test 结论：</strong>N {pct(sm["N"]["em"])}，I {pct(sm["I"]["em"])}，B {pct(sm["B"]["em"])}。B 相对 N 修复 {nb["counts"]["repaired"]} 题、退化 {nb["counts"]["regressed"]} 题；相对 I 修复 {ib["counts"]["repaired"]} 题、退化 {ib["counts"]["regressed"]} 题。本次没有观察到进化技能超过静态基线的测试收益。B 输出更短且平均延迟较低，但这不能代替准确率提升。</div>'
        source_hashes['searchqa_test']={"directory":str(test_dir.resolve()),"freeze_id":frozen['freeze_id'],"results_sha256":hashlib.sha256((test_dir/'results.json').read_bytes()).hexdigest()}
        save(assets/'searchqa_test_results.json',test_summary)
        save(assets/'searchqa_test_comparison.json',read(test_dir/'comparison.json'))
        save(assets/'searchqa_frozen.json',frozen)
    categories=Counter((r['dataset'],r['split'],r['condition'],r['manual']['category']) for r in all_cases if r['manual'])
    category_rows=[[ds,split,c,esc(cat),n] for (ds,split,c,cat),n in sorted(categories.items())]
    failures=[r for r in all_cases if r['em']==0]
    save(assets/'complete_failure_inventory.json',failures)
    save(assets/'dev_instruction_diagnoses.json',diagnosis_export)
    if test_diagnoses:
        save(assets/'test_instruction_diagnoses.json',test_diagnoses)
    save(assets/'evolution_analysis.json',evolution_export)
    import csv
    with (assets/'complete_failure_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['dataset','split','condition','observation_condition','id','question','gold_answers','predicted_answer','em','f1','category','diagnosis']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in failures:
            rev=r['manual'] or {};w.writerow({**{k:r[k] for k in keys if k in r},'category':rev.get('category','格式错误' if not r['format_ok'] else '未作语义归因'),'diagnosis':rev.get('note','保留全部 EM=0 样例；严格 EM 失败本身不证明推理原因。')})
    compact=[]
    for r in all_cases:
        compact.append({k:r[k] for k in ['dataset','split','condition','observation_condition','id','question','gold_answers','predicted_answer','em','f1','format_ok','context_truncated','raw_response','visible_context','manual','skill_diagnosis']})
    payload=json.dumps(compact,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    stats={'dev_unique_questions':60,'dev_observations':180,'test_unique_questions':len(test_summary['test_ids']) if test_summary else 0,'test_observations':sum(x['n'] for x in test_summary['conditions'].values()) if test_summary else 0,'all_failures':len(failures),'dev_B_reviewed':len(diagnosis_export),'test_reviewed':len(test_review_map),'test_B_instruction_diagnoses':len(test_diagnoses),'skill_svg_count':16}
    audit={'generated_at':datetime.now().astimezone().isoformat(),'stats':stats,'sources':source_hashes,'test':test_summary,'C_run':False,'hotpotqa_test':'blocked_missing_course_partition','pdf_data_link_audit':{'pages':'19–23','dataset_text':'HotpotQA (provided)','download_links':[],'embedded_attachments':[],'contact':'jchen349@connect.hkust-gz.edu.cn'},'core_source_modified':False}
    save(assets/'hotpotqa_preflight.json',read(ROOT/'Baseline/runs/part12/hotpotqa/preflight.json'))
    save(output.with_suffix('.audit.json'),audit)
    if test_summary:
        b=test_summary['conditions']['B']; hero=f'{b["correct"]}/{b["n"]}';hero_note=f'SearchQA test · B EM {pct(b["em"])} / F1 {pct(b["f1"])}'
    else:
        hero='待评测';hero_note='SearchQA test 尚未完成'
    downloads=[]
    if test_summary:
        downloads += [('searchqa_test_results.json','SearchQA test 结果'),('searchqa_test_comparison.json','配对变化与区间'),('searchqa_frozen.json','冻结清单')]
    if test_diagnoses:
        downloads.append(('test_instruction_diagnoses.json',f'{len(test_diagnoses)} 道 B test 指令诊断'))
    downloads.append(('hotpotqa_preflight.json','HotpotQA 缺失数据预检'))
    test_downloads='<p>'+' · '.join(f'<a download href="{esc(assets.name)}/{name}">{label}</a>' for name,label in downloads)+'</p>'
    content=TEMPLATE.replace('__TEST_FINDING__',test_finding).replace('__METRICS__',table(['数据集','划分','条件','n','正确','EM','F1','均值 s','P50 s','P95 s','格式失败'],metrics)).replace('__TRANSITIONS__',table(['数据集','划分','对照','修复','退化','均正确','均错误','EM 差 pp'],transitions)).replace('__EVOLUTION__',''.join(sections)).replace('__CATEGORIES__',table(['数据集','划分','条件','人工观察类别','错误观测数'],category_rows)).replace('__COSTS__',table(['数据集','阶段','实际调用','输入 tokens','生成 tokens'],cost_rows)).replace('__DATA__',payload).replace('__HERO__',hero).replace('__HERO_NOTE__',hero_note).replace('__TEST_STATUS__',test_status).replace('__ASSETS__',esc(assets.name)).replace('__DATE__',esc(audit['generated_at'])).replace('__FAILURES__',str(len(failures))).replace('__TEST_REVIEWED__',str(len(test_review_map))).replace('__AUDIT__',esc(json.dumps(audit,ensure_ascii=False,indent=2)))
    content=content.replace('__TEST_DOWNLOADS__',test_downloads)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(content,encoding='utf-8')
    return {'output':str(output),'assets':str(assets),'stats':stats}


TEMPLATE='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project 5 · Part 1 & 2 实验与证据报告</title><style>
:root{--ink:#172b42;--muted:#52667a;--line:#dce4eb;--bg:#f4f7fa;--accent:#236a83;--good:#176b59;--warn:#82541b}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--accent)}header{background:#14334f;color:white;padding:46px max(24px,calc((100vw - 1200px)/2));}header p{max-width:820px;color:#d4e0e9}h1{font-size:32px;line-height:1.35;margin:12px 0}h2{font-size:24px;margin:0 0 20px}h3{font-size:20px}h4{margin:0;font-size:16px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;margin:0}.big{font-size:38px;line-height:1.2;margin:16px 0;font-weight:650}.hero{display:flex;gap:48px;align-items:center;justify-content:space-between}.hero-metric{min-width:220px;border-left:1px solid #688399;padding-left:28px}nav{position:sticky;top:0;z-index:4;background:white;border-bottom:1px solid var(--line);display:flex;gap:25px;padding:12px 24px;justify-content:center;overflow-x:auto;white-space:nowrap}nav a{text-decoration:none;font-size:14px}main{max-width:1248px;margin:auto;padding:32px 24px}section{margin:0 0 42px;scroll-margin-top:70px}.panel,.dataset{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;margin:18px 0}.notice{padding:16px 20px;border-left:4px solid #b38238;background:#fff7e9}.good{border-left-color:#398d74;background:#edf7f3}.muted,small{color:var(--muted)}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:24px}.finding{padding:24px;background:#eff6f8;border-radius:8px}.scroll{max-width:100%;overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#edf2f6;white-space:nowrap}td{overflow-wrap:anywhere}td:nth-child(-n+3){white-space:nowrap}.round{border-top:1px solid var(--line);padding-top:12px;margin-top:20px}.change{padding:16px 0;border-bottom:1px dashed var(--line)}.diagram{max-width:100%;overflow:auto;background:#fff}.diagram svg{display:block;width:100%;min-width:720px;height:auto}svg{width:100%;height:auto}details{margin:12px 0}summary{cursor:pointer;color:var(--accent);font-weight:550}.skill{padding:14px;border:1px solid var(--line);border-radius:7px;margin:12px 0}.gallery>summary{padding:14px;background:#eaf2f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;font-size:12px;padding:15px;background:#f3f5f8;border-radius:6px}code{font-family:Consolas,monospace}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center}select,input,button{font:inherit;border:1px solid #bdccd8;border-radius:6px;padding:9px 11px;background:white;color:var(--ink)}input{flex:1;min-width:180px}button{cursor:pointer}.case{border:1px solid var(--line);background:white;border-radius:9px;padding:20px;margin:16px 0}.case-title{font-size:16px;font-weight:600;line-height:1.55}.tag{display:inline-block;background:#eaf1f6;border-radius:5px;padding:2px 7px;font-size:12px;margin-right:7px}.bad{background:#fff0e9;color:#7a422b}.answers{display:grid;grid-template-columns:1fr 1fr;gap:14px}.answer{padding:12px;background:#f3f6f8;border-radius:6px;overflow-wrap:anywhere}.answer b{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}.case .note{border-left:3px solid #75a1b3;padding:8px 12px;background:#f1f7f9}.case .id{font:11px Consolas,monospace;overflow-wrap:anywhere;color:var(--muted)}footer{color:var(--muted);font-size:12px;padding:24px 0;border-top:1px solid var(--line)}.checklist{padding-left:20px}.checklist li{margin:10px 0}.method-flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.method-flow span{padding:10px 14px;background:#eaf2f6;border-radius:6px}.empty{padding:30px;color:var(--muted)}
@media(max-width:700px){header{padding:30px 20px}h1{font-size:26px}.hero{display:block}.hero-metric{margin-top:22px;padding-left:18px}.twocol,.answers{grid-template-columns:1fr}main{padding:22px 12px}.panel,.dataset{padding:16px}nav{justify-content:flex-start;gap:20px}.controls>*{max-width:100%}.big{font-size:32px}td,th{padding:8px}}
@media print{nav,.controls,button{display:none}body{background:white;font-size:10pt}header{background:white;color:#172b42;padding:10px}header p{color:#52667a}main{padding:0}.panel,.dataset{break-inside:auto;border:0;padding:8px}.case{break-inside:avoid}.diagram svg{min-width:0}section{margin-bottom:22px}}
</style></head><body><header><div class="hero"><div><p class="eyebrow">AIAA4051 · Project 5 · Fixed baseline study</p><h1>Part 1 &amp; Part 2<br>复现、技能演化与完整失败分析</h1><p>基于实际保存的模型响应、冻结记录和逐轮技能。开发集用于选择与诊断；测试集只用于冻结后的描述性评估。此报告不包含 Explore C 的实验成绩。</p></div><div class="hero-metric"><p class="eyebrow">Held-out evaluation</p><p class="big">__HERO__</p><p>__HERO_NOTE__</p></div></div></header><nav><a href="#status">完成范围</a><a href="#method">固定流程</a><a href="#results">结果与对照</a><a href="#evolution">逐轮技能图</a><a href="#failures">全部错误与证据</a><a href="#cost">成本与复现</a></nav><main>
<section id="status"><h2>01 · 已完成什么，缺少什么</h2><div class="notice good">__TEST_STATUS__。两套数据的 N/I/B dev 实验、三轮候选分析、16 份技能流程图以及 18 道 B dev 错题的技能指令对应分析已整理。</div><div class="notice"><strong>HotpotQA 课程 test 尚缺。</strong>课程 PDF 第 20 页仅写 “HotpotQA (provided)”，第 19–23 页没有数据下载链接，PDF 也无嵌入附件。负责人邮箱为 jchen349@connect.hkust-gz.edu.cn。现有公开 distractor 的 100 train / 30 dev 属于开发材料；课程划分确认后仍需单独完成正式实验。本报告不把 dev 或其他项目数据替代课程 test。</div><div class="panel"><ul class="checklist"><li>Part 1：共享初始化、N/I/B、3 轮 × 每轮 2 个新候选、dev 选择、逐轮图和候选差异已具备；SearchQA 的 test 状态见上方。</li><li>Part 2：完整 dev 错误、人工观察分类、具体技能指令与输出的对应分析已完成；全部已运行 test 的 EM=0 记录均保留，人工复核覆盖数见错误区。</li><li>Part 3：未运行 C。SearchQA 测试前已封存现有 Explore 方法源代码与配置；后续不能根据本报告 test 案例修改方法后仍声称对同一 test 是未见评估。</li><li>最终课程验收仍需要 HotpotQA 课程数据与测试、Part 3 对照、服务端模型/GPU 记录以及六页报告和演示材料。</li></ul></div></section>
<section id="method"><h2>02 · 固定流程与数据边界</h2><div class="panel"><div class="method-flow"><span>共享初始化：模板 + 自动技能</span>→<span>父代执行 train</span>→<span>抽取 3 个失败反馈</span>→<span>模型生成 2 个候选</span>→<span>完整 dev 评分与选择</span>→<span>3 轮后冻结最佳技能</span>→<span>独立 test</span></div><p><strong>train：</strong>提供模型执行结果及训练参考答案，供技能修改使用。每轮两个父代使用相同 20 题，固定 seed=42。<strong>dev：</strong>用于候选排名、父代/幸存者选择、最终选择与方法诊断。反复选择同一 30 题会带来选择偏差，因此 dev 不是无偏泛化成绩。<strong>test：</strong>固定 ID、冻结技能与方法后评估，不更新技能或经验记忆。</p><p><strong>N</strong> 无技能；<strong>I</strong> 为上游原始空节点/空路由模板，不是人工编写的完整技能；<strong>B</strong> 为原版失败反馈流程的课程规模包装。HotpotQA 是给定上下文的问答适配。</p><p>执行器与优化器使用 qwen3.5-9b；窗口 32768；上下文预算 4096 tokens；QA 输出上限 2048；技能生成上限 3072；技能与反馈各 2048；并发 1。当前 train/dev 的全部上下文均容纳在预算内。test 的截断逐题记录。SearchQA 部分技能有 4 条路由，超过提示中建议的 3 条；硬校验实际约束结构合法性和 token 长度，不强制该路由数量建议。</p><p><strong>答案抽取：</strong>取最后一对完整 answer 标签内的文字；缺少标签时回退到最后一个非空行，再回退到去除首尾空白的整段响应。EM 在小写、标点、英语冠词和空白归一化后比较；F1 使用词元重叠。HotpotQA 的 yes/no/noanswer 按其专门规则处理。所有指标保留原始标准，人工语义复核不改分。</p></div></section>
<section id="results"><h2>03 · EM、F1、延迟与逐题变化</h2>__TEST_FINDING__<div class="panel">__METRICS__<p class="muted">延迟单位为秒，包含 API 网络与服务端处理；B 的 dev 指标使用最佳技能已保存的评估记录，不额外重复调用。格式失败仍在总题数分母内。</p></div><div class="panel"><h3>相同题目上的正确性变化</h3>__TRANSITIONS__<p class="muted">“修复”=左组 EM=0、右组 EM=1；“退化”反之。EM 差以百分点计。dev 配对变化受选择偏差影响；test 配对变化为冻结后描述性结果。不能据此证明某条技能指令的因果效果。</p></div></section>
<section id="evolution"><h2>04 · 初始化到最终技能：逐轮图与真实变更</h2><p>图展示技能定义中的任务路由、Use when 条件、节点顺序及明确写出的回退。它不是模型内部执行轨迹。每个候选都可查看实际文本差异和全部正确性变化案例；SVG 可独立下载用于报告。</p>__EVOLUTION__</section>
<section id="failures"><h2>05 · 完整失败清单与技能指令诊断</h2><div class="panel"><p>包含全部 <strong>__FAILURES__</strong> 条 EM=0 观测，按数据集、划分和 N/I/B 分开。dev 的最终 B 共 18 道已全部人工复核并对应到具体技能指令；test 人工复核记录数为 <strong>__TEST_REVIEWED__</strong>。未复核的样例标为“未作语义归因”，不会自动归为推理错误。</p>__CATEGORIES__<p class="muted">每条已复核错误指定一个主要观察类别；重叠原因和不确定标注写入诊断说明。姓名、粒度和参考歧义均保留官方 EM/F1。指令与输出的对应仅支持诊断假设，不证明内部节点是否执行。</p><p><a download href="__ASSETS__/complete_failure_inventory.csv">下载完整错误 CSV</a> · <a download href="__ASSETS__/complete_failure_inventory.json">下载完整错误 JSON</a> · <a download href="__ASSETS__/dev_instruction_diagnoses.json">下载 18 题指令证据</a> · <a download href="__ASSETS__/evolution_analysis.json">下载逐轮变更证据</a></p></div><div class="controls"><select id="dataset" aria-label="数据集"><option value="all">全部数据集</option><option value="searchqa">SearchQA</option><option value="hotpotqa">HotpotQA</option></select><select id="split" aria-label="划分"><option value="all">dev + test</option><option value="dev">dev</option><option value="test">test</option></select><select id="condition" aria-label="条件"><option value="B">B 最终技能</option><option value="all">N / I / B</option><option value="N">N 无技能</option><option value="I">I 初始模板</option></select><select id="outcome" aria-label="结果"><option value="fail">EM=0</option><option value="all">全部结果</option><option value="pass">EM=1</option></select><input id="query" placeholder="搜索 ID、题目、答案或类别" aria-label="搜索"><button id="reset">重置</button><button onclick="window.print()">打印</button></div><p id="count" class="muted"></p><div id="cases"></div></section>
<section id="cost"><h2>06 · 成本、冻结与复现证据</h2><div class="panel">__COSTS__<p class="muted">实际调用去掉共享初始化的重复逻辑记账；每阶段包含成功与失败请求。以上开发成本是本报告两次选定开发运行，不是整个项目历史累计消耗。GPU 小时未知，不以 API wall time 替代。合成检查、历史失败运行另存原账本。</p><p>SearchQA 使用原始推理代码快照评测 test；冻结清单锁定 N/I/B 的技能、配置、结果、源代码和数据。网络修复只是进程内绑定 WLAN 地址，原 endpoint、请求体、解码和评分保持一致。新入口独立记录驱动代码 SHA256。原四组冻结入口保留；此处是 Part 1/2 的 N/I/B 独立评测。</p><p>服务端权重 revision 与精度仍未披露，模型 ID 一致不足以证明跨日期权重相同。该局限必须写入最终报告。固定 ID 数量上限为 100/30/50；本次没有新增训练样例或修改已选技能。</p>__TEST_DOWNLOADS__<details><summary>来源、数量校验与运行标识</summary><pre>__AUDIT__</pre></details></div></section><footer>生成于 __DATE__ · 离线 HTML，图表与逐题证据已嵌入；独立下载文件在同名 assets 文件夹。此报告是 Part 1/2 实验材料，不代替课程最终六页报告。</footer></main><script id="report-data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('report-data').textContent);const $=id=>document.getElementById(id);const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const q=$('query').value.toLowerCase().trim();const filtered=rows.filter(r=>($('dataset').value==='all'||r.dataset===$('dataset').value)&&($('split').value==='all'||r.split===$('split').value)&&($('condition').value==='all'||r.condition===$('condition').value)&&($('outcome').value==='all'||($('outcome').value==='fail'?r.em===0:r.em===1))&&(!q||[r.id,r.question,r.predicted_answer,r.gold_answers.join(' '),r.manual?.category||''].join(' ').toLowerCase().includes(q)));$('count').textContent=`显示 ${filtered.length} / ${rows.length} 条观测；同一道题的 N/I/B 分开列出。`;$('cases').innerHTML=filtered.map(r=>`<article class="case"><p><span class="tag">${escape(r.dataset)}</span><span class="tag">${r.split}</span><span class="tag">${r.condition}</span><span class="tag ${r.em===0?'bad':''}">EM ${r.em} / F1 ${(r.f1*100).toFixed(2)}%</span></p><p class="case-title">${escape(r.question)}</p><p class="id">${escape(r.id)}</p><div class="answers"><div class="answer"><b>预测</b>${escape(r.predicted_answer)}</div><div class="answer"><b>参考</b>${escape(r.gold_answers.join(' / '))}</div></div>${r.manual?`<p class="note"><strong>${escape(r.manual.category)}</strong>：${escape(r.manual.note)}</p>`:r.em===0?'<p class="muted">未作语义归因：已列入完整错误清单，EM=0 本身不证明某个节点出错。</p>':''}${r.skill_diagnosis?`<details><summary>对应技能指令与诊断依据</summary><p>${escape(r.skill_diagnosis.note)}</p><p class="muted">下列为真实最终技能原文；内部节点执行不可观测。</p>${Object.entries(r.skill_diagnosis.instruction_quotes).map(([n,t])=>`<h4>${escape(n)}</h4><pre>${escape(t)}</pre>`).join('')}</details>`:''}<details><summary>原始响应与实际可见上下文</summary><p class="muted">答案标签${r.format_ok?'完整':'缺失'}；上下文${r.context_truncated?'截断':'完整'}。</p><h4>原始响应</h4><pre>${escape(r.raw_response)}</pre><h4>模型可见上下文</h4><pre>${escape(r.visible_context)}</pre></details></article>`).join('')||'<p class="empty">当前筛选没有匹配记录。</p>';}
for(const id of ['dataset','split','condition','outcome','query'])$(id).addEventListener(id==='query'?'input':'change',render);$('reset').onclick=()=>{for(const id of ['dataset','split'])$(id).value='all';$('condition').value='B';$('outcome').value='fail';$('query').value='';render();};render();
</script></body></html>'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--test-dir')
    p.add_argument('--test-review')
    args=p.parse_args()
    print(json.dumps(build(args.output,args.test_dir,args.test_review),ensure_ascii=False,indent=2))
