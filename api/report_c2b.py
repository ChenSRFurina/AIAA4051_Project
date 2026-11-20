"""Offline detailed B/C/C2b train/dev report. Reads completed audited evidence."""
from __future__ import annotations

from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from io import StringIO
import zipfile

from run_c2b import ROOT, SESSION
from part12 import read, write
from report_part12 import skill_svg


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(v) + '</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def curve_svg(ds, series):
    # Report-only dependencies are isolated from the running inference environment.
    sys.path.insert(0, str(ROOT / 'api/runtime/report-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'B': '#78828d', 'C': '#397e9b', 'C2b': '#b0502d'}
    with plt.rc_context({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'C2b-dev-report'}):
        fig, ax = plt.subplots(figsize=(7.8, 3.8), layout='constrained')
        for method, points in series.items():
            ax.plot([p['round'] for p in points], [100*p['em'] for p in points],
                    color=colors[method], label=method, linewidth=2,
                    linestyle={'B': '-', 'C': '--', 'C2b': ':'}[method],
                    marker={'B': 'o', 'C': '^', 'C2b': 's'}[method],
                    markersize={'B': 5, 'C': 8, 'C2b': 10}[method], markerfacecolor='none')
        ax.set(title=f'{ds}: retained best dev EM', xlabel='Completed evolution round', ylabel='Dev exact match (%)',
               ylim=(40, 100), xticks=[0, 1, 2, 3], xticklabels=['Init', '1', '2', '3'])
        ax.grid(axis='y', color='#e4e9ed')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(title='Method', ncols=3, loc='lower right', frameon=False)
        stream = StringIO()
        fig.savefig(stream, format='svg', metadata={'Date': None, 'Description': 'Fixed dev, seed 42; validation-selected best at each complete round. Source: audited histories.'})
        plt.close(fig)
        return stream.getvalue()


def build():
    output = ROOT / 'Explore/runs/reports/C2b_results_20261110.html'
    assets = output.with_name(output.stem + '_assets')
    assets.mkdir(parents=True, exist_ok=True)
    data = {ds: read(SESSION / ds / 'analysis.json') for ds in ['searchqa', 'hotpotqa']}
    assert all(a['status'] == a['audit']['status'] == 'passed' for a in data.values())
    metrics, budgets, pair_rows, matched_rows, train_rows, candidate_rows, quality, repeat = [], [], [], [], [], [], [], []
    browser = {}
    health_rows = []
    overview = []
    graphs = []
    total_calls, total_tokens, total_prompt = 0, 0, 0
    downloads = []
    for ds, a in data.items():
        m = a['metrics']
        overview.append(f'<li><strong>{esc(ds)}</strong>：B {m["B"]["correct"]}/30，C {m["C"]["correct"]}/30，C2b <strong>{m["C2b"]["correct"]}/30</strong>。C2b 相对 B 的 EM 差为 {(m["C2b"]["em"]-m["B"]["em"])*100:+.2f} 个百分点，相对 C 为 {(m["C2b"]["em"]-m["C"]["em"])*100:+.2f} 个百分点。</li>')
        for method in ['B', 'C', 'C2b']:
            v = m[method]; c = a['costs'][method]
            health = a['health'][method]
            health_rows.append([ds, method, health['physical_calls'], *[health['length_by_phase'].get(phase, 0) for phase in ['train', 'dev', 'optimize']], health['qa_by_split']['train']['format_failures'], health['qa_by_split']['dev']['format_failures'], health['qa_by_split']['dev']['length_format_failures']])
            metrics.append([ds, method, f'{v["correct"]}/{v["n"]}', pct(v['em']), pct(v['f1']), f'{v["mean_latency_seconds"]:.2f}', f'{v["p50_latency_seconds"]:.2f}', f'{v["p95_latency_seconds"]:.2f}', v['infrastructure_errors'], v['provider_refusals'], v['format_failures']])
            budgets.append([ds, method, c['logical']['calls'], c['physical']['physical_calls'], f'{c["logical"]["generated_tokens"]:,}', f'{c["physical"]["generated_tokens"]:,}', f'{c["physical"]["prompt_tokens"]:,}', f'{c["wall_seconds_sum"]/3600:.3f}', c['failures']])
            changes = [x for x in a['mutations'][method] if x['accepted']]
            quality.append([ds, method, len(changes), sum(bool(x['discarded_sections']) for x in changes), sum(bool(x['new_gold_lines']) for x in changes), sum(bool(x['training_answer_reuse_flags']) for x in changes), sum(x['no_op'] for x in changes), sum(x['attempts'] for x in a['mutations'][method]), sum(t['status'] != 'ok' for x in a['mutations'][method] for t in x['attempt_log'])])
            for r in a['mutations'][method]:
                candidate_rows.append([ds, method, r['round'], r['child'], r.get('operator', '未接受'), r.get('parent', '—'), pct(r.get('child_em')), pct(r.get('child_f1')), r.get('retained', False), r['attempts']])
                if r['accepted']:
                    graph_path = Path(a['paths'][method]) / 'skills' / (r['child'] + '.json')
                    graph_name = ds + '_' + method + '_' + r['child'] + '.svg'
                    (assets / graph_name).write_text(skill_svg(read(graph_path), f'{ds} / {method} / {r["child"]}'), encoding='utf-8')
                    r['svg'] = assets.name + '/' + graph_name
            for generation in range(1, 4):
                tr = [r['metrics'] for r in a['training'][method] if r['round'] == generation]
                n = sum(r['n'] for r in tr)
                train_rows.append([ds, method, generation, n, sum(r['correct'] for r in tr), pct(sum(r['em'] * r['n'] for r in tr)/n), pct(sum(r['f1'] * r['n'] for r in tr)/n)])
            point = a['matched']['checkpoints'][method]
            matched_rows.append([ds, method, a['matched']['call_cap'], a['matched']['generated_token_cap'], point['round'], point['best_id'], pct(point['em']), pct(point['f1']), point['unused_calls'], point['unused_generated_tokens']])
            run = Path(a['paths'][method])
            graph = skill_svg(read(run / 'best_skill.json'), f'{ds} · {method} final dev-selected skill')
            name = ds + '_' + method + '_best.svg'
            (assets / name).write_text(graph, encoding='utf-8')
            shutil.copyfile(run / 'best_skill.md', assets / (ds + '_' + method + '_best.md'))
            graphs.append(f'<details><summary>{ds} / {method} 最终技能图与 Markdown</summary><img src="{assets.name}/{name}" alt="{ds} {method} 最终技能定义图"><a download href="{assets.name}/{ds}_{method}_best.md">下载技能 Markdown</a></details>')
        total_calls += a['costs']['C2b']['physical']['physical_calls']
        total_tokens += a['costs']['C2b']['physical']['generated_tokens']
        total_prompt += a['costs']['C2b']['physical']['prompt_tokens']
        for mode in ['comparisons', 'matched']:
            comps = a['comparisons'] if mode == 'comparisons' else a['matched']['comparisons']
            for name, comp in comps.items():
                c = comp['counts']; ci = comp['paired_bootstrap_95']
                pair_rows.append([ds, '完整三轮' if mode == 'comparisons' else '共同预算截点', name.replace('_to_', ' → '), c['repaired'], c['regressed'], pct(comp['repair_rate']), pct(comp['regression_rate']), c['both_correct'], c['both_wrong'], f'{comp["em_difference_C_minus_B"]*100:+.2f}', f'[{ci[0]*100:.1f}, {ci[1]*100:.1f}]'])
        for method, r in a['initial_repeat'].items():
            repeat.append([ds, method + ' → C2b', r['n'], r['raw_response_equal'], r['predicted_answer_equal'], r['EM_equal']])
        (assets / (ds + '_curve.svg')).write_text(curve_svg(ds, a['curves']), encoding='utf-8')
        write(assets / (ds + '_analysis.json'), a)
        write(assets / (ds + '_policy.json'), a['audit']['policy'])
        downloads.extend([(ds + ' 完整审计 JSON', ds + '_analysis.json'), (ds + ' 运行前封存策略', ds + '_policy.json')])
        browser[ds] = {'rows': a['dev_rows'], 'mutations': a['mutations'],
                       'feedback': {'C': a['original_C_feedback']['exposures'], 'C2b': a['audit']['exposures']}}
    coverage = []
    for ds, a in data.items():
        old = a['original_C_feedback']; c = old['counts']; new = a['audit']['counts']
        coverage.append([ds, 'C', c['selected_count'], c['same_question'] + c['similar_task'], pct(c['paired_rate']), len({p['failure_id'] for e in old['exposures'] for p in e['packets'] if p['pair_type'] != 'unpaired'}), sum(e['same_question'] + e['similar_task'] == 0 for e in old['exposures']), old['failure_context_survives_packet'], max(e['feedback_tokens'] for e in old['exposures'])])
        coverage.append([ds, 'C2b', new['selected'], new['paired'], pct(new['paired_rate']), new['unique_paired_questions'], new['operators_without_reference'], new['failure_context_present'], new['feedback_max_tokens']])
    diagnostic_rows = [[ds, name, n] for ds, a in data.items() for name, n in a['audit']['counts']['diagnoses'].items()]
    memory_rows = [[ds, m['round'], m['records'], m['success_observations'], m['success_questions']]
                   for ds, a in data.items() for m in a['audit']['memory_sizes']]
    assert data['searchqa']['audit']['policy']['method_hash'] == data['hotpotqa']['audit']['policy']['method_hash']
    for relative in data['searchqa']['audit']['policy']['method_files']:
        target = assets / 'method_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SESSION / 'searchqa/method_source' / relative, target)
    shutil.copyfile(ROOT / 'api/C2B.md', assets / 'C2B.md')
    downloads.append(('C2b 实现与运行说明', 'C2B.md'))
    hashseed_check = ROOT / 'Explore/runs/reports/C2b_hashseed_validation_20261110.json'
    if hashseed_check.exists():
        shutil.copyfile(hashseed_check, assets / 'hashseed_validation.json')
        downloads.append(('运行中已完成反馈的跨进程重放检查（覆盖数见文件）', 'hashseed_validation.json'))
    manual_path = SESSION / 'manual_review.json'
    manual = read(manual_path) if manual_path.exists() else {'summary': [], 'cases': []}
    manual_html = ''.join('<p>' + esc(s) + '</p>' for s in manual.get('summary', []))
    manual_cases = []
    for case in manual.get('cases', []):
        a = data[case['dataset']]
        row = next(r for r in a['dev_rows']['C2b'] if r['id'] == case['id'])
        assert digest_response(row['raw_response']) == case['response_sha256']
        manual_cases.append([case['dataset'], case['id'], case['category'], case['note']])
    reviewed = {(c['dataset'], c['id']) for c in manual.get('cases', [])}
    required = {(ds, r['id']) for ds, a in data.items() for r in a['dev_rows']['C2b'] if r['em'] == 0}
    required |= {(ds, r['id']) for ds, a in data.items() for comparison in a['comparisons'].values() for r in comparison['changes']}
    assert required <= reviewed, 'Complete the final-error and changed-case review before publishing the report'
    manual_html += ('<details><summary>展开最终 dev 错题及差异题的逐题复核</summary>' +
                    table(['数据集', '题目 ID', '复核类别', '证据与限制'], manual_cases) + '</details>') if manual_cases else ''
    finding_rows = []
    for finding in manual.get('candidate_findings', []):
        candidate = next(x for x in data[finding['dataset']]['mutations'][finding['method']] if x['child'] == finding['candidate'])
        assert digest_response(candidate['adopted_skill']) == finding['skill_sha256']
        finding_rows.append([finding['dataset'], finding['candidate'], finding['failure_id'], finding['category'], finding['note']])
    finding_html = '<h3>训练反馈与实际技能的具体复核</h3>' + table(['数据集', '候选', '训练题 ID', '发现', '证据与限制'], finding_rows) if finding_rows else ''
    downloads_html = ''.join(f'<li><a download href="{assets.name}/{name}">{esc(title)}</a></li>' for title, name in downloads)
    cfg = data['searchqa']['config']
    policies = table(['数据集', '共享协议代码 hash', 'C2b 方法 hash', '已验证检查'], [[ds, a['audit']['policy']['protocol_code_hash'], a['audit']['policy']['method_hash'], '；'.join(a['audit']['checks'])] for ds, a in data.items()])
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C2b train/dev 实验对比报告</title><style>
:root{--ink:#173444;--muted:#526877;--line:#d7e1e7;--accent:#a74726}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#f5f7f9;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}header{background:#173e50;color:white;padding:42px max(22px,calc((100% - 1220px)/2))}h1{font-size:32px;line-height:1.4;max-width:950px}h2{font-size:24px;margin:0 0 18px}h3{font-size:18px}p{max-width:1100px}nav{position:sticky;top:0;background:white;border-bottom:1px solid var(--line);padding:12px 22px;display:flex;gap:23px;overflow-x:auto;white-space:nowrap;z-index:2}a{color:#216382}main{max-width:1264px;padding:24px 22px 65px;margin:auto}section{padding:30px 0;scroll-margin-top:65px}.note{padding:18px 22px;background:#e7f0f4;border-left:3px solid #407e98;margin:18px 0}.limits{color:var(--muted);font-size:14px}.scroll{overflow:auto;max-width:100%;background:white}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px 14px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);min-width:85px}th{background:#e9f0f4;white-space:nowrap}td{overflow-wrap:anywhere}tr:hover td{background:#f3f7f9}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:center;margin:16px 0}select,input{max-width:100%;padding:9px;border:1px solid #aebfc9;background:white;color:var(--ink);font:inherit}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;border:1px solid var(--line);padding:16px;font:14px/1.7 ui-monospace,Consolas,monospace;max-height:650px;overflow:auto}details{border-top:1px solid var(--line);padding:12px 0}summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}.tag{color:var(--accent);font-weight:600}.response{background:white;border:1px solid var(--line);padding:15px;min-width:0}.response h3{margin-top:0}#question{font-size:18px;font-weight:600}footer{color:var(--muted);padding:24px 0;font-size:14px}code{overflow-wrap:anywhere}@media(max-width:720px){header{padding:25px 18px}h1{font-size:25px}main{padding:15px}.grid{grid-template-columns:1fr}nav{gap:18px}th,td{padding:10px}h2{font-size:21px}}@media print{nav,.controls{display:none}section{break-inside:avoid}pre{max-height:none}body{background:white}}
</style></head><body><header><div>Project 5 · Option A · train/dev only</div><h1>C2b：诊断与证据反馈<br>B / 原版 C / C2b 完整对比</h1><p>__DATE__ · 两套数据各三轮；沿用原预算、共享初始化和选择规则。没有执行 test，也没有追加评审模型调用。</p></header><nav><a href="#result">主要结果</a><a href="#protocol">实验边界</a><a href="#evolution">逐轮变化</a><a href="#feedback">反馈审计</a><a href="#cases">逐题对比</a><a href="#skills">技能变化</a><a href="#cost">预算与复现</a></nav><main>
<section id="result"><h2>1 · 主要结果</h2><ul>__OVERVIEW__</ul><div class="note">这是一次在已用于选择的 dev 上进行的方法迭代验证，样本量每套 30 道、训练 seed=42。任何改善都只能称为本次 dev 改善；不等同于独立 test 提升，也不能由这一次复合改动证明某个模块的因果贡献。</div>__METRICS__<p class="limits">延迟单位：秒，来自各自被选技能的原始 dev 观测；相同技能命中验证缓存时沿用原观测，未额外重测。历史 B/C 与本次 C2b 的服务负载可能不同。EM/F1 为原评分规则，未为名称别名或集合顺序改分。</p><h3>逐题修复与退化</h3>__PAIRS__<p class="limits">箭头左侧为参照。修复率=左侧错误中被 C2b 修复的比例；退化率=左侧正确中被 C2b 答错的比例。区间是固定 dev 题目上的 2,000 次配对 bootstrap 95% 区间，单位百分点；未包含训练 seed 或反复使用 dev 造成的不确定性。</p>__MANUAL__</section>
<section id="protocol"><h2>2 · C2b 改了什么，哪些保持固定</h2><ol><li>对训练失败标注保守诊断：答案类型、token 顺序、名称／粒度重叠、格式与未能定位的内容错误；不更改评分。</li><li>从完整训练上下文和响应中，按问题内容词、引用短语、年份及训练答案线索提取相关原文片段；记录字符位置与来源 hash。训练 gold 仅用于诊断与定位，QA 推理输入不含 gold。</li><li>用“触发条件 → 证据检查 → 行为 → 应保留的正确行为”组织反馈，并强调 mutation_other / mutation_graph 的章节范围。语义检查只记录结果，不拒绝、重抽或增加候选。</li><li>无成功参照时仍使用有证据和不确定性说明的失败反馈，不再逐字回退 B。配对检索本身保持原样，未实现 C2c。</li></ol><p>__CONFIG__</p><p>共享初始化按原账本计入每个方法的逻辑预算，物理费用不重复计算。C2b 通过独立反馈模块接入每个数据集原 B/C 的精确代码快照。原版 C 的文件和运行结果保持不变。</p><div class="note">修正前次文字说明：原 C 的首轮摘录上限为 <strong>160 token</strong>，不是 160 字符。本次改变的是本地优化器反馈的取材与组织方式；API 上下文窗口没有改变。</div><h3>相同第一轮输入的重复性检查</h3>__REPEAT__<p class="limits">每个比较含 2 个初始化父代 × 20 道训练题。输入技能、可见上下文及生成参数完全匹配；输出是否一致用于观察服务重复性。服务没有公开权重 revision，不能证明历史与当前权重完全相同。</p></section>
<section id="evolution"><h2>3 · 三轮演化与训练观察</h2><div class="grid">__CURVES__</div><p class="limits">纵轴 dev EM（%），横轴完整轮次。折线为每轮保留的最佳技能；重叠线可能被遮挡，请结合下方候选表。来源：各方法 history.json，本次报告生成时读取。</p><h3>所有候选，不只展示获胜者</h3>__CANDIDATES__<h3>输出截断与格式失败</h3>__HEALTH__<p class="limits">统计各条件新发出的物理请求和去重、未缓存的 QA 观测，不含共享初始化。length 表示服务返回达到输出上限；格式失败是原解析器未获得合规答案。两者不一定相等。候选的这些失败按原评分计入，不是最终所选技能的错误数；没有扩大 token 上限或追加重试。</p><h3>各轮父代 train 表现</h3>__TRAIN__<p class="limits">每轮 2 个父代在同一批 20 道题上各答一次，共 40 条观测。同轮按观测汇总；不同轮题目会变化。它不是最终技能在全部 100 道 train 上的独立评估，不能把跨轮变化当作同一训练集学习曲线。</p></section>
<section id="feedback"><h2>4 · 配对、证据和实际采用情况</h2>__COVERAGE__<p class="limits">按实际算子使用的父代反馈统计；同一父代被两个算子使用时重复计入曝光。原 C 全回退 B 的文本可能含上下文，但没有结构化 actual_packet，因此“结构化失败上下文”不等同于原 C 所有可见证据。覆盖率变化可能来自模型响应和父代演化，C2b 没有改变配对规则。</p>__QUALITY__<p class="limits">“范围外修改”指生成文档中部分章节未被采用，非整个候选无效。“新增 gold 行”是文本风险标记，需结合语义复核，不等于测试答案泄漏。统计的是实际采用技能相对父代的新增行，notes 不算实现。</p><div class="controls"><label>数据集 <select id="feedback-dataset"><option>searchqa</option><option>hotpotqa</option></select></label><label>方法 <select id="feedback-method"><option>C2b</option><option>C</option></select></label><label>调用 <select id="feedback-index"></select></label></div><p id="feedback-meta"></p><pre id="feedback-text"></pre><details><summary>查看结构化配对材料与完整原始记录</summary><pre id="feedback-packets"></pre></details></section>
<section id="cases"><h2>5 · 全部 dev 逐题对比与完整响应</h2><div class="controls"><label>数据集 <select id="case-dataset"><option>searchqa</option><option>hotpotqa</option></select></label><label>筛选 <select id="case-filter"><option value="all">全部题目</option><option value="wrong">C2b 错题</option><option value="changed">相对 B 或 C 正误变化</option></select></label><label>题目 <select id="case-index"></select></label></div><p id="question"></p><p id="gold"></p><div id="responses"></div><details><summary>相同的可见上下文</summary><pre id="case-context"></pre></details><p class="limits">此处仅供事后评估，不进入 C2b 训练记忆、反馈或 checkpoint 选择。报告包含所有 60 道 dev 题及三种方法的响应，避免只展示成功案例。</p></section>
<section id="skills"><h2>6 · 生成说明与实际技能差异</h2><div class="controls"><label>数据集 <select id="skill-dataset"><option>searchqa</option><option>hotpotqa</option></select></label><label>方法 <select id="skill-method"><option>C2b</option><option>C</option><option>B</option></select></label><label>候选 <select id="skill-index"></select></label></div><p id="skill-meta"></p><h3>实际采用的差异</h3><pre id="skill-diff"></pre><details><summary>此候选相对父代的逐题修复与退化</summary><pre id="skill-transitions"></pre></details><details><summary>此候选 dev 输出截断或格式异常的完整记录</summary><pre id="skill-anomalies"></pre></details><details><summary>优化器 notes 与实际采用的完整技能</summary><pre id="skill-notes"></pre><pre id="skill-full"></pre></details><details><summary>此候选的技能定义图</summary><img id="skill-graph" alt="当前候选的技能定义图"></details><h3>最终技能定义图</h3>__GRAPHS__<p class="limits">图展示技能文件中的节点和工作流定义，不声称观测到了模型内部节点执行轨迹。</p></section>
<section id="cost"><h2>7 · 预算、公平比较与复现</h2><p>本次新增 C2b 物理调用 <strong>__TOTAL_CALLS__</strong> 次，服务报告的输入 token <strong>__TOTAL_PROMPT__</strong>、生成 token <strong>__TOTAL_TOKENS__</strong>。离线诊断、审计及报告没有模型费用。</p>__BUDGETS__<p class="limits">“逻辑”含共享初始化一次；“物理”只计该条件新发出的请求。耗时为物理请求 wall time 之和，不是 GPU 小时。生成 token 相同上限不代表实际生成量相等。</p><h3>三方法共同调用／生成 token 上限的次要分析</h3>__MATCHED__<p class="limits">两个上限分别取 B/C/C2b 最终逻辑用量的最小值，然后各自取同时满足双上限的最新完整轮次（初始化为第 0 轮）。只使用该轮原本由 dev 选择的最佳技能，排除未完成轮次。这是账本约束比较，不是精确相同实际费用，也不是重新训练。</p><h3>代码与输入审计</h3>__POLICIES__<ul>__DOWNLOADS__</ul><p>复现入口：<code>python api/run_c2b.py searchqa --output 新目录 --source-ip 本机校园网IPv4</code>，HotpotQA 将数据集改为 <code>hotpotqa</code>。运行会校验封存配置和初始化，不会覆盖旧目录。查看 <code>api/C2B.md</code> 获取完整说明。</p><div class="note">结论边界：单 seed、小 dev、dev 参与选择、历史对照、服务 revision 未公开。C2b 同时改变诊断、取材和指令组织，尚不能分离各项的贡献。没有运行 C2c，也没有新的 test 结果；后续若要报告泛化收益，应在方法冻结后使用未见评测集。</div></section><footer>离线 HTML，无网络依赖。表格、图与逐题内容均来源于已完成运行；可下载完整审计 JSON 和技能文件。__DATE__</footer></main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);
function options(el,items){el.replaceChildren(...items.map(([v,t])=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o;}));}
function feedbackOptions(){const xs=DATA[$('feedback-dataset').value].feedback[$('feedback-method').value];options($('feedback-index'),xs.map((x,i)=>[i,`第 ${x.round} 轮 · ${x.child||x.child_id}`]));renderFeedback();}
function renderFeedback(){let xs=DATA[$('feedback-dataset').value].feedback[$('feedback-method').value],x=xs[+$('feedback-index').value];$('feedback-meta').textContent=`父代：${x.parent||x.parent_id}；实际反馈 ${x.feedback_tokens} tokens；同题参照 ${x.same_question}，跨题参照 ${x.similar_task}，无参照 ${x.unpaired}`;$('feedback-text').textContent=x.actual_feedback;$('feedback-packets').textContent=JSON.stringify(x.packets,null,2);}
function caseOptions(){const rows=DATA[$('case-dataset').value].rows;const by=m=>Object.fromEntries(rows[m].map(x=>[x.id,x]));const b=by('B'),c=by('C');let xs=rows.C2b.filter(x=>$('case-filter').value==='all'||($('case-filter').value==='wrong'?x.em===0:x.em!==b[x.id].em||x.em!==c[x.id].em));options($('case-index'),xs.map(x=>[x.id,`${x.em?'正确':'错误'} · ${x.question.slice(0,95)}`]));renderCase();}
function renderCase(){const rows=DATA[$('case-dataset').value].rows,id=$('case-index').value;const selected=Object.fromEntries(['B','C','C2b'].map(m=>[m,rows[m].find(x=>x.id===id)]));const x=selected.C2b;if(!x){$('question').textContent='此筛选没有匹配题目';$('responses').replaceChildren();$('gold').textContent='';$('case-context').textContent='';return;}$('question').textContent=x.question;$('gold').textContent=`ID: ${id} · 参考答案：${x.gold_answers.join(' / ')}`;$('case-context').textContent=x.visible_context;$('responses').replaceChildren(...['B','C','C2b'].map(m=>{let r=selected[m],d=document.createElement('div');d.className='response';let h=document.createElement('h3');h.textContent=`${m} · EM ${r.em} · F1 ${r.f1.toFixed(4)} · ${r.latency_seconds.toFixed(2)} 秒`;let p=document.createElement('p');p.textContent='预测：'+r.predicted_answer;let pre=document.createElement('pre');pre.textContent=r.raw_response;d.append(h,p,pre);return d;}));}
function skillOptions(){const xs=DATA[$('skill-dataset').value].mutations[$('skill-method').value];options($('skill-index'),xs.map((x,i)=>[i,`${x.child} · EM ${x.child_em===undefined?'未接受':(x.child_em*100).toFixed(2)+'%'}`]));renderSkill();}
function renderSkill(){const x=DATA[$('skill-dataset').value].mutations[$('skill-method').value][+$('skill-index').value];$('skill-meta').textContent=x.accepted?`父代 ${x.parent}；保留 ${x.retained}；无变化 ${x.no_op}；被舍弃章节 ${x.discarded_sections.join(', ')||'无'}；新增 gold 行 ${x.new_gold_lines.length}；修复 ${x.dev_transitions.counts.repaired} / 退化 ${x.dev_transitions.counts.regressed}`:'候选在原有尝试上限内未接受';$('skill-diff').textContent=x.diff||'无实际差异';$('skill-transitions').textContent=JSON.stringify(x.dev_transitions||{},null,2);$('skill-anomalies').textContent=x.output_anomalies?.length?JSON.stringify(x.output_anomalies,null,2):'此候选没有输出截断或格式失败记录';$('skill-notes').textContent=JSON.stringify({notes:x.notes||[],attempts:x.attempt_log,training_answer_reuse_flags:x.training_answer_reuse_flags||[]},null,2);$('skill-full').textContent=x.adopted_skill||'';$('skill-graph').hidden=!x.svg;if(x.svg)$('skill-graph').src=x.svg;}
['feedback-dataset','feedback-method'].forEach(id=>$(id).addEventListener('change',feedbackOptions));$('feedback-index').addEventListener('change',renderFeedback);['case-dataset','case-filter'].forEach(id=>$(id).addEventListener('change',caseOptions));$('case-index').addEventListener('change',renderCase);['skill-dataset','skill-method'].forEach(id=>$(id).addEventListener('change',skillOptions));$('skill-index').addEventListener('change',renderSkill);feedbackOptions();caseOptions();skillOptions();
</script></body></html>'''
    values = {
        'DATE': datetime.now().strftime('%Y-%m-%d %H:%M'), 'OVERVIEW': ''.join(overview),
        'METRICS': table(['数据集', '方法', '正确/n', 'EM', 'F1', '平均秒', 'p50秒', 'p95秒', '基础设施错误', '拒答', '格式失败'], metrics),
        'PAIRS': table(['数据集', '比较口径', '参照 → 新方法', '修复', '退化', '修复率', '退化率', '都对', '都错', 'ΔEM百分点', '95%区间'], pair_rows),
        'MANUAL': manual_html,
        'CONFIG': f'固定设置：seed={cfg["seed"]}；每轮每父代 {cfg["train_batch_size"]} 道 train；失败 k={cfg["failure_k"]}；3 轮 × 2 个候选；QA 输入上下文上限 {cfg["qa_context_max_tokens"]} tokens，QA 输出 {cfg["qa_max_tokens"]}，算子输出 {cfg["operator_max_tokens"]}，技能和反馈各 {cfg["feedback_max_tokens"]}；QA temperature={cfg["qa_temperature"]}，算子 temperature={cfg["operator_temperature"]}，并发 1，最多 2 次尝试。每方法预算：600 次调用、700,000 生成 tokens、9 小时。生存选择按 EM，平分优先父代和原顺序；最终最佳按 EM 与当前顺序。父代排名仍使用上游 EM/F1 规则。',
        'REPEAT': table(['数据集', '比较', '相同输入数', '原响应相同', '预测相同', 'EM相同'], repeat),
        'CURVES': ''.join(f'<img src="{assets.name}/{ds}_curve.svg" alt="{ds} 四个检查点的 dev EM">' for ds in data),
        'CANDIDATES': table(['数据集', '方法', '轮', '候选', '算子', '父代', 'dev EM', 'dev F1', '被保留', '尝试'], candidate_rows),
        'HEALTH': table(['数据集', '方法', '新物理调用', 'train length', 'dev length', '算子 length', 'train 格式失败', 'dev 格式失败', 'dev 二者重叠'], health_rows),
        'TRAIN': table(['数据集', '方法', '轮', '父代观测数', '正确', 'train EM', 'train F1'], train_rows),
        'COVERAGE': table(['数据集', '方法', '失败曝光', '配对曝光', '覆盖率', '不同配对题', '无参照调用', '结构化失败上下文', '最大反馈token'], coverage)
                    + finding_html
                    + '<p class="limits">原文片段存在不代表关键证据完整，也不证明语义相关或因果关系；诊断和检索结果仍需结合完整记录复核。</p>'
                    + '<h3>C2b 实际曝光的诊断类别</h3>' + table(['数据集', '保守诊断类别', '曝光次数'], diagnostic_rows)
                    + '<h3>C2b 各轮训练记忆</h3>' + table(['数据集', '轮', '记忆记录', '成功观测', '不同成功题目'], memory_rows),
        'QUALITY': table(['数据集', '方法', '已接受候选', '含范围外修改', '新增gold相关行', '训练答案字面复用标记', '无变化候选', '总算子尝试', '无效/失败算子尝试'], quality)
                   + '<p class="limits">字面复用标记：实际新增行包含所选训练失败的参考答案或预测短语（归一化后至少 4 字符，排除布尔词与常用拒答模板）。它可定位复制风险，但常用词也可能命中，需人工复核；未作为候选拒绝或重试条件。</p>',
        'GRAPHS': ''.join(graphs), 'TOTAL_CALLS': total_calls, 'TOTAL_TOKENS': f'{total_tokens:,}', 'TOTAL_PROMPT': f'{total_prompt:,}',
        'BUDGETS': table(['数据集', '方法', '逻辑调用', '新物理调用', '逻辑生成token', '新生成token', '新输入token', '请求小时', '失败请求'], budgets),
        'MATCHED': table(['数据集', '方法', '共同调用上限', '共同生成上限', '轮', '最佳技能', 'dev EM', 'dev F1', '未用调用', '未用生成token'], matched_rows),
        'POLICIES': policies, 'DOWNLOADS': downloads_html,
        'DATA': json.dumps(browser, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')}
    for key, value in values.items(): page = page.replace('__' + key + '__', str(value))
    assert '__DATA__' not in page
    output.write_text(page, encoding='utf-8')
    if manual_path.exists(): shutil.copyfile(manual_path, assets / 'manual_review.json')
    shutil.copyfile(ROOT / 'Explore/runs/reports/C2b_offline_replay_20261110.json', assets / 'offline_replay.json')
    manifest = {str(p.relative_to(output.parent)).replace('\\', '/'): digest_response(p.read_bytes())
                for p in [output] + sorted(assets.rglob('*')) if p.is_file()}
    write(output.with_suffix('.checksums.json'), manifest)
    bundle = output.with_name(output.stem + '_bundle.zip')
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in [output, output.with_suffix('.checksums.json')] + sorted(assets.rglob('*')):
            if p.is_file(): archive.write(p, p.relative_to(output.parent))
    print(json.dumps({'html': str(output), 'bundle': str(bundle), 'new_calls': total_calls, 'generated_tokens': total_tokens}, ensure_ascii=False))


def digest_response(value):
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


if __name__ == '__main__':
    build()
