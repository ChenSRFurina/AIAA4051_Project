"""Render the offline report, replay and audit appendix from verified final evidence."""
import difflib
import html
import json
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/final_project"
E=lambda v:html.escape(str(v),quote=True)
def read(n): return json.loads((OUT/(n+".json")).read_text(encoding="utf-8"))
def save(n,s): (OUT/n).write_text(s,encoding="utf-8")
def percent(x): return f"{100*x:.2f}"
def tbl(head,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+E(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+E(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def embedded(v): return json.dumps(v,ensure_ascii=False,separators=(',',':')).replace('<',r'\u003c')
def details(title,body): return '<details><summary>'+E(title)+'</summary>'+body+'</details>'
def pre(x):return '<pre>'+E(x)+'</pre>'

CSS="""*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f3ee;color:#1c303a;font:16px/1.55 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1160px;margin:auto;padding:42px clamp(18px,4vw,54px) 65px}h1,h2,h3,p{margin:0 0 .7em}h1,h2{line-height:1.14}h1{font-size:clamp(32px,4.4vw,56px);letter-spacing:-.035em}h2{font-size:27px;margin-top:1.15em}h3{font-size:18px}small,.label{color:#47716b;font-size:13px}.label{text-transform:uppercase;letter-spacing:.09em;font-weight:700}a{color:#006858}nav,.controls{display:flex;align-items:center;gap:12px 19px;flex-wrap:wrap}nav{padding:12px 0;margin:14px 0;border-block:1px solid #a7b9b1}button,select,input{font:inherit;border:1px solid #a8b9b3;padding:5px 9px;border-radius:4px;background:white;color:inherit}button,summary{cursor:pointer}button:disabled{opacity:.45}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;background:white;font-size:.88em}td,th{text-align:left;padding:7px 10px;vertical-align:top;border-bottom:1px solid #cdd7d0}th{background:#e2ebe4;font-weight:650}code,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#fff;padding:14px;border-left:3px solid #7ba195;font:14px/1.45 Consolas,monospace}details{padding:8px 0;border-bottom:1px solid #d1d9d4}summary{font-weight:600}section{margin-top:22px}.lead{font:1.24em/1.5 Georgia,'Microsoft YaHei',serif;max-width:80ch}article.case{border-top:1px solid #b9c9c0;padding:14px 0}.diagram{max-width:100%;max-height:650px;display:block;background:white}.sub{color:#49625f}.rule{border-left:4px solid #1d7766;background:#e9f0e9;padding:12px 16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:650px){.grid{grid-template-columns:1fr}}"""
def shell(title,body,extra='',lang='en'):
    return f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{E(title)}</title><style>{CSS}{extra}</style></head><body>{body}</body></html>'

def svg_graph(dataset,arm,g):
    routes=g["routes"]
    w=840
    def lines(text,width):return textwrap.wrap(text,width=width,break_long_words=False) or [""]
    def txt(x,y,rows,style=""):
        return ''.join(f'<text x="{x}" y="{y+i*20}" {style}>{E(row)}</text>' for i,row in enumerate(rows))
    mark='<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#317366"/></marker></defs>'
    objects=[mark,txt(18,28,[f"{dataset}  /  {arm}  /  {g['id']}"],'font-weight="700"'),txt(18,60,['Select a task graph using its “Use when” instruction.'])]
    y=88
    if not routes:
        objects.append(txt(18,y+25,["No task graph or execution nodes in this skill.","The global guidance is available in the appendix."]))
        y+=75
    for route in routes:
        cond=lines(route["use_when"],88)
        objects.append(f'<path d="M24 70 V{y+22} H48" fill="none" stroke="#317366" marker-end="url(#arr)"/>')
        objects.extend([f'<rect x="52" y="{y}" width="765" height="{37+20*len(cond)}" fill="#e5efe9"/>',txt(65,y+22,[route["task_type"]],'font-weight="700"'),txt(65,y+44,cond)])
        y+=49+len(cond)*20
        for j,node in enumerate(route["sequence"]):
            objects.extend([f'<path d="M84 {y-12} v12" stroke="#317366" marker-end="url(#arr)"/>',
                            f'<rect x="68" y="{y+2}" width="745" height="32" fill="white" stroke="#aac0b6"><title>{E(g["nodes"].get(node,""))}</title></rect>',
                            txt(79,y+24,[f"{j+1}. {node}"])])
            y+=46
        y+=12
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{y}" viewBox="0 0 {w} {y}" role="img"><title>{E(dataset+" "+arm+" "+g["id"])}</title><g font-family="Arial,sans-serif" font-size="15" fill="#19352e">'+''.join(objects)+'</g></svg>'

def appendix(summary,graphs,inventory):
    gp=OUT/"graphs"; gp.mkdir(exist_ok=True)
    for ds,methods in graphs.items():
        for arm,timeline in methods.items():
            for key,g in timeline["graphs"].items():
                (gp/f"{ds}-{arm}-{key}.svg").write_text(svg_graph(ds,arm,g),encoding="utf8")
    body="""<main><header><p class="label">AIAA 4051 · Project 5</p><h1>Evidence appendix</h1><p class="lead">Inspect the graph used at each generation and every EM-zero observation of the validation-selected baseline.</p></header>
    <nav><a href="report.pdf">Six-page paper</a><a href="demo.html">Guided demonstration</a><a href="#fail">Failure review</a><a href="#files">Reproduction files</a></nav>
    <h2>Skill evolution</h2><div class="controls"><label>Dataset <select id="ds"><option>hotpotqa</option><option>searchqa</option></select></label><label>Method <select id="method"><option>B</option><option>C</option><option>C2b</option><option>C2c</option></select></label><label>Candidate <select id="candidate"></select></label></div><div id="evolution"></div>
    <h2 id="fail">Complete failure inventory</h2><p>Each record has one primary label. Development notes interpret the recorded response. Test labels describe the question, prediction and reference after freezing. A reference concern is a review flag. All original EM and F1 values are retained. The literal context excerpt is selected by text overlap.</p><div class="controls"><select id="errds" aria-label="Failure dataset"><option>hotpotqa</option><option>searchqa</option></select><select id="split" aria-label="Failure split"><option>dev</option><option>test</option></select><select id="category" aria-label="Failure category"></select><input id="find" aria-label="Search failures" placeholder="Question or original ID"></div><p id="freq"></p><div class="controls"><button id="prev">Previous</button><span id="pages"></span><button id="next">Next</button></div><div id="cases"></div>
    <h2 id="files">Reproduction files</h2><nav><a href="README.md">Runbook</a><a href="fixed_ids.json">Fixed IDs</a><a href="summary.json">Results, budgets and freezes</a><a href="predictions.json">Paired predictions</a><a href="failure_inventory.csv">Every reviewed failure (CSV)</a><a href="skill_graphs.json">Complete skill graphs</a><a href="operator_inputs.json">Recorded operator inputs</a><a href="gpu_log.json">GPU service log</a><a href="demo_evidence.json">Demonstration evidence</a></nav></main>"""
    js="""const el=id=>document.getElementById(id),graphs=JSON.parse(el('gs').textContent),fail=JSON.parse(el('fs').textContent),labels=JSON.parse(el('lb').textContent);
    function h(tag,text='',cls=''){let v=document.createElement(tag);v.textContent=text;v.className=cls;return v}function opt(s,t){let o=h('option',t);o.value=s;return o}function detail(title,value){let d=h('details');d.append(h('summary',title),h('pre',value));return d}
    function loadOpts(){const g=graphs[el('ds').value][el('method').value];el('candidate').replaceChildren(...Object.keys(g.graphs).map(k=>opt(k,k+' · '+g.graphs[k].correct+' correct')));el('candidate').value=g.points.at(-1).best_id;graph()}
    function graph(){let ds=el('ds').value,a=el('method').value,t=graphs[ds][a],g=t.graphs[el('candidate').value],box=el('evolution');let img=h('img');img.src='graphs/'+ds+'-'+a+'-'+g.id+'.svg';img.alt=g.id+' routing and ordered execution nodes';img.className='diagram';let title=h('h3','Generation '+g.round+' · '+g.operator);let changes=g.changes.map(x=>x.type+' · '+x.question+' ['+x.id+']'+'\\n'+x.before+' → '+x.after+'; reference '+x.gold.join(' / ')).join('\\n\\n');let meta='Parent '+(g.parent.join(', ')||'—')+' · validation correct '+g.correct+' / '+(ds==='searchqa'?30:150);let trend=t.points.map(p=>'R'+p.round+'  '+p.best_id+'  ('+Math.round(p.em*(ds==='searchqa'?30:150))+')').join(' → ');box.replaceChildren(h('p',trend,'rule'),title,h('p',meta),img,detail('Node conditions and complete adopted skill',g.text),detail('All development repairs and regressions against this parent ('+g.changes.length+')',changes||'No changed EM observations.'))}
    el('ds').onchange=el('method').onchange=loadOpts;el('candidate').onchange=graph;loadOpts();el('category').append(opt('all','All categories'),...Object.keys(labels).map(k=>opt(k,labels[k][0])));
    let page=0;function cases(){let selected=fail.filter(r=>r.dataset===el('errds').value&&r.split===el('split').value),counts={};selected.forEach(r=>counts[r.category]=(counts[r.category]||0)+1);el('freq').textContent=Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([k,v])=>labels[k][0]+' '+v).join(' · ');let q=el('find').value.trim().toLowerCase(),rows=selected.filter(r=>(el('category').value==='all'||r.category===el('category').value)&&(!q||(r.id+' '+r.question).toLowerCase().includes(q)));page=Math.min(page,Math.max(0,Math.ceil(rows.length/8)-1));el('pages').textContent=rows.length+' records · '+(rows.length?page+1:0)+' / '+Math.ceil(rows.length/8);el('prev').disabled=!page;el('next').disabled=(page+1)*8>=rows.length;let b=el('cases');b.replaceChildren();for(const r of rows.slice(page*8,page*8+8)){let d=h('article','','case');d.append(h('small',r.id+' · '+labels[r.category][0]),h('h3',r.question),h('p','B: '+r.predicted_answer),h('p','Reference: '+r.gold_answers.join(' / ')+' · EM 0 · F1 '+r.f1.toFixed(3)),h('p',r.diagnosis),detail('Recorded model output and literal passage excerpt',r.raw_response+'\\n\\n'+r.context_excerpt));b.append(d)}} for(let k of ['errds','split','category','find'])el(k).oninput=()=>{page=0;cases()};el('prev').onclick=()=>{page--;cases()};el('next').onclick=()=>{page++;cases()};cases();"""
    for key,value in (('gs',graphs),('fs',inventory),('lb',summary["taxonomy"])): body+=f'<script type="application/json" id="{key}">{embedded(value)}</script>'
    save("appendix.html",shell("GraphSkillEvo · Evidence appendix",body+'<script>'+js+'</script>'))


def demo(summary,graphs):
    e=read("demo_evidence"); packet=e["packet"]
    arm=graphs["hotpotqa"]["C2c"]["graphs"]; parent=arm[e["parent"]]; child=arm[e["child"]]
    added=[l[1:] for l in difflib.unified_diff(parent["text"].splitlines(),child["text"].splitlines()) if l.startswith('+') and not l.startswith('+++')]
    chosen=next(x for x in child["changes"] if x["before"]=="operas")
    stages=[
      ("训练中的一个复数","失败样本",f'<p class="lead">{E(packet["question"])}</p>'+tbl(["C2c 父代回答","训练参考","EM"],[[packet["predicted"]," / ".join(packet["train_gold"]),0]])+'<p>模型找到了职业，评分器保留单复数差异。这个训练样本能够支持对输出形式的检查；它无法证明模型漏掉了推理步骤。</p>'),
      ("优化器真正收到的参照","训练反馈",f'<p class="lead">{E(packet["success"]["question"])}</p><p>同一方法此前答对了这道训练题，回答为 <b>{E(packet["success"]["predicted"])}</b>。检索器把两题放进同一份反馈。这份参照展示了一次简短作答，但无法单独解释失败题的复数问题。</p>'+details("展开当次算子实际收到的反馈",pre(e["feedback"]))),
      ("第三轮采用了什么修改","技能变异",'<p>此次算子只更新全局指导。以下为实际采用的新增文字。</p>'+pre(next(l for l in added if 'plural' in l))+'<p>这条指令提到了训练例中的单复数，也要求匹配 gold。部署时模型看不到参考答案，且评分器本身忽略大小写。这个不准确的要求保留在冻结技能及报告中。</p>'),
      ("同一子代带来两次修复与一次退化","验证集",tbl(["变化","问题","父代","子代","参考"],[[x["type"],x["question"],x["before"],x["after"]," / ".join(x["gold"])] for x in child["changes"]])+f'<p>验证正确数从 {parent["correct"]}/150 变为 {child["correct"]}/150。第一类现象可以由答案形式解释。一次变异同时改了多条规则，逐题变化不能单独归因到某一句。</p>'),
      ("同预算下怎样解读最终结果","冻结测试",tbl(["数据集","B EM / F1","原版 C EM / F1","C2c EM / F1"],[[ds,*[percent(m["arms"][a]["test"]["em"])+" / "+percent(m["arms"][a]["test"]["f1"]) for a in ("B","C","C2c")]] for ds,m in summary["datasets"].items()])+'<p>表内为百分数。原版 C 是事先指定的主比较，C2c 是扩展。本次 HotpotQA 上，原版 C 对 B 各修复、退化 20 题；C2c 对 B 修复 55 题、退化 34 题。共同开发预算下 B、C 在 HotpotQA 选中相同文本，复用同一份测试观测。</p>')]
    parts=[]
    for i,(title,label,body) in enumerate(stages):
        parts.append(f'<article class="step" id="s{i}"'+(' hidden' if i else '')+f'><p class="label">{i+1} / 5 · {E(label)}</p><h1>{E(title)}</h1>{body}</article>')
    body='<main><nav><a href="report.pdf">课程报告</a><a href="appendix.html">逐题证据</a><a href="demo_evidence.json">原始反馈</a></nav>'+''.join(parts)+'<div class="controls"><button id="back">上一步</button><button id="go">下一步</button><span id="pos" aria-live="polite"></span></div></main>'
    js="""let i=0;const steps=[...document.querySelectorAll('.step')];function show(n){i=Math.max(0,Math.min(4,n));steps.forEach((s,k)=>s.hidden=k!==i);document.getElementById('back').disabled=!i;document.getElementById('go').disabled=i===4;document.getElementById('pos').textContent=(i+1)+' / 5';}document.getElementById('back').onclick=()=>show(i-1);document.getElementById('go').onclick=()=>show(i+1);document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(i+1);if(e.key==='ArrowLeft')show(i-1)});show(0);"""
    save("demo.html",shell("训练反馈与冻结测试 · 现场演示",body+"<script>"+js+"</script>",".step{min-height:340px;margin:35px 0}.step h1{margin:12px 0 24px}","zh-CN"))



def index(summary):
    import csv
    rows = []
    for ds, dataset in summary["datasets"].items():
        for arm, condition in dataset["arms"].items():
            test = condition["test"]
            rows.append([ds, arm, condition["dev"]["correct"], test["correct"], test["n"],
                         percent(test["em"]), percent(test["f1"]), f"{test['mean_latency_seconds']:.3f}",
                         condition["test_source"]])
    columns = ["dataset","arm","dev_correct","test_correct","test_n","test_em_pct","test_f1_pct","mean_latency_s","test_source"]
    with (OUT/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    links='<nav><a href="report.pdf">英文论文 PDF · 6 页</a><a href="report.html">论文网页版</a><a href="appendix.html">技能图与错题附录</a><a href="demo.html">五步离线演示</a><a href="README.md">复现与讲解指南</a></nav>'
    body='<main><p class="label">AIAA 4051 · 2026 Fall · Project 5</p><h1>成功参照能帮助模型从失败中改进技能吗？</h1><p class="lead">同一台 RTX 4070 Ti SUPER、同一份 Qwen3.5-9B 四比特权重。基线 B 与原版 C 的主比较覆盖两套固定测试集。</p>'+links
    body+='<h2>固定测试集对照</h2><p>EM 和 F1 以百分数计，延迟是单题实测均值。N 为无技能，I 为初始技能，B 为失败反馈，C 为原版成功／失败对照。C2b 增加可观察诊断与证据，C2c 进一步按题目所求类型检索参照。</p>'
    body+=tbl(["数据集","方法","dev 正确","test 正确 / 总数","test EM","test F1","秒/题"],[r[:3]+[f"{r[3]} / {r[4]}"]+r[5:8] for r in rows])
    body+='<p class="rule">HotpotQA 上，原版 C 相对 B 修复 20 题、退化 20 题。C2c 修复 55 题、退化 34 题，净增 21 题。SearchQA 上原版 C 净减 2 题，C2c 净减 5 题。一次训练种子下，结果显示明显的数据集差异。</p>'
    body+='<h2>相同开发预算下的原版 B / C</h2>'
    matches=[]
    for ds,d in summary["datasets"].items():
        for a,m in d["matched"].items():
            c=m["checkpoint"]
            matches.append([ds,a,c["round"],f"{c['cost']['calls']:,} / {d['caps']['calls']:,}",f"{c['cost']['generated_tokens']:,} / {d['caps']['tokens']:,}",percent(m["metrics"]["em"]),percent(m["metrics"]["f1"])])
    body+=tbl(["数据集","方法","轮次","逻辑调用 / 上限","生成 tokens / 上限","test EM","test F1"],matches)
    body+='<p>取两组均可负担的最后一个完整轮次。HotpotQA 的 B 第三轮和 C 第二轮选中完全相同的文本，1,000 个测试提示均通过逐字校验，因此复用 B 的实际预测。SearchQA 的 C2b 与 C2c 同样共享一次测试观测。</p>'
    body+='<h2>课程要求对应材料</h2>'+tbl(["课程部分","可以核查的内容"],[["Part 1 · 复现","两套数据、三轮选择、N/I/B、逐轮候选流程图、每个子代修复和退化的原题。"],["Part 2 · 失败分析","B 在 SearchQA dev/test 的 6/15 题及 HotpotQA 的 45/382 题，共 448 条错题、原始响应和诊断。"],["Part 3 · 优化过程","冻结原 C、两个后续变体、真实优化器输入、配对覆盖、共同调用与 token 预算和配对测试。"],["Part 4 · 报告与演示","英文六页论文和五步离线演示。演示可展开训练反馈，并展示同一子代带来的收益与退化。"]])
    body+='<nav><a href="comparison.csv">对照表 CSV</a><a href="failure_inventory.csv">错题 CSV</a><a href="fixed_ids.json">固定题目 ID</a><a href="protocols.json">协议与配置</a><a href="summary.json">完整指标</a><a href="gpu_log.json">GPU 启停台账</a></nav><p class="sub">2026-11-25 汇总。本机全部有记录时段合计 '+f'{summary["gpu"]["local_hours"]:.3f}'+' GPU 小时。此前队友 API 的 GPU 时间未获提供。SearchQA 的 50 个 ID 曾用于早期评测；新完成的本机结果属于同一划分上的复现。</p></main>'
    save("index.html",shell("Project 5 · 最终实验与课程材料",body,lang="zh-CN"))


if __name__=="__main__":
    from final_course_report import paper
    summary=read("summary")
    graphs=read("skill_graphs")
    appendix(summary,graphs,read("failure_inventory"))
    demo(summary,graphs)
    index(summary)
    paper(summary)
    print("Rendered course report, comparison tables, evidence appendix, skill diagrams and demonstration.")
