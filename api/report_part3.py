"""Offline HTML for the sealed Option-A experiment; requires completed runs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import difflib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Baseline'))
from baseline.io import read_json,digest
from baseline.evaluation import aggregate
from part3_analysis import checked_rows,checkpoint_data
from report_part12 import skill_svg,lineage_svg,table,pct,esc,save
from run_part3 import BASELINES,SESSION


def curve_svg(dataset,b,c):
    bits=['<rect width="740" height="300" fill="white"/>',f'<text x="65" y="24" font-size="17">{dataset} · retained best dev EM</text>']
    for v in [0,.25,.5,.75,1]:
        y=240-v*180
        bits.append(f'<path d="M 65 {y} H 690" stroke="#e2e8ef"/><text x="20" y="{y+4}" font-size="12">{v*100:.0f}%</text>')
    for name,points,color in [('B',b,'#7d8793'),('C',c,'#1e6f89')]:
        pts=' '.join(f'{85+i*190},{240-p["em"]*180}' for i,p in enumerate(points))
        bits.append(f'<polyline points="{pts}" stroke="{color}" stroke-width="3" fill="none"/>')
        for i,p in enumerate(points):
            x,y=85+i*190,240-p['em']*180
            bits.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/><text x="{x}" y="{y+(-13 if name=="C" else 22)}" fill="{color}" text-anchor="middle" font-size="12">{name} {pct(p["em"])}</text>')
    for i in range(4):bits.append(f'<text x="{85+i*190}" y="265" text-anchor="middle" font-size="12">{"Init" if i==0 else "Round "+str(i)}</text>')
    bits.append('<text x="270" y="290" font-size="12">Evolution round · B gray / C blue</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 300" font-family="Arial,sans-serif" role="img">'+''.join(bits)+'</svg>'


def resolve_test(test_dir,condition):
    folder=test_dir/condition
    if (folder/'outputs.json').exists():return read_json(folder/'outputs.json')
    return read_json(read_json(folder/'reuse.json')['source_path'])


def paired_metric_row(dataset,split,label,comp):
    cc=comp['counts'];ci=comp['paired_bootstrap_95']
    return [dataset,split,label,cc['repaired'],cc['regressed'],
            pct(comp['repair_rate']) if comp['repair_rate'] is not None else 'NA',
            pct(comp['regression_rate']) if comp['regression_rate'] is not None else 'NA',
            cc['both_correct'],cc['both_wrong'],
            f"{comp['em_difference_C_minus_B']*100:+.2f}",
            f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"]


def service_metadata(run):
    calls=[json.loads(line) for line in (Path(run)/'calls.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    responses=[c['raw_response'] for c in calls if isinstance(c.get('raw_response'),dict)]
    return {'models':sorted({r['model'] for r in responses if r.get('model')}),
            'system_fingerprints':sorted({r['system_fingerprint'] for r in responses if r.get('system_fingerprint')}),
            'finish_reasons':dict(Counter(c.get('finish_reason','unknown') for c in calls)),
            'requests':len(calls)}


def initial_input_repeat(b_run,c_run):
    pairs=[]
    for parent in ['g0000_i0000','g0000_i0001']:
        paths=[Path(r)/f'evaluations/g1/{parent}/train/outputs.jsonl' for r in [b_run,c_run]]
        rows=[[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()] for p in paths]
        assert len(rows[0])==len(rows[1])==20
        for a,b in zip(*rows):
            assert all(a[k]==b[k] for k in ['id','skill_hash','visible_context_hash','generation_config'])
            pairs.append((a,b))
    return {'n':len(pairs),'raw_response_equal':sum(a['raw_response']==b['raw_response'] for a,b in pairs),
            'predicted_answer_equal':sum(a['predicted_answer']==b['predicted_answer'] for a,b in pairs),
            'EM_equal':sum(a['em']==b['em'] for a,b in pairs),
            'scope':'Same initial skills, first-round train inputs and generation config; observed reproducibility check, not proof of identical server weights.'}


def build(output,session=SESSION):
    output,session=Path(output).resolve(),Path(session).resolve()
    assets=output.parent/(output.stem+'_assets');assets.mkdir(parents=True,exist_ok=True)
    data={ds:read_json(session/ds/'analysis.json') for ds in BASELINES}
    assert all(a['status']=='passed' and a['feedback']['status']=='passed' for a in data.values())
    cfg=data['searchqa']['common_config']
    displayed_config_keys=['seed','train_batch_size','failure_k','max_attempts','qa_context_max_tokens','qa_max_tokens','operator_max_tokens','skill_max_tokens','feedback_max_tokens','qa_temperature','operator_temperature','top_p','concurrency']
    assert all(a['common_config'][k]==cfg[k] for a in data.values() for k in displayed_config_keys)
    settings=(f"固定设置：seed={cfg['seed']}，每父代每轮 {cfg['train_batch_size']} 道 train，失败抽样 k={cfg['failure_k']}；"
              f"QA 可见上下文上限 {cfg['qa_context_max_tokens']:,} tokens，QA 输出上限 {cfg['qa_max_tokens']:,}，算子输出上限 {cfg['operator_max_tokens']:,}，"
              f"技能与反馈上限均为 {cfg['skill_max_tokens']:,} tokens；QA temperature={cfg['qa_temperature']}，算子 temperature={cfg['operator_temperature']}，"
              f"top_p={cfg['top_p']}，并发={cfg['concurrency']}，最多 {cfg['max_attempts']} 次尝试。模型为 qwen3.5-9b，服务上下文 32,768 tokens，关闭 thinking。")
    test=read_json(session/'searchqa/test/results.json')
    assert test['status']=='completed'
    comparisons=test['comparisons']
    metrics=[];pairs=[];budget=[];prefix=[];coverage=[];exposures=[];case_rows=[];galleries=[];round_rows=[];health=[]
    total_new=Counter()
    for ds,a in data.items():
        base=BASELINES[ds];c_run=session/ds/'C'
        for condition in ['N','I']:
            s=read_json(base/condition/'summary.json');m=s['dev']
            metrics.append([ds,'dev',condition,m['correct'],m['n'],pct(m['em']),pct(m['f1']),f"{m['mean_latency_seconds']:.2f}",f"{m['p50_latency_seconds']:.2f}",f"{m['p95_latency_seconds']:.2f}"])
        for condition in ['B','C']:
            m=a[condition]['dev'];cost=a[condition]['cost']
            metrics.append([ds,'dev',condition,m['correct'],m['n'],pct(m['em']),pct(m['f1']),f"{m['mean_latency_seconds']:.2f}",f"{m['p50_latency_seconds']:.2f}",f"{m['p95_latency_seconds']:.2f}"])
            run=base/'B' if condition=='B' else c_run
            ledger=[json.loads(l) for l in (run/'ledger.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
            actual={k:sum(e.get(k,0) for e in ledger if e.get('physical_calls',0)) for k in ['physical_calls','prompt_tokens','generated_tokens']}
            if condition=='C':total_new.update(actual)
            budget.append([ds,condition,cost['calls'],cost['physical_calls'],f"{cost['prompt_tokens']:,}",f"{cost['generated_tokens']:,}",f"{actual['generated_tokens']:,}"])
        for label,comp in [('完整三轮',a['dev_comparison']),('共同预算截点',a['matched']['dev_comparison'])]:
            pairs.append(paired_metric_row(ds,'dev',label,comp))
        m=a['matched']
        for cond in ['B','C']:
            x=m[cond];prefix.append([ds,cond,m['call_cap'],f"{m['generated_token_cap']:,}",x['round'],x['best_id'],x['cost']['calls'],f"{x['cost']['generated_tokens']:,}",pct(x['em']),x['unused_calls'],f"{x['unused_generated_tokens']:,}"])
        f=a['feedback'];ct=f['counts']
        fallback=sum(e['same_question']+e['similar_task']==0 for e in f['exposures'])
        coverage.append([ds,f['operator_exposures'],f['actual_attempt_exposures'],f['unique_parent_feedbacks'],ct['selected_count'],ct['same_question'],ct['similar_task'],ct['unpaired'],pct(ct['paired_rate']) if ct['paired_rate'] is not None else 'NA',fallback,f['unknown_question_type']])
        for ex in f['exposures']:
            exposures.append(dict(ex,dataset=ds))
        b_rows=read_json(base/'B/final_dev_outputs.json');c_rows=read_json(c_run/'final_dev_outputs.json')
        b_by_id={r['id']:r for r in b_rows}
        checked_rows(c_rows,ds)
        for c in c_rows:
            b=b_by_id[c['id']]
            state={(0,0):'both_wrong',(0,1):'repaired',(1,0):'regressed',(1,1):'both_correct'}[(b['hard'],c['hard'])]
            case_rows.append({'dataset':ds,'split':'dev','comparison':'full','id':c['id'],'question':c['question'],'gold':c['gold_answers'],'B':b['predicted_answer'],'C':c['predicted_answer'],'B_em':b['em'],'C_em':c['em'],'B_f1':b['f1'],'C_f1':c['f1'],'B_raw':b['raw_response'],'C_raw':c['raw_response'],'context':c['visible_context'],'state':state})
        init=read_json(base/'shared_init/population.json');hist=read_json(c_run/'history.json')
        attempts=[t for h in hist for o in h['operator_records'] for t in o['attempts']]
        qa=[json.loads(line) for path in (c_run/'evaluations').rglob('outputs.jsonl') for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        health.append([ds,len(hist),len(qa),sum(bool(r['error']) for r in qa),sum(bool(r.get('provider_refusal')) for r in qa),sum(not r['format_ok'] for r in qa),sum(t['status']!='ok' for t in attempts),len(list((c_run/'cache_events').glob('*.json'))),service_metadata(c_run)['finish_reasons'].get('length',0)])
        points,records,rows_by_id=checkpoint_data(c_run,base/'shared_init')
        _,_,b_observations=checkpoint_data(base/'B',base/'shared_init')
        mb=b_observations[a['matched']['B']['best_id']]
        mc=rows_by_id[a['matched']['C']['best_id']]
        checked_rows(mb,ds);checked_rows(mc,ds)
        for b,c in zip(mb,mc):
            assert b['id']==c['id'] and b['visible_context_hash']==c['visible_context_hash']
            state={(0,0):'both_wrong',(0,1):'repaired',(1,0):'regressed',(1,1):'both_correct'}[(b['hard'],c['hard'])]
            case_rows.append({'dataset':ds,'split':'dev','comparison':'matched','id':c['id'],'question':c['question'],'gold':c['gold_answers'],'B':b['predicted_answer'],'C':c['predicted_answer'],'B_em':b['em'],'C_em':c['em'],'B_f1':b['f1'],'C_f1':c['f1'],'B_raw':b['raw_response'],'C_raw':c['raw_response'],'context':c['visible_context'],'state':state})
        curve=curve_svg(ds,a['B_curve'],a['C_curve']);lineage=lineage_svg(ds+' / C',init,hist)
        (assets/f'{ds}-BC-curve.svg').write_text(curve,encoding='utf-8')
        (assets/f'{ds}-C-lineage.svg').write_text(lineage,encoding='utf-8')
        graph_cards=[]
        for key,item in records.items():
            graph=read_json(c_run/'skills'/(key+'.json'))
            assert graph['hash']==digest(item['content'])
            svg=skill_svg(graph,f'{ds} / C / {key} / dev EM {pct(item["val_hard"])}')
            (assets/f'{ds}-C-{key}.svg').write_text(svg,encoding='utf-8')
            graph_cards.append(f'<details class="skill"><summary>{key} · {item["origin"]} · dev EM {pct(item["val_hard"])}</summary><div class="diagram">{svg}</div><a download href="{assets.name}/{ds}-C-{key}.svg">下载 SVG</a><details><summary>技能原文</summary><pre>{esc(item["content"])}</pre></details></details>')
        for h in hist:
            for item in h['candidates']:
                parent=records[item['parent_ids'][0]]
                diff='\n'.join(difflib.unified_diff(parent['content'].splitlines(),item['content'].splitlines(),fromfile=parent['id'],tofile=item['id'],lineterm=''))
                round_rows.append([ds,h['generation'],item['id'],item['origin'],esc(', '.join(item['parent_ids'])),pct(item['val_hard']),pct(item['val_soft']),'保留' if item['id'] in h['survivor_ids'] else '淘汰',f'<details><summary>实际差异</summary><pre>{esc(diff)}</pre></details>'])
        bskill=(base/'B/best_skill.md').read_text(encoding='utf-8');cskill=(c_run/'best_skill.md').read_text(encoding='utf-8')
        final_diff='\n'.join(difflib.unified_diff(bskill.splitlines(),cskill.splitlines(),fromfile='B final',tofile='C final',lineterm=''))
        galleries.append(f'<article class="panel"><h3>{ds}</h3>{curve}<p>B 最终 {a["B"]["best"]["id"]}；C 最终 {a["C"]["best"]["id"]}。C 相对共同初始化最佳技能的 dev 变化为 {(points[-1]["em"]-points[0]["em"])*100:+.2f} pp。</p><div class="diagram">{lineage}</div><details><summary>B/C 最终技能的实际文本差异</summary><pre>{esc(final_diff or "内容相同")}</pre></details><details class="gallery"><summary>C 初始化及全部候选的 8 份流程图</summary>'+''.join(graph_cards)+'</details></article>')
        save(assets/f'{ds}-analysis.json',a)
        save(assets/f'{ds}-policy.json',read_json(session/ds/'policy.json'))
    testdir=session/'searchqa/test'
    frozen=read_json(session/'searchqa/test_frozen.json')
    save(assets/'test_frozen.json',frozen)
    prior=Path(frozen['baseline_test_dir'])
    for condition in ['N','I','B','C']:
        m=test['conditions']['C'] if condition=='C' else test['baseline_conditions'][condition]
        metrics.append(['searchqa','test',condition,m['correct'],m['n'],pct(m['em']),pct(m['f1']),f"{m['mean_latency_seconds']:.2f}",f"{m['p50_latency_seconds']:.2f}",f"{m['p95_latency_seconds']:.2f}"])
    for condition,m in test['conditions'].items():
        total_new.update({k:m['cost'].get(k,0) for k in ['physical_calls','prompt_tokens','generated_tokens']})
    for label,key in [('完整三轮','B_to_C'),('共同预算截点','matched_B_to_C')]:
        pairs.append(paired_metric_row('searchqa','test',label,comparisons[key]))
    for label,b_rows,c_rows in [('full',read_json(prior/'B/outputs.json'),resolve_test(testdir,'C')),('matched',resolve_test(testdir,'B_prefix'),resolve_test(testdir,'C_prefix'))]:
        checked_rows(c_rows,'searchqa');checked_rows(b_rows,'searchqa')
        for b,c in zip(b_rows,c_rows):
            assert b['id']==c['id'] and b['visible_context_hash']==c['visible_context_hash']
            state={(0,0):'both_wrong',(0,1):'repaired',(1,0):'regressed',(1,1):'both_correct'}[(b['hard'],c['hard'])]
            case_rows.append({'dataset':'searchqa','split':'test','comparison':label,'id':c['id'],'question':c['question'],'gold':c['gold_answers'],'B':b['predicted_answer'],'C':c['predicted_answer'],'B_em':b['em'],'C_em':c['em'],'B_f1':b['f1'],'C_f1':c['f1'],'B_raw':b['raw_response'],'C_raw':c['raw_response'],'context':c['visible_context'],'state':state})
    review_path=session/'manual_review.json'
    review=read_json(review_path) if review_path.exists() else {'status':'not_reviewed','pairs':[],'cases':[]}
    assert review['status'] in {'verified','not_reviewed'}, 'Manual review is still in progress'
    paired_ids={(e['dataset'],p['failure_id']) for e in exposures for p in e['packets'] if p['pair_type']!='unpaired'}
    assert len(review.get('pairs',[]))<=10
    assert len({(p['dataset'],p['failure_id']) for p in review.get('pairs',[])})==len(review.get('pairs',[]))
    assert all((p['dataset'],p['failure_id']) in paired_ids for p in review.get('pairs',[]))
    review_by_id={(r['dataset'],r['split'],r['id']):r for r in review.get('cases',[])}
    for row in case_rows:
        r=review_by_id.get((row['dataset'],row['split'],row['id']))
        if r and row['comparison']=='full':
            assert r['C']==row['C'] and r['C_response_sha256']==digest(row['C_raw'].encode())
            row['review']=r
    reviewed=[r for r in case_rows if r['comparison']=='full' and r.get('review')]
    review_scope=(f"已复核 {sum(r['C_em']==0 for r in reviewed)} 条最终 C 错误及额外 {sum(r['state']=='repaired' for r in reviewed)} 条修复案例。"
                  if reviewed else '尚未进行逐题语义复核。')
    reused=[r for r in reviewed if r['review'].get('annotation_reused_from')]
    assert all(r['B_raw']==r['C_raw'] for r in reused)
    if reused:
        review_scope+=f'其中 {len(reused)} 条与原 B 缓存观测完全相同，校验技能、输入及原响应 SHA256 后复用既有诊断。'
    review_scope+='逐题备注不改分，也不把输出推断当成已观测的内部节点。'
    save(assets/'paired_cases.json',case_rows)
    save(assets/'optimizer_feedback_exposures.json',exposures)
    save(assets/'test_results.json',test)
    save(assets/'manual_review.json',review)
    import csv
    with (assets/'paired_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['dataset','split','comparison','id','question','gold','B','C','B_em','C_em','B_f1','C_f1','state']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(case_rows)
    btest=test['baseline_conditions']['B'];ctest=test['conditions']['C'];delta=(ctest['em']-btest['em'])*100
    verdict='提高' if delta>0 else '下降' if delta<0 else '持平'
    findings=f'SearchQA 固定 50 题 test：B {pct(btest["em"])} → C {pct(ctest["em"])}（{delta:+.2f} pp，{verdict}）。修复 {comparisons["B_to_C"]["counts"]["repaired"]} 题、退化 {comparisons["B_to_C"]["counts"]["regressed"]} 题。'
    unchanged_datasets=[]
    for ds,a in data.items():
        unchanged=a['C_curve'][-1]['skill_hash']==a['C_curve'][0]['skill_hash']
        findings+=f' {ds} dev：B {a["B"]["dev"]["correct"]}/30，C {a["C"]["dev"]["correct"]}/30。'
        if unchanged:
            unchanged_datasets.append(ds)
    if unchanged_datasets:
        findings+=' '+ '、'.join(unchanged_datasets)+' 的 C 最终仍为共同初始化中的最佳技能，三轮未选出更好的新技能；B/C 差异不能解释为 C 演化带来的提升。'
    audit={'generated_at':datetime.now().astimezone().isoformat(),'status':'completed_available_data','method_modified':False,'new_cost':dict(total_new),'searchqa_test':test,'datasets':{ds:{'baseline':str(BASELINES[ds]),'experiment':str(session/ds),'code_hash':a['C']['code_hash'],'config_hash':a['C']['config_hash'],'pairing':a['feedback']['counts'],'service':service_metadata(session/ds/'C'),'initial_input_repeat':initial_input_repeat(BASELINES[ds]/'B',session/ds/'C')} for ds,a in data.items()},'hotpotqa_test':'missing_course_partition','manual_pairs_reviewed':len(review.get('pairs',[])),'manual_cases_reviewed':len(review.get('cases',[]))}
    save(output.with_suffix('.audit.json'),audit)
    pack={'cases':case_rows,'exposures':exposures,'review':review}
    payload=json.dumps(pack,ensure_ascii=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    counts_stats='；'.join(f'{ds}：{a["feedback"]["observations"]} 条真实 train 输出、{a["feedback"]["memory_records"]} 条去重记忆，实际成功配对涉及 {len({p["failure_id"] for e in a["feedback"]["exposures"] for p in e["packets"] if p["pair_type"] != "unpaired"})} 道不同失败题目，审计通过' for ds,a in data.items())
    quote_stats='；'.join(f'{ds}：完整成功响应可机械验证的引文 {a["feedback"]["verifiable_quote_in_full_reference"]} 次，压缩后仍可验证 {a["feedback"]["verifiable_quote_survives_packet"]} 次' for ds,a in data.items())
    content=TEMPLATE.replace('__REVIEW_SCOPE__',esc(review_scope)).replace('__TOTAL_COST__',f'本次新增 {total_new["physical_calls"]:,} 次实际请求、{total_new["prompt_tokens"]:,} 输入 tokens、{total_new["generated_tokens"]:,} 生成 tokens，包含两套 C 优化与 SearchQA C test；不重复计入共享初始化或既有 N/I/B 观测。').replace('__SETTINGS__',esc(settings)).replace('__FINDINGS__',findings).replace('__HERO__',f'{ctest["correct"]}/{ctest["n"]}').replace('__DELTA__',f'{delta:+.2f} pp').replace('__METRICS__',table(['数据集','划分','条件','正确','n','EM','F1','均值 s','P50 s','P95 s'],metrics)).replace('__PAIRS__',table(['数据集','划分','比较','修复','退化','修复率','退化率','均正确','均错误','C−B pp','配对 95% 区间 pp'],pairs)).replace('__BUDGET__',table(['数据集','方法','逻辑调用','新增实际调用','逻辑输入 tokens','逻辑生成 tokens','新增生成 tokens'],budget)).replace('__PREFIX__',table(['数据集','方法','共同调用上限','共同生成上限','选中轮','技能','已用调用','已用生成','dev EM','剩余调用','剩余生成'],prefix)).replace('__COVERAGE__',table(['数据集','算子次数','实际尝试','唯一父代反馈','所选失败曝光','同题成功','同类成功','无配对','配对率','全回退算子','unknown 类型'],coverage)).replace('__REPEAT__','；'.join(f'{ds}：相同初始技能、输入和生成配置的 40 条首轮 train 输出中，完整响应一致 {v["initial_input_repeat"]["raw_response_equal"]}/40，EM 一致 {v["initial_input_repeat"]["EM_equal"]}/40' for ds,v in audit['datasets'].items())).replace('__HEALTH__',table(['数据集','完成轮数','新增 QA 输出','最终请求错误','服务拒绝','QA 格式失败','算子失败尝试','dev 缓存事件','length 结束请求'],health)).replace('__ROUNDS__',table(['数据集','轮','候选','算子','父代','dev EM','dev F1','本轮选择','变更'],round_rows)).replace('__GALLERIES__',''.join(galleries)).replace('__COUNTS__',counts_stats).replace('__QUOTES__',quote_stats).replace('__ASSETS__',esc(assets.name)).replace('__DATA__',payload).replace('__AUDIT__',esc(json.dumps(audit,ensure_ascii=False,indent=2))).replace('__DATE__',esc(audit['generated_at']))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(content,encoding='utf-8')
    return {'output':str(output),'assets':str(assets),'findings':findings,'new_cost':dict(total_new),'paired_rows':len(case_rows)}


TEMPLATE='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project 5 · Part 3 封存 C 对照实验</title><style>
:root{--ink:#182d42;--muted:#52667a;--line:#dbe4eb;--accent:#1e6f89}*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--accent)}header{background:#163a4e;color:white;padding:44px max(24px,calc((100vw - 1220px)/2));display:flex;justify-content:space-between;gap:36px}header p{color:#d3e2e9;max-width:820px}h1{font-size:32px;line-height:1.4;margin:10px 0}h2{font-size:24px;margin-top:0}h3{font-size:20px}.eyebrow{font-size:12px;letter-spacing:2px}.hero-metric{min-width:210px;border-left:1px solid #7696a5;padding-left:28px}.big{font-size:42px;font-weight:650;line-height:1.2;margin:18px 0}nav{position:sticky;top:0;background:white;border-bottom:1px solid var(--line);padding:12px 22px;display:flex;justify-content:center;gap:25px;overflow:auto;white-space:nowrap;z-index:4}nav a{text-decoration:none;font-size:14px}main{max-width:1268px;margin:auto;padding:28px 24px}section{margin-bottom:42px;scroll-margin-top:72px}.panel{background:white;border:1px solid var(--line);border-radius:9px;padding:24px;margin:18px 0}.notice{padding:18px 20px;border-left:4px solid #3d8796;background:#eaf4f6;margin:16px 0}.warn{border-left-color:#b2843c;background:#fff7e9}.muted,small{color:var(--muted)}.scroll{overflow:auto;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf3f6;white-space:nowrap}td{overflow-wrap:anywhere}td:nth-child(-n+3){white-space:nowrap}details{margin:12px 0}summary{cursor:pointer;color:var(--accent);font-weight:550}pre{font-size:12px;line-height:1.65;white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;background:#f2f5f8;border-radius:6px}.diagram{overflow:auto}.diagram svg{min-width:720px;width:100%;height:auto}svg{width:100%;height:auto}.skill{border:1px solid var(--line);border-radius:7px;padding:14px;margin:12px 0}.gallery>summary{padding:14px;background:#edf4f7}.controls{display:flex;flex-wrap:wrap;gap:10px}input,select,button{font:inherit;border:1px solid #bdccd7;background:white;padding:9px 11px;border-radius:6px;color:var(--ink)}input{flex:1;min-width:160px}button{cursor:pointer}.case{background:white;border:1px solid var(--line);border-radius:8px;padding:20px;margin:15px 0}.tag{display:inline-block;padding:2px 8px;border-radius:4px;background:#eaf1f5;font-size:12px;margin-right:7px}.answers{display:grid;grid-template-columns:1fr 1fr;gap:12px}.answer{background:#f1f5f8;padding:13px;border-radius:6px;overflow-wrap:anywhere}.answer b{display:block;font-size:12px;margin-bottom:5px}.id{font:11px Consolas,monospace;color:var(--muted);overflow-wrap:anywhere}.note{border-left:3px solid #7fa9b8;padding:10px 14px;background:#eef6f8}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}li{margin:8px 0}footer{border-top:1px solid var(--line);padding:25px 0;font-size:12px;color:var(--muted)}@media(max-width:700px){header{display:block;padding:28px 20px}h1{font-size:26px}.hero-metric{margin-top:20px;padding-left:18px}.big{font-size:34px}main{padding:20px 12px}.panel{padding:16px}.answers,.twocol{grid-template-columns:1fr}nav{justify-content:flex-start}.controls>*{max-width:100%}}@media print{nav,.controls,button{display:none}body{background:white}header{background:white;color:#182d42;padding:10px}header p{color:#52667a}main{padding:0}.panel{border:0;padding:8px}.case{break-inside:avoid}.diagram svg{min-width:0}}
</style></head><body><header><div><p class="eyebrow">AIAA4051 / GraphSkillEvo / Option A</p><h1>Part 3 · 成功与失败对照反馈<br>封存 C 方法的真实实验</h1><p>从共同初始化出发，保留失败抽样与选择规则，只改变优化器收到的反馈。完整三轮结果与共同预算截点分开报告。</p></div><div class="hero-metric"><p class="eyebrow">SearchQA · C test</p><p class="big">__HERO__</p><p>相对 B：__DELTA__</p></div></header><nav><a href="#result">主要结果</a><a href="#implementation">实现核对</a><a href="#budget">预算公平性</a><a href="#feedback">实际反馈</a><a href="#skills">技能演化</a><a href="#cases-section">逐题对照</a><a href="#audit">复现审计</a></nav><main>
<section id="result"><h2>01 · 真实结果</h2><div class="notice">__FINDINGS__</div><div class="notice warn"><strong>HotpotQA 尚无课程 test。</strong>本报告完成公开 distractor 100 train / 30 dev 上的 B/C 开发对照，不能把这些结果称为课程最终测试。SearchQA 的 N/I/B test 已在前一阶段完成；C 的实现与配置在该测试之前封存，本次按用户确认运行原封存方法。</div><div class="panel">__METRICS__<p class="muted">dev 参与技能选择；test 是冻结后的描述性结果。N/I/B test 引用先前真实观测，C 新增请求或同技能缓存复用单独记账。延迟包含网络与服务端处理，不能等同 GPU 时间。复用技能的 dev 延迟沿用原观测，不能当作本次重新计时的加速；跨日服务负载也可能不同。</p></div><div class="panel"><h3>B → C 修复与退化</h3>__PAIRS__<p class="muted">修复=B 错 C 对，退化=B 对 C 错；修复率分母为 B 的错题数，退化率分母为 B 的正确题数。计数仅按 EM，同为 EM=0 的样例仍可能有 F1 变化。95% 区间来自固定题目上的 2000 次配对 bootstrap，不代表多训练种子的波动，也没有消除 dev 选择偏差。</p></div></section>
<section id="implementation"><h2>02 · 当前代码已实现什么，本次优化了什么</h2><p>__SETTINGS__</p><div class="twocol"><div class="panel"><h3>封存方法：保持原规格</h3><ul><li>本条件的 train 经验记忆，每题最多两份不同成功和两份不同失败。</li><li>失败仍按原随机 k=3 抽样；优先同题同上下文成功，再按问题类型和 Jaccard≥0.2 配同类成功。</li><li>成功只作为对照证据；不把 EM=1 当作完整推理正确的证明。</li><li>全部无成功参照时精确回退到 B 的失败反馈。</li><li>同一轮两个父代先全部执行，再构造反馈，避免执行顺序造成配对机会不对称。</li></ul></div><div class="panel"><h3>新增运行与审计能力</h3><ul><li>校验原代码快照、预先封存的 C 方法、配置、共享初始化和数据。</li><li>串行队列运行，两个数据集不同时占用模型请求并发。</li><li>验证训练记忆来源、失败抽样和反馈长度，并核对实际文本确实进入 mutation 提示。</li><li>按实际使用的父代统计配对覆盖率，区分算子曝光与唯一父代反馈。</li><li>预先声明共同调用/生成 token 截点规则，补充冻结后的对应技能比较。</li></ul></div></div><div class="panel"><h3>代码审阅发现的限制</h3><p>当前反馈按字段保留前部片段，最初每字段 160 tokens；更紧的总预算下还会缩短。后置答案或关键证据可能丢失。问题类型规则对 SearchQA 的线索式题目可能返回 unknown，同类配对因此受限。显式引文标志只做字面检查，不代表语义支持；换行或标点差异可能造成漏检，压缩也可能删除原引文。</p><p>这些属于会改变优化器输入的方法调整。本次遵守用户确认，保持封存算法；未把已看过的 test 错题写入反馈或新规则。后续若调整上述算法，需要新的独立评估或明确标为探索性结果。</p></div></section>
<section id="budget"><h2>03 · 同上限、实际消耗与共同预算截点</h2><p>__TOTAL_COST__</p><div class="panel"><p>B/C 使用相同配置：每阶段最多 600 次逻辑调用、700,000 生成 tokens、9 小时；三轮、每轮两候选、相同训练 batch、单次 QA/算子输出上限与重试规则。共享初始化在每个方法的逻辑预算中计入一次；只有真实新增请求计入物理消耗。</p>__BUDGET__<p class="muted">上表是优化阶段，含逻辑共享初始化。N/I 和 pilot 不算 C 的新优化费用；test 费用另在审计结果记录。相同预算上限不等于实际消耗完全相等。</p></div><div class="panel"><h3>共同预算矩形：预先声明的补充比较</h3><p>共同调用上限取 B/C 最终调用数较小值，共同生成上限取最终生成 tokens 较小值；每个方法选择同时不超过两项上限的最后一个完整轮次，初始化为轮 0。再使用该轮由 dev 选出的最佳技能。规则在 C 训练开始前保存，不用 test 成绩选择截点；不完整轮次不参与。</p>__PREFIX__<p class="muted">这匹配的是共同累计上限，而非强行消耗到完全相同 token 数。表中保留未用额度；不通过额外无用调用补齐成本。SearchQA 截点技能在 C test 前冻结，内容相同的技能复用已有 test 观测，不冒充独立重复实验。</p></div></section>
<section id="feedback"><h2>04 · 成功参照是否真正送进了优化器</h2><div class="panel">__COVERAGE__<p>__COUNTS__。</p><p>__QUOTES__。</p><p class="muted">分母是实际 mutation 父代所选失败的曝光次数；同一父代被两个算子使用时计两次曝光，同时另列唯一父代反馈数。无失败的父代不添加虚构分母；主覆盖率按算子计数，不按重试次数重复加权，实际尝试次数另列。下方展示实际压缩后文本和原诊断包，可核对丢失的信息。上游拼接 Reflection Context 时仅去除首尾空白，审计按这一规则核对正文。</p></div><div class="controls"><select id="feedback-dataset" aria-label="反馈数据集"><option value="all">全部数据集</option><option value="searchqa">SearchQA</option><option value="hotpotqa">HotpotQA</option></select><select id="pair-type" aria-label="配对类型"><option value="all">全部配对</option><option value="same_question">同题</option><option value="similar_task">同类</option><option value="unpaired">无配对</option></select></div><p id="feedback-count" class="muted"></p><div id="feedback-list"></div><div id="manual-pairs"></div></section>
<section id="skills"><h2>05 · 三轮候选、保留技能与图结构变化</h2><div class="panel">__ROUNDS__<p class="muted">技能差异来自最终采用的文本，不把优化器 notes 中未采用的改动当成真实变化。流程图展示技能定义与明确条件，不冒充模型内部执行轨迹。</p></div>__GALLERIES__</section>
<section id="cases-section"><h2>06 · 相同题目上的 B/C 输出与失败清单</h2><p>完整三轮主比较与共同预算截点分开筛选。test 错误只做事后描述，不反馈到优化器。__REVIEW_SCOPE__</p><div class="controls"><select id="ds" aria-label="对照数据集"><option value="all">全部数据集</option><option value="searchqa">SearchQA</option><option value="hotpotqa">HotpotQA</option></select><select id="split" aria-label="划分"><option value="all">dev + test</option><option value="dev">dev</option><option value="test">test</option></select><select id="comparison" aria-label="比较范围"><option value="full">完整三轮</option><option value="matched">共同预算截点</option></select><select id="state" aria-label="变化"><option value="changed">修复与退化</option><option value="repaired">修复</option><option value="regressed">退化</option><option value="C_failed">C 的全部错题</option><option value="all">全部</option></select><input id="query" aria-label="搜索" placeholder="搜索题目、ID 或答案"><button onclick="window.print()">打印</button></div><p id="case-count" class="muted"></p><div id="case-list"></div><p><a download href="__ASSETS__/paired_cases.csv">下载逐题对照 CSV</a> · <a download href="__ASSETS__/paired_cases.json">下载全部响应 JSON</a> · <a download href="__ASSETS__/optimizer_feedback_exposures.json">实际反馈曝光</a> · <a download href="__ASSETS__/manual_review.json">人工复核记录</a></p></section>
<section id="audit"><h2>07 · 审计、版本与可复现边界</h2><div class="panel"><h3>运行完整性</h3>__HEALTH__<p class="muted">QA 输出覆盖本次各轮 train/dev 实际执行；不把共享初始化或缓存复用算作新增样本。算子失败尝试包含格式校验失败，所有重试均计入调用与 token。缓存事件表示相同技能沿用已有 dev 观测，可能节省一次完整 dev 评测。</p></div><div class="panel"><p>本次没有修改 Explore 反馈算法、失败抽样、候选选择、QA 提示词或固定模型配置。SearchQA 与 HotpotQA 分别使用原 B 的代码快照；C 复用对应 shared_init，不读取 B 分叉后产生的训练成功轨迹。</p><p>__REPEAT__。这些是观测一致性检查，不能证明服务端权重完全相同。</p><p>服务端权重 revision、精度及实际 GPU 占用仍未提供；模型 ID 与服务 fingerprint 一致不足以证明跨日期权重完全相同。固定单一 seed、30 道 dev 以及重复 dev 选择限制了结论强度。没有模型权重训练，也不是 JitRL 的数值 logits 更新。</p><p><a download href="__ASSETS__/searchqa-analysis.json">SearchQA 完整审计</a> · <a download href="__ASSETS__/hotpotqa-analysis.json">HotpotQA 完整审计</a> · <a download href="__ASSETS__/searchqa-policy.json">SearchQA 预声明规则</a> · <a download href="__ASSETS__/hotpotqa-policy.json">HotpotQA 预声明规则</a> · <a download href="__ASSETS__/test_results.json">test 结果与来源</a> · <a download href="__ASSETS__/test_frozen.json">test 前冻结清单</a></p><details><summary>全部运行标识、预算与来源摘要</summary><pre>__AUDIT__</pre></details></div></section><footer>生成于 __DATE__ · HTML 自包含，可离线阅读；下载文件在同名 assets 目录。HotpotQA 课程 test 与最终课程报告仍待完成。</footer></main><script id="report-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id),e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const names={repaired:'B 错 → C 对',regressed:'B 对 → C 错',both_correct:'均正确',both_wrong:'均未完全匹配'};
function cases(){const q=$('query').value.toLowerCase().trim();const rows=data.cases.filter(r=>($('ds').value==='all'||r.dataset===$('ds').value)&&($('split').value==='all'||r.split===$('split').value)&&r.comparison===$('comparison').value&&($('state').value==='all'||($('state').value==='changed'?['repaired','regressed'].includes(r.state):$('state').value==='C_failed'?r.C_em===0:r.state===$('state').value))&&(!q||[r.id,r.question,r.B,r.C,r.gold.join(' ')].join(' ').toLowerCase().includes(q)));$('case-count').textContent=`显示 ${rows.length} 条配对记录。`;$('case-list').innerHTML=rows.map(r=>`<article class="case"><p><span class="tag">${e(r.dataset)}</span><span class="tag">${e(r.split)}</span><span class="tag">${names[r.state]}</span></p><strong>${e(r.question)}</strong><p class="id">${e(r.id)}</p><div class="answers"><div class="answer"><b>B · EM ${r.B_em} / F1 ${(r.B_f1*100).toFixed(2)}%</b>${e(r.B)}</div><div class="answer"><b>C · EM ${r.C_em} / F1 ${(r.C_f1*100).toFixed(2)}%</b>${e(r.C)}</div></div><p>参考：${e(r.gold.join(' / '))}</p>${r.review?`<p class="note"><b>${e(r.review.category)}</b>：${e(r.review.note)}</p>`:''}<details><summary>两组原始响应与相同输入</summary><h4>B</h4><pre>${e(r.B_raw)}</pre><h4>C</h4><pre>${e(r.C_raw)}</pre><h4>可见上下文</h4><pre>${e(r.context)}</pre></details></article>`).join('')||'<p class="muted">当前筛选没有匹配记录。</p>';}
function feedback(){let count=0;const items=data.exposures.filter(x=>$('feedback-dataset').value==='all'||x.dataset===$('feedback-dataset').value).map(x=>{const ps=x.packets.filter(p=>$('pair-type').value==='all'||p.pair_type===$('pair-type').value);count+=ps.length;if(!ps.length)return '';return `<details class="panel exposure"><summary>${e(x.dataset)} · round ${x.round} · ${e(x.parent_id)} → ${e(x.child_id)} · ${x.feedback_tokens} tokens</summary>${ps.map(p=>`<p><span class="tag">${e(p.pair_type)}</span><span class="id">${e(p.failure_id)}</span> → ${e(p.reference_id||'无成功参照')}</p><pre>${e(JSON.stringify(p.actual_packet||p.diagnostic_packet,null,2))}</pre>`).join('')}<details><summary>本次实际反馈全文</summary><pre>${e(x.actual_feedback)}</pre></details></details>`;});$('feedback-count').textContent=`显示 ${count} 个所选失败曝光；下列按 mutation 调用分组。`;$('feedback-list').innerHTML=items.join('')||'<p class="muted">当前筛选没有匹配反馈。</p>';}
for(const id of ['ds','split','comparison','state','query'])$(id).addEventListener(id==='query'?'input':'change',cases);for(const id of ['feedback-dataset','pair-type'])$(id).addEventListener('change',feedback);cases();feedback();if(data.review.pairs?.length)$('manual-pairs').innerHTML='<div class="panel"><h3>固定顺序抽查的训练配对</h3>'+data.review.pairs.map(r=>`<p><b>${e(r.dataset)} / ${e(r.failure_id)}</b>：${e(r.note)}</p>`).join('')+'<p class="muted">抽查只验证材料与诊断边界，不修改方法或技能。</p></div>';
</script></body></html>'''


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--session',default=str(SESSION))
    args=p.parse_args();print(json.dumps(build(args.output,args.session),ensure_ascii=False,indent=2))
