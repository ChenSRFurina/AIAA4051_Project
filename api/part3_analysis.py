"""Read-only audits for sealed B/C experiments and common-budget checkpoints."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Baseline'))
sys.path.insert(0,str(ROOT/'Explore'))
from baseline.io import digest,read_json,read_jsonl
from baseline.evaluation import aggregate,score
from baseline.analysis import compare
from baseline.feedback import selected_failures
from baseline.budget import TokenCounter
from explore.feedback import evidence_check


def checked_rows(rows,dataset):
    assert rows and len({r['id'] for r in rows})==len(rows)
    for row in rows:
        assert row['dataset']==dataset and not row['error']
        s=score(row['raw_response'],row['gold_answers'],dataset)
        assert row['em']==s['em'] and abs(row['f1']-s['f1'])<1e-9
        assert digest(row['visible_context'])==row['visible_context_hash']
    return aggregate(rows)


def checkpoint_data(run,initialization):
    run,initialization=Path(run),Path(initialization)
    init=read_json(initialization/'population.json')
    best=max(init,key=lambda r:r['val_hard'])
    points=[{'round':0,'best_id':best['id'],'skill_hash':digest(best['content']),
             'cost':read_json(initialization/'summary.json')['cost'],'em':best['val_hard'],'f1':best['val_soft']}]
    records={r['id']:r for r in init}
    cache=read_json(initialization/'validation_cache.json')
    rows_by_id={r['id']:cache[digest(r['content'])[:16]] for r in init}
    for h in read_json(run/'history.json'):
        for c in h['candidates']:
            records[c['id']]=c
            path=run/'evaluations'/f"g{h['generation']}"/c['id']/'dev/outputs.jsonl'
            if path.exists():
                rows_by_id[c['id']]=read_jsonl(path)
            else:
                # Identical-skill validation caching is an observation reuse.
                prior=next(k for k,r in records.items() if k!=c['id'] and digest(r['content'])==digest(c['content']) and k in rows_by_id)
                rows_by_id[c['id']]=rows_by_id[prior]
        r=records[h['best_id']]
        points.append({'round':h['generation'],'best_id':r['id'],'skill_hash':digest(r['content']),
                       'cost':h['cost_at_round'],'em':r['val_hard'],'f1':r['val_soft']})
    return points,records,rows_by_id


def choose_checkpoint(points,calls,tokens):
    feasible=[p for p in points if p['cost']['calls']<=calls and p['cost']['generated_tokens']<=tokens]
    if not feasible:
        raise ValueError('Common caps cannot cover shared initialization')
    chosen=max(feasible,key=lambda p:p['round'])
    return {**chosen,'unused_calls':calls-chosen['cost']['calls'],
            'unused_generated_tokens':tokens-chosen['cost']['generated_tokens']}


def matched_checkpoints(b_points,c_points,b_cost,c_cost):
    calls=min(b_cost['calls'],c_cost['calls'])
    tokens=min(b_cost['generated_tokens'],c_cost['generated_tokens'])
    return {'call_cap':calls,'generated_token_cap':tokens,
            'B':choose_checkpoint(b_points,calls,tokens),'C':choose_checkpoint(c_points,calls,tokens),
            'interpretation':'Common cumulative call/token ceilings; realized costs may differ because only complete rounds are eligible. Not exactly equal realized tokens.'}


def compact_comparison(left,right):
    c=compare(left,right)
    return {k:v for k,v in c.items() if k!='transitions'} | {'changes':[{'id':r['id'],'transition':r['transition'],'question':r['B']['question'],'gold_answers':r['B']['gold_answers'],'B':r['B']['predicted_answer'],'C':r['C']['predicted_answer']} for r in c['transitions'] if r['transition'] in {'repaired','regressed'}]}


def audit_feedback(c_run,b_run,counter,train_ids):
    c_run,b_run=Path(c_run),Path(b_run)
    config=read_json(c_run/'config.json')
    hist=read_json(c_run/'history.json')
    observations={}
    parent_reports={}
    for h in hist:
        generation=h['generation']
        batch=read_json(c_run/'train_batches'/f'{generation}.json')
        assert batch==read_json(b_run/'train_batches'/f'{generation}.json'), 'B/C train batches differ'
        for parent in h['parent_population']:
            folder=c_run/'evaluations'/f'g{generation}'/parent['id']/'train'
            rows=read_jsonl(folder/'outputs.jsonl')
            assert [r['id'] for r in rows]==batch['ids']
            assert all(r['split']=='train' and r['condition']=='C' and r['id'] in train_ids and not r.get('cached') for r in rows)
            for r in rows:
                observations[r['observation_id']]=r
            report=read_json(folder/'feedback.json')
            expected=[r['id'] for r in selected_failures(rows,config['failure_k'],batch['seed'])]
            assert report['selected_failure_ids']==expected, 'Failure selection differs from pinned B rule'
            assert report['feedback_tokens']==counter.count(report['actual_feedback'])<=config['feedback_max_tokens']
            assert sum(report[k] for k in ['same_question','similar_task','unpaired'])==len(expected)
            parent_reports[(generation,parent['id'])]=report
    for r in read_json(c_run/'memory.json'):
        original=observations[r['observation_id']]
        assert all(r[k]==original[k] for k in original)
        assert not r.get('error') and not r.get('cached')
    for report in parent_reports.values():
        for packet in report['packets']:
            f=packet['failure']; original=observations[f['observation_id']]
            assert original['id']==packet['failure_id'] and original['em']==0
            ref=packet['success_reference']
            if ref:
                success=observations[ref['observation_id']]
                assert success['em']==1 and success['round']<=original['round'] and success['id'] in train_ids
                if packet['pair_type']=='same_question':
                    assert success['id']==original['id'] and success['visible_context_hash']==original['visible_context_hash']
    exposures=[]
    for h in hist:
        for operator in h['operator_records']:
            parent_id=operator['parent_ids'][0]
            report=parent_reports[(h['generation'],parent_id)]
            verified_attempts=0
            for attempt in operator['attempts']:
                prompt=attempt.get('prompt')
                if prompt:
                    # The pinned upstream _format_reflection_context strips only
                    # outer whitespace before adding the Reflection Context block.
                    assert report['actual_feedback'].strip() in prompt
                    verified_attempts+=1
            assert verified_attempts, 'Cannot verify actual optimizer feedback exposure'
            actual={p['failure_id']:p for p in report['actual_packets']}
            packet_stats=[]
            for packet in report['packets']:
                a=actual.get(packet['failure_id'])
                ref=a['success_reference'] if a else None
                quote_survives=evidence_check({'raw_response':ref['response_excerpt'],'visible_context':ref.get('visible_context_excerpt','')}) if ref else False
                packet_stats.append({'failure_id':packet['failure_id'],'pair_type':packet['pair_type'],
                    'task_type':packet['task_type'],'similarity':packet['similarity'],
                    'reference_id':packet['success_reference']['example_id'] if packet['success_reference'] else None,
                    'quote_in_full_reference':packet['observable_checks']['reference_has_verifiable_quote'],
                    'quote_survives_bounded_packet':quote_survives,
                    'context_excerpt_present':bool(a and a['failure'].get('visible_context_excerpt')),
                    'actual_packet':a,'diagnostic_packet':packet})
            exposures.append({'round':h['generation'],'child_id':operator['child_id'],'parent_id':parent_id,
                'accepted':operator['accepted'],'verified_attempts':verified_attempts,
                'selected_count':len(report['selected_failure_ids']),
                **{k:report[k] for k in ['same_question','similar_task','unpaired','feedback_tokens']},
                'packets':packet_stats,'actual_feedback':report['actual_feedback']})
    counts={k:sum(e[k] for e in exposures) for k in ['selected_count','same_question','similar_task','unpaired']}
    counts['paired_rate']=(counts['same_question']+counts['similar_task'])/counts['selected_count'] if counts['selected_count'] else None
    all_packets=[p for e in exposures for p in e['packets']]
    return {'status':'passed','observations':len(observations),'memory_records':len(read_json(c_run/'memory.json')),
            'unique_parent_feedbacks':len({(e['round'],e['parent_id']) for e in exposures}),
            'operator_exposures':len(exposures),'actual_attempt_exposures':sum(e['verified_attempts'] for e in exposures),
            'counts':counts,'unknown_question_type':sum(p['task_type']=='unknown' for p in all_packets),
            'verifiable_quote_in_full_reference':sum(p['quote_in_full_reference'] for p in all_packets),
            'verifiable_quote_survives_packet':sum(p['quote_survives_bounded_packet'] for p in all_packets),
            'failure_context_survives_packet':sum(p['context_excerpt_present'] for p in all_packets),
            'exposures':exposures,
            'checks':['exact B/C training batches','unchanged random failure IDs','train-only own observations','no future success reference','same-question context equality','feedback token cap','actual text in optimizer prompt','memory provenance'],
            'source_policy':'Inspect actual mutation-parent exposures, not every unused parent feedback. Repeated use of one parent counts as repeated operator exposure.'}


def analyze(baseline,experiment):
    baseline,experiment=Path(baseline),Path(experiment)
    b_run,c_run=baseline/'B',experiment/'C'
    b,c=read_json(b_run/'summary.json'),read_json(c_run/'summary.json')
    assert b['status']==c['status']=='completed' and b['completed_rounds']==c['completed_rounds']==3
    for k in ['config_hash','data_hash','code_hash','tokenizer_hash','upstream_commit','demo']:
        assert b[k]==c[k]
    assert read_json(b_run/'shared_initialization.json')['population_hash']==read_json(c_run/'shared_initialization.json')['population_hash']
    dataset=b['dataset']
    b_rows,c_rows=read_json(b_run/'final_dev_outputs.json'),read_json(c_run/'final_dev_outputs.json')
    assert checked_rows(b_rows,dataset)==b['dev'] and checked_rows(c_rows,dataset)==c['dev']
    bp,br,bo=checkpoint_data(b_run,baseline/'shared_init')
    cp,cr,co=checkpoint_data(c_run,baseline/'shared_init')
    matched=matched_checkpoints(bp,cp,b['cost'],c['cost'])
    matched['dev_comparison']=compact_comparison(bo[matched['B']['best_id']],co[matched['C']['best_id']])
    config=read_json(c_run/'config.json')
    counter=TokenCounter(config['model']['tokenizer_path'])
    assert counter.identity==c['tokenizer_hash']
    feedback=audit_feedback(c_run,b_run,counter,set(read_json(ROOT/'Baseline/data/prepared'/dataset/'train_ids.json')))
    return {'status':'passed','dataset':dataset,'B':b,'C':c,'B_curve':bp,'C_curve':cp,
            'dev_comparison':compact_comparison(b_rows,c_rows),'matched':matched,'feedback':feedback,
            'baseline':str(baseline.resolve()),'experiment':str(experiment.resolve()),
            'common_config':config,'C_failures':[r for r in c_rows if r['em']==0]}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--experiment',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    result=analyze(args.baseline,args.experiment)
    from part12 import write
    write(args.output,result)
    print(json.dumps({'status':result['status'],'dataset':result['dataset'],'dev':result['dev_comparison'],'pairing':result['feedback']['counts'],'matched':result['matched']},ensure_ascii=False,indent=2))
