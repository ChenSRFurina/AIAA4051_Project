"""Read-only diagnosis of C feedback, replaying only each round's past memory."""
from collections import Counter
import difflib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Baseline'))
sys.path.insert(0,str(ROOT/'Explore'))
from baseline.io import digest,read_json,read_jsonl,write_json
from explore.memory import TrainMemory
from explore.pairing import task_type,jaccard,pair_failure
from graphskillevo.graph_skill import split_graph_skill
from part3_analysis import checkpoint_data,compact_comparison
from run_part3 import SESSION,BASELINES


def diagnose(dataset):
    run=SESSION/dataset/'C';a=read_json(SESSION/dataset/'analysis.json')
    hist=read_json(run/'history.json')
    ids=read_json(ROOT/'Baseline/data/prepared'/dataset/'train_ids.json')
    memory=TrainMemory(dataset,ids)
    rounds=[];failures=[];mutations=[]
    _,records,dev_rows=checkpoint_data(run,BASELINES[dataset]/'shared_init')
    for h in hist:
        observations={}
        for parent in h['parent_population']:
            rows=read_jsonl(run/f"evaluations/g{h['generation']}/{parent['id']}/train/outputs.jsonl")
            observations[parent['id']]={r['id']:r for r in rows}
            for row in rows:memory.add(row)
        successes=memory.successes()
        rounds.append({'round':h['generation'],'memory_records':len(memory.records()),'success_observations':len(successes),'success_questions':len({r['id'] for r in successes})})
        for op in h['operator_records']:
            parent_id=op['parent_ids'][0]
            feedback=read_json(run/f"evaluations/g{h['generation']}/{parent_id}/train/feedback.json")
            for packet in feedback['packets']:
                f=observations[parent_id][packet['failure_id']]
                selected=pair_failure(f,memory)
                assert selected['pair_type']==packet['pair_type']
                if selected['success_reference']:
                    assert selected['success_reference']['observation_id']==packet['success_reference']['observation_id']
                kind=task_type(f['question'])
                different=[r for r in successes if r['id']!=f['id']]
                same_kind=[r for r in different if kind!='unknown' and task_type(r['question'])==kind]
                best_any=max((jaccard(f['question'],r['question']) for r in different),default=0)
                best_kind=max((jaccard(f['question'],r['question']) for r in same_kind),default=0)
                actual=next((p for p in feedback['actual_packets'] if p['failure_id']==f['id']),None)
                failures.append({'round':h['generation'],'child':op['child_id'],'parent':parent_id,
                    'failure_id':f['id'],'question':f['question'],'type':kind,'pair_type':packet['pair_type'],
                    'failure_em':f['em'],'failure_f1':f['f1'],'prediction':f['predicted_answer'],'gold':f['gold_answers'],
                    'successes_available':len(successes),'same_kind_available':len(same_kind),
                    'max_jaccard_any_kind':best_any,'max_jaccard_same_kind':best_kind,
                    'success':packet['success_reference'],'actual_packet':actual,
                    'feedback_tokens':feedback['feedback_tokens']})
            child=records[op['child_id']];parent=records[parent_id]
            raw=op['attempts'][-1]['parsed']['new_skill']
            raw_parts=split_graph_skill(raw);adopted=split_graph_skill(child['content'])
            forbidden=['node_lists','task_graphs'] if child['origin']=='mutation_other' else ['global_guidance']
            discarded=[key for key in forbidden if getattr(raw_parts,key)!=getattr(adopted,key)]
            added=[s[1:] for s in difflib.unified_diff(parent['content'].splitlines(),child['content'].splitlines(),n=0) if s.startswith('+') and not s.startswith('+++')]
            comp=compact_comparison(dev_rows[parent_id],dev_rows[child['id']])
            mutations.append({'round':h['generation'],'child':child['id'],'operator':child['origin'],'parent':parent_id,
                'parent_em':parent['val_hard'],'child_em':child['val_hard'],
                'no_op':digest(parent['content'])==digest(child['content']),
                'discarded_sections':discarded,'adopted_added_lines':added,
                'introduced_gold_instructions':[s for s in added if 'gold' in s.lower()],
                'notes':op['attempts'][-1]['parsed'].get('notes'),
                'dev_transitions':comp['counts'],'feedback_tokens':feedback['feedback_tokens'],
                'paired_failures':feedback['same_question']+feedback['similar_task']})
    assert memory.records()==read_json(run/'memory.json'), 'Memory replay must match saved final bank'
    paired=[f for f in failures if f['pair_type']!='unpaired']
    return {'dataset':dataset,'memory_replay':'exact_match','source_code_hash':a['C']['code_hash'],
        'rounds':rounds,'failures':failures,'mutations':mutations,
        'summary':{'selected_failure_exposures':len(failures),'paired_exposures':len(paired),
            'unique_paired_failures':len({f['failure_id'] for f in paired}),
            'pair_types':dict(Counter(f['pair_type'] for f in failures)),
            'unknown_type_exposures':sum(f['type']=='unknown' for f in failures),
            'fallback_operators':sum(m['paired_failures']==0 for m in mutations),
            'no_op_candidates':sum(m['no_op'] for m in mutations),
            'candidates_with_scope_discard':sum(bool(m['discarded_sections']) for m in mutations),
            'candidates_with_new_gold_reference':sum(bool(m['introduced_gold_instructions']) for m in mutations),
            'unique_paired_f1':{f['failure_id']:f['failure_f1'] for f in paired},
            'unpaired_with_any_kind_jaccard_ge_02':sum(f['pair_type']=='unpaired' and f['max_jaccard_any_kind']>=.2 for f in failures),
            'unpaired_with_eligible_kind_candidate':sum(f['pair_type']=='unpaired' and f['same_kind_available']>0 for f in failures)},
        'interpretation':'Counts describe this seed and stored observations. No new model calls, no test-based tuning, no causal proof from candidate score differences alone.'}


if __name__=='__main__':
    result={ds:diagnose(ds) for ds in ['searchqa','hotpotqa']}
    output=ROOT/'Explore/runs/reports/C2_feedback_diagnosis_20261110.json'
    write_json(output,result)
    print(json.dumps({'output':str(output),'summaries':{k:v['summary'] for k,v in result.items()}},ensure_ascii=False,indent=2))
