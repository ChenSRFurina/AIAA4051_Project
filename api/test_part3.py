"""Freeze C and predeclared common-budget skills, then evaluate SearchQA once.

Baseline N/I/B test observations retain their original provenance. A skill with
identical content shares its observations; it is never counted as a fresh run.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import part12
from run_part3 import ROOT,BASELINES,SESSION,PRIOR_SEAL,setup,POLICY,sha


def freeze(experiment,output,source_ip=None):
    experiment,output=Path(experiment).resolve(),Path(output).resolve()
    part12.require(not output.exists(),'Part 3 test freeze already exists')
    p,config,baseline,snapshot,prior=setup('searchqa',source_ip)
    policy=part12.read(experiment/'policy.json')
    part12.require(policy['policy_id']==p.digest({k:v for k,v in policy.items() if k!='policy_id'}),'Run policy changed')
    part12.require(all(policy[k]==v for k,v in POLICY.items()),'Budget selection policy changed after training began')
    from part3_analysis import analyze,checkpoint_data
    audit=analyze(baseline,experiment)
    part12.write(experiment/'analysis.json',audit)
    _,br,_=checkpoint_data(baseline/'B',baseline/'shared_init')
    _,cr,_=checkpoint_data(experiment/'C',baseline/'shared_init')
    final=(experiment/'C/best_skill.md').read_text(encoding='utf-8')
    skills={'C':final,'B_prefix':br[audit['matched']['B']['best_id']]['content'],
            'C_prefix':cr[audit['matched']['C']['best_id']]['content']}
    record={'scope':'part3_sealed_C_and_budget_prefix','dataset':'searchqa','config':config,
            'signature':part12.read(experiment/'C/run_meta.json'),'protocol_root':str(snapshot),
            'C_run':str(experiment/'C'),'C_artifacts':p.artifact_hashes(experiment/'C'),
            'prior_baseline_freeze':str(PRIOR_SEAL),'prior_baseline_freeze_id':prior['freeze_id'],
            'baseline_test_dir':str(ROOT/'Baseline/runs/part12/searchqa/test'),
            'policy':policy,'matched':audit['matched'],
            'skills':{k:{'content':v,'skill_hash':p.digest(v)} for k,v in skills.items()},
            'data_dir':prior['data_dir'],'test_ids':prior['test_ids'],
            'driver_sha256':sha(__file__),'analysis_driver_sha256':sha(ROOT/'api/part3_analysis.py'),
            'created_at':datetime.now(timezone.utc).isoformat(),
            'selection':'final and prefix skills selected from dev before C test exposure; prefix policy declared before C training',
            'test_policy':'read_only_no_optimization'}
    record['freeze_id']=p.digest(record)
    part12.write(output,record)
    return record


def verify(frozen):
    p=part12.bootstrap(frozen['protocol_root'])
    part12.require(frozen['freeze_id']==p.digest({k:v for k,v in frozen.items() if k!='freeze_id'}),'Part 3 test manifest changed')
    part12.require(sha(__file__)==frozen['driver_sha256'],'Test driver changed')
    part12.require(sha(ROOT/'api/part3_analysis.py')==frozen['analysis_driver_sha256'],'Analysis driver changed after test freeze')
    part12.require(p.tree_fingerprint()==frozen['signature']['code_hash'],'Protocol snapshot changed')
    part12.require(p.artifact_hashes(Path(frozen['C_run']))==frozen['C_artifacts'],'C evidence changed after freeze')
    prior=part12.read(frozen['prior_baseline_freeze'])
    part12.require(prior['freeze_id']==frozen['prior_baseline_freeze_id'],'Baseline freeze differs')
    part12.verify(prior)
    data,manifest=p.load_data(frozen['data_dir'],'searchqa',demo=False)
    part12.require(p.digest(manifest)==frozen['signature']['data_hash'],'Dataset changed')
    part12.require([r['id'] for r in data['test']]==frozen['test_ids'],'Fixed test IDs changed')
    return p,data


def evaluate(manifest,output,source_ip=None):
    frozen=part12.read(manifest)
    p,data=verify(frozen)
    output=Path(output).resolve()
    p.ensure_empty(output)
    config=frozen['config']
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False,interpolate=False)
    if source_ip:
        part12.install_source_transport(source_ip,config['model']['base_url'])
    counter=p.TokenCounter(config['model']['tokenizer_path'])
    part12.require(counter.identity==frozen['signature']['tokenizer_hash'],'Tokenizer changed')
    prior_results=part12.read(Path(frozen['baseline_test_dir'])/'results.json')
    part12.require(prior_results['status']=='completed' and prior_results['freeze_id']==frozen['prior_baseline_freeze_id'],'Baseline test incomplete/different')
    by_hash={}
    rows_by_condition={}
    source_hashes={}
    for condition in ['N','I','B']:
        path=Path(frozen['baseline_test_dir'])/condition/'outputs.json'
        rows=part12.read(path)
        part12.require([r['id'] for r in rows]==frozen['test_ids'],'Prior test IDs mismatch')
        part12.require(all(r['skill_hash']==rows[0]['skill_hash'] and r['split']=='test' and not r['error'] for r in rows),'Prior test observations invalid')
        prior=part12.read(frozen['prior_baseline_freeze'])
        part12.require(rows[0]['skill_hash']==prior['conditions'][condition]['skill_hash'],'Prior test skill differs from baseline freeze')
        part12.require(all(prior_results['conditions'][condition][k]==v for k,v in p.aggregate(rows).items()),'Prior test metrics differ from saved observations')
        rows_by_condition[condition]=rows
        by_hash[rows[0]['skill_hash']]=(condition,rows,str(path))
        source_hashes[condition]={'path':str(path),'sha256':sha(path),'freeze_id':prior_results['freeze_id']}
    results={}
    try:
        for condition,item in frozen['skills'].items():
            if item['skill_hash'] in by_hash:
                source,rows,path=by_hash[item['skill_hash']]
                results[condition]={**p.aggregate(rows),'new_calls':0,'reused_from':source,'source_path':path,
                                    'cost':{'calls':0,'physical_calls':0,'prompt_tokens':0,'generated_tokens':0,'reasoning_tokens':0}}
                part12.write(output/condition/'reuse.json',{'source':source,'source_path':path,'source_sha256':sha(path),'new_observations':0,'skill_hash':item['skill_hash']})
            else:
                folder=output/condition
                budget=p.Budget(folder/'ledger.jsonl',config['budget'])
                client=p.ChatClient(config,budget,counter,folder/'calls.jsonl',demo=False)
                rows=[]
                for example in data['test']:
                    rows.append(p.run_qa(client,example,item['content'],split='test',round_no=-1,
                                        seed=config['seed'],condition=condition,out_dir=folder))
                    part12.write(output/'progress.json',{'condition':condition,'done':len(rows),'expected':len(data['test']),'cost':budget.totals()})
                    print(json.dumps({'condition':condition,'done':len(rows),'expected':len(data['test'])}),flush=True)
                part12.write(folder/'outputs.json',rows)
                p.inventory(rows,folder/'analysis')
                results[condition]={**p.aggregate(rows),'new_calls':budget.totals()['physical_calls'],'cost':budget.totals(),'reused_from':None}
                by_hash[item['skill_hash']]=(condition,rows,str(folder/'outputs.json'))
            rows_by_condition[condition]=rows
        from part3_analysis import compact_comparison
        comparisons={'B_to_C':compact_comparison(rows_by_condition['B'],rows_by_condition['C']),
                     'matched_B_to_C':compact_comparison(rows_by_condition['B_prefix'],rows_by_condition['C_prefix'])}
        verify(frozen)
        part12.require(all(sha(v['path'])==v['sha256'] for v in source_hashes.values()),'Baseline test changed during C evaluation')
        report={'status':'completed','freeze_id':frozen['freeze_id'],'manifest':str(Path(manifest).resolve()),
                'conditions':results,'baseline_conditions':prior_results['conditions'],'comparisons':comparisons,
                'prior_sources':source_hashes,'matched':frozen['matched'],'test_n':len(data['test']),
                'finished_at':datetime.now(timezone.utc).isoformat(),'gpu_hours':None}
        part12.write(output/'results.json',report)
        return report
    except Exception as exc:
        part12.write(output/'incomplete.json',{'error':str(exc),'complete_conditions':list(results),'not_a_complete_test_result':True})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','test'])
    p.add_argument('--experiment',default=str(SESSION/'searchqa'))
    p.add_argument('--frozen',default=str(SESSION/'searchqa/test_frozen.json'))
    p.add_argument('--output')
    p.add_argument('--source-ip')
    args=p.parse_args()
    if args.action=='freeze':
        result=freeze(args.experiment,args.frozen,args.source_ip)
        print(json.dumps({'status':'frozen','freeze_id':result['freeze_id'],'matched':result['matched']},ensure_ascii=False,indent=2))
    else:
        result=evaluate(args.frozen,args.output or SESSION/'searchqa/test',args.source_ip)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
