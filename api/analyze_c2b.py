"""Read-only train/dev audit and B/C/C2b comparison; no test evaluation."""
from __future__ import annotations

from collections import Counter
import difflib
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Baseline'))
sys.path.insert(0, str(ROOT / 'Explore'))
from baseline.io import digest, read_json, read_jsonl, write_json
from baseline.evaluation import aggregate
from baseline.feedback import selected_failures
from baseline.budget import TokenCounter
from graphskillevo.types import SkillIndividual
from graphskillevo.selection import sorted_by_validation, sample_parent, select_survivors, child_operator_sequence
from graphskillevo.graph_skill import split_graph_skill, adopt_other, adopt_graph
from graphskillevo.envs.searchqa.evaluator import normalize_answer
from explore_v2.feedback import DiagnosticFeedback
from part3_analysis import checked_rows, checkpoint_data, compact_comparison, choose_checkpoint
from report_part3 import service_metadata, initial_input_repeat
from run_part3 import BASELINES, SESSION as OLD_SESSION, sha
from run_c2b import SESSION


def output_health(run):
    """Audit physical requests and unique, uncached QA observations only."""
    calls = read_jsonl(run / 'calls.jsonl')
    by_id = {r['call_id']: r for r in calls}
    assert len(by_id) == len(calls)
    qa = {}
    for path in (run / 'evaluations').rglob('outputs.jsonl'):
        for row in read_jsonl(path):
            if row.get('cached'): continue
            used = [a.get('usage', {}).get('call_id') for a in row.get('attempts', [])]
            if not any(call_id in by_id for call_id in used): continue
            key = row['observation_id']
            if key in qa: assert qa[key]['raw_response'] == row['raw_response']
            qa[key] = row
    return {
        'physical_calls': len(calls),
        'length_by_phase': dict(Counter(r['phase'] for r in calls if r.get('finish_reason') == 'length')),
        'qa_by_split': {split: {
            'n': sum(r['split'] == split for r in qa.values()),
            'format_failures': sum(r['split'] == split and not r['format_ok'] for r in qa.values()),
            'length_format_failures': sum(r['split'] == split and not r['format_ok'] and
                any(a.get('usage', {}).get('finish_reason') == 'length' for a in r.get('attempts', []))
                for r in qa.values())} for split in ['train', 'dev']},
        'scope': 'new physical requests and unique uncached QA observations; excludes shared initialization'
    }


def mutations(run, init):
    points, records, rows = checkpoint_data(run, init)
    result = []
    for h in read_json(run / 'history.json'):
        for op in h['operator_records']:
            attempt_log = [{'attempt': i + 1, 'status': a['status'], 'error': a.get('error')}
                           for i, a in enumerate(op['attempts'])]
            if not op['accepted']:
                result.append({'round': h['generation'], 'child': op['child_id'], 'accepted': False,
                               'attempts': len(op['attempts']), 'attempt_log': attempt_log})
                continue
            child, parent = records[op['child_id']], records[op['parent_ids'][0]]
            attempt = next(a for a in reversed(op['attempts']) if a['status'] == 'ok')
            candidate = attempt['parsed']['new_skill'].rstrip() + '\n'
            adopted = adopt_other(parent['content'], candidate) if child['origin'] == 'mutation_other' else adopt_graph(parent['content'], candidate)
            assert adopted == child['content']
            raw_parts, adopted_parts = split_graph_skill(candidate), split_graph_skill(adopted)
            forbidden = ['node_lists', 'task_graphs'] if child['origin'] == 'mutation_other' else ['global_guidance']
            discarded = [key for key in forbidden if getattr(raw_parts, key) != getattr(adopted_parts, key)]
            lines = list(difflib.unified_diff(parent['content'].splitlines(), adopted.splitlines(), fromfile=parent['id'], tofile=child['id'], lineterm=''))
            added = [s[1:] for s in lines if s.startswith('+') and not s.startswith('+++')]
            folder = run / 'evaluations' / f"g{h['generation']}" / parent['id'] / 'train'
            exposed_ids = set(read_json(folder / 'feedback.json')['selected_failure_ids'])
            added_normalized = ' ' + normalize_answer(' '.join(added)) + ' '
            copy_flags, seen_phrases = [], set()
            for observation in read_jsonl(folder / 'outputs.jsonl'):
                if observation['id'] not in exposed_ids: continue
                for source, phrases in [('gold', observation['gold_answers']), ('prediction', [observation['predicted_answer']])]:
                    for phrase in phrases:
                        normalized = normalize_answer(phrase)
                        if len(normalized) < 4 or normalized in {'true', 'false', 'yes', 'no', 'unknown', 'not found in context', 'cannot be determined from the text'}:
                            continue
                        if normalized not in seen_phrases and (' ' + normalized + ' ') in added_normalized:
                            seen_phrases.add(normalized)
                            copy_flags.append({'phrase': phrase, 'source': source, 'failure_id': observation['id'],
                                               'status': 'literal_reuse_flag_requires_semantic_review'})
            result.append({'round': h['generation'], 'child': child['id'], 'operator': child['origin'],
                           'parent': parent['id'], 'accepted': True, 'attempts': len(op['attempts']),
                           'attempt_log': attempt_log,
                           'parent_em': parent['val_hard'], 'child_em': child['val_hard'], 'child_f1': child['val_soft'],
                           'retained': child['id'] in h['survivor_ids'], 'no_op': digest(parent['content']) == digest(adopted),
                           'discarded_sections': discarded, 'new_gold_lines': [s for s in added if 'gold' in s.lower()],
                           'training_answer_reuse_flags': copy_flags,
                           'diff': '\n'.join(lines), 'notes': attempt['parsed'].get('notes'),
                           'dev_transitions': compact_comparison(rows[parent['id']], rows[child['id']]),
                           'output_anomalies': [r for r in rows[child['id']] if not r['format_ok'] or
                               any(a.get('usage', {}).get('finish_reason') == 'length' for a in r.get('attempts', []))],
                           'adopted_skill': adopted, 'raw_candidate': candidate})
    return result


def audit_c2b(dataset, run, baseline, counter):
    config = read_json(run / 'config.json')
    history = read_json(run / 'history.json')
    policy = read_json(run.parent / 'policy.json')
    assert read_json(run.parent / 'completed.json')['policy_id'] == policy['policy_id']
    assert digest({k: v for k, v in policy.items() if k != 'policy_id'}) == policy['policy_id']
    for relative, expected in policy['method_files'].items():
        assert sha(run.parent / 'method_source' / relative) == expected
        assert sha(ROOT / relative) == expected
    assert digest(policy['method_files']) == policy['method_hash']
    assert counter.identity == read_json(run / 'run_meta.json')['tokenizer_hash']
    assert policy['config_hash'] == digest(config)
    train_ids = read_json(ROOT / 'Baseline/data/prepared' / dataset / 'train_ids.json')
    exposures, parent_feedbacks, train_metrics, memory_sizes = [], {}, [], []
    with tempfile.TemporaryDirectory() as tmp:
        replay = DiagnosticFeedback(dataset, train_ids, tmp, config)
        for h in history:
            generation = h['generation']
            batch = read_json(run / 'train_batches' / f'{generation}.json')
            assert batch == read_json(baseline / 'B/train_batches' / f'{generation}.json')
            parents = [SkillIndividual(**p) for p in h['parent_population']]
            candidates = [SkillIndividual(**p) for p in h['candidates']]
            ranked = sorted_by_validation(parents)
            rng = random.Random(config['seed'] + generation * 3000)
            assert [o['operator'] for o in h['operator_records']] == child_operator_sequence(2)
            for op in h['operator_records']:
                assert op['parent_ids'] == [sample_parent(ranked, rng).id]
                assert len(op['attempts']) <= config['max_attempts']
            survivors = select_survivors(parents, candidates, population_size=2)
            survivors.sort(key=lambda item: item.survivor_score, reverse=True)
            assert [p.id for p in survivors] == h['survivor_ids']
            assert max(survivors, key=lambda p: p.val_hard).id == h['best_id']
            batches = []
            for parent in parents:
                folder = run / 'evaluations' / f'g{generation}' / parent.id / 'train'
                rows = read_jsonl(folder / 'outputs.jsonl')
                assert [r['id'] for r in rows] == batch['ids']
                assert all(r['condition'] == 'C2b' and r['split'] == 'train' and r['id'] in train_ids and not r['cached'] for r in rows)
                train_metrics.append({'round': generation, 'parent': parent.id, 'metrics': checked_rows(rows, dataset)})
                batches.append((rows, folder, parent.id))
            replay.observe([r for rows, _, _ in batches for r in rows])
            memory_sizes.append({'round': generation, 'records': len(replay.memory.records()),
                                 'success_observations': len(replay.memory.successes()),
                                 'success_questions': len({r['id'] for r in replay.memory.successes()})})
            for rows, folder, parent_id in batches:
                text, fresh = replay.build(rows, [], tmp, batch['seed'], config, counter)
                stored = read_json(folder / 'feedback.json')
                assert text == stored['actual_feedback']
                assert fresh == {k: v for k, v in stored.items() if k != 'actual_feedback'}
                assert fresh['selected_failure_ids'] == [r['id'] for r in selected_failures(rows, config['failure_k'], batch['seed'])]
                assert counter.count(text) == stored['feedback_tokens'] <= config['feedback_max_tokens']
                parent_feedbacks[generation, parent_id] = stored
            for op in h['operator_records']:
                f = parent_feedbacks[generation, op['parent_ids'][0]]
                verified = 0
                for attempt in op['attempts']:
                    if attempt.get('prompt'):
                        assert f['actual_feedback'].strip() in attempt['prompt']
                        verified += 1
                assert verified >= 1
                compact_packets = []
                for packet, actual in zip(f['packets'], f['actual_packets']):
                    for side in ['failure', 'success']:
                        original = packet['failure' if side == 'failure' else 'success_reference']
                        if original is None: continue
                        assert original['split'] == 'train' and original['condition'] == 'C2b'
                        assert original['round'] <= generation and original['id'] in train_ids
                        for name, field in [('context', 'visible_context'), ('response', 'raw_response')]:
                            for span in actual['evidence'].get(side + '_' + name, []):
                                assert original[field][slice(*span['at'])] == span['text']
                    compact_packets.append({'failure_id': packet['failure_id'], 'pair_type': packet['pair_type'],
                        'diagnosis': packet['diagnosis'], 'actual': actual,
                        'full_failure': packet['failure'], 'full_success': packet['success_reference']})
                exposures.append({'round': generation, 'child': op['child_id'], 'parent': op['parent_ids'][0],
                                  'attempts_with_verified_feedback': verified, 'feedback_tokens': f['feedback_tokens'],
                                  'actual_feedback': f['actual_feedback'], 'packets': compact_packets,
                                  **{k: f[k] for k in ['same_question', 'similar_task', 'unpaired']}})
        assert replay.memory.records() == read_json(run / 'memory.json')
    packets = [p for e in exposures for p in e['packets']]
    paired = [p for p in packets if p['pair_type'] != 'unpaired']
    diag = Counter(p['diagnosis']['kind'] for p in packets)
    counts = {'selected': len(packets), 'paired': len(paired), 'paired_rate': len(paired) / len(packets) if packets else None,
              'unique_paired_questions': len({p['failure_id'] for p in paired}),
              'operators_without_reference': sum(e['same_question'] + e['similar_task'] == 0 for e in exposures),
              'feedback_max_tokens': max(e['feedback_tokens'] for e in exposures),
              'failure_context_present': sum(bool(p['actual']['evidence'].get('failure_context')) for p in packets),
              'success_response_present': sum(bool(p['actual']['evidence'].get('success_response')) for p in paired),
              'unpaired': sum(e['unpaired'] for e in exposures), 'diagnoses': dict(diag)}
    return {'status': 'passed', 'checks': ['sealed method files', 'exact shared config/tokenizer', 'B train batches',
            'parent sampling', 'EM survivor/final selection', 'random failure selection', 'original C pairing replay',
            'full feedback determinism', 'own train-only memory', 'no future references', 'literal source offsets',
            'feedback cap', 'actual optimizer exposure', 'memory exact replay'],
            'policy': policy, 'counts': counts, 'exposures': exposures, 'train_metrics': train_metrics,
            'memory_sizes': memory_sizes}


def analyze(dataset):
    baseline = BASELINES[dataset]
    paths = {'B': baseline / 'B', 'C': OLD_SESSION / dataset / 'C', 'C2b': SESSION / dataset / 'C2b'}
    summaries = {k: read_json(p / 'summary.json') for k, p in paths.items()}
    rows = {k: read_json(p / 'final_dev_outputs.json') for k, p in paths.items()}
    points, records, checkpoints = {}, {}, {}
    costs, metrics, changes, training, health = {}, {}, {}, {}, {}
    for method, path in paths.items():
        s = summaries[method]
        assert s['status'] == 'completed' and s['completed_rounds'] == 3
        assert checked_rows(rows[method], dataset) == s['dev']
        for key in ['config_hash', 'data_hash', 'code_hash', 'tokenizer_hash', 'upstream_commit', 'demo']:
            assert s[key] == summaries['B'][key]
        assert read_json(path / 'config.json') == read_json(baseline / 'B/config.json')
        assert read_json(path / 'shared_initialization.json')['population_hash'] == read_json(baseline / 'B/shared_initialization.json')['population_hash']
        assert (path / 'best_skill.md').read_text(encoding='utf-8') == s['best']['content']
        points[method], records[method], checkpoints[method] = checkpoint_data(path, baseline / 'shared_init')
        assert points[method][-1]['best_id'] == s['best']['id']
        ledger = read_jsonl(path / 'ledger.jsonl')
        assert all(e.get('phase') != 'test' for e in ledger), 'This comparison contains training/dev runs only'
        total = {k: sum(e.get(k, 0) for e in ledger) for k in s['cost']}
        assert total == s['cost']
        config = read_json(path / 'config.json')
        assert total['calls'] <= config['budget']['max_calls'] and total['generated_tokens'] <= config['budget']['max_generated_tokens']
        physical = {k: sum(e.get(k, 0) for e in ledger if e.get('physical_calls')) for k in s['cost']}
        costs[method] = {'logical': total, 'physical': physical,
                         'failures': sum(e.get('status') != 'ok' for e in ledger if e.get('physical_calls')),
                         'wall_seconds_sum': sum(e.get('wall_seconds', 0) for e in ledger if e.get('physical_calls')),
                         'by_phase': {phase: {k: sum(e.get(k, 0) for e in ledger if e.get('phase') == phase and e.get('physical_calls'))
                                            for k in ['calls', 'prompt_tokens', 'generated_tokens']}
                                      for phase in sorted({e.get('phase', 'unknown') for e in ledger})}}
        metrics[method] = s['dev']
        health[method] = output_health(path)
        assert health[method]['physical_calls'] == physical['physical_calls']
        changes[method] = mutations(path, baseline / 'shared_init')
        training[method] = []
        for h in read_json(path / 'history.json'):
            for parent in h['parent_population']:
                tr = read_jsonl(path / 'evaluations' / f"g{h['generation']}" / parent['id'] / 'train/outputs.jsonl')
                training[method].append({'round': h['generation'], 'parent': parent['id'], 'metrics': checked_rows(tr, dataset)})
    counter = TokenCounter(config['model']['tokenizer_path'])
    audit = audit_c2b(dataset, paths['C2b'], baseline, counter)
    call_cap = min(s['cost']['calls'] for s in summaries.values())
    token_cap = min(s['cost']['generated_tokens'] for s in summaries.values())
    chosen = {k: choose_checkpoint(p, call_cap, token_cap) for k, p in points.items()}
    comp = {left + '_to_C2b': compact_comparison(rows[left], rows['C2b']) for left in ['B', 'C']}
    common_comp = {left + '_to_C2b': compact_comparison(checkpoints[left][chosen[left]['best_id']],
                                                      checkpoints['C2b'][chosen['C2b']['best_id']]) for left in ['B', 'C']}
    old = read_json(OLD_SESSION / dataset / 'analysis.json')['feedback']
    result = {'status': 'passed', 'dataset': dataset, 'scope': 'train_dev_only', 'config': config,
              'paths': {k: str(p) for k, p in paths.items()}, 'summaries': summaries, 'metrics': metrics,
              'curves': points, 'costs': costs, 'mutations': changes, 'training': training, 'health': health,
              'comparisons': comp, 'matched': {'call_cap': call_cap, 'generated_token_cap': token_cap,
                                             'checkpoints': chosen, 'comparisons': common_comp},
              'audit': audit, 'original_C_feedback': old, 'dev_rows': rows,
              'initial_repeat': {k: initial_input_repeat(paths[k], paths['C2b']) for k in ['B', 'C']},
              'service': {k: service_metadata(p) for k, p in paths.items()},
              'limitations': ['one seed and 30 dev examples per dataset', 'dev used for selection',
                              'historical B/C controls; service revision undisclosed', 'no new test results',
                              'train summaries are parent-batch rollouts, not final full-train evaluation']}
    write_json(SESSION / dataset / 'analysis.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['searchqa', 'hotpotqa', 'all'], default='all', nargs='?')
    args = parser.parse_args()
    for ds in ['searchqa', 'hotpotqa'] if args.dataset == 'all' else [args.dataset]:
        a = analyze(ds)
        print(json.dumps({'dataset': ds, 'status': a['status'], 'metrics': a['metrics'],
                          'feedback': a['audit']['counts'], 'comparison': a['comparisons']}, ensure_ascii=False))
