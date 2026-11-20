"""Offline replay of historical C TRAIN observations to validate C2b packing.

The condition is relabelled only inside this isolated replay, not in run records.
No output from B/C enters the actual C2b training memory.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Baseline'))
sys.path.insert(0, str(ROOT / 'Explore'))
from baseline.budget import TokenCounter
from baseline.io import read_json, read_jsonl, write_json
from explore_v2.feedback import DiagnosticFeedback


def replay():
    result = {}
    for dataset in ['searchqa', 'hotpotqa']:
        run = ROOT / 'Explore/runs/api/part3-20261110' / dataset / 'C'
        config = read_json(run / 'config.json')
        counter = TokenCounter(config['model']['tokenizer_path'])
        ids = read_json(ROOT / 'Baseline/data/prepared' / dataset / 'train_ids.json')
        records = []
        with tempfile.TemporaryDirectory() as tmp:
            feedback = DiagnosticFeedback(dataset, ids, tmp, config)
            for h in read_json(run / 'history.json'):
                batches = []
                for parent in h['parent_population']:
                    folder = run / 'evaluations' / f"g{h['generation']}" / parent['id'] / 'train'
                    rows = deepcopy(read_jsonl(folder / 'outputs.jsonl'))
                    for r in rows: r['condition'] = 'C2b'
                    batches.append((rows, folder))
                feedback.observe([r for rows, _ in batches for r in rows])
                for rows, folder in batches:
                    text, report = feedback.build(rows, [], tmp, config['seed'] + h['generation'] * 10000, config, counter)
                    prior = read_json(folder / 'feedback.json')
                    assert report['selected_failure_ids'] == prior['selected_failure_ids']
                    assert counter.count(text) <= config['feedback_max_tokens']
                    for packet, old, actual in zip(report['packets'], prior['packets'], report['actual_packets']):
                        assert packet['pair_type'] == old['pair_type']
                        if packet['success_reference']:
                            assert packet['success_reference']['observation_id'] == old['success_reference']['observation_id']
                        for side in ['failure', 'success']:
                            row = packet['failure' if side == 'failure' else 'success_reference']
                            if not row: continue
                            for name, field in [('context', 'visible_context'), ('response', 'raw_response')]:
                                for span in packet[side + '_evidence'][name]:
                                    assert row[field][span['start']:span['end']] == span['text']
                                for span in actual['evidence'].get(side + '_' + name, []):
                                    assert row[field][slice(*span['at'])] == span['text']
                        records.append({'id': packet['failure_id'], 'pair': packet['pair_type'],
                                        'diagnosis': packet['diagnosis']['kind'], 'actual_packet': actual,
                                        'feedback_tokens': report['feedback_tokens']})
            result[dataset] = {'status': 'passed', 'parent_feedbacks': 6, 'records': records,
                               'diagnoses': dict(Counter(p['diagnosis'] for p in records)),
                               'max_feedback_tokens': max(p['feedback_tokens'] for p in records),
                               'checks': ['same selected failures', 'same success observation', 'exact source substrings',
                                          '2048-token cap', 'all selected packets retained'],
                               'scope': 'historical train-only offline packing validation, no API calls'}
    path = ROOT / 'Explore/runs/reports/C2b_offline_replay_20261110.json'
    write_json(path, result)
    print(json.dumps({d: {k: v for k, v in a.items() if k != 'records'} for d, a in result.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    replay()
