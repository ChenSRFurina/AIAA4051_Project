"""Input integrity tests; no hosted model calls or test-set observations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Baseline'))
sys.path.insert(0, str(ROOT / 'Explore'))
from baseline.budget import TokenCounter
from baseline.io import digest
from baseline.feedback import selected_failures
from explore.memory import TrainMemory
from explore.pairing import pair_failure
from explore_v2.diagnostics import diagnose
from explore_v2.evidence import evidence
from explore_v2.feedback import DiagnosticFeedback


def row(id='f', prediction='False', gold='Example Institute'):
    context = '[DOC] Irrelevant\n' + 'Unrelated detail. ' * 80 + '\n[DOC] Institute\nThe Example Institute operates the Science Museum.'
    return {'id': id, 'dataset': 'searchqa', 'split': 'train', 'condition': 'C2b', 'round': 1,
            'question': 'The Science Museum is operated by this institution', 'predicted_answer': prediction,
            'gold_answers': [gold], 'format_ok': True, 'em': 0, 'f1': 0, 'hard': 0,
            'raw_response': 'The museum is operated by Example Institute. <answer>' + prediction + '</answer>',
            'visible_context': context, 'visible_context_hash': digest(context),
            'observation_id': id, 'skill_hash': 'skill', 'cached': False, 'error': ''}


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = TokenCounter(ROOT / 'api/runtime/tokenizer')
        cls.config = {'pairing': {'per_outcome_limit': 2, 'jaccard_threshold': .2},
                      'failure_k': 3, 'feedback_max_tokens': 2048}

    def test_answer_type_and_boolean_preservation(self):
        d = diagnose(row())
        self.assertEqual(d['kind'], 'answer_type_mismatch')
        self.assertIn('genuine yes/no', d['preserve'])
        r = row(); r['question'] = 'Is the museum open?'
        self.assertNotEqual(diagnose(r)['kind'], 'answer_type_mismatch')

    def test_order_alias_and_uncertainty(self):
        self.assertEqual(diagnose(row(prediction='Beta and Alpha', gold='Alpha and Beta'))['kind'], 'token_order_only')
        self.assertEqual(diagnose(row(prediction='Alex Smith', gold='Smith'))['kind'], 'name_or_granularity_overlap')
        self.assertEqual(diagnose(row(prediction='X', gold='Y'))['kind'], 'unresolved_content_error')

    def test_late_evidence_and_literal_offsets(self):
        r = row(); e = evidence(r, self.counter)
        self.assertTrue(any(s['start'] > 500 and 'operates' in s['text'] for s in e['context']))
        for spans in e.values():
            for s in spans:
                self.assertEqual(r[s['source']][s['start']:s['end']], s['text'])
                self.assertEqual(digest(r[s['source']]), s['source_hash'])

    def test_selection_pairing_budget_and_no_mutation(self):
        rows = [row(str(i)) for i in range(5)]
        before = deepcopy(rows)
        success = row('0', 'Example Institute'); success.update(em=1, hard=1, observation_id='success', skill_hash='second')
        with tempfile.TemporaryDirectory() as tmp:
            feedback = DiagnosticFeedback('searchqa', [r['id'] for r in rows], tmp, self.config)
            feedback.observe(rows + [success])
            text, report = feedback.build(rows, [], tmp, 42, self.config, self.counter)
            self.assertEqual(report['selected_failure_ids'], [r['id'] for r in selected_failures(rows, 3, 42)])
            self.assertLessEqual(self.counter.count(text), 2048)
            for packet, actual in zip(report['packets'], report['actual_packets']):
                expected = pair_failure(packet['failure'], feedback.memory, .2)
                self.assertEqual(expected['pair_type'], packet['pair_type'])
                self.assertEqual(actual['failure_id'], packet['failure_id'])
                self.assertEqual(actual['question'], packet['failure']['question'])
            self.assertEqual(rows, before)

    def test_memory_boundaries(self):
        memory = TrainMemory('searchqa', ['f'], condition='C2b')
        for key, value in [('split', 'dev'), ('split', 'test'), ('condition', 'B'), ('condition', 'C'), ('id', 'foreign')]:
            r = row(); r[key] = value
            with self.assertRaises(ValueError): memory.add(r)

    def test_future_success_excluded(self):
        memory = TrainMemory('searchqa', ['f'], condition='C2b')
        r = row(prediction='Example Institute'); r.update(round=2, hard=1, em=1)
        memory.add(r)
        self.assertEqual(pair_failure(row(), memory)['pair_type'], 'unpaired')


if __name__ == '__main__':
    unittest.main()
