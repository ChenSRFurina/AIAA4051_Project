"""C2b changes only optimizer reflection input, including unpaired failures."""
from __future__ import annotations

import json
from pathlib import Path
from baseline.feedback import BaselineFeedback, selected_failures
from baseline.io import write_json
from explore.memory import TrainMemory
from explore.pairing import pair_failure, task_type
from .diagnostics import diagnose
from .evidence import evidence

INSTRUCTION = '''Revise the skill using these TRAIN diagnostics. Gold labels locate training discrepancies only; the deployed skill has no gold. Infer targets from the question and checks from supplied evidence. Never copy question-specific answers/entities or solved examples into the skill.
Turn supported diagnoses into conditional, reusable checks: trigger -> evidence check -> action; state which valid behavior to preserve. String/order/name differences alone do not prove a failed reasoning step. EM=1 does not verify the success reasoning. Retrieved spans are exact excerpts, not verified entailment or internal execution traces. Unpaired cases have no success evidence; keep uncertainty, do not invent a comparison.
Respect this call's operator: mutation_other changes ONLY Global Guidance; leave Node Lists and Task Graphs verbatim. mutation_graph changes ONLY Node Lists and Task Graphs; leave Global Guidance verbatim. Put the intended repair inside the allowed sections. Return the complete new_skill and brief notes linking actual in-scope changes to failure IDs. Do not claim changes only in notes.
'''


def compact_span(span, counter, limit):
    # Preserve a literal source prefix; never fabricate or paraphrase evidence.
    text = span['text']
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    result = {'at': [span['start'], span['start'] + lo], 'text': text[:lo]}
    if span.get('document'): result['doc'] = counter.clip(span['document'], 25)
    return result


def pack_packet(packet, counter, maximum):
    f, s = packet['failure'], packet['success_reference']
    d = packet['diagnosis']
    base = {'failure_id': packet['failure_id'], 'pair_type': packet['pair_type'],
            'question': f['question'], 'predicted': f['predicted_answer'],
            'train_gold': f['gold_answers'], 'EM': f['em'], 'F1': f['f1'],
            'diagnosis': {'kind': d['kind'], 'confidence': d['confidence'],
                          'check': d['check'], 'preserve': d['preserve']},
            'success': None if s is None else {'id': s['id'], 'predicted': s['predicted_answer'],
                'question': s['question'] if packet['pair_type'] == 'similar_task' else 'same question/context', 'EM': 1},
            'evidence': {}, 'omitted': []}
    dumps = lambda p: json.dumps(p, ensure_ascii=False, separators=(',', ':'))
    # Questions and answers are retained verbatim for ordinary items. Explicitly
    # flag exceptional long fields; never silently slice a serialized packet.
    for limit in [240, 160, 100, 64]:
        if counter.count(dumps(base)) <= maximum - 100:
            break
        for key in ['question', 'predicted']:
            original = f['question' if key == 'question' else 'predicted_answer']
            base[key] = counter.clip(original, limit)
            if base[key] != original and key + '_truncated' not in base['omitted']:
                base['omitted'].append(key + '_truncated')
        base['train_gold'] = [counter.clip(a, limit) for a in f['gold_answers']]
        if base['train_gold'] != f['gold_answers'] and 'train_gold_truncated' not in base['omitted']:
            base['omitted'].append('train_gold_truncated')
        if s:
            base['success']['predicted'] = counter.clip(s['predicted_answer'], limit)
            if packet['pair_type'] == 'similar_task':
                base['success']['question'] = counter.clip(s['question'], limit)
    if counter.count(dumps(base)) > maximum:
        raise ValueError('C2b cannot retain the selected failure identity and diagnostics within the unchanged cap')
    channels = [('failure_context', packet['failure_evidence']['context']),
                ('failure_response', packet['failure_evidence']['response'])]
    if s:
        # Same-question context is identical; retain more response evidence instead.
        if packet['pair_type'] != 'same_question':
            channels.append(('success_context', packet['success_evidence']['context']))
        channels.append(('success_response', packet['success_evidence']['response']))
    for index in range(4):
        for key, spans in channels:
            if index >= len(spans):
                continue
            accepted = False
            for length in [90, 64, 40, 24]:
                snippet = compact_span(spans[index], counter, length)
                base['evidence'].setdefault(key, []).append(snippet)
                if counter.count(dumps(base)) <= maximum - 20:
                    accepted = True
                    break
                base['evidence'][key].pop()
                if not base['evidence'][key]:
                    del base['evidence'][key]
            if not accepted:
                continue
    # Missing channels are explicit, not mistaken for evidence of absence.
    base['omitted'] += [key for key, spans in channels if spans and key not in base['evidence']]
    if counter.count(dumps(base)) > maximum:
        raise ValueError('C2b diagnostic packet overflow')
    return base


def pack_feedback(packets, counter, maximum):
    if not packets:
        return '', []
    header_cost = counter.count(INSTRUCTION)
    allowance = (maximum - header_cost - 16) // len(packets)
    while allowance > 80:
        actual = [pack_packet(p, counter, allowance) for p in packets]
        text = INSTRUCTION + '\n' + json.dumps(actual, ensure_ascii=False, separators=(',', ':'))
        if counter.count(text) <= maximum:
            return text, actual
        allowance -= 8
    raise ValueError('C2b feedback cannot fit all selected failures')


class DiagnosticFeedback(BaselineFeedback):
    name = 'C2b_diagnostic_evidence_feedback_v1'

    def __init__(self, dataset, train_ids, output, config):
        self.memory = TrainMemory(dataset, train_ids, condition='C2b', per_outcome_limit=config['pairing']['per_outcome_limit'])
        self.output, self.counter = Path(output), 0

    def observe(self, rows):
        for row in rows:
            self.memory.add(row)
        write_json(self.output / 'memory.json', self.memory.records())

    def build(self, rows, items, out_dir, seed, config, counter):
        failures = selected_failures(rows, config['failure_k'], seed)
        packets, counts = [], {'same_question': 0, 'similar_task': 0, 'unpaired': 0}
        for failure in failures:
            if failure['split'] != 'train' or failure['condition'] != 'C2b':
                raise ValueError('C2b feedback accepts own train observations only')
            pair = pair_failure(failure, self.memory, config['pairing']['jaccard_threshold'])
            reference = pair['success_reference']
            counts[pair['pair_type']] += 1
            packets.append({'failure_id': failure['id'], 'pair_type': pair['pair_type'],
                            'similarity': pair['similarity'], 'task_type': task_type(failure['question']),
                            'failure': failure, 'success_reference': reference, 'diagnosis': diagnose(failure),
                            'failure_evidence': evidence(failure, counter),
                            'success_evidence': evidence(reference, counter) if reference else None})
        text, actual = pack_feedback(packets, counter, config['feedback_max_tokens'])
        report = {'mode': self.name, 'selected_failure_ids': [f['id'] for f in failures], **counts,
                  'feedback_tokens': counter.count(text), 'packets': packets, 'actual_packets': actual,
                  'pairing_unchanged': True, 'unpaired_policy': 'diagnostic failure-only input; no invented success',
                  'note': 'Diagnostics are hypotheses, not scores; no rejection, resampling, extra calls or selection changes.'}
        self.counter += 1
        write_json(self.output / 'feedback' / f'{self.counter:04d}.json', {**report, 'actual_feedback': text})
        return text, report
