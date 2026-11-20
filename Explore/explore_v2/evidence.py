"""Select exact source spans from full TRAIN observations, not their prefixes.

Training labels may locate diagnostic evidence; they never enter QA prompts.
Lexical retrieval is an aid to inspection, not proof of entailment or causality.
"""
from __future__ import annotations

import math
import re
from baseline.io import digest

STOP = set('a an the is are was were be been in on of for to and or by with as at from this that these those it its what which who where when how does did do has have had both named called'.split())


def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - STOP


def source_spans(text):
    """Line/sentence offsets in the original string (no reconstructed quotes)."""
    for match in re.finditer(r'[^\n.!?]+(?:[.!?]+|(?=\n)|$)', text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            yield start, end


def bounded_span(text, start, end, query, counter, limit):
    """An exact substring around hits if the source sentence is too long."""
    if counter.count(text[start:end]) <= limit:
        return start, end
    hits = [m.start() + start for m in re.finditer(r'[a-z0-9]+', text[start:end].lower()) if m.group() in query]
    pivot = hits[0] if hits else start
    left = max(start, pivot - 100)
    # Binary search source character offsets; tokenizer.decode can normalize text.
    lo, hi = left, end
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[left:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return left, lo


def select_spans(row, field, counter, per_span=100, count=4):
    text = row[field]
    q = terms(row['question'])
    answers = terms(' '.join(row['gold_answers']) + ' ' + row['predicted_answer'])
    locations = list(source_spans(text))
    frequencies = {t: sum(t in terms(text[a:b]) for a, b in locations) for t in q}
    phrases = re.findall(r'"([^"]+)"', row['question'].lower())
    candidates = []
    for start, end in locations:
        words = terms(text[start:end])
        score = sum(1 + math.log((1 + len(locations)) / (1 + frequencies[t])) for t in q & words) + 3 * len(answers & words)
        score += 12 * sum(p in text[start:end].lower() for p in phrases)
        score += 6 * sum(t.isdigit() for t in q & words)
        if field == 'raw_response':
            if re.search(r'therefore|conclusion|final answer', text[start:end], re.I): score += 12
            if re.search(r'\b(identify|find|determine)\b', text[start:end], re.I): score -= 6
            if '<answer>' in text[start:end].lower(): score += 8
        candidates.append((score, start, end, words))
    # Greedy novel-term coverage avoids spending all space on repeated answer hits.
    selected, covered = [], set()
    while candidates and len(selected) < count:
        best = max(candidates, key=lambda c: (c[0] + 2 * len((c[3] & (q | answers)) - covered), -c[1]))
        candidates.remove(best)
        _, start, end, words = best
        start, end = bounded_span(text, start, end, q | answers, counter, per_span)
        if end <= start:
            continue
        title = None
        if field == 'visible_context':
            doc = text.rfind('[DOC]', 0, start + 1)
            if doc >= 0:
                header = text[doc:min(len(text), doc + 240)].split('\n')[0]
                if '[TLE]' in header: header = header.split('[TLE]', 1)[1].split('[PAR]', 1)[0]
                title = header.strip()
        selected.append({'source': field, 'start': start, 'end': end, 'text': text[start:end], 'document': title,
                         'source_hash': digest(text), 'observation_id': row['observation_id'],
                         'retrieval': 'lexical_train_label_assisted_not_entailment'})
        covered |= words
    return selected


def evidence(row, counter):
    return {'context': select_spans(row, 'visible_context', counter),
            'response': select_spans(row, 'raw_response', counter, per_span=85, count=3)}
