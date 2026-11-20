"""Conservative observable labels, never a judge or a replacement scorer."""
from __future__ import annotations

from collections import Counter
import re
from graphskillevo.envs.searchqa.evaluator import normalize_answer


def diagnose(row):
    prediction = normalize_answer(row['predicted_answer'])
    golds = [normalize_answer(a) for a in row['gold_answers']]
    question = row['question'].strip().lower()
    boolean_request = bool(re.match(r'^(is|are|was|were|did|does|do|has|have|can|could|will|would)\b', question))
    bools = {'true', 'false', 'yes', 'no'}
    result = {'kind': 'unresolved_content_error', 'confidence': 'low',
              'observation': 'EM=0; the available output alone does not identify a failed reasoning step.',
              'check': 'Verify the requested target and each necessary relation against the supplied passages; keep the diagnosis uncertain where evidence is missing.',
              'preserve': 'Preserve supported evidence and valid reasoning; do not invent a new task-wide restriction.'}
    if not row['format_ok']:
        result.update(kind='answer_format', confidence='high', observation='The required answer tag is absent.',
                      check='Keep evidence-based reasoning and place the final answer in the required answer tag.')
    elif prediction in bools and all(g not in bools for g in golds) and not boolean_request:
        result.update(kind='answer_type_mismatch', confidence='high',
                      observation='The response is boolean, while this training target is non-boolean and the question is not an explicit yes/no request.',
                      check='Infer the requested answer type from the question or clue; for an entity/fact request extract the missing entity/fact rather than confirming the clue.',
                      preserve='Keep yes/no responses for genuine yes/no questions. Do not copy the training answer into the skill.')
    elif prediction and any(Counter(prediction.split()) == Counter(g.split()) for g in golds):
        result.update(kind='token_order_only', confidence='high',
                      observation='Normalized answer token multisets match; EM differs by order. This does not establish a reasoning failure.',
                      check='Check whether the question or evidence requires an order; do not impose an arbitrary reference order on unrelated questions.',
                      preserve='Preserve all supported entities and relations. Exact-match scoring is unchanged.')
    elif prediction and any((' ' + prediction + ' ') in (' ' + g + ' ') or
                            (' ' + g + ' ') in (' ' + prediction + ' ') for g in golds if g):
        result.update(kind='name_or_granularity_overlap', confidence='medium',
                      observation='One normalized answer contains the other; alias, name length or granularity may explain the mismatch, but semantic equivalence is unverified.',
                      check='Check the referent and requested granularity against the question and source. Do not infer a missing hop solely from the answer-string difference.',
                      preserve='Retain correctly supported intermediate relations; do not globally shorten names or remove needed qualifiers.')
    elif row['f1'] >= .8:
        result.update(kind='high_overlap_unresolved', confidence='low',
                      observation='High answer-token overlap with EM=0; wording and substantive differences both remain possible.',
                      check='Locate the differing qualifier or relation in the original evidence before changing reasoning rules.')
    result['inference_trace_observed'] = False
    return result
