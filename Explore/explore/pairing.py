from __future__ import annotations

import re


def task_type(question):
    """Observable question-only rules; never read HotpotQA's gold type/answers."""
    q = question.lower().strip()
    if re.match(r"^(is|are|was|were|did|does|do|has|have|can)\b", q):
        return "yes_no"
    if re.search(r"\b(more|less|earlier|later|first|older|younger|both|compare)\b", q):
        return "comparison"
    if re.search(r"\b(how many|when|year|date|number)\b", q):
        return "count_date"
    if re.search(r"\b(whose|wife|husband|mentor|father|mother|founded by|named after)\b", q):
        return "relation_chain"
    if re.match(r"^(who|where|what|which)\b", q):
        return "entity_attribute"
    return "unknown"


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(left, right):
    a, b = words(left), words(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def pair_failure(failure, memory, threshold=.2):
    successes = [r for r in memory.successes() if r["dataset"] == failure["dataset"] and r["round"] <= failure["round"]]
    same = [r for r in successes if r["id"] == failure["id"] and r["visible_context_hash"] == failure["visible_context_hash"]]
    order = lambda r: (-r["round"], r["id"], r["skill_hash"], r["observation_id"])
    if same:
        return {"pair_type": "same_question", "success_reference": sorted(same, key=order)[0], "similarity": 1.0}
    kind = task_type(failure["question"])
    candidates = [(jaccard(failure["question"], r["question"]), r) for r in successes
                  if kind != "unknown" and r["id"] != failure["id"] and task_type(r["question"]) == kind]
    candidates = [(similarity, r) for similarity, r in candidates if similarity >= threshold]
    if candidates:
        # Same-round first, then similarity, then fixed provenance order.
        similarity, reference = sorted(candidates, key=lambda pair: (pair[1]["round"] != failure["round"], -pair[0], *order(pair[1])))[0]
        return {"pair_type": "similar_task", "success_reference": reference, "similarity": similarity}
    return {"pair_type": "unpaired", "success_reference": None, "similarity": None}
