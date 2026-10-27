"""Complete failure inventories and paired repair/regression counts."""
from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path
from graphskillevo.envs.searchqa.evaluator import normalize_answer
from .io import write_json
from .evaluation import aggregate


def diagnose(row):
    observation = {"format_ok": row["format_ok"], "context_truncated": row["context_truncated"],
                   "infrastructure_error": row["error"], "gold_literal_visible": any(
                       normalize_answer(a) in normalize_answer(row["visible_context"]) for a in row["gold_answers"])}
    if row["error"]:
        category, reason = "infrastructure_error", "The request failed; retained in the EM/F1 denominator."
    elif not row["format_ok"]:
        category, reason = "answer_extraction", "Required answer tags are absent; upstream fallback was applied."
    else:
        category, reason = "unknown_insufficient_evidence", "EM=0 alone does not identify a faulty reasoning step."
    hypotheses = []
    if not observation["gold_literal_visible"]:
        hypotheses.append("Gold literal absent from visible context; aliases, paraphrases and truncation require inspection.")
    return {"primary_category": category, "secondary_hypotheses": hypotheses, "observations": observation,
            "diagnostic_explanation": reason, "skill_node": None, "node_attribution_status": "unassigned_output_only_hypothesis"}


def inventory(rows, output):
    output = Path(output)
    failures = [{**row, **diagnose(row)} for row in rows if row["em"] == 0]
    write_json(output / "failures.json", failures)
    write_json(output / "failure_taxonomy.json", {
        "complete": True, "evaluated_n": len(rows), "failed_n": len(failures),
        "primary_counts": dict(Counter(r["primary_category"] for r in failures)),
        "policy": "One primary label per failure. Semantic attribution is left uncertain until evidence is reviewed. No automatic broad causal claims.",
        "manual_review_fields": ["entity_confusion", "relation_chain", "constraint_omission", "unsupported_inference", "skill_rule_inapplicable"]})
    output.mkdir(parents=True, exist_ok=True)
    keys = ["id", "question", "gold_answers", "predicted_answer", "em", "f1", "primary_category", "diagnostic_explanation", "skill_node"]
    with (output / "failures.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(failures)
    write_json(output / "metrics.json", aggregate(rows))
    return failures


def compare(left, right, seed=42):
    a, b = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    if not a or set(a) != set(b) or len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired comparison requires identical unique fixed IDs")
    for key in a:
        if a[key]["visible_context_hash"] != b[key]["visible_context_hash"] or a[key]["gold_answers"] != b[key]["gold_answers"]:
            raise ValueError("Paired inputs/references differ")
    counts = Counter()
    transitions = []
    for key in sorted(a):
        state = { (0, 1): "repaired", (1, 0): "regressed", (1, 1): "both_correct", (0, 0): "both_wrong" }[(a[key]["hard"], b[key]["hard"])]
        counts[state] += 1
        transitions.append({"id": key, "transition": state, "B": a[key], "C": b[key]})
    n_failed = sum(not r["hard"] for r in left)
    n_correct = sum(r["hard"] for r in left)
    differences = [b[key]["em"] - a[key]["em"] for key in sorted(a)]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
    return {"n": len(a), "counts": {key: counts[key] for key in ["repaired", "regressed", "both_correct", "both_wrong"]},
            "repair_rate": counts["repaired"] / n_failed if n_failed else None,
            "regression_rate": counts["regressed"] / n_correct if n_correct else None,
            "em_difference_C_minus_B": sum(differences) / len(a), "paired_bootstrap_95": [boot[49], boot[1949]],
            "uncertainty_scope": "Question-level resampling; not variability across training seeds.", "transitions": transitions}
