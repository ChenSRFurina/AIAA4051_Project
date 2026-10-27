from __future__ import annotations

import json
import re
from pathlib import Path
from baseline.feedback import BaselineFeedback, fit_parts, selected_failures
from baseline.io import write_json
from .memory import TrainMemory
from .pairing import pair_failure, task_type

INSTRUCTION = """Use the selected training failures and available success references to revise the parent graph-structured skill.
Identify shared task conditions, a supported or suspected harmful deviation, a relevant skill node or missing rule, and successful behavior to preserve.
Add narrowly applicable checks. Differences alone are not errors. EM=1 does not verify every reasoning step.
Similar-task references support abstract strategy comparison only, not causal step replacement.
Node attribution from single-turn output is a diagnosis hypothesis, not an observed internal execution trace.
If unpaired, use the original failure feedback. If evidence is weak, record uncertainty and avoid broad prohibitions.
Never copy dataset answers, question-specific entities or solved examples into the skill.
Preserve the required structure. Return new_skill and notes linking changes to supplied evidence IDs.
"""

SHORT_INSTRUCTION = ("Compare train failures with success references. Preserve effective behavior; add narrow checks for supported deviations. "
                     "EM=1 is not verified reasoning. Similar-task comparisons and node attribution are hypotheses. "
                     "Unpaired: use original failure feedback. Avoid broad bans or copying answers/entities. "
                     "Return new_skill and evidence-linked notes.\n")


def pack_feedback(packets, counter, maximum):
    """Shrink fields, never slice serialized JSON or silently lose a selected pair."""
    for instruction in [INSTRUCTION, SHORT_INSTRUCTION]:
        for limit in [160, 100, 64, 40, 24, 16, 8]:
            packed = []
            for p in packets:
                f, s = p["failure"], p["success_reference"]
                compact = {"failure_id": p["failure_id"], "pair_type": p["pair_type"],
                           "failure": {"question": counter.clip(f["question"], limit),
                                       "response_excerpt": counter.clip(f["response_excerpt"], limit),
                                       "predicted_answer": counter.clip(f["predicted_answer"], limit),
                                       "gold_answers": [counter.clip(a, limit) for a in f["gold_answers"]], "em": 0},
                           "success_reference": None if s is None else {
                               "example_id": s["example_id"], "response_excerpt": counter.clip(s["response_excerpt"], limit),
                               "predicted_answer": counter.clip(s["predicted_answer"], limit), "em": 1},
                           "observable_checks": {"answer_format_ok": p["observable_checks"]["answer_format_ok"],
                                                 "reference_has_verifiable_quote": p["observable_checks"]["reference_has_verifiable_quote"],
                                                 "diagnosis_status": "hypothesis_to_check"}}
                if limit >= 40:
                    compact["failure"]["visible_context_excerpt"] = counter.clip(f["visible_context_excerpt"], limit)
                    if s:
                        compact["success_reference"]["question"] = counter.clip(s["question"], limit)
                        compact["success_reference"]["visible_context_excerpt"] = counter.clip(s["visible_context_excerpt"], limit)
                if p.get("original_failure_feedback"):
                    compact["original_failure_feedback"] = counter.clip(p["original_failure_feedback"], limit)
                packed.append(compact)
            text = instruction + "\n" + json.dumps(packed, ensure_ascii=False, separators=(",", ":"))
            if counter.count(text) <= maximum:
                return text, packed
    raise ValueError("Feedback cap cannot retain all selected failure IDs and references; adjust BOTH methods in the pilot")


def evidence_check(reference):
    if reference is None:
        return False
    text = re.sub(r"<answer>.*?</answer>", "", reference["raw_response"], flags=re.S | re.I)
    # Only explicit quoted spans are mechanically verifiable. No claim of semantic support.
    quotes = re.findall(r'"([^"\n]{12,})"|“([^”\n]{12,})”', text)
    return any((left or right).strip() in reference["visible_context"] for left, right in quotes)


def excerpt(row, counter, maximum):
    if row is None:
        return None
    return {"example_id": row["id"], "skill_hash": row["skill_hash"], "round": row["round"],
            "visible_context_hash": row["visible_context_hash"], "observation_id": row["observation_id"],
            "question": counter.clip(row["question"], maximum),
            "visible_context_excerpt": counter.clip(row["visible_context"], maximum),
            "response_excerpt": counter.clip(row["raw_response"], maximum),
            "predicted_answer": counter.clip(row["predicted_answer"], maximum), "gold_answers": row["gold_answers"], "em": row["em"]}


class ContrastiveFeedback(BaselineFeedback):
    name = "success_failure_contrastive"

    def __init__(self, dataset, train_ids, output, config):
        self.memory = TrainMemory(dataset, train_ids, per_outcome_limit=config["pairing"]["per_outcome_limit"])
        self.output = Path(output)
        self.counter = 0

    def observe(self, rows):
        for row in rows:
            self.memory.add(row)
        write_json(self.output / "memory.json", self.memory.records())

    def build(self, rows, items, out_dir, seed, config, counter):
        failures = selected_failures(rows, config["failure_k"], seed)
        packets = []
        counts = {"same_question": 0, "similar_task": 0, "unpaired": 0}
        for failure in failures:
            pair = pair_failure(failure, self.memory, config["pairing"]["jaccard_threshold"])
            reference = pair["success_reference"]
            counts[pair["pair_type"]] += 1
            verified_quote = evidence_check(reference)
            packet = {"failure_id": failure["id"], "pair_type": pair["pair_type"], "similarity": pair["similarity"],
                      "task_type": task_type(failure["question"]), "failure": excerpt(failure, counter, 160),
                      "success_reference": excerpt(reference, counter, 160), "observable_checks": {
                          "answer_format_ok": failure["format_ok"], "reference_has_verifiable_quote": verified_quote,
                          "reference_evidence_status": "quoted_span_present_semantic_support_unverified" if verified_quote else "answer_only_or_unverified",
                          "diagnosis_status": "hypothesis_to_check", "node_execution_observed": False}}
            packets.append(packet)
            if pair["pair_type"] == "unpaired":
                original, _ = super().build([failure], items, out_dir, seed, config, counter)
                packet["original_failure_feedback"] = original
        if not packets:
            text, actual_packets = "", []
        elif counts["unpaired"] == len(packets):
            # With no success references, exactly retain the baseline feedback behavior.
            text, _ = super().build(rows, items, out_dir, seed, config, counter)
            actual_packets = []
        else:
            text, actual_packets = pack_feedback(packets, counter, config["feedback_max_tokens"])
        report = {"mode": self.name, "selected_failure_ids": [r["id"] for r in failures], **counts,
                  "feedback_tokens": counter.count(text), "packets": packets, "actual_packets": actual_packets,
                  "note": "Packets are complete logged diagnostics; actual bounded optimizer text is saved separately."}
        self.counter += 1
        write_json(self.output / "feedback" / f"{self.counter:04d}.json", {**report, "actual_feedback": text})
        return text, report
