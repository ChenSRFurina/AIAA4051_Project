"""Extend C2b's success-reference retrieval for declarative QA clues."""
from pathlib import Path
import re

from baseline.feedback import selected_failures
from baseline.io import write_json
from explore.memory import TrainMemory
from explore.pairing import jaccard, pair_failure, task_type
from explore_v2.diagnostics import diagnose
from explore_v2.evidence import evidence
from explore_v2.feedback import DiagnosticFeedback, pack_feedback


# Only inspect the requested target or an explicit personal pronoun. A word
# elsewhere in the clue can describe a distractor rather than the target.
TARGET = re.compile(r"\b(?:this|these|which|what)\s+((?:[a-z-]+\s+){0,2}[a-z-]+)", re.I)
CATEGORIES = {
    "person": set("man woman author writer novelist poet musician composer singer actor actress artist leader president king queen monarch scientist physicist inventor explorer son daughter guitarist painter director politician player athlete".split()),
    "place": set("country countries nation state city capital town island river mountain continent province lake sea ocean region".split()),
    "work": set("book novel poem play song album film movie painting story opera symphony series".split()),
    "group": set("company team group band organization agency party university school".split()),
    "word": set("word term phrase verb adverb adjective noun prefix suffix expression saying".split()),
    "time": set("year month day decade century date".split()),
    "object": set("instrument device machine tool coin fruit substance material metal mineral animal bird plant tree flower food".split()),
}


def target_kind(question):
    q = question.lower().strip()
    if re.match(r"^(?:is|are|was|were|did|does|do|has|have|can)\b", q):
        return "yes_no"
    if re.match(r"^(?:how many|how much)\b", q):
        return "quantity"
    if q.startswith("when "):
        return "time"
    if q.startswith("where "):
        return "place"
    if q.startswith("who "):
        return "person"
    for match in TARGET.finditer(q):
        for word in match[1].split():
            if word in {"is", "was", "were", "are", "of", "in", "as", "for", "had", "has"}:
                break
            for category, nouns in CATEGORIES.items():
                if word in nouns:
                    return category
    if re.match(r"^(?:he|she|his|her)\b", q) or re.search(r"\b(?:him|herself|himself)\b", q):
        return "person"
    if re.match(r"^(?:term|word|phrase|the (?:adverb|adjective|verb|noun))\b", q):
        return "word"
    return None


def retrieve(failure, memory, threshold, used):
    original = pair_failure(failure, memory, threshold)
    if original["success_reference"] is not None:
        return original, "original"
    category = target_kind(failure["question"])
    if not category:
        return original, "unpaired"
    candidates = [
        row for row in memory.successes()
        if row["dataset"] == failure["dataset"] and row["split"] == "train"
        and row["round"] <= failure["round"] and row["id"] != failure["id"]
        and row["format_ok"] and target_kind(row["question"]) == category
    ]
    if not candidates:
        return original, "unpaired"
    def order(row):
        return (row["id"] in used, -jaccard(failure["question"], row["question"]),
                -row["round"], row["id"], row["skill_hash"], row["observation_id"])
    best = min(candidates, key=order)
    return {"pair_type": "similar_task", "success_reference": best,
            "similarity": jaccard(failure["question"], best["question"])}, "target_kind:" + category


class ClueFeedback(DiagnosticFeedback):
    name = "C2c_clue_reference"

    def __init__(self, dataset, train_ids, output, config):
        self.memory = TrainMemory(dataset, train_ids, condition="C2c",
                                  per_outcome_limit=config["pairing"]["per_outcome_limit"])
        self.output, self.counter = Path(output), 0

    def build(self, rows, items, out_dir, seed, config, counter):
        failures = selected_failures(rows, config["failure_k"], seed)
        packets, used = [], set()
        counts = dict(same_question=0, similar_task=0, unpaired=0)
        for failure in failures:
            if failure["split"] != "train" or failure["condition"] != "C2c":
                raise ValueError("C2c requires its own train observations")
            pair, rule = retrieve(failure, self.memory, config["pairing"]["jaccard_threshold"], used)
            reference = pair["success_reference"]
            if reference:
                used.add(reference["id"])
            counts[pair["pair_type"]] += 1
            packets.append({**pair, "retrieval_rule": rule, "failure_id": failure["id"],
                            "task_type": task_type(failure["question"]), "failure": failure,
                            "diagnosis": diagnose(failure), "failure_evidence": evidence(failure, counter),
                            "success_evidence": evidence(reference, counter) if reference else None})
        text, actual = pack_feedback(packets, counter, config["feedback_max_tokens"])
        report = {"mode": self.name, "selected_failure_ids": [r["id"] for r in failures],
                  **counts, "feedback_tokens": counter.count(text), "packets": packets, "actual_packets": actual}
        self.counter += 1
        write_json(self.output / "feedback" / f"{self.counter:04d}.json", {**report, "actual_feedback": text})
        return text, report
