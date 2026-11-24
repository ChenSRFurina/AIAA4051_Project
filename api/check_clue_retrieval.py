"""Paired retrieval replay on actual mutation-parent TRAIN exposures."""
import argparse
from collections import Counter
from pathlib import Path

from run_clue_comparison import ROOT
from baseline.budget import TokenCounter
from baseline.feedback import selected_failures
from baseline.io import read_json, read_jsonl, write_json
from explore.memory import TrainMemory
from explore.pairing import pair_failure
from explore_c2c import retrieve


def replay(run, arm, counter):
    config = read_json(run / "config.json")
    dataset = config["dataset"]
    ids = read_json(ROOT / "Baseline/data/prepared" / dataset / "train_ids.json")
    memory = TrainMemory(dataset, ids, condition=arm)
    counts = Counter()
    exposures = []
    observed = {}
    verified_prompts = 0
    for h in read_json(run / "history.json"):
        g = h["generation"]
        batch = read_json(run / "train_batches" / f"{g}.json")
        parents = {}
        for parent in h["parent_population"]:
            folder = run / "evaluations" / f"g{g}" / parent["id"] / "train"
            rows = read_jsonl(folder / "outputs.jsonl")
            assert [r["id"] for r in rows] == batch["ids"]
            for r in rows:
                assert r["condition"] == arm and not r["cached"] and not r["error"]
                memory.add(r)
                observed[r["observation_id"]] = r
            parents[parent["id"]] = rows, read_json(folder / "feedback.json")
        for operator in h["operator_records"]:
            rows, feedback = parents[operator["parent_ids"][0]]
            failures = selected_failures(rows, config["failure_k"], batch["seed"])
            assert [r["id"] for r in failures] == feedback["selected_failure_ids"]
            assert counter.count(feedback["actual_feedback"]) == feedback["feedback_tokens"] <= config["feedback_max_tokens"]
            assert len(failures) == len(feedback["actual_packets"]) == len(feedback["packets"])
            for attempt in operator["attempts"]:
                if attempt.get("prompt"):
                    assert feedback["actual_feedback"].strip() in attempt["prompt"]
                    verified_prompts += 1
            used = set()
            for failure, pkt, actual in zip(failures, feedback["packets"], feedback["actual_packets"]):
                old = pair_failure(failure, memory, config["pairing"]["jaccard_threshold"])
                new, rule = retrieve(failure, memory, config["pairing"]["jaccard_threshold"], used)
                old_ref, new_ref = old["success_reference"], new["success_reference"]
                if new_ref:
                    assert observed[new_ref["observation_id"]]["hard"] == 1
                    assert new_ref["round"] <= g and new_ref["id"] in ids
                    used.add(new_ref["id"])
                if old_ref:
                    assert old_ref["observation_id"] == new_ref["observation_id"]
                expected = old if arm == "C2b" else new
                assert pkt["pair_type"] == expected["pair_type"] == actual["pair_type"]
                reference = expected["success_reference"]
                assert (pkt["success_reference"] or {}).get("observation_id") == (reference or {}).get("observation_id")
                assert (actual["success"] or {}).get("id") == (reference or {}).get("id")
                counts["selected"] += 1
                counts["original_pairs"] += bool(old_ref)
                counts["expanded_pairs"] += bool(new_ref)
                exposures.append({"round":g, "child":operator["child_id"], "failure":failure["id"],
                                  "question":failure["question"], "original_reference":old_ref["id"] if old_ref else None,
                                  "expanded_reference":new_ref["id"] if new_ref else None,
                                  "success_question":new_ref["question"] if new_ref else None, "rule":rule})
    assert memory.records() == read_json(run / "memory.json")
    return {"counts":dict(counts), "exposures":exposures, "verified_attempt_prompts":verified_prompts,
            "distinct_failed_questions_paired":{key:len({e["failure"] for e in exposures if e[key]})
                                               for key in ("original_reference","expanded_reference")},
            "distinct_references":{key:len({e[key] for e in exposures if e[key]})
                                   for key in ("original_reference","expanded_reference")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reports = {}
    for dataset in ("searchqa","hotpotqa"):
        path = ROOT / "Explore/runs/api/c2b-20261110" / dataset / "C2b"
        counter = TokenCounter(read_json(path / "config.json")["model"]["tokenizer_path"])
        reports[dataset] = replay(path, "C2b", counter)
        print(dataset, reports[dataset]["counts"], flush=True)
    write_json(args.output, {"scope":"Fixed historical C2b train trajectories, zero new model calls", "datasets":reports})
