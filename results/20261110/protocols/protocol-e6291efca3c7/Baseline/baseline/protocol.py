"""Freeze all four conditions before any held-out test requests."""
from __future__ import annotations

from pathlib import Path
from .io import digest, read_json, read_jsonl, write_json
from .config import config_hash
from .data import load_data
from .budget import Budget, ByteCounter, TokenCounter, BudgetExceeded
from .client import ChatClient
from .evaluation import run_qa, aggregate
from .analysis import compare, inventory
from .trainer import tree_fingerprint, ensure_empty


def artifact_hashes(run):
    paths = [run / "best_skill.md", run / "summary.json", run / "config.json", run / "run_meta.json", run / "final_dev_outputs.json"]
    paths += sorted((run / "operators").glob("*.json"))
    paths += [run / name for name in ["history.json", "ledger.jsonl", "calls.jsonl", "memory.json", "feedback_design.json", "shared_initialization.json"] if (run / name).exists()]
    paths += sorted((run / "feedback").glob("*.json"))
    return {str(p.relative_to(run)).replace("\\", "/"): digest(p.read_bytes()) for p in paths}


def freeze(runs, data_dir, output, *, demo=False):
    if Path(output).exists():
        raise ValueError("Freeze manifest already exists; use a new explicit destination")
    if set(runs) != {"N", "I", "B", "C"}:
        raise ValueError("Freeze requires exactly N, I, B, C")
    result, signature = {}, None
    for condition, path in runs.items():
        path = Path(path).resolve()
        summary = read_json(path / "summary.json")
        meta = read_json(path / "run_meta.json")
        current = {key: meta[key] for key in ["config_hash", "data_hash", "code_hash", "tokenizer_hash", "upstream_commit", "demo"]}
        if summary["condition"] != condition or bool(meta["demo"]) != bool(demo):
            raise ValueError("Condition/demo mismatch")
        if summary["status"] not in {"completed", "static_complete"} and not summary["status"].startswith("budget_stopped:"):
            raise ValueError("Cannot freeze an unfinished run")
        if signature is not None and signature != current:
            raise ValueError("Conditions do not share the exact model/config/data/code protocol")
        signature = current
        result[condition] = {"run": str(path), "artifacts": artifact_hashes(path),
                             "skill_hash": digest((path / "best_skill.md").read_text(encoding="utf-8"))}
    config = read_json(Path(runs["B"]) / "config.json")
    _, manifest = load_data(data_dir, config["dataset"], demo=demo)
    if manifest.get("test_pending") or not manifest["counts"]["test"]:
        raise ValueError("Course HotpotQA held-out test labels/partition are still pending; cannot freeze final evaluation")
    if digest(manifest) != signature["data_hash"] or tree_fingerprint() != signature["code_hash"]:
        raise ValueError("Data/code changed after training")
    b_init = read_json(Path(runs["B"]) / "shared_initialization.json")
    c_init = read_json(Path(runs["C"]) / "shared_initialization.json")
    if b_init["population_hash"] != c_init["population_hash"]:
        raise ValueError("B/C must start from the same initialization")
    record = {"signature": signature, "conditions": result, "dataset": config["dataset"],
              "test_ids_hash": digest(read_json(Path(data_dir) / "test_ids.json")), "config": config,
              "data_dir": str(Path(data_dir).resolve()), "selection": "validation_only", "test_memory_policy": "read_only_no_update"}
    record["freeze_id"] = digest(record)
    write_json(output, record)
    return record


def test_frozen(frozen_path, output):
    frozen = read_json(frozen_path)
    if frozen["freeze_id"] != digest({k: v for k, v in frozen.items() if k != "freeze_id"}):
        raise ValueError("Freeze manifest modified")
    if tree_fingerprint() != frozen["signature"]["code_hash"]:
        raise ValueError("Code/prompts changed after freeze")
    config = frozen["config"]
    splits, manifest = load_data(frozen["data_dir"], frozen["dataset"], demo=frozen["signature"]["demo"])
    if digest(manifest) != frozen["signature"]["data_hash"]:
        raise ValueError("Frozen data changed")
    for item in frozen["conditions"].values():
        if artifact_hashes(Path(item["run"])) != item["artifacts"]:
            raise ValueError("Frozen skills/run evidence changed")
    ensure_empty(output)
    output = Path(output)
    counter = ByteCounter() if frozen["signature"]["demo"] else TokenCounter(config["model"]["tokenizer_path"])
    if counter.identity != frozen["signature"]["tokenizer_hash"]:
        raise ValueError("Tokenizer changed after freeze")
    results, all_rows = {}, {}
    for condition, item in frozen["conditions"].items():
        run_dir = output / condition
        budget = Budget(run_dir / "ledger.jsonl", config["budget"])
        client = ChatClient(config, budget, counter, run_dir / "calls.jsonl", demo=frozen["signature"]["demo"])
        skill = (Path(item["run"]) / "best_skill.md").read_text(encoding="utf-8")
        rows = []
        try:
            for example in splits["test"]:
                rows.append(run_qa(client, example, skill, split="test", round_no=-1, seed=config["seed"],
                                   condition=condition, out_dir=run_dir))
        except BudgetExceeded as exc:
            write_json(run_dir / "incomplete.json", {"reason": str(exc), "evaluated": len(rows), "expected": len(splits["test"]),
                                                      "not_a_full_test_result": True})
            raise
        write_json(run_dir / "outputs.json", rows)
        inventory(rows, run_dir / "analysis")
        results[condition] = {**aggregate(rows), "cost": budget.totals(), "gpu_hours": None}
        all_rows[condition] = rows
    write_json(output / "comparison.json", compare(all_rows["B"], all_rows["C"], config["seed"]))
    report = {"freeze_id": frozen["freeze_id"], "dataset": frozen["dataset"], "demo": frozen["signature"]["demo"],
              "conditions": results, "gpu_hours": None}
    write_json(output / "results.json", report)
    return report


def freeze_design(baseline_run, output, config):
    baseline_run = Path(baseline_run)
    summary = read_json(baseline_run / "summary.json")
    if summary["condition"] != "B" or summary["config_hash"] != config_hash(config):
        raise ValueError("Feedback design must reference the matching baseline B development run")
    if summary["status"] != "completed" and not summary["status"].startswith("budget_stopped:"):
        raise ValueError("Complete the baseline development run before freezing feedback design")
    from explore.feedback import INSTRUCTION
    record = {"baseline_run": str(baseline_run.resolve()), "baseline_summary_hash": digest(summary),
              "complete_dev_failures_hash": digest(read_json(baseline_run / "dev_analysis/failures.json")),
              "config_hash": config_hash(config), "pairing": config["pairing"], "instruction_hash": digest(INSTRUCTION),
              "failure_sampler": "pinned_upstream_random_k3", "code_hash": tree_fingerprint(),
              "note": "Records the implemented method specification. Does not claim that semantic failure labels were manually reviewed."}
    if Path(output).exists():
        raise ValueError("Feedback design manifest already exists")
    write_json(output, record)
    return record


def check_design(path, config):
    from explore.feedback import INSTRUCTION
    record = read_json(path)
    if record["config_hash"] != config_hash(config) or record["code_hash"] != tree_fingerprint() or record["instruction_hash"] != digest(INSTRUCTION):
        raise ValueError("Feedback design changed after its development freeze")
    baseline = Path(record["baseline_run"])
    if digest(read_json(baseline / "summary.json")) != record["baseline_summary_hash"] or digest(read_json(baseline / "dev_analysis/failures.json")) != record["complete_dev_failures_hash"]:
        raise ValueError("Baseline development evidence changed")
    return record
