"""Finalize an audited B checkpoint after the known metrics replacement failure.

No inference or selection is rerun. This deliberately rejects incomplete rounds,
unrelated crashes, missing observations, and changed evaluation/training code.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Baseline"))
from baseline.analysis import inventory
from baseline.config import config_hash
from baseline.evaluation import aggregate
from baseline.io import digest, read_json, read_jsonl, write_json
from baseline.trainer import tree_fingerprint
from graphskillevo.selection import select_survivors
from graphskillevo.types import SkillIndividual


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recover(run):
    run = Path(run).resolve()
    folder = run / "B"
    summary = read_json(folder / "summary.json")
    config = read_json(folder / "config.json")
    history = read_json(folder / "history.json")
    population = read_json(folder / "population.json")
    rows = read_json(folder / "final_dev_outputs.json")
    log = (run / "B.log").read_text(encoding="utf-8")
    require(summary["status"] == "running" and summary["condition"] == "B", "Expected interrupted B checkpoint")
    require("PermissionError: [WinError 5]" in log and "metrics.json.tmp" in log and
            'return self.save_final(population, status)' in log, "Not the known final-save failure")
    require(len(history) == summary["completed_rounds"] == config["rounds"], "Rounds incomplete")
    require([h["generation"] for h in history] == list(range(1, config["rounds"] + 1)), "Noncontiguous rounds")
    require(config_hash(config) == summary["config_hash"], "Config mismatch")
    require(not (folder / "partial_generation.json").exists(), "Partial generation exists")
    # Verify the recorded original code snapshot; only I/O and transport fixes
    # may differ now. Preserve the ORIGINAL inference hash in the final summary.
    snapshot = ROOT / "api/runtime" / ("protocol-" + summary["code_hash"][:12])
    manifest = read_json(snapshot / "snapshot.json")
    hashes = {name.replace("\\", "/"): digest((snapshot / name).read_bytes()) for name in manifest["files"]}
    require(digest(hashes) == summary["code_hash"], "Original code snapshot is not intact")
    changed = [name for name, h in hashes.items() if digest((ROOT / name).read_bytes()) != h]
    require(set(changed) <= {"Baseline/baseline/io.py", "Baseline/baseline/client.py"},
            "Recovery requires unchanged training, selection, evaluation, and prompts")
    dev = read_json(ROOT / "Baseline/data/prepared" / config["dataset"] / "dev.json")
    dev_ids = {r["id"] for r in dev}
    cached = read_json(run / "shared_init/validation_cache.json")
    for p in (folder / "evaluations").glob("g*/*/dev/outputs.jsonl"):
        values = read_jsonl(p)
        require(len(values) == len(dev_ids) and {r["id"] for r in values} == dev_ids, f"Incomplete dev observations: {p}")
        cached[values[0]["skill_hash"][:16]] = values
    for generation in history:
        parents = [SkillIndividual(**p) for p in generation["parent_population"]]
        children = [SkillIndividual(**p) for p in generation["candidates"]]
        require(len(children) == config["population_size"], "Missing candidate")
        for individual in parents + children:
            values = cached[individual.hash]
            metrics = aggregate(values)
            require(metrics["em"] == individual.val_hard and metrics["f1"] == individual.val_soft,
                    f"Candidate metrics mismatch: {individual.id}")
        batch = read_json(folder / "train_batches" / f"{generation['generation']}.json")["ids"]
        for parent in parents:
            values = read_jsonl(folder / "evaluations" / f"g{generation['generation']}" / parent.id / "train/outputs.jsonl")
            require([v["id"] for v in values] == batch, f"Incomplete train batch: {parent.id}")
        survivors = select_survivors(parents, children, population_size=config["population_size"])
        survivors.sort(key=lambda p: p.survivor_score, reverse=True)
        require([p.id for p in survivors] == generation["survivor_ids"], "Survivor selection mismatch")
        require(max(survivors, key=lambda p: p.val_hard).id == generation["best_id"], "Best selection mismatch")
    require([p["id"] for p in population] == history[-1]["survivor_ids"], "Final population mismatch")
    best = max(population, key=lambda p: p["val_hard"])
    require(best == summary["best"], "Best record mismatch")
    require((folder / "best_skill.md").read_text(encoding="utf-8") == best["content"], "Best skill mismatch")
    require(rows == cached[SkillIndividual(**best).hash], "Final outputs differ from evaluated best")
    require(len(rows) == len(dev_ids) and {r["id"] for r in rows} == dev_ids, "Incomplete final dev")
    metrics = aggregate(rows)
    require(metrics == summary["dev"] == read_json(folder / "dev_analysis/metrics.json.tmp"), "Final metrics mismatch")
    require(metrics["infrastructure_errors"] == 0, "Unresolved final infrastructure errors")
    ledger = read_jsonl(folder / "ledger.jsonl")
    require({k: sum(r.get(k, 0) for r in ledger) for k in summary["cost"]} == summary["cost"], "Cost mismatch")
    evidence = [folder / name for name in ("summary.json", "history.json", "population.json", "final_dev_outputs.json", "ledger.jsonl", "calls.jsonl")]
    audit = folder / "recovery"
    require(not audit.exists(), "Recovery audit already exists; inspect before retrying")
    audit.mkdir()
    # Snapshot the original bytes before modifying any derived output.
    (audit / "summary.before.json").write_bytes((folder / "summary.json").read_bytes())
    proof = {"status": "validated", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "original_inference_code_hash": summary["code_hash"], "recovery_code_hash": tree_fingerprint(),
             "changed_original_code_files": changed, "new_model_calls": 0,
             "reason": "All configured rounds and observations complete; final metrics atomic replacement raised WinError 5.",
             "original_evidence_sha256": {str(p.relative_to(run)): digest(p.read_bytes()) for p in evidence},
             "checks": ["original code snapshot", "unchanged scoring/selection/prompts", "full train batches",
                        "all candidate dev observations", "survivors and best", "final metrics", "ledger accounting"]}
    write_json(audit / "audit.json", proof)
    inventory(rows, folder / "dev_analysis")
    result = dict(summary, status="completed", finalization_recovery="recovery/audit.json")
    write_json(folder / "summary.json", result)
    write_json(audit / "audit.json", dict(proof, status="completed", final_summary_sha256=digest((folder / "summary.json").read_bytes())))
    return {"status": "completed", "run": str(run), "new_model_calls": 0, "dev": metrics}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.run), ensure_ascii=False, indent=2))
