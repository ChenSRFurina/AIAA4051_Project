"""Check the self-contained submission using the Python standard library."""
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/final_project"
ARMS = ("N", "I", "B", "C", "C2b", "C2c")


def read(name):
    return json.loads((FINAL / (name + ".json")).read_text(encoding="utf-8"))


def validate(condition, detail):
    if not condition:
        raise ValueError(detail)


def verify():
    summary, fixed, rows = read("summary"), read("fixed_ids"), read("predictions")
    evolution, failures = read("skill_graphs"), read("failure_inventory")
    origin = json.loads((ROOT / "results/hotpotqa_course/split_ids.json").read_text(encoding="utf-8"))
    exclude = json.loads((ROOT / "results/hotpotqa_course/excluded_old_dev_ids.json").read_text(encoding="utf-8"))
    validate(all(origin[k] == fixed["hotpotqa"][i] for i,k in
                 (("train","train_ids"),("dev","val_ids"),("test","test_ids"))), "HotpotQA ID release disagrees")
    validate(len(set(exclude)) == 30 and set(exclude).isdisjoint(origin["test_ids"]), "Previously seen ID in test")
    counts, misses, num_graphs = {}, {}, 0
    for dataset, metrics in summary["datasets"].items():
        data_ids = fixed[dataset]
        expected = (300,150,1000) if dataset == "hotpotqa" else (100,30,50)
        flat = [i for s in ("train","dev","test") for i in data_ids[s]]
        validate(tuple(len(data_ids[s]) for s in ("train","dev","test")) == expected and len(set(flat)) == sum(expected), dataset+" split")
        for split in ("dev","test"):
            for arm in ARMS:
                observed, overall = rows[dataset][split][arm], metrics["arms"][arm][split]
                validate([r["id"] for r in observed] == data_ids[split], f"{dataset} {arm} {split} order")
                validate(all(r["skill_hash"] == metrics["arms"][arm]["skill_hash"] for r in observed), "Frozen skill mismatch")
                for metric in ("em","f1"):
                    validate(math.isclose(mean(r[metric] for r in observed), overall[metric], abs_tol=1e-10), metric)
                validate(sum(r["em"] for r in observed) == overall["correct"] and len(observed)==overall["n"],"Count mismatch")
                if arm == "B":
                    misses.update({(dataset,split,r["id"]):r for r in observed if not r["em"]})
        for arm, t in evolution[dataset].items():
            for gid, g in t["graphs"].items():
                num_graphs += 1
                svg = FINAL/"graphs"/f"{dataset}-{arm}-{gid}.svg"
                validate(svg.is_file() and "<svg" in svg.read_text(encoding="utf-8"), str(svg))
            for point in t["points"]:
                g = t["graphs"][point["best_id"]]
                validate(hashlib.sha256(g["text"].encode()).hexdigest()==point["skill_hash"], "Skill serialization")
        for arm, match in metrics["matched"].items():
            cp, src = match["checkpoint"],metrics["arms"][match["test_source"]]
            validate(cp["cost"]["calls"]<=metrics["caps"]["calls"] and cp["cost"]["generated_tokens"]<=metrics["caps"]["tokens"], "Cap exceeded")
            validate(cp["skill_hash"]==src["skill_hash"] and match["metrics"]==src["test"],"Invalid reused observation")
        counts[dataset]=expected
    seen = set()
    for failure in failures:
        key=(failure["dataset"],failure["split"],failure["id"])
        validate(key in misses and key not in seen,"Unexpected failure")
        seen.add(key)
        validate(failure["predicted_answer"]==misses[key]["predicted_answer"] and failure["gold_answers"]==misses[key]["gold_answers"], "Failure/prediction mismatch")
        validate(failure["category"] in summary["taxonomy"] and failure["diagnosis"].strip(),"Missing review")
        validate(hashlib.sha256(failure["raw_response"].encode()).hexdigest()==failure["response_sha256"],"Response digest")
    validate(seen == set(misses), "Incomplete inventory")
    with (FINAL/"failure_inventory.csv").open(encoding="utf-8-sig",newline="") as f:
        validate([(r["dataset"],r["split"],r["id"]) for r in csv.DictReader(f)] ==
                 [(r["dataset"],r["split"],r["id"]) for r in failures], "CSV review order")
    for name in ("index.html","report.html","appendix.html","demo.html"):
        source=(FINAL/name).read_text(encoding="utf-8")
        for link in re.findall(r'href="([^"#?]+)(?:#[^"]*)?"',source):
            if "://" not in link:
                validate((FINAL/link).is_file(), f"Missing link {name}: {link}")
    print(f"Verified two fixed splits, 24 arm/split outputs, {num_graphs} graphs and {len(seen)} failures.")


if __name__=="__main__":
    verify()
