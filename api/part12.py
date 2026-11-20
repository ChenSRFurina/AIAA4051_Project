"""Freeze and evaluate N/I/B for course Parts 1/2, without running an optimizer.

The exact training protocol snapshot is imported, including its QA/scoring code.
Explore's implemented method is frozen with that snapshot BEFORE test exposure.
This is a baseline-only result, not a completed Part 3 B/C comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    """Independent orchestration I/O; never changes the archived protocol."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, suffix=".tmp", delete=False) as f:
        f.write(body)
        temp = Path(f.name)
    for attempt in range(8):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.1 * 2 ** attempt, 1))


def require(value, message):
    if not value:
        raise ValueError(message)


def bootstrap(snapshot):
    snapshot = Path(snapshot).resolve()
    if "baseline" in sys.modules:
        require(Path(sys.modules["baseline"].__file__).resolve().parents[2] == snapshot,
                "Run different protocol snapshots in separate processes")
    sys.path.insert(0, str(snapshot / "Baseline"))
    from baseline import protocol as p
    return p


def freeze(run, data, output, snapshot=None):
    run, data, output = Path(run).resolve(), Path(data).resolve(), Path(output).resolve()
    require(not output.exists(), "Freeze already exists; use a new destination")
    meta = read(run / "B/run_meta.json")
    snapshot = Path(snapshot or ROOT / "api/runtime" / ("protocol-" + meta["code_hash"][:12])).resolve()
    p = bootstrap(snapshot)
    require(p.tree_fingerprint() == meta["code_hash"], "Protocol snapshot differs from training code")
    conditions, signature = {}, None
    config = read(run / "B/config.json")
    splits, manifest = p.load_data(data, config["dataset"], demo=meta["demo"])
    require(not manifest.get("test_pending") and bool(splits["test"]),
            "Missing course held-out test partition/labels; dev cannot substitute for test")
    require(p.digest(manifest) == meta["data_hash"], "Data changed since training; rerun affected conditions")
    for condition in ["N", "I", "B"]:
        folder = run / condition
        summary, current_meta = read(folder / "summary.json"), read(folder / "run_meta.json")
        current = {k: current_meta[k] for k in ["config_hash", "data_hash", "code_hash", "tokenizer_hash", "upstream_commit", "demo"]}
        require(summary["condition"] == current_meta["condition"] == condition, "Wrong condition")
        require(summary["status"] in {"completed", "static_complete"}, "Incomplete baseline run")
        require(all(summary[k] == current[k] for k in current), "Summary/meta signature mismatch")
        require(signature is None or signature == current, "N/I/B protocol mismatch")
        signature = current
        require(read(folder / "config.json") == config and p.config_hash(config) == current["config_hash"], "Config mismatch")
        skill = (folder / "best_skill.md").read_text(encoding="utf-8")
        require(skill == summary["best"]["content"], "Selected skill mismatch")
        rows = read(folder / "final_dev_outputs.json")
        require([r["id"] for r in rows] == [r["id"] for r in splits["dev"]], "Incomplete dev IDs/order")
        require(all(r["split"] == "dev" and not r["error"] for r in rows), "Unresolved dev error")
        require(p.aggregate(rows) == summary["dev"], "Dev metrics changed")
        conditions[condition] = {"run": str(folder), "artifacts": p.artifact_hashes(folder), "skill_hash": p.digest(skill)}
    method_files = {str(f.relative_to(snapshot)).replace("\\", "/"): sha(f)
                    for f in sorted((snapshot / "Explore/explore").glob("*.py"))}
    require(method_files, "Missing C method source to seal before test exposure")
    record = {
        "scope": "part1_part2_baseline_only", "conditions": conditions, "signature": signature,
        "protocol_root": str(snapshot), "dataset": config["dataset"], "config": config,
        "data_dir": str(data), "test_ids": [r["id"] for r in splits["test"]],
        "test_ids_hash": p.digest([r["id"] for r in splits["test"]]),
        "created_at": datetime.now(timezone.utc).isoformat(), "driver_sha256": sha(__file__),
        "selection": "validation_only", "test_memory_policy": "read_only_no_update",
        "future_C": {"status": "implementation_sealed_not_executed", "files": method_files,
                     "policy": "C feedback/pairing/config are sealed before baseline test exposure. Do not use test cases or scores to revise C. Any revised method needs a new untouched evaluation or explicit exploratory labeling; Part 3 remains incomplete."},
    }
    record["freeze_id"] = p.digest(record)
    write(output, record)
    return {"status": "baseline_frozen", "dataset": record["dataset"], "test_n": len(splits["test"]), "freeze_id": record["freeze_id"]}


def verify(frozen):
    p = bootstrap(frozen["protocol_root"])
    require(frozen["scope"] == "part1_part2_baseline_only" and set(frozen["conditions"]) == {"N", "I", "B"}, "Wrong freeze scope")
    require(frozen["freeze_id"] == p.digest({k: v for k, v in frozen.items() if k != "freeze_id"}), "Freeze manifest changed")
    require(sha(__file__) == frozen["driver_sha256"], "Evaluation driver changed after freeze")
    require(p.tree_fingerprint() == frozen["signature"]["code_hash"], "Archived protocol changed")
    require(p.config_hash(frozen["config"]) == frozen["signature"]["config_hash"], "Frozen config changed")
    splits, manifest = p.load_data(frozen["data_dir"], frozen["dataset"], demo=frozen["signature"]["demo"])
    require(p.digest(manifest) == frozen["signature"]["data_hash"] and not manifest.get("test_pending"), "Frozen data changed")
    require([r["id"] for r in splits["test"]] == frozen["test_ids"], "Test IDs/order changed")
    for item in frozen["conditions"].values():
        require(p.artifact_hashes(Path(item["run"])) == item["artifacts"], "Frozen baseline evidence changed")
    return p, splits


def install_source_transport(source_ip, base_url):
    """A process-local socket binding for the original urllib-based client.

    Keeps original endpoint, headers and JSON payload. No model code is patched.
    Disable ambient proxies and reject redirects to another origin.
    """
    from urllib.parse import urlsplit
    target = urlsplit(base_url)
    address = ipaddress.ip_address(target.hostname)
    networks = [ipaddress.ip_network(s) for s in ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]]
    source = ipaddress.ip_address(source_ip)
    require(target.scheme == "http" and any(address in n for n in networks), "Binding only supports the configured private HTTP endpoint")
    require(source.version == 4 and not source.is_unspecified and not source.is_multicast, "Invalid local IPv4")
    allowed = target.netloc

    class Handler(urllib.request.HTTPHandler):
        def http_open(self, req):
            require(req.host == allowed, "Transport destination changed")
            def connection(host, **kwargs):
                return http.client.HTTPConnection(host, source_address=(source_ip, 0), **kwargs)
            return self.do_open(connection, req)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Model endpoint redirect rejected")

    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({}), Handler(), NoRedirect()))
    os.environ["PROJECT5_API_SOURCE_IP"] = source_ip  # New snapshot's identical optional transport.


def evaluate(frozen_path, output, source_ip=None):
    frozen, output = read(frozen_path), Path(output).resolve()
    p, splits = verify(frozen)
    p.ensure_empty(output)
    config = frozen["config"]
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    if source_ip and not frozen["signature"]["demo"]:
        install_source_transport(source_ip, config["model"]["base_url"])
    counter = p.ByteCounter() if frozen["signature"]["demo"] else p.TokenCounter(config["model"]["tokenizer_path"])
    require(counter.identity == frozen["signature"]["tokenizer_hash"], "Tokenizer differs from frozen run")
    write(output / "execution.json", {"freeze_id": frozen["freeze_id"], "protocol_root": frozen["protocol_root"],
          "driver_sha256": sha(__file__), "python": sys.executable, "source_ip": source_ip,
          "started_at": datetime.now(timezone.utc).isoformat(), "scope": frozen["scope"]})
    results, outputs = {}, {}
    try:
        for condition in ["N", "I", "B"]:
            folder = output / condition
            budget = p.Budget(folder / "ledger.jsonl", config["budget"])
            client = p.ChatClient(config, budget, counter, folder / "calls.jsonl", demo=frozen["signature"]["demo"])
            skill = (Path(frozen["conditions"][condition]["run"]) / "best_skill.md").read_text(encoding="utf-8")
            rows = []
            for example in splits["test"]:
                rows.append(p.run_qa(client, example, skill, split="test", round_no=-1,
                                     seed=config["seed"], condition=condition, out_dir=folder))
                # Progress is operational only; no scores are used for adaptation.
                write(output / "progress.json", {"condition": condition, "done": len(rows), "expected": len(splits["test"]), "cost": budget.totals()})
                print(json.dumps({"condition": condition, "done": len(rows), "expected": len(splits["test"])}), flush=True)
            write(folder / "outputs.json", rows)
            p.inventory(rows, folder / "analysis")
            outputs[condition] = rows
            results[condition] = {**p.aggregate(rows), "cost": budget.totals(), "gpu_hours": None}
        verify(frozen)
        comparisons = {}
        for left, right in [("N", "I"), ("N", "B"), ("I", "B")]:
            c = p.compare(outputs[left], outputs[right], config["seed"])
            comparisons[left + "_to_" + right] = {"left_condition": left, "right_condition": right,
                "counts": c["counts"], "repair_rate": c["repair_rate"], "regression_rate": c["regression_rate"],
                "em_difference_right_minus_left": c["em_difference_C_minus_B"], "paired_bootstrap_95": c["paired_bootstrap_95"],
                "uncertainty_scope": c["uncertainty_scope"],
                "transitions": [{"id": t["id"], "transition": t["transition"]} for t in c["transitions"]]}
        write(output / "comparison.json", comparisons)
        report = {"status": "completed", "scope": frozen["scope"], "freeze_id": frozen["freeze_id"],
                  "frozen_manifest": str(Path(frozen_path).resolve()), "dataset": frozen["dataset"],
                  "demo": frozen["signature"]["demo"], "conditions": results,
                  "test_ids": frozen["test_ids"], "finished_at": datetime.now(timezone.utc).isoformat(), "gpu_hours": None}
        write(output / "results.json", report)
        return report
    except Exception as exc:
        write(output / "incomplete.json", {"status": "incomplete", "error": str(exc),
              "completed_conditions": list(results), "not_a_complete_test_result": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    f = commands.add_parser("freeze")
    f.add_argument("--run", required=True)
    f.add_argument("--data", required=True)
    f.add_argument("--snapshot")
    f.add_argument("--output", required=True)
    t = commands.add_parser("test")
    t.add_argument("--frozen", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--source-ip")
    args = parser.parse_args()
    result = (freeze(args.run, args.data, args.output, args.snapshot) if args.action == "freeze"
              else evaluate(args.frozen, args.output, args.source_ip))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
