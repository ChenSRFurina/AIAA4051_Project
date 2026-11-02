"""Read experiment progress without loading credentials or making API calls."""
import argparse
import json
from pathlib import Path


def ledger_rows(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live append can leave only the final line incomplete for a moment.
            if index != len(lines) - 1:
                raise
    return rows


def inspect(run):
    stages = {}
    physical = []
    for name in ("pilot", "shared_init", "N", "I", "B"):
        folder = run / name
        rows = ledger_rows(folder / "ledger.jsonl")
        summary_path = folder / ("pilot_summary.json" if name == "pilot" else "summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        status = summary.get("status") or ("completed" if name == "pilot" and summary else
                                           "incomplete_or_running" if folder.exists() else "not_started")
        stages[name] = {"status": status, "calls": sum(row.get("calls", 0) for row in rows),
                        "physical_calls": sum(row.get("physical_calls", 0) for row in rows),
                        "failed_requests": sum(row.get("status") == "error" and row.get("physical_calls", 0) > 0 for row in rows),
                        "generated_tokens": sum(row.get("generated_tokens", 0) for row in rows)}
        if "completed_rounds" in summary:
            stages[name]["completed_rounds"] = summary["completed_rounds"]
        if "dev" in summary:
            stages[name]["dev"] = summary["dev"]
        physical.extend(row for row in rows if row.get("physical_calls", 0) > 0)
    return {"path": str(run.resolve()), "stages": stages, "physical_cost": {
        key: sum(row.get(key, 0) for row in physical)
        for key in ("physical_calls", "prompt_tokens", "generated_tokens", "reasoning_tokens")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "Baseline/runs/api")
    parser.add_argument("--run", type=Path, help="Inspect a specific run directory instead of latest per dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = {}
    if args.run:
        if not args.run.is_dir():
            parser.error("--run must point to an existing run directory")
        reports[args.run.parent.name] = inspect(args.run)
    else:
        for dataset in ("searchqa", "hotpotqa"):
            root = args.root / dataset
            runs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
            if runs:
                reports[dataset] = inspect(runs[-1])
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    if not reports:
        print("No runs found.")
    for dataset, report in reports.items():
        print(f"{dataset}: {report['path']}")
        for name, stage in report["stages"].items():
            print(f"  {name:12} {stage['status']:24} calls={stage['calls']} physical={stage['physical_calls']} failed_requests={stage['failed_requests']}")
            if "dev" in stage:
                dev = stage["dev"]
                print(f"    dev n={dev['n']} EM={dev['em']:.4f} F1={dev['f1']:.4f} unresolved_errors={dev['infrastructure_errors']} provider_refusals={dev.get('provider_refusals', 0)}")
        print("  Physical cost:", json.dumps(report["physical_cost"]))


if __name__ == "__main__":
    main()
