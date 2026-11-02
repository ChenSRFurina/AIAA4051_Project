"""Small explicit API checks; never run benchmark data or expose credentials."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Baseline"))
from baseline.budget import Budget, TokenCounter
from baseline.client import ChatClient, ModelError
from baseline.config import load_config
from baseline.credentials import api_key
from baseline.io import write_json
from baseline.trainer import ensure_empty, tree_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "Baseline/configs/searchqa_api.json"))
    parser.add_argument("--models", action="store_true", help="List visible model IDs, no generation")
    parser.add_argument("--operators", action="store_true", help="Also generate and validate one skill and one mutation (2 extra paid calls)")
    parser.add_argument("--output", help="Fresh directory for accounting and smoke evidence")
    args = parser.parse_args()
    # Listing models needs credentials but no local tokenizer.
    config = load_config(args.config, demo=args.models)
    if config["backend"] != "api":
        raise ValueError("Use an API config for api/check.py")
    if args.models:
        key = api_key(config)
        request = urllib.request.Request(config["model"]["base_url"].rstrip("/") + "/models",
                                         headers={"Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise ModelError(f"Model listing failed: HTTP {exc.code}") from None
        except Exception as exc:
            raise ModelError(f"Model listing failed: {type(exc).__name__}") from None
        ids = sorted(item["id"] for item in data.get("data", []) if isinstance(item.get("id"), str))
        print(json.dumps({"configured_model": config["model"]["id"], "listed": config["model"]["id"] in ids,
                          "qwen35_model_ids": [item for item in ids if item.startswith("qwen3.5-")]}, indent=2))
        return
    output = Path(args.output) if args.output else ROOT / "api/runs" / datetime.now(timezone.utc).strftime("check-%Y%m%dT%H%M%S%fZ")
    ensure_empty(output)
    counter = TokenCounter(config["model"]["tokenizer_path"])
    limits = {"max_calls": 4 if args.operators else 2,
              "max_generated_tokens": 160 + (2 * config["operator_max_tokens"] if args.operators else 0),
              "max_wall_hours": 0.5}
    budget = Budget(output / "ledger.jsonl", limits)
    client = ChatClient(config, budget, counter, output / "calls.jsonl")
    write_json(output / "config.json", config)
    report = {"status": "started", "model": config["model"]["id"], "backend": "api",
              "synthetic_prompts_only": True, "not_a_benchmark": True, "limits": limits,
              "code_hash": tree_fingerprint(),
              "tokenizer_hash": counter.identity, "tokenizer_note": config["model"]["tokenizer_note"]}
    try:
        text, _ = client.chat("Follow the instruction exactly.", "Reply with the single word OK.",
                              maximum=32, stage="service_smoke", seed=config["seed"], temperature=0)
        if text.strip().upper() != "OK":
            raise ModelError("Plain text smoke response did not match OK")
        text, _ = client.chat("Return one JSON object only.", 'Return exactly {"skills":["ok"]}.',
                              maximum=128, stage="initialize", seed=config["seed"], temperature=0)
        if json.loads(text) != {"skills": ["ok"]}:
            raise ModelError("JSON mode smoke response did not match the expected object")
        if args.operators:
            from baseline import VENDOR
            from baseline.operators import CourseOperators
            operators = CourseOperators(client, config)
            parent = (VENDOR / "graphskillevo/initial_skills/searchqa/init_skill.md").read_text(encoding="utf-8")
            skill, _ = operators.initial_attempt(parent)
            (output / "generated_skill.md").write_text(skill, encoding="utf-8")
            child = operators.mutation_graph(parent=skill, context=(
                "Synthetic smoke example: the context said Paris but the agent answered London. "
                "Add a narrow verification step using only supplied evidence. Preserve useful behaviors."))
            (output / "mutated_skill.md").write_text(child.new_skill, encoding="utf-8")
        report["status"] = "passed"
        report["operators_checked"] = args.operators
    except Exception as exc:
        # The detailed call records already contain sanitized client errors.
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["cost"] = budget.totals()
        write_json(output / "summary.json", report)
    print(json.dumps({**report, "output": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ModelError, ValueError) as exc:
        print(f"API check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
