from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .data import prepare
from .io import read_json, read_jsonl, write_json
from .trainer import CourseTrainer, ensure_empty
from .protocol import freeze, test_frozen, freeze_design, check_design
from .analysis import compare
from .gpu import gpu_event


def enable_explore():
    explore = Path(__file__).resolve().parents[2] / "Explore"
    if not (explore / "explore").exists():
        raise ValueError("Keep Baseline and Explore beside each other")
    sys.path.insert(0, str(explore))


def pilot(config, data_dir, output, demo=False):
    runner = CourseTrainer(config, data_dir, output, condition="pilot", demo=demo)
    batch = runner.data["train"][:10]
    from graphskillevo.types import SkillIndividual
    summaries, evaluated = {}, []
    from .evaluation import aggregate
    for name, skill in [("N", ""), ("I", runner.initial_content)]:
        individual = SkillIndividual("pilot_" + name, skill, "pilot")
        rows = runner.evaluate_items(individual, batch, "train", 0, runner.output / name)
        summaries[name] = aggregate(rows)
        evaluated.append((individual, rows, runner.output / name))
    for attempt in range(config["max_attempts"]):
        try:
            skill, _ = runner.operators.initial_attempt(runner.initial_content)
            break
        except (ValueError, RuntimeError):
            if attempt + 1 == config["max_attempts"]:
                raise
    initial = SkillIndividual("pilot_generated", skill, "pilot_generated")
    rows = runner.evaluate_items(initial, batch, "train", 0, runner.output / "generated")
    summaries["generated_initial"] = aggregate(rows)
    parent, parent_rows, parent_out = evaluated[1]
    parent.train_reflection_context, _ = runner.feedback.build(parent_rows, batch, parent_out, config["seed"], config, runner.counter)
    for attempt in range(config["max_attempts"]):
        try:
            child = runner.operators.mutation_graph(parent=parent.content, context=parent.train_reflection_context)
            break
        except (ValueError, RuntimeError):
            if attempt + 1 == config["max_attempts"]:
                raise
    candidate = SkillIndividual("pilot_candidate", child.new_skill, "pilot_mutation")
    rows = runner.evaluate_items(candidate, batch, "train", 0, runner.output / "candidate")
    summaries["candidate"] = aggregate(rows)
    result = {"pilot_train_ids": [r["id"] for r in batch], "conditions": summaries,
              "cost": runner.budget.totals(), "demo": demo, "gpu_hours": None,
              "note": "Train-only pilot. Inspect truncation, usage, latency and backend compatibility before freezing real config."}
    write_json(runner.output / "pilot_summary.json", result)
    return result


def demo_run(output):
    enable_explore()
    from explore.feedback import ContrastiveFeedback
    ensure_empty(output)
    output = Path(output).resolve()
    results = {}
    for dataset in ["searchqa", "hotpotqa"]:
        root = output / dataset
        paths = {}
        for split, count in [("train", 12), ("dev", 6), ("test", 6)]:
            rows = [{"id": f"{dataset}_{split}_{i:03d}", "question": f"What is the documented value for synthetic item {split} {i}?",
                     "context": f"[DOC] Synthetic fixture {split} {i}\nDocumented value: Value{i}", "answers": [f"Value{i}"]} for i in range(count)]
            paths[split] = root / "raw" / f"{split}.json"
            write_json(paths[split], rows)
        prepare(paths, root / "data", dataset, demo=True)
        config_path = root / "demo_config.json"
        write_json(config_path, {"dataset": dataset, "rounds": 1, "train_batch_size": 8,
                                "model": {"id": "SYNTHETIC-MOCK-NO-BENCHMARK", "revision": "demo"}})
        config = load_config(config_path, demo=True)
        CourseTrainer(config, root / "data", root / "shared_init", condition="shared_initialization", demo=True).initialize()
        for condition in ["N", "I"]:
            CourseTrainer(config, root / "data", root / condition, condition=condition, demo=True).static()
        CourseTrainer(config, root / "data", root / "B", condition="B", demo=True).train_course(root / "shared_init")
        freeze_design(root / "B", root / "feedback_design.json", config)
        feedback = ContrastiveFeedback(dataset, [r["id"] for r in read_json(root / "data/train.json")], root / "C", config)
        CourseTrainer(config, root / "data", root / "C", condition="C", demo=True, feedback=feedback).train_course(root / "shared_init")
        freeze({k: root / k for k in ["N", "I", "B", "C"]}, root / "data", root / "frozen.json", demo=True)
        results[dataset] = test_frozen(root / "frozen.json", root / "test")
    write_json(output / "SYNTHETIC_ONLY.json", {"warning": "Pipeline smoke test only. These are not SearchQA/HotpotQA benchmark results.", "results": results})
    return {"status": "synthetic_end_to_end_passed", "datasets": list(results), "output": str(output)}


def parser(explore=False):
    root = argparse.ArgumentParser(description="Project5 bounded GraphSkillEvo experiments")
    commands = root.add_subparsers(dest="action", required=True)
    for action in (["train", "freeze-design"] if explore else ["initialize", "train", "static", "pilot"]):
        item = commands.add_parser(action)
        item.add_argument("--config", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--demo", action="store_true", help="Synthetic data/backend only")
        if action != "freeze-design":
            item.add_argument("--data", required=True)
        if action == "train":
            item.add_argument("--initialization", required=True)
            if explore:
                item.add_argument("--design", required=True)
        if action == "static":
            item.add_argument("--condition", choices=["N", "I"], required=True)
        if action == "freeze-design":
            item.add_argument("--baseline-run", required=True)
    if not explore:
        item = commands.add_parser("prepare")
        item.add_argument("--dataset", choices=["searchqa", "hotpotqa"], required=True)
        item.add_argument("--train", required=True)
        item.add_argument("--dev", required=True)
        item.add_argument("--test")
        item.add_argument("--output", required=True)
        item = commands.add_parser("freeze")
        for condition in ["N", "I", "B", "C"]:
            item.add_argument("--" + condition, required=True)
        item.add_argument("--data", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--demo", action="store_true")
        item = commands.add_parser("test")
        item.add_argument("--frozen", required=True)
        item.add_argument("--output", required=True)
        item = commands.add_parser("compare")
        item.add_argument("--baseline", required=True)
        item.add_argument("--contrastive", required=True)
        item.add_argument("--output", required=True)
        item = commands.add_parser("gpu")
        item.add_argument("operation", choices=["start", "stop", "status"])
        item.add_argument("--ledger", required=True)
        item.add_argument("--session", default="a6000")
        item = commands.add_parser("demo")
        item.add_argument("--output", required=True)
    return root


def main(explore=False):
    args = parser(explore).parse_args()
    if args.action == "demo":
        result = demo_run(args.output)
    elif args.action == "prepare":
        result = prepare({key: getattr(args, key) for key in ["train", "dev", "test"]}, args.output, args.dataset)
    elif args.action == "freeze":
        result = freeze({key: getattr(args, key) for key in ["N", "I", "B", "C"]}, args.data, args.output, demo=args.demo)
    elif args.action == "test":
        result = test_frozen(args.frozen, args.output)
    elif args.action == "compare":
        result = compare(read_json(args.baseline), read_json(args.contrastive))
        write_json(args.output, result)
    elif args.action == "gpu":
        result = gpu_event(args.ledger, args.operation, args.session)
    else:
        config = load_config(args.config, demo=args.demo)
        if args.action == "freeze-design":
            enable_explore()
            result = freeze_design(args.baseline_run, args.output, config)
        elif args.action == "pilot":
            result = pilot(config, args.data, args.output, demo=args.demo)
        else:
            condition = "C" if explore else (getattr(args, "condition", None) or ("B" if args.action == "train" else "shared_initialization"))
            feedback = None
            if explore:
                enable_explore()
                design = check_design(args.design, config)
                from explore.feedback import ContrastiveFeedback
                train_ids = [r["id"] for r in read_json(Path(args.data) / "train.json")]
                feedback = ContrastiveFeedback(config["dataset"], train_ids, args.output, config)
            trainer = CourseTrainer(config, args.data, args.output, condition=condition, demo=args.demo, feedback=feedback)
            if explore:
                write_json(Path(args.output) / "feedback_design.json", design)
            if args.action == "initialize":
                result = [as_record.to_record() for as_record in trainer.initialize()]
            elif args.action == "train":
                result = trainer.train_course(args.initialization)
            else:
                result = trainer.static()
    print(json.dumps(result, ensure_ascii=False, indent=2))
