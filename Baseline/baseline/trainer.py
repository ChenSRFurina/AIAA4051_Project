"""Course wrapper around pinned trainer interfaces and unchanged selection functions.

The outer loop is bounded: upstream's indefinite initialization/parent-resampling
loops are replaced for BOTH B and C. Mutation prompts and section adoption remain
upstream; the only B/C method hook is train_reflection_context construction.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import random
from pathlib import Path

from graphskillevo.config import EvolutionConfig
from graphskillevo.trainer import EvolutionarySkillTrainer
from graphskillevo.types import SkillIndividual
from graphskillevo.selection import child_operator_sequence, sorted_by_validation, normalized_rank_weights, select_survivors
from . import VENDOR, UPSTREAM_COMMIT
from .analysis import inventory
from .budget import Budget, BudgetExceeded, ByteCounter, TokenCounter
from .client import ChatClient, ModelError
from .config import config_hash
from .data import load_data
from .evaluation import run_qa, aggregate
from .feedback import BaselineFeedback
from .graph_export import export_skill, export_lineage
from .io import digest, read_json, read_jsonl, write_json
from .operators import CourseOperators


def tree_fingerprint():
    base = Path(__file__).resolve().parents[1]
    explore = base.parent / "Explore"
    selected = []
    for folder in [base / "baseline", base / "tests", base / "vendor/GraphSkillEvo/graphskillevo", explore / "explore"]:
        if folder.exists():
            selected.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".md"})
    selected.extend(p for p in [base / "run.py", base / "download_data.py", explore / "run.py"] if p.exists())
    return digest({str(p.relative_to(base.parent)).replace("\\", "/"): digest(p.read_bytes()) for p in sorted(selected)})


def ensure_empty(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh run directory; silent overwrite/resume would invalidate accounting")
    output.mkdir(parents=True, exist_ok=True)


class CourseTrainer(EvolutionarySkillTrainer):
    def __init__(self, config, data_dir, output, *, condition, demo=False, feedback=None):
        self.course = config
        self.data, self.manifest = load_data(data_dir, config["dataset"], demo=demo)
        ensure_empty(output)
        self.output, self.condition, self.demo = Path(output).resolve(), condition, demo
        self.budget = Budget(self.output / "ledger.jsonl", config["budget"])
        self.counter = ByteCounter() if demo else TokenCounter(config["model"]["tokenizer_path"])
        self.client = ChatClient(config, self.budget, self.counter, self.output / "calls.jsonl", demo=demo)
        self.feedback = feedback or BaselineFeedback()
        operators = CourseOperators(self.client, config)
        evolution = EvolutionConfig(population_size=2, generations=config["rounds"], train_eval_num=config["train_batch_size"],
                                    reflection_failure_k=config["failure_k"], graph_retry_attempts=config["max_attempts"])
        super().__init__({"out_root": str(self.output), "seed": config["seed"], "env": config["dataset"]}, evolution, None, operators=operators)
        self.val_rows = {}
        self.initial_content = (VENDOR / "graphskillevo/initial_skills/searchqa/init_skill.md").read_text(encoding="utf-8")
        self.operators.validate(self.initial_content)
        self.signature = {"config_hash": config_hash(config), "data_hash": digest(self.manifest),
                          "code_hash": tree_fingerprint(), "tokenizer_hash": self.counter.identity,
                          "upstream_commit": UPSTREAM_COMMIT, "demo": demo}
        write_json(self.output / "config.json", config)
        write_json(self.output / "run_meta.json", {**self.signature, "condition": condition, "dataset": config["dataset"],
                                                   "concurrency": 1, "gpu_hours": None,
                                                   "gpu_accounting": "Inference wall time is not GPU occupancy. Use the separate service/session GPU ledger."})

    def evaluate_items(self, individual, items, split, generation, output):
        rows = []
        for item in items:
            rows.append(run_qa(self.client, item, individual.content, split=split, round_no=generation,
                               seed=self.course["seed"], condition=self.condition, out_dir=output))
        metrics = aggregate(rows)
        write_json(Path(output) / "metrics.json", metrics)
        return rows

    def evaluate_validation(self, individual, generation):
        if individual.hash in self.val_rows:
            rows = deepcopy(self.val_rows[individual.hash])
            for row in rows:
                row["cached"] = True
            write_json(self.output / "cache_events" / f"g{generation}_{individual.id}.json", {
                "kind": "deterministic_validation_reuse", "skill_hash": individual.hash,
                "source_observation_ids": [r["observation_id"] for r in rows], "new_observations": 0})
        else:
            rows = self.evaluate_items(individual, self.data["dev"], "dev", generation,
                                       self.output / "evaluations" / f"g{generation}" / individual.id / "dev")
            self.val_rows[individual.hash] = rows
        metrics = aggregate(rows)
        individual.val_hard, individual.val_soft = metrics["em"], metrics["f1"]
        individual.rank_score = individual.survivor_score = individual.val_hard
        return rows

    def initialize(self):
        base = SkillIndividual("g0000_i0000", self.initial_content, "initial_skill")
        population = [base]
        errors = []
        for attempt in range(self.course["max_attempts"]):
            try:
                skill, record = self.operators.initial_attempt(self.initial_content)
                population.append(SkillIndividual("g0000_i0001", skill, "llm_initialization"))
                write_json(self.output / "initialization.json", record)
                break
            except (ValueError, ModelError) as exc:
                errors.append({"attempt": attempt + 1, "error": str(exc)})
                write_json(self.output / "initialization_errors.json", errors)
        if len(population) != 2:
            raise RuntimeError("Initialization failed within the common total-attempt cap; inspect calls.jsonl")
        for item in population:
            self.evaluate_validation(item, 0)
            export_skill(item.content, self.output / "skills" / item.id)
        write_json(self.output / "population.json", [asdict(item) for item in population])
        write_json(self.output / "validation_cache.json", self.val_rows)
        write_json(self.output / "summary.json", {"status": "shared_initialization_complete", "cost": self.budget.totals(), **self.signature})
        return population

    def load_initialization(self, folder):
        folder = Path(folder)
        meta = read_json(folder / "run_meta.json")
        if {k: meta[k] for k in self.signature} != self.signature:
            raise ValueError("Shared initialization differs in data/model/config/tokenizer/code/demo signature")
        if read_json(folder / "summary.json")["status"] != "shared_initialization_complete":
            raise ValueError("Incomplete shared initialization")
        population = [SkillIndividual(**row) for row in read_json(folder / "population.json")]
        for item in population:
            self.operators.validate(item.content)
        self.val_rows = read_json(folder / "validation_cache.json")
        self.budget.import_shared(read_jsonl(folder / "ledger.jsonl"))
        write_json(self.output / "shared_initialization.json", {"path": str(folder.resolve()),
                   "population_hash": digest([asdict(p) for p in population]), "logical_cost_charged": True})
        return population

    def generation_train(self, population, generation):
        seed = self.course["seed"] + generation * 10000
        # Same batch and order for BOTH parents. Independent of outcomes and method.
        batch = random.Random(seed).sample(self.data["train"], min(self.course["train_batch_size"], len(self.data["train"])))
        write_json(self.output / "train_batches" / f"{generation}.json", {"seed": seed, "ids": [i["id"] for i in batch]})
        results = []
        for individual in population:
            out = self.output / "evaluations" / f"g{generation}" / individual.id / "train"
            rows = self.evaluate_items(individual, batch, "train", generation, out)
            metrics = aggregate(rows)
            individual.train_hard, individual.train_soft = metrics["em"], metrics["f1"]
            individual.train_eval_dir = str(out)
            results.append((individual, rows, out))
        # Observe all parents before pairing; no asymmetry from evaluation order.
        self.feedback.observe([row for _, rows, _ in results for row in rows])
        for individual, rows, out in results:
            text, report = self.feedback.build(rows, batch, out, seed, self.course, self.counter)
            individual.train_reflection_context = text
            individual.train_examples = [{"id": key} for key in report["selected_failure_ids"]]
            write_json(out / "feedback.json", {**report, "actual_feedback": text})

    def offspring(self, population, generation):
        ranked = sorted_by_validation(population)
        weights = normalized_rank_weights(ranked)
        rng = random.Random(self.course["seed"] + generation * 3000)
        children, records = [], []
        for index, operator in enumerate(child_operator_sequence(2)):
            child_id = f"g{generation:04d}_i{index:04d}"
            parent_ids, payload, context = self._sample_operator_inputs(operator, ranked, weights, rng)
            record = {"child_id": child_id, "operator": operator, "parent_ids": parent_ids, "selection_context": context, "attempts": []}
            path = self.output / "operators" / child_id
            for attempt in range(self.course["max_attempts"]):
                self.operators.call_seed = self.course["seed"] + generation * 1000 + index * 10 + attempt
                try:
                    result = self._invoke_operator(self.operators, operator, payload, context)
                    self.operators.validate(result.new_skill)
                    child = SkillIndividual(child_id, result.new_skill, operator, parent_ids=parent_ids, generation=generation)
                    children.append(child)
                    record["attempts"].append({"status": "ok", "prompt": result.prompt, "parsed": result.parsed,
                                               "raw_response": result.raw_response, "usage": result.usage})
                    export_skill(child.content, self.output / "skills" / child.id)
                    break
                except (ValueError, ModelError) as exc:
                    detail = exc.to_record() if hasattr(exc, "to_record") else {"error": str(exc)}
                    record["attempts"].append({"status": "invalid_or_failed", **detail})
                finally:
                    write_json(path.with_suffix(".json"), record)
            record["accepted"] = any(c.id == child_id for c in children)
            write_json(path.with_suffix(".json"), record)
            records.append(record)
        return children, records

    def save_final(self, population, status):
        best = self._best_individual(population)  # Upstream final tie handling: EM only, stable current order.
        export_skill(best.content, self.output / "best_skill")
        write_json(self.output / "population.json", [asdict(p) for p in population])
        write_json(self.output / "history.json", self.history)
        export_lineage(self.history, self.output / "evolution.mmd")
        rows = self.val_rows[best.hash]
        write_json(self.output / "final_dev_outputs.json", rows)
        inventory(rows, self.output / "dev_analysis")
        result = {"status": status, "condition": self.condition, "dataset": self.course["dataset"], "best": asdict(best),
                  "completed_rounds": len(self.history), "cost": self.budget.totals(), **self.signature,
                  "dev": aggregate(rows), "gpu_hours": None}
        write_json(self.output / "summary.json", result)
        return result

    def train_course(self, initialization):
        population = self.load_initialization(initialization)
        for item in population:
            export_skill(item.content, self.output / "skills" / item.id)
        status = "completed"
        validated_children = []
        generation = 0
        try:
            for generation in range(1, self.course["rounds"] + 1):
                validated_children = []
                self.generation_train(population, generation)
                children, records = self.offspring(population, generation)
                for child in children:
                    self.evaluate_validation(child, generation)
                    validated_children.append(child)
                survivors = select_survivors(population, children, population_size=2)
                survivors.sort(key=lambda item: float(item.survivor_score if item.survivor_score is not None else -1), reverse=True)
                record = {"generation": generation, "parent_population": [asdict(p) for p in population],
                          "candidates": [asdict(c) for c in children], "operator_records": records,
                          "survivor_ids": [p.id for p in survivors], "best_id": self._best_individual(survivors).id,
                          "cost_at_round": self.budget.totals()}
                self.history.append(record)
                population = survivors
                self.save_final(population, "running")
        except BudgetExceeded as exc:
            status = "budget_stopped:" + str(exc)
            if validated_children:
                population = select_survivors(population, validated_children, population_size=2)
            write_json(self.output / "partial_generation.json", {"generation": generation, "reason": str(exc),
                       "fully_validated_candidates": [asdict(c) for c in validated_children],
                       "note": "Incomplete candidates are excluded; fully validated candidates remain eligible by upstream selection."})
        return self.save_final(population, status)

    def static(self):
        skill = "" if self.condition == "N" else self.initial_content
        individual = SkillIndividual("static_" + self.condition, skill, self.condition)
        self.evaluate_validation(individual, 0)
        return self.save_final([individual], "static_complete")
