"""Correctness checks for scientific protocol, not assertions about benchmark gains."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

BASELINE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASELINE), str(BASELINE.parent / "Explore")]
from baseline.budget import Budget, BudgetExceeded, ByteCounter
from baseline.client import ChatClient, ModelError
from baseline.config import load_config
from baseline.data import normalize_item, prepare, load_data, check_splits
from baseline.evaluation import qa_prompts, score, truncate_context
from baseline.feedback import selected_failures
from baseline.io import digest, read_json, read_jsonl, write_json
from baseline.analysis import compare
from baseline.operators import CourseOperators
from baseline.protocol import test_frozen, check_design
from baseline.cli import demo_run
from graphskillevo.evaluation import _sample_failure_rows
from graphskillevo.graph_skill import split_graph_skill
from explore.memory import TrainMemory
from explore.pairing import pair_failure, task_type
from explore.feedback import pack_feedback, INSTRUCTION


def observation(item_id="q1", *, hard=1, round_no=1, condition="C", context="Evidence says Paris.", response=None):
    response = response or ("Evidence: Paris. <answer>Paris</answer>" if hard else "<answer>London</answer>")
    return {"id": item_id, "dataset": "searchqa", "split": "train", "condition": condition, "round": round_no,
            "skill_hash": digest(response), "visible_context_hash": digest(context), "visible_context": context,
            "question": "What is the capital city of France?", "raw_response": response,
            "predicted_answer": "Paris" if hard else "London", "gold_answers": ["Paris"], "em": hard, "hard": hard,
            "observation_id": digest([item_id, response, round_no]), "error": "", "cached": False, "format_ok": True,
            "f1": float(hard), "latency_seconds": .1}


class ProtocolTests(unittest.TestCase):
    def test_json_schema_only_applies_to_skill_generation(self):
        from baseline.client import operator_response_format
        self.assertIsNone(operator_response_format("qa"))
        self.assertIsNone(operator_response_format("service_smoke"))
        init = operator_response_format("initialize")["schema"]
        self.assertEqual(init["properties"]["skills"]["maxItems"], 1)
        mutation = operator_response_format("graphskillevo_graph_mutation_graph")["schema"]
        self.assertEqual(mutation["required"], ["new_skill", "notes"])
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(demo=True)
            config["model"]["structured_operators"] = True
            ledger = Budget(Path(directory) / "ledger.jsonl", config["budget"])
            log = Path(directory) / "calls.jsonl"
            client = ChatClient(config, ledger, ByteCounter(), log, demo=True)
            client.chat("s", "u", maximum=2048, stage="initialize", seed=42, temperature=.2)
            client.chat("s", "u", maximum=64, stage="qa", seed=42, temperature=0)
            calls = read_jsonl(log)
            self.assertIn("response_format", calls[0]["request"])
            self.assertNotIn("response_format", calls[1]["request"])

    def test_upstream_failure_sampler_is_shared_exactly(self):
        rows = [{"id": str(i), "hard": i % 2} for i in range(12)]
        for seed in [0, 42, 10042]:
            self.assertEqual(selected_failures(rows, 3, seed), _sample_failure_rows(rows, failure_k=3, seed=seed))

    def test_answer_extraction_normalization_and_hotpot_yes_no(self):
        result = score("<answer>bad</answer>\n<answer>The Eiffel Tower!</answer>", ["Eiffel Tower"], "searchqa")
        self.assertEqual((result["em"], result["f1"]), (1, 1))
        self.assertEqual(score("reasoning\nParis", ["Paris"], "searchqa")["em"], 1)
        self.assertEqual(score("<answer>yes</answer>", ["yes indeed"], "hotpotqa")["f1"], 0)
        self.assertAlmostEqual(score("<answer>red fox</answer>", ["fox"], "searchqa")["f1"], 2 / 3)

    def test_gold_and_supporting_facts_never_enter_execution_prompt(self):
        item = {"question": "Who is named here?", "context": "[DOC] Ordinary context", "answers": ["SECRET_GOLD"],
                "metadata": {"supporting_facts": ["SECRET_SUPPORT_LABEL"], "type": "SECRET_GOLD_TYPE"}}
        system, user, _ = qa_prompts(item, "")
        self.assertNotIn("SECRET", system + user)
        self.assertNotIn("EVALUATION RESULT", system + user)

    def test_truncation_matches_upstream_boundary_behavior(self):
        import ast
        from baseline import VENDOR
        module = ast.parse((VENDOR / "graphskillevo/envs/searchqa/rollout.py").read_text(encoding="utf-8"))
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_truncate_context")
        scope = {"_MAX_CONTEXT_CHARS": 6000}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream_truncation", "exec"), scope)
        context = "[DOC]" + "x" * 30 + "[DOC]" + "y" * 40
        self.assertEqual(truncate_context(context, 40), scope["_truncate_context"](context, 40))
        self.assertTrue(truncate_context("z" * 50, 40).endswith("...[truncated]"))

    def test_hotpot_sentence_locations_and_labels_are_kept_off_prompt(self):
        item = normalize_item({"_id": "a1", "question": "Where?", "answer": "Paris", "context": [["Title", ["Sentence 0.", "Sentence 1."]]],
                               "supporting_facts": [["Title", 1]]}, "hotpotqa")
        self.assertIn("[1] Sentence 1.", item["context"])
        self.assertEqual(item["metadata"]["sentences"][1]["sentence_index"], 1)
        self.assertEqual(item["metadata"]["supporting_facts"], [["Title", 1]])

    def test_partition_overlap_and_unsafe_ids_rejected(self):
        with self.assertRaises(ValueError):
            check_splits({"train": [{"id": "same", "question": "a"}], "dev": [{"id": "same", "question": "b"}], "test": []}, test_pending=True)
        with self.assertRaises(ValueError):
            normalize_item({"id": "../../x", "question": "q", "answers": ["a"], "context": "x"}, "searchqa")

    def test_data_freeze_rejects_tampering_and_never_fills_from_other_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ["train", "dev", "test"]:
                paths[name] = root / f"{name}.json"
                write_json(paths[name], [{"id": name + "1", "question": name + " q", "context": "context", "answers": ["x"]}])
            prepare(paths, root / "prepared", "searchqa", demo=True)
            data, manifest = load_data(root / "prepared", "searchqa", demo=True)
            self.assertEqual(manifest["counts"], {"train": 1, "dev": 1, "test": 1})
            data["test"][0]["answers"] = ["tampered"]
            write_json(root / "prepared/test.json", data["test"])
            with self.assertRaises(ValueError):
                load_data(root / "prepared", "searchqa", demo=True)

    def test_memory_rejects_dev_test_foreign_method_and_cached_outputs(self):
        memory = TrainMemory("searchqa", ["q1"])
        for split in ["dev", "test"]:
            with self.assertRaises(ValueError):
                memory.add(dict(observation(), split=split))
        with self.assertRaises(ValueError):
            memory.add(observation(condition="B"))
        self.assertFalse(memory.add(dict(observation(), cached=True)))
        self.assertFalse(memory.records())

    def test_memory_deduplicates_responses_and_caps_each_outcome(self):
        memory = TrainMemory("searchqa", ["q1"])
        memory.add(observation())
        memory.add(observation())
        self.assertEqual(len(memory.records()), 1)
        for i in range(2, 5):
            memory.add(observation(round_no=i, response=f"response{i}"))
        self.assertEqual(len(memory.records()), 2)
        self.assertEqual({r["round"] for r in memory.records()}, {3, 4})

    def test_pairing_same_context_latest_then_similarity_then_unpaired(self):
        memory = TrainMemory("searchqa", ["q1", "q2"])
        memory.add(observation(round_no=1))
        memory.add(observation(round_no=2, response="New evidence Paris <answer>Paris</answer>"))
        failure = observation(hard=0, round_no=2)
        pair = pair_failure(failure, memory)
        self.assertEqual(pair["pair_type"], "same_question")
        self.assertEqual(pair["success_reference"]["round"], 2)
        # Same ID with different context is not a same-question or similar-task reference.
        self.assertEqual(pair_failure(dict(failure, visible_context_hash="different"), memory)["pair_type"], "unpaired")
        memory.add(observation("q2"))
        self.assertEqual(pair_failure(dict(failure, id="new", visible_context_hash="different"), memory)["pair_type"], "similar_task")

    def test_task_type_uses_observable_question_only(self):
        self.assertEqual(task_type("Which person was born earlier?"), "comparison")
        self.assertEqual(task_type("Is this correct?"), "yes_no")
        self.assertEqual(task_type("unstructured clue"), "unknown")

    def test_structured_feedback_stays_valid_under_tight_cap(self):
        from explore.feedback import excerpt
        counter = ByteCounter()
        packets = []
        for i in range(3):
            packets.append({"failure_id": f"f{i}", "pair_type": "same_question", "failure": excerpt(observation(f"f{i}", hard=0), counter, 160),
                            "success_reference": excerpt(observation(f"f{i}"), counter, 160),
                            "observable_checks": {"answer_format_ok": True, "reference_has_verifiable_quote": False}})
        text, actual = pack_feedback(packets, counter, 2048)
        self.assertLessEqual(counter.count(text), 2048)
        serialized = text[text.index("["):]
        parsed = json.loads(serialized)
        self.assertEqual(len(parsed), 3)
        self.assertTrue(all(p["success_reference"]["em"] == 1 and p["success_reference"]["predicted_answer"] for p in parsed))

    def test_all_unpaired_feedback_exactly_falls_back_to_baseline(self):
        from baseline.feedback import BaselineFeedback
        from explore.feedback import ContrastiveFeedback
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = observation(hard=0)
            write_json(root / "predictions/q1/conversation.json", [{"role": "assistant", "content": row["raw_response"]}])
            item = {"id": "q1", "question": row["question"], "answers": row["gold_answers"]}
            config = load_config(demo=True)
            baseline = BaselineFeedback()
            contrastive = ContrastiveFeedback("searchqa", ["q1"], root / "contrastive", config)
            contrastive.observe([row])
            expected, _ = baseline.build([row], [item], root, 42, config, ByteCounter())
            actual, report = contrastive.build([row], [item], root, 42, config, ByteCounter())
            self.assertTrue(expected)
            self.assertEqual(actual, expected)
            self.assertEqual(report["unpaired"], 1)

    def test_future_memories_do_not_leak_back_into_earlier_pairing(self):
        memory = TrainMemory("searchqa", ["q1"])
        memory.add(observation(round_no=3))
        self.assertEqual(pair_failure(observation(hard=0, round_no=1), memory)["pair_type"], "unpaired")

    def test_budget_blocks_before_next_request_and_charges_shared_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Budget(Path(directory) / "ledger.jsonl", {"max_calls": 2, "max_generated_tokens": 15, "max_wall_hours": 1})
            ledger.preflight(10)
            ledger.finish({"calls": 1, "generated_tokens": 8, "physical_calls": 1}, 10)
            with self.assertRaises(BudgetExceeded):
                ledger.preflight(10)
            ledger.import_shared([{"calls": 1, "generated_tokens": 2, "physical_calls": 1}])
            self.assertEqual(ledger.totals()["physical_calls"], 1)
            with self.assertRaises(BudgetExceeded):
                ledger.preflight(1)

    def test_failed_http_request_is_logged_and_charged_full_reserved_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(demo=True)
            ledger = Budget(Path(directory) / "ledger.jsonl", config["budget"])
            client = ChatClient(config, ledger, ByteCounter(), Path(directory) / "calls.jsonl")
            with patch("urllib.request.urlopen", side_effect=TimeoutError("simulated")), patch("baseline.gpu.gpu_event", return_value={"open_sessions": ["a6000"], "gpu_hours": 0, "cap_hours": 40}):
                with self.assertRaises(ModelError):
                    client.chat("system", "user", maximum=10, stage="qa", seed=42, temperature=0)
            self.assertEqual(ledger.totals()["calls"], 1)
            self.assertEqual(ledger.totals()["generated_tokens"], 10)
            self.assertEqual(read_jsonl(Path(directory) / "calls.jsonl")[0]["accounting"]["status"], "error")

    def test_reasoning_usage_is_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(demo=True)
            ledger = Budget(Path(directory) / "ledger.jsonl", config["budget"])
            client = ChatClient(config, ledger, ByteCounter(), Path(directory) / "calls.jsonl", demo=True)
            response = {"model": config["model"]["id"], "usage": {"prompt_tokens": 9, "completion_tokens": 10,
                         "completion_tokens_details": {"reasoning_tokens": 7}}, "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            with patch.object(client, "_mock", return_value=response):
                client.chat("s", "u", maximum=10, stage="qa", seed=42, temperature=0)
            self.assertEqual(ledger.totals()["generated_tokens"], 10)
            self.assertEqual(ledger.totals()["reasoning_tokens"], 7)

    def test_upstream_operator_scope_is_enforced(self):
        from baseline import VENDOR
        parent = (VENDOR / "graphskillevo/initial_skills/searchqa/init_skill.md").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(demo=True)
            ledger = Budget(Path(directory) / "ledger.jsonl", config["budget"])
            client = ChatClient(config, ledger, ByteCounter(), Path(directory) / "calls.jsonl", demo=True)
            operator = CourseOperators(client, config)
            other = operator.mutation_other(parent=parent, context="failure")
            graph = operator.mutation_graph(parent=parent, context="failure")
            self.assertEqual(split_graph_skill(other.new_skill).node_lists, split_graph_skill(parent).node_lists)
            self.assertEqual(split_graph_skill(graph.new_skill).global_guidance, split_graph_skill(parent).global_guidance)

    def test_paired_comparison_preserves_counts_and_undefined_denominators(self):
        left = [observation("q1", hard=0), observation("q2", hard=1)]
        right = [observation("q1", hard=1), observation("q2", hard=0)]
        result = compare(left, right)
        self.assertEqual(result["counts"]["repaired"], 1)
        self.assertEqual(result["counts"]["regressed"], 1)
        self.assertEqual(result["repair_rate"], 1)
        with self.assertRaises(ValueError):
            compare(left, right[:1])

    def test_complete_synthetic_workflow_and_freeze_tamper_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "demo"
            result = demo_run(root)
            self.assertEqual(result["status"], "synthetic_end_to_end_passed")
            for dataset in ["searchqa", "hotpotqa"]:
                report = read_json(root / dataset / "test/results.json")
                self.assertTrue(report["demo"])
                self.assertEqual(set(report["conditions"]), {"N", "I", "B", "C"})
                self.assertEqual(report["conditions"]["C"]["n"], 6)
                config = read_json(root / dataset / "demo_config.json")
                check_design(root / dataset / "feedback_design.json", load_config(root / dataset / "demo_config.json", demo=True))
            from baseline.trainer import CourseTrainer
            config = load_config(root / "searchqa/demo_config.json", demo=True)
            trainer = CourseTrainer(config, root / "searchqa/data", root / "searchqa/invalid_B", condition="B", demo=True)
            # The original trainer resamples forever here; this wrapper stops at 2 attempts per child.
            with patch.object(trainer.operators, "_single_skill_operator", side_effect=ValueError("invalid candidate")) as invoke:
                summary = trainer.train_course(root / "searchqa/shared_init")
                self.assertEqual(invoke.call_count, 4)
                self.assertEqual(summary["completed_rounds"], 1)
                history = read_json(root / "searchqa/invalid_B/history.json")
                self.assertEqual(history[0]["candidates"], [])
            # A frozen skill cannot be silently edited before test.
            skill_path = root / "searchqa/B/best_skill.md"
            skill_path.write_text(skill_path.read_text(encoding="utf-8") + "changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                test_frozen(root / "searchqa/frozen.json", root / "searchqa/retest")


if __name__ == "__main__":
    unittest.main()
