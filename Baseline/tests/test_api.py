"""Hosted-backend regressions; these tests never call the real API or read its key."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE))
from baseline.budget import Budget, BudgetExceeded, ByteCounter
from baseline.client import ChatClient, ModelError, ProviderRefusal, operator_response_format
from baseline.config import load_config
from baseline.credentials import api_key
from baseline.io import read_jsonl

FAKE_KEY = "unit-test-placeholder-not-a-real-key"


class APITests(unittest.TestCase):
    def config(self):
        config = load_config(demo=True)
        config["backend"] = "api"
        config["model"].update(id="dated-test-model", api_key_env="QWEN_API_KEY",
                               base_url="https://example.invalid/v1", structured_operators="json_object")
        return config

    def response(self, model="dated-test-model"):
        return io.BytesIO(json.dumps({"model": model,
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}).encode())

    def test_hosted_call_skips_gpu_uses_auth_and_server_usage_without_logging_key(self):
        with tempfile.TemporaryDirectory() as folder:
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            with patch("baseline.client.api_key", return_value=FAKE_KEY):
                client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl")
            with patch("baseline.gpu.gpu_event", side_effect=AssertionError("API has no local GPU")), \
                 patch("urllib.request.urlopen", return_value=self.response()) as request:
                result, _ = client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            self.assertEqual(result, "OK")
            self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer " + FAKE_KEY)
            self.assertNotIn("response_format", json.loads(request.call_args.args[0].data))
            self.assertEqual(budget.totals()["generated_tokens"], 3)
            self.assertEqual(budget.totals()["prompt_tokens"], 11)
            self.assertNotIn(FAKE_KEY, (root / "calls.jsonl").read_text())

    def test_hosted_json_mode_omits_local_schema_but_qa_is_plain(self):
        self.assertEqual(operator_response_format("initialize", "json_object"), {"type": "json_object"})
        self.assertEqual(operator_response_format("graphskillevo_graph_mutation_graph", "json_object"), {"type": "json_object"})
        self.assertIsNone(operator_response_format("qa", "json_object"))
        self.assertIn("schema", operator_response_format("initialize"))

    def test_credentials_use_explicit_repo_file_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch("baseline.credentials.REPO_ROOT", Path(folder)):
            (Path(folder) / ".env").write_text("QWEN_API_KEY='fake-${DO_NOT_INTERPOLATE}'\n")
            self.assertEqual(api_key(self.config()), "fake-${DO_NOT_INTERPOLATE}")
            os.environ["QWEN_API_KEY"] = FAKE_KEY
            self.assertEqual(api_key(self.config()), FAKE_KEY)

    def test_schema_mode_constrains_initial_skill_array_and_mutation_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config()
            config["model"]["structured_operators"] = "json_schema"
            with patch("baseline.client.api_key", return_value=FAKE_KEY):
                client = ChatClient(config, Budget(Path(folder) / "ledger.jsonl", config["budget"]),
                                    ByteCounter(), Path(folder) / "calls.jsonl")
            with patch("urllib.request.urlopen", return_value=self.response()) as request:
                client.chat("s", "u", maximum=16, stage="initialize", seed=42, temperature=0)
            response_format = json.loads(request.call_args.args[0].data)["response_format"]
            self.assertEqual(response_format["type"], "json_schema")
            self.assertTrue(response_format["json_schema"]["strict"])
            schema = response_format["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["skills"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["properties"]["skills"],
                             {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 1})
            mutation = operator_response_format("graphskillevo_graph_mutation_graph", "json_schema")
            self.assertEqual(mutation["json_schema"]["schema"]["required"], ["new_skill", "notes"])
            self.assertIsNone(operator_response_format("qa", "json_schema"))

    def test_missing_credentials_fail_before_any_network_or_accounting(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
             patch("baseline.credentials.REPO_ROOT", Path(folder)), patch("urllib.request.urlopen") as request:
            config = self.config()
            budget = Budget(Path(folder) / "ledger.jsonl", config["budget"])
            with self.assertRaisesRegex(ValueError, "Missing QWEN_API_KEY"):
                ChatClient(config, budget, ByteCounter(), Path(folder) / "calls.jsonl")
            request.assert_not_called()
            self.assertEqual(budget.totals()["calls"], 0)

    def test_http_failure_is_redacted_and_charged_at_full_cap(self):
        with tempfile.TemporaryDirectory() as folder, patch("baseline.client.api_key", return_value=FAKE_KEY):
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl")
            with patch("urllib.request.urlopen", side_effect=TimeoutError("fake error " + FAKE_KEY)):
                with self.assertRaises(ModelError) as caught:
                    client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            self.assertNotIn(FAKE_KEY, str(caught.exception))
            self.assertNotIn(FAKE_KEY, (root / "calls.jsonl").read_text())
            self.assertEqual(budget.totals()["generated_tokens"], 16)

    def test_local_backend_still_requires_gpu_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load_config(demo=True)
            budget = Budget(Path(folder) / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), Path(folder) / "calls.jsonl")
            with patch("baseline.gpu.gpu_event", return_value={"open_sessions": [], "gpu_hours": 0, "cap_hours": 40}), \
                 patch("urllib.request.urlopen") as request:
                with self.assertRaises(BudgetExceeded):
                    client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            request.assert_not_called()

    def test_api_config_resolves_tokenizer_from_repository_and_rejects_credential_urls(self):
        path = BASELINE / "configs/searchqa_api.json"
        config = load_config(path, demo=True)
        self.assertEqual(Path(config["model"]["tokenizer_path"]), BASELINE.parent / "api/runtime/tokenizer")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "invalid.json"
            for url in ["http://example.invalid/v1", "https://u:p@example.invalid/v1", "https://example.invalid/v1?api_key=fake"]:
                target.write_text(json.dumps({"backend": "api", "model": {"base_url": url}}))
                with self.assertRaisesRegex(ValueError, "HTTPS endpoint"):
                    load_config(target, demo=True)

    def test_api_rejects_model_switch_and_records_actual_usage(self):
        with tempfile.TemporaryDirectory() as folder, patch("baseline.client.api_key", return_value=FAKE_KEY):
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl")
            with patch("urllib.request.urlopen", return_value=self.response("unexpected-model")):
                with self.assertRaisesRegex(ModelError, "Served model ID differs"):
                    client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            self.assertEqual(read_jsonl(root / "ledger.jsonl")[0]["status"], "error")
            self.assertEqual(budget.totals()["generated_tokens"], 3)

    def test_window_guard_releases_reservation_without_a_physical_call(self):
        with tempfile.TemporaryDirectory() as folder, patch("baseline.client.api_key", return_value=FAKE_KEY):
            root, config = Path(folder), self.config()
            config["model"].update(context_window=100, context_safety_tokens=16)
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl")
            with patch("urllib.request.urlopen", return_value=self.response()) as request:
                with self.assertRaisesRegex(ValueError, "exceeds context window"):
                    client.chat("s" * 60, "u" * 30, maximum=16, stage="qa", seed=42, temperature=0)
                request.assert_not_called()
                self.assertEqual(budget.reserved, 0)
                self.assertEqual(budget.totals()["physical_calls"], 0)
                client.chat("s" * 30, "u" * 38, maximum=16, stage="qa", seed=42, temperature=0)
                self.assertEqual(request.call_count, 1)
                config["model"]["structured_operators"] = "json_schema"
                with self.assertRaisesRegex(ValueError, "exceeds context window"):
                    client.chat("s", "u", maximum=16, stage="initialize", seed=42, temperature=0)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(budget.reserved, 0)
            self.assertEqual(budget.totals()["physical_calls"], 1)

    def test_token_policy_requires_valid_window_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for override in (
                {"qa_context_max_tokens": 4096},
                {"qa_context_max_tokens": True, "model": {"context_window": 32768}},
                {"qa_context_max_tokens": -1, "model": {"context_window": 32768}},
                {"model": {"context_window": 32768, "context_safety_tokens": -1}},
                {"model": {"context_window": 100, "context_safety_tokens": 512}},
            ):
                with self.subTest(override=override):
                    path.write_text(json.dumps(override), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path, demo=True)
            for dataset in ("searchqa", "hotpotqa"):
                config = load_config(BASELINE / "configs" / f"{dataset}_api.json", demo=True)
                self.assertEqual(config["qa_context_max_tokens"], 4096)
                self.assertEqual(config["qa_max_tokens"], 2048)
                self.assertEqual(config["model"]["context_safety_tokens"], 512)

    def test_extra_initialization_items_are_rejected_without_silent_repair(self):
        from baseline.operators import CourseOperators
        from graphskillevo.operators import GraphSkillGenerationError
        from baseline import VENDOR
        with tempfile.TemporaryDirectory() as folder:
            config = self.config()
            budget = Budget(Path(folder) / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), Path(folder) / "calls.jsonl", demo=True)
            parent = (VENDOR / "graphskillevo/initial_skills/searchqa/init_skill.md").read_text()
            operator = CourseOperators(client, config)
            with patch.object(client, "chat", return_value=(json.dumps({"skills": [parent, "}]}" ]}), {})):
                with self.assertRaisesRegex(GraphSkillGenerationError, "exactly one initial skill"):
                    operator.initial_attempt(parent)

    def test_invalid_initial_graph_retry_receives_the_validation_error(self):
        from baseline.operators import CourseOperators
        from baseline import VENDOR
        with tempfile.TemporaryDirectory() as folder:
            config = self.config()
            client = ChatClient(config, Budget(Path(folder) / "ledger.jsonl", config["budget"]), ByteCounter(), Path(folder) / "calls.jsonl", demo=True)
            operator = CourseOperators(client, config)
            parent = (VENDOR / "graphskillevo/initial_skills/searchqa/init_skill.md").read_text()
            valid = client._mock("", "", 4096, "initialize", 42)["choices"][0]["message"]["content"]
            invalid = valid.replace("1. VerifyEvidence", "2. VerifyEvidence")
            with patch.object(client, "chat", return_value=(invalid, {})):
                with self.assertRaises(ValueError) as caught:
                    operator.initial_attempt(parent)
            errors = [{"attempt": 1, "error": str(caught.exception)}]
            with patch.object(client, "chat", return_value=(valid, {})) as request:
                skill, _ = operator.initial_attempt(parent, recent_errors=errors)
            operator.validate(skill)
            self.assertIn(str(caught.exception), request.call_args.args[1])
            self.assertNotIn('"full skill document", "..."', request.call_args.args[0])
            self.assertNotIn("For mutation", request.call_args.args[1])

    def test_api_cooldown_waits_without_adding_hidden_http_attempts(self):
        with tempfile.TemporaryDirectory() as folder, patch("baseline.client.api_key", return_value=FAKE_KEY):
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl")
            with patch("urllib.request.urlopen", side_effect=[TimeoutError("transient"), self.response()]) as request, \
                 patch("baseline.client.time.sleep") as sleep:
                with self.assertRaises(ModelError):
                    client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
                text, _ = client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            self.assertEqual(text, "OK")
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once()
            self.assertGreater(sleep.call_args.args[0], 0)
            self.assertEqual(budget.totals()["calls"], 2)
            self.assertEqual(budget.totals()["generated_tokens"], 19)

    def test_exhausted_qa_requests_preserve_evidence_and_abort_scoring(self):
        from baseline.evaluation import run_qa, InfrastructureError
        with tempfile.TemporaryDirectory() as folder:
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl", demo=True)
            item = {"id": "network-case", "dataset": "searchqa", "question": "Where?",
                    "context": "Paris", "answers": ["Paris"]}
            with patch.object(client, "chat", side_effect=ModelError("network unavailable")) as request:
                with self.assertRaises(InfrastructureError):
                    run_qa(client, item, "", split="dev", round_no=0, seed=42, condition="N", out_dir=root)
            self.assertEqual(request.call_count, 2)
            row = json.loads((root / "predictions/network-case/result.json").read_text())
            self.assertEqual(len(row["attempts"]), 2)
            self.assertEqual(row["error"], "network unavailable")

    def test_transient_qa_failure_can_recover_within_existing_attempt_cap(self):
        from baseline.evaluation import run_qa
        with tempfile.TemporaryDirectory() as folder:
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl", demo=True)
            item = {"id": "network-case", "dataset": "searchqa", "question": "Where?",
                    "context": "Paris", "answers": ["Paris"]}
            with patch.object(client, "chat", side_effect=[ModelError("transient"), ("<answer>Paris</answer>", {})]) as request:
                row = run_qa(client, item, "", split="dev", round_no=0, seed=42, condition="N", out_dir=root)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(row["em"], 1)
            self.assertEqual(row["error"], "")

    def test_known_provider_rejection_is_classified_without_logging_body(self):
        with tempfile.TemporaryDirectory() as folder, patch("baseline.client.api_key", return_value=FAKE_KEY):
            root, config = Path(folder), self.config()
            budget = Budget(root / "ledger.jsonl", config["budget"])
            client = ChatClient(config, budget, ByteCounter(), root / "calls.jsonl")
            body = json.dumps({"error": {"code": "data_inspection_failed", "message": "sensitive " + FAKE_KEY}}).encode()
            error = urllib.error.HTTPError("https://example.invalid", 400, "Bad Request", {}, io.BytesIO(body))
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(ProviderRefusal):
                    client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            log = (root / "calls.jsonl").read_text()
            self.assertNotIn(FAKE_KEY, log)
            self.assertNotIn("sensitive", log)
            self.assertEqual(read_jsonl(root / "calls.jsonl")[0]["provider_error_code"], "data_inspection_failed")
            self.assertEqual(budget.totals()["generated_tokens"], 16)

    def test_unrecognized_http_400_is_not_a_scored_refusal(self):
        with tempfile.TemporaryDirectory() as folder, patch("baseline.client.api_key", return_value=FAKE_KEY):
            root, config = Path(folder), self.config()
            client = ChatClient(config, Budget(root / "ledger.jsonl", config["budget"]), ByteCounter(), root / "calls.jsonl")
            error = urllib.error.HTTPError("https://example.invalid", 400, "Bad Request", {},
                                          io.BytesIO(b'{"error":{"code":"invalid_parameter"}}'))
            with patch("urllib.request.urlopen", side_effect=error):
                with self.assertRaises(ModelError) as caught:
                    client.chat("s", "u", maximum=16, stage="qa", seed=42, temperature=0)
            self.assertNotIsInstance(caught.exception, ProviderRefusal)

    def test_exhausted_provider_refusals_are_zero_scores_not_infrastructure_failures(self):
        from baseline.evaluation import run_qa, aggregate
        from baseline.analysis import diagnose
        with tempfile.TemporaryDirectory() as folder:
            root, config = Path(folder), self.config()
            client = ChatClient(config, Budget(root / "ledger.jsonl", config["budget"]), ByteCounter(), root / "calls.jsonl", demo=True)
            item = {"id": "refusal-case", "dataset": "hotpotqa", "question": "Where?",
                    "context": "Paris", "answers": ["Paris"]}
            with patch.object(client, "chat", side_effect=ProviderRefusal("HTTP 400 (data_inspection_failed)")) as request:
                row = run_qa(client, item, "", split="dev", round_no=0, seed=42, condition="N", out_dir=root)
            self.assertEqual(request.call_count, 2)
            self.assertTrue(row["provider_refusal"])
            self.assertEqual(row["error"], "")
            self.assertEqual((row["em"], row["f1"]), (0, 0))
            metrics = aggregate([row])
            self.assertEqual(metrics["n"], 1)
            self.assertEqual(metrics["provider_refusals"], 1)
            self.assertEqual(metrics["infrastructure_errors"], 0)
            self.assertEqual(diagnose(row)["primary_category"], "provider_refusal")

    def test_provider_refusal_then_success_clears_final_refusal_flag(self):
        from baseline.evaluation import run_qa
        with tempfile.TemporaryDirectory() as folder:
            root, config = Path(folder), self.config()
            client = ChatClient(config, Budget(root / "ledger.jsonl", config["budget"]), ByteCounter(), root / "calls.jsonl", demo=True)
            item = {"id": "refusal-case", "dataset": "hotpotqa", "question": "Where?",
                    "context": "Paris", "answers": ["Paris"]}
            with patch.object(client, "chat", side_effect=[ProviderRefusal("rejected"), ("<answer>Paris</answer>", {})]):
                row = run_qa(client, item, "", split="dev", round_no=0, seed=42, condition="N", out_dir=root)
            self.assertFalse(row["provider_refusal"])
            self.assertTrue(row["attempts"][0]["provider_refusal"])
            self.assertEqual(row["em"], 1)


if __name__ == "__main__":
    unittest.main()
