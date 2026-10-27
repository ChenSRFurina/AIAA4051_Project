"""One HTTP request per accounted attempt; compatible with a local vLLM server."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import uuid
from .io import append_jsonl, digest


def operator_response_format(stage):
    """Optional common syntax constraint; QA answer tags remain unconstrained."""
    if stage == "initialize":
        schema = {"type": "object", "properties": {"skills": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 1}},
                  "required": ["skills"], "additionalProperties": False}
    elif stage.startswith("graphskillevo_graph_"):
        schema = {"type": "object", "properties": {"new_skill": {"type": "string"}, "notes": {"type": "array", "items": {"type": "string"}}},
                  "required": ["new_skill", "notes"], "additionalProperties": False}
    else:
        return None
    # llama.cpp supports schema-constrained json_object responses.
    return {"type": "json_object", "schema": schema}


class ModelError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, config, budget, counter, log_path, *, demo=False):
        self.config, self.budget, self.counter = config, budget, counter
        self.log_path, self.demo = log_path, demo

    def chat(self, system, user, *, maximum, stage, seed, temperature, phase=None):
        if not self.demo:
            from .gpu import gpu_event
            from .budget import BudgetExceeded
            gpu = gpu_event(self.config["gpu_ledger_path"], "status", "a6000")
            if not gpu["open_sessions"]:
                raise BudgetExceeded("Start the shared GPU service allocation ledger before real model requests")
            if gpu["gpu_hours"] >= gpu["cap_hours"]:
                raise BudgetExceeded("project_40_gpu_hour_cap")
        self.budget.preflight(maximum)
        call_id, start = uuid.uuid4().hex, time.monotonic()
        payload = {"model": self.config["model"]["id"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "max_tokens": maximum, "temperature": temperature, "top_p": self.config["top_p"], "seed": seed}
        extra = self.config["model"]["extra_body"]
        if set(extra) & set(payload):
            self.budget.finish({"call_id": call_id, "calls": 0, "physical_calls": 0, "stage": stage, "status": "invalid_config"}, maximum)
            raise ValueError("extra_body may not override frozen core request fields")
        payload.update(extra)
        if self.config["model"].get("structured_operators", False):
            structured = operator_response_format(stage)
            if structured is not None:
                payload["response_format"] = structured
        phase = phase or ("initialize" if stage == "initialize" else "optimize")
        record = {"call_id": call_id, "stage": stage, "phase": phase, "demo": self.demo, "request": payload,
                  "prompt_hash": digest([system, user]), "raw_response": None}
        event = {"call_id": call_id, "stage": stage, "phase": phase, "calls": 1, "physical_calls": 1,
                 "prompt_tokens": self.counter.count(system + user), "generated_tokens": maximum,
                 "reasoning_tokens": 0, "usage_status": "reserved_conservative", "shared": False}
        try:
            if self.demo:
                data = self._mock(system, user, maximum, stage, seed)
            else:
                body = json.dumps(payload).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                key = os.environ.get(self.config["model"]["api_key_env"])
                if key:
                    headers["Authorization"] = "Bearer " + key
                request = urllib.request.Request(self.config["model"]["base_url"].rstrip("/") + "/chat/completions", data=body, headers=headers)
                with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                    data = json.load(response)
            record["raw_response"] = data
            usage = data.get("usage") or {}
            completion = usage.get("completion_tokens")
            if isinstance(completion, int) and completion >= 0:
                event["generated_tokens"] = completion
                event["usage_status"] = "server_reported"
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] >= 0:
                event["prompt_tokens"] = usage["prompt_tokens"]
            event["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            # Reasoning is a subset of completion_tokens in this protocol, never add twice.
            choice = data["choices"][0]
            text = choice["message"].get("content")
            if text is None or not isinstance(text, str):
                raise ModelError("Server returned no text content")
            if data.get("model") not in {None, self.config["model"]["id"]}:
                raise ModelError("Served model ID differs from frozen model ID")
            if event["generated_tokens"] > maximum or event["reasoning_tokens"] > event["generated_tokens"]:
                raise ModelError("Backend token accounting violates requested completion cap")
            if not self.demo and completion is None:
                raise ModelError("Real server must return completion usage including reasoning; request charged at full cap")
            record["finish_reason"] = choice.get("finish_reason")
            event["status"] = "ok"
            return text, {**usage, "call_id": call_id, "finish_reason": choice.get("finish_reason")}
        except Exception as exc:
            event["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise ModelError(record["error"]) from exc
        finally:
            event["wall_seconds"] = time.monotonic() - start
            record["accounting"] = event
            append_jsonl(self.log_path, record)
            self.budget.finish(event, maximum)

    def _mock(self, system, user, maximum, stage, seed):
        """Synthetic fixtures: useful for pipeline validation, no benchmark claims."""
        if stage == "qa":
            match = re.search(r"Documented value: ([^\n]+)", user)
            value = match.group(1) if match else "unknown"
            good = "VerifyEvidence" in system or int(digest(user)[:2], 16) % 3 == 0
            text = f"Evidence: Documented value: {value}\n<answer>{value if good else 'unknown'}</answer>"
        else:
            skill = ("# Synthetic QA Skill\n\n## Global Guidance\nRead the supplied context. Return concise answer tags.\n\n"
                     "## Node Lists\n\n### VerifyEvidence\nCheck the requested relation against visible evidence.\n\n"
                     "## Task Graphs\n\n### QA\n**Use when:** A question has supplied context.\n**Workflow:**\n1. VerifyEvidence\n")
            text = json.dumps({"skills": [skill]} if stage == "initialize" else {"new_skill": skill, "notes": ["Synthetic change; not an experimental result."]})
        text = self.counter.clip(text, maximum)
        usage = {"prompt_tokens": self.counter.count(system + user), "completion_tokens": self.counter.count(text),
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        return {"model": self.config["model"]["id"], "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": usage}
