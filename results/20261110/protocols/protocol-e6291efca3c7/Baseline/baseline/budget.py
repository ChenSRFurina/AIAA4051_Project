"""Preflight reservation, including failed requests and shared initialization costs."""
from __future__ import annotations

import time
from .io import append_jsonl, read_jsonl


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path, limits):
        self.path, self.limits = path, limits
        self.started = time.monotonic()
        self.events = read_jsonl(path)
        self.reserved = 0

    def totals(self):
        return {key: sum(e.get(key, 0) for e in self.events)
                for key in ["calls", "prompt_tokens", "generated_tokens", "reasoning_tokens", "physical_calls"]}

    def preflight(self, maximum):
        totals = self.totals()
        if totals["calls"] + 1 > self.limits["max_calls"]:
            raise BudgetExceeded("model_call_limit")
        if totals["generated_tokens"] + self.reserved + maximum > self.limits["max_generated_tokens"]:
            raise BudgetExceeded("generation_token_reservation_limit")
        if time.monotonic() - self.started >= self.limits["max_wall_hours"] * 3600:
            raise BudgetExceeded("wall_time_limit")
        self.reserved += maximum

    def finish(self, event, maximum):
        self.reserved -= maximum
        self.events.append(event)
        append_jsonl(self.path, event)

    def import_shared(self, events):
        totals = self.totals()
        if totals["calls"] + sum(e.get("calls", 0) for e in events) > self.limits["max_calls"]:
            raise BudgetExceeded("shared_initialization_exceeds_calls")
        if totals["generated_tokens"] + sum(e.get("generated_tokens", 0) for e in events) > self.limits["max_generated_tokens"]:
            raise BudgetExceeded("shared_initialization_exceeds_tokens")
        for original in events:
            event = dict(original, physical_calls=0, shared=True, source_call_id=original.get("call_id"))
            event["call_id"] = f"shared-{original.get('call_id')}"
            self.events.append(event)
            append_jsonl(self.path, event)


class ByteCounter:
    """Demo-only counter. Never used to claim real tokenizer lengths."""
    identity = "synthetic-utf8-bytes"

    def count(self, text):
        return len(text.encode("utf-8"))

    def clip(self, text, limit):
        return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


class TokenCounter:
    def __init__(self, path):
        from transformers import AutoTokenizer
        from pathlib import Path
        from .io import digest
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        files = [p for p in Path(path).iterdir() if p.is_file() and ("token" in p.name or p.name in {"vocab.json", "merges.txt", "special_tokens_map.json"})]
        self.identity = digest({p.name: digest(p.read_bytes()) for p in sorted(files)})

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def clip(self, text, limit):
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:limit]
        result = self.tokenizer.decode(ids, skip_special_tokens=False)
        while self.count(result) > limit and ids:
            ids.pop()
            result = self.tokenizer.decode(ids, skip_special_tokens=False)
        return result
