"""Single-turn QA, upstream answer extraction, label-free execution prompts."""
from __future__ import annotations

import re
import time
from pathlib import Path
from graphskillevo.envs.searchqa.evaluator import evaluate, normalize_answer
from graphskillevo.prompts import load_prompt
from .budget import BudgetExceeded
from .client import ModelError, ProviderRefusal
from .io import digest, append_jsonl, write_json


class InfrastructureError(RuntimeError):
    """A failed request must not become a benchmark model-error observation."""


def truncate_context(context, max_chars=6000):
    # Exact pinned SearchQA boundary rule, without importing upstream model backends.
    if len(context) <= max_chars:
        return context
    result = ""
    for doc in context.split("[DOC]"):
        candidate = result + "[DOC]" + doc if result else doc
        if len(candidate) > max_chars:
            break
        result = candidate
    return result or context[:max_chars] + "\n...[truncated]"


def truncate_context_tokens(context, counter, limit):
    """Keep the original prefix, preferring complete documents within the token cap."""
    if counter.count(context) <= limit:
        return context
    result = ""
    for document in re.split(r"(?=\[DOC\])", context):
        candidate = result + document
        if counter.count(candidate) > limit:
            break
        result = candidate
    # If even the first document is too long, use the tokenizer's bounded prefix.
    return result if result.strip() else counter.clip(context, limit)


def qa_prompts(item, skill, *, config=None, counter=None):
    section = f"## Skill\n{skill.strip()}\n\n" if skill.strip() else ""
    system = load_prompt("rollout_system", env="searchqa").format(skill_section=section)
    if config is not None and "qa_context_max_tokens" in config:
        if counter is None:
            raise ValueError("Token-based QA context requires the pinned token counter")
        visible = truncate_context_tokens(item["context"], counter, config["qa_context_max_tokens"])
    else:
        visible = truncate_context(item["context"])
    return system, f"## Context\n{visible}\n\n## Question\n{item['question']}", visible


def score(response, answers, dataset):
    metrics = evaluate(response, answers)
    if dataset == "hotpotqa":
        # Official HotpotQA answer-only F1 special-cases yes/no/noanswer.
        predicted = normalize_answer(metrics["predicted_answer"])
        golds = [normalize_answer(a) for a in answers]
        if predicted in {"yes", "no", "noanswer"} or any(a in {"yes", "no", "noanswer"} for a in golds):
            metrics["f1"] = max(0.0 if predicted != g and (predicted in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"})
                                else evaluate(response, [a])["f1"] for a, g in zip(answers, golds))
    return metrics


def run_qa(client, item, skill, *, split, round_no, seed, condition, out_dir):
    system, user, visible = qa_prompts(item, skill, config=client.config, counter=client.counter)
    start, response, error, usage = time.monotonic(), "", "", {}
    attempts, provider_refusal = [], False
    for attempt in range(client.config["max_attempts"]):
        try:
            response, usage = client.chat(system, user, maximum=client.config["qa_max_tokens"], stage="qa", seed=seed,
                                          temperature=client.config["qa_temperature"], phase=split)
            attempts.append({"attempt": attempt + 1, "status": "ok", "usage": usage})
            error = ""
            provider_refusal = False
            break
        except BudgetExceeded:
            raise
        except ModelError as exc:
            error = str(exc)
            provider_refusal = isinstance(exc, ProviderRefusal)
            attempts.append({"attempt": attempt + 1, "status": "error", "error": error,
                             "provider_refusal": provider_refusal})
    refusal_reason = error if provider_refusal else ""
    if provider_refusal:
        error = ""
    metrics = score(response, item["answers"], item["dataset"])
    if error or provider_refusal:
        metrics.update(em=0.0, f1=0.0)
    row = {"id": item["id"], "dataset": item["dataset"], "split": split, "condition": condition,
           "round": round_no, "skill_hash": digest(skill), "question": item["question"],
           "visible_context": visible, "visible_context_hash": digest(visible), "context_truncated": visible != item["context"],
           "raw_response": response, "response": response, **metrics,
           "hard": int(metrics["em"]), "soft": metrics["f1"], "gold_answers": item["answers"],
           "format_ok": bool(re.search(r"<answer>.*?</answer>", response, re.S | re.I)),
           "error": error, "provider_refusal": provider_refusal, "refusal_reason": refusal_reason,
           "fail_reason": error or refusal_reason or ("EM=0" if metrics["em"] == 0 else ""),
           "latency_seconds": time.monotonic() - start, "generation_config": {
               "model": client.config["model"], "temperature": client.config["qa_temperature"], "seed": seed,
               "max_tokens": client.config["qa_max_tokens"], "tokenizer_hash": client.counter.identity},
           "attempts": attempts, "observation_id": digest([item["id"], digest(skill), round_no, condition, usage.get("call_id")]),
           "cached": False}
    prediction_dir = Path(out_dir) / "predictions" / item["id"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "system.txt").write_text(system, encoding="utf-8")
    (prediction_dir / "user.txt").write_text(user, encoding="utf-8")
    write_json(prediction_dir / "result.json", row)
    write_json(prediction_dir / "conversation.json", [
        {"role": "assistant", "content": response},
        {"role": "system", "content": f"[EVALUATION RESULT]\nGold answers: {item['answers']!r}\nEM: {row['em']}\nF1: {row['f1']}"},
    ])
    append_jsonl(Path(out_dir) / "outputs.jsonl", row)
    if error:
        raise InfrastructureError(
            f"QA infrastructure failure after {len(attempts)} attempts for {item['id']}; "
            f"diagnostics saved to {prediction_dir / 'result.json'}. No complete score was produced."
        )
    return row


def aggregate(rows):
    if not rows:
        raise ValueError("Cannot score an empty evaluation")
    values = sorted(r["latency_seconds"] for r in rows)
    def percentile(p):
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"n": len(rows), "correct": sum(r["hard"] for r in rows), "em": sum(r["em"] for r in rows) / len(rows),
            "f1": sum(r["f1"] for r in rows) / len(rows), "mean_latency_seconds": sum(values) / len(values),
            "p50_latency_seconds": percentile(.5), "p95_latency_seconds": percentile(.95),
            "infrastructure_errors": sum(bool(r["error"]) for r in rows),
            "provider_refusals": sum(bool(r.get("provider_refusal")) for r in rows),
            "format_failures": sum(not r["format_ok"] for r in rows)}
