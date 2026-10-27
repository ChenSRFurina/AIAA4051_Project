from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from .io import read_json, digest

DEFAULT = {
    "dataset": "searchqa", "seed": 42, "rounds": 3, "population_size": 2,
    "train_batch_size": 20, "failure_k": 3, "max_attempts": 2,
    "qa_max_tokens": 1024, "operator_max_tokens": 3072,
    "skill_max_tokens": 2048, "feedback_max_tokens": 2048,
    "qa_temperature": 0.0, "operator_temperature": 0.2, "top_p": 1.0,
    "timeout_seconds": 120, "concurrency": 1,
    "gpu_ledger_path": "runs/gpu_accounting.jsonl",
    "model": {"id": "SET_MODEL_ID", "revision": "SET_REVISION", "base_url": "http://127.0.0.1:8000/v1",
              "api_key_env": "PROJECT5_API_KEY", "tokenizer_path": "SET_LOCAL_TOKENIZER",
              "precision": "SET_PRECISION", "server_version": "SET_VERSION", "extra_body": {}, "structured_operators": False},
    "budget": {"max_calls": 600, "max_generated_tokens": 700000, "max_wall_hours": 9},
    "pairing": {"jaccard_threshold": 0.2, "per_outcome_limit": 2},
}


def load_config(path=None, *, demo=False):
    config = deepcopy(DEFAULT)
    if path:
        for key, value in read_json(path).items():
            if key not in config:
                raise ValueError(f"Unknown config key: {key}")
            if isinstance(config[key], dict):
                unknown = set(value) - set(config[key])
                if unknown:
                    raise ValueError(f"Unknown {key} keys: {unknown}")
                config[key].update(value)
            else:
                config[key] = value
    if config["dataset"] not in {"searchqa", "hotpotqa"}:
        raise ValueError("Only searchqa and hotpotqa are supported")
    if config["population_size"] != 2 or not 0 <= config["rounds"] <= 3:
        raise ValueError("Course protocol requires population_size=2 and 0<=rounds<=3")
    for key in ["train_batch_size", "failure_k", "max_attempts", "qa_max_tokens", "operator_max_tokens",
                "skill_max_tokens", "feedback_max_tokens", "timeout_seconds"]:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_attempts"] > 2 or config["failure_k"] != 3 or config["train_batch_size"] > 100:
        raise ValueError("Use k=3, at most 2 attempts and at most 100 training IDs")
    if config["concurrency"] != 1:
        raise ValueError("This audited runner currently supports sequential concurrency=1")
    if config["qa_temperature"] != 0.0:
        raise ValueError("QA validation caching requires the frozen deterministic decoding policy qa_temperature=0")
    if config["operator_temperature"] < 0 or not 0 < config["top_p"] <= 1:
        raise ValueError("Invalid sampling parameters")
    gpu_path = Path(config["gpu_ledger_path"])
    if not gpu_path.is_absolute():
        config["gpu_ledger_path"] = str((Path(__file__).resolve().parents[1] / gpu_path).resolve())
    for key, value in config["budget"].items():
        if value <= 0:
            raise ValueError(f"budget.{key} must be positive")
    if not 0 <= config["pairing"]["jaccard_threshold"] <= 1 or config["pairing"]["per_outcome_limit"] != 2:
        raise ValueError("Invalid frozen pairing rules")
    if not demo:
        for key in ["id", "revision", "tokenizer_path", "precision", "server_version"]:
            if str(config["model"][key]).startswith("SET_"):
                raise ValueError(f"Freeze model.{key} after the pilot before a real run")
        if not Path(config["model"]["tokenizer_path"]).exists():
            raise ValueError("model.tokenizer_path must point to a locally pinned tokenizer")
    return config


def config_hash(config):
    return digest(config)
