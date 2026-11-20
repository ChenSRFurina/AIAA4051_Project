from __future__ import annotations

from copy import deepcopy
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from urllib.parse import urlsplit
from .io import read_json, digest

DEFAULT = {
    "backend": "local",
    "dataset": "searchqa", "seed": 42, "rounds": 3, "population_size": 2,
    "train_batch_size": 20, "failure_k": 3, "max_attempts": 2,
    "qa_max_tokens": 1024, "operator_max_tokens": 3072,
    "skill_max_tokens": 2048, "feedback_max_tokens": 2048,
    "qa_temperature": 0.0, "operator_temperature": 0.2, "top_p": 1.0,
    "timeout_seconds": 120, "concurrency": 1,
    "api_retry_delay_seconds": 2.0,
    "gpu_ledger_path": "runs/gpu_accounting.jsonl",
    "model": {"id": "SET_MODEL_ID", "revision": "SET_REVISION", "base_url": "http://127.0.0.1:8000/v1",
              "api_key_env": "PROJECT5_API_KEY", "tokenizer_path": "SET_LOCAL_TOKENIZER",
              "precision": "SET_PRECISION", "server_version": "SET_VERSION", "extra_body": {}, "structured_operators": False,
              "tokenizer_note": "Local tokenizer for the served model"},
    "budget": {"max_calls": 600, "max_generated_tokens": 700000, "max_wall_hours": 9},
    "pairing": {"jaccard_threshold": 0.2, "per_outcome_limit": 2},
}


def load_config(path=None, *, demo=False):
    config = deepcopy(DEFAULT)
    if path:
        for key, value in read_json(path).items():
            if key not in config and key != "qa_context_max_tokens":
                raise ValueError(f"Unknown config key: {key}")
            if isinstance(config.get(key), dict):
                unknown = set(value) - set(config[key])
                if key == "model":
                    unknown -= {"allow_private_http", "context_window", "context_safety_tokens"}
                if unknown:
                    raise ValueError(f"Unknown {key} keys: {unknown}")
                config[key].update(value)
            else:
                config[key] = value
    if config["dataset"] not in {"searchqa", "hotpotqa"}:
        raise ValueError("Only searchqa and hotpotqa are supported")
    if config["backend"] not in {"local", "api"}:
        raise ValueError("backend must be local or api")
    if config["model"]["structured_operators"] not in (True, False, "json_object", "json_schema"):
        raise ValueError("structured_operators must be a boolean, json_object or json_schema")
    if config["backend"] == "api":
        model = config["model"]
        endpoint = urlsplit(model["base_url"])
        private_http = False
        if endpoint.scheme == "http" and model.get("allow_private_http") is True:
            try:
                address = IPv4Address(endpoint.hostname or "")
            except ValueError:
                pass
            else:
                private_http = any(address in IPv4Network(network)
                                   for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
        if ((endpoint.scheme != "https" and not private_http) or not endpoint.hostname
                or endpoint.username is not None or endpoint.password is not None
                or endpoint.query or endpoint.fragment):
            raise ValueError("API base_url must be an HTTPS endpoint or an explicitly enabled private IPv4 HTTP endpoint, without credentials, query or fragment")
        if not config["model"]["api_key_env"]:
            raise ValueError("API mode requires model.api_key_env")
        tokenizer = Path(config["model"]["tokenizer_path"])
        if not tokenizer.is_absolute():
            config["model"]["tokenizer_path"] = str((Path(__file__).resolve().parents[2] / tokenizer).resolve())
    context_window = config["model"].get("context_window")
    if context_window is not None and (type(context_window) is not int or context_window <= 0):
        raise ValueError("model.context_window must be a positive integer")
    safety = config["model"].get("context_safety_tokens", 512)
    if type(safety) is not int or safety <= 0:
        raise ValueError("model.context_safety_tokens must be a positive integer")
    if "qa_context_max_tokens" in config:
        if type(config["qa_context_max_tokens"]) is not int or config["qa_context_max_tokens"] <= 0:
            raise ValueError("qa_context_max_tokens must be a positive integer")
        if context_window is None:
            raise ValueError("Token-based QA context requires model.context_window")
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
    if context_window is not None and max(config["qa_max_tokens"], config["operator_max_tokens"]) + safety >= context_window:
        raise ValueError("Completion caps and safety margin leave no room for model input")
    if not isinstance(config["api_retry_delay_seconds"], (int, float)) or not 0 <= config["api_retry_delay_seconds"] <= 60:
        raise ValueError("api_retry_delay_seconds must be between 0 and 60")
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
