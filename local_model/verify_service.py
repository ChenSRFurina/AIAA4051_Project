from pathlib import Path
import json
import sys
import urllib.request

PROJECT = Path(r"F:\Document\HKUST\2026fall\NLP\Project\AIAA4051_Project")
sys.path.insert(0, str(PROJECT / "Baseline"))
from baseline.config import load_config
from baseline.budget import TokenCounter, Budget
from baseline.client import ChatClient
from baseline.io import write_json

config = load_config(PROJECT / "Baseline/configs/searchqa.json")
counter = TokenCounter(config["model"]["tokenizer_path"])
output = PROJECT / "local_model/checks"
output.mkdir(parents=True, exist_ok=True)
budget = Budget(output / "ledger.jsonl", config["budget"])
client = ChatClient(config, budget, counter, output / "calls.jsonl")
health = json.load(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5))
models = json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5))
if config["model"]["id"] not in [m["id"] for m in models["data"]]:
    raise RuntimeError("Served model alias does not match the fixed experiment config")
text, usage = client.chat("Answer the user's question concisely.", "What is 2 + 2? Put the final number in <answer> tags.",
                          maximum=64, stage="service_smoke", phase="diagnostic", seed=42, temperature=0)
if "4" not in text or "<answer>" not in text or "<think>" in text:
    raise RuntimeError(f"Unexpected non-thinking QA response: {text!r}")
sample = "A quoted fact: Paris is in France. 中文测试。"
request = urllib.request.Request("http://127.0.0.1:8000/tokenize", data=json.dumps({"content": sample, "add_special": False, "parse_special": True}).encode(), headers={"Content-Type": "application/json"})
tokens = json.load(urllib.request.urlopen(request, timeout=10))["tokens"]
tokenizer_count = counter.count(sample)
if tokenizer_count != len(tokens):
    raise RuntimeError(f"Token counts differ between fixed tokenizer and serving GGUF: {tokenizer_count} != {len(tokens)}")
result = {"health": health, "served_alias": config["model"]["id"], "response": text, "usage": usage,
          "tokenizer_count_matches_gguf": True, "sample_tokens": tokenizer_count,
          "tokenizer_hash": counter.identity, "cost": budget.totals(), "real_model": True}
write_json(output / "service_validation.json", result)
print(json.dumps(result, indent=2, ensure_ascii=False))
