from pathlib import Path
import json
import sys

project = Path(r"F:\Document\HKUST\2026fall\NLP\Project\AIAA4051_Project")
sys.path.insert(0, str(project / "Baseline"))
from baseline.config import load_config
from baseline.budget import TokenCounter, Budget
from baseline.client import ChatClient

config = load_config(project / "Baseline/configs/searchqa.json")
root = project / "local_model/checks"
client = ChatClient(config, Budget(root / "ledger.jsonl", config["budget"]), TokenCounter(config["model"]["tokenizer_path"]), root / "calls.jsonl")
text, usage = client.chat("Return valid JSON following the supplied schema.", 'Return {"skills":["ok"]}.', maximum=64,
                          stage="initialize", phase="diagnostic", seed=42, temperature=.2)
if json.loads(text) != {"skills": ["ok"]}:
    raise RuntimeError(f"Unexpected schema response: {text!r}")
print("Real backend JSON schema: OK", usage)
