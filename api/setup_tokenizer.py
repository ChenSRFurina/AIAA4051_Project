"""Download a pinned tokenizer only. No weights or model serving components."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
FILES = ("tokenizer.json", "tokenizer_config.json")
DESTINATION = ROOT / "api/runtime/tokenizer"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "source.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and (previous["repo"] != REPO or previous["revision"] != REVISION):
        raise ValueError("Existing tokenizer has a different source; preserve it before using a new directory")
    hashes = {}
    for name in FILES:
        path = DESTINATION / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}")
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            # Parse before adopting a download, so an HTML error is not a tokenizer.
            json.loads(temporary.read_text(encoding="utf-8"))
            temporary.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous and hashes[name] != previous["sha256"][name]:
            raise ValueError(f"Pinned tokenizer changed: {name}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DESTINATION, local_files_only=True, trust_remote_code=False)
    assert tokenizer.encode("Project 5 测试", add_special_tokens=False)
    manifest = {"repo": REPO, "revision": REVISION, "sha256": hashes,
                "purpose": "Proxy for skill/feedback lengths; hosted API usage is authoritative for billed tokens"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", "path": str(DESTINATION), "weights_downloaded": False}))


if __name__ == "__main__":
    main()
