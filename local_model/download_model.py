from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import math
import os
import time
import requests

ROOT = Path(r"F:\Document\HKUST\2026fall\NLP\Qwen3.5_9B_4bit")
MODEL = "Qwen3.5-9B-Q4_K_M.gguf"
QUANT_REV = "3885219b6810b007914f3a7950a8d1b469d598a5"
BASE_REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SHA256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
SIZE = 5680522464
CHUNK = 32 * 1024 * 1024
URL = f"https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/{QUANT_REV}/{MODEL}"


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def chunk_download(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    path = ROOT / ".download-parts" / f"{index:04d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    temporary = path.with_suffix(".incomplete")
    for attempt in range(5):
        try:
            with requests.get(URL + f"?download=true&part={index}", headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                expected_range = f"bytes {start}-{end}/{SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Invalid range response for part {index}: {response.status_code}, {response.headers.get('Content-Range')}")
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if temporary.stat().st_size != end - start + 1:
                raise RuntimeError("Incomplete range download")
            temporary.replace(path)
            return path
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(10, 2 ** attempt))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".download-parts").mkdir(exist_ok=True)
    destination = ROOT / MODEL
    if not destination.exists():
        count = math.ceil(SIZE / CHUNK)
        print(f"Downloading {SIZE/1e9:.2f} GB in {count} resumable ranges", flush=True)
        completed = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(chunk_download, i) for i in range(count)]
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(f"ranges {completed}/{count} ({completed/count:.1%})", flush=True)
        partial = destination.with_suffix(".gguf.assembling")
        with partial.open("wb") as output:
            for index in range(count):
                with (ROOT / ".download-parts" / f"{index:04d}.part").open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
        print("Verifying full SHA256", flush=True)
        if partial.stat().st_size != SIZE or hash_file(partial) != SHA256:
            raise RuntimeError("Full GGUF checksum mismatch")
        partial.replace(destination)
    elif hash_file(destination) != SHA256:
        raise RuntimeError("Existing GGUF does not match pinned checksum")
    tokenizer = ROOT / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    hashes = {}
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]:
        path = tokenizer / name
        if not path.exists():
            response = requests.get(f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{BASE_REV}/{name}", timeout=(20, 120))
            response.raise_for_status()
            path.write_bytes(response.content)
        hashes[name] = hash_file(path)
        print("Tokenizer ready:", name, flush=True)
    manifest = {"base_repo": "Qwen/Qwen3.5-9B", "base_revision": BASE_REV,
                "quant_repo": "unsloth/Qwen3.5-9B-GGUF", "quant_revision": QUANT_REV,
                "quantization": "Q4_K_M", "gguf": str(destination), "gguf_sha256": SHA256,
                "bytes": SIZE, "tokenizer": str(tokenizer), "tokenizer_sha256": hashes}
    (ROOT / "deployment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Model and tokenizer verified", flush=True)


if __name__ == "__main__":
    main()
