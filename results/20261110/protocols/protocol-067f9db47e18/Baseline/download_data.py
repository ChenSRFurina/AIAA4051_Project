"""Download pinned public data. HotpotQA public gold test labels are unavailable.

Raw files stay outside git; no source partition is repurposed as held-out test.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
import urllib.request

from baseline import VENDOR
from baseline.data import normalize_item, prepare
from baseline.io import read_json, write_json

SOURCES = {
    "searchqa": ("lucadiliello/searchqa", "c1a979068ba118d85467179b704031d113d689cc", [
        ("train", "data/train-00000-of-00001-55e7116bea868a35.parquet"),
        ("validation", "data/validation-00000-of-00001-2092e81367c2ca98.parquet")]),
    "hotpotqa": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab", [
        ("train", "distractor/train-00000-of-00002.parquet"),
        ("train", "distractor/train-00001-of-00002.parquet"),
        ("validation", "distractor/validation-00000-of-00001.parquet")]),
}


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Project5-reproducible-data/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
                expected = response.headers.get("Content-Length")
                downloaded = 0
                started = time.monotonic()
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if time.monotonic() - started > 15:
                        print(f"  {destination.name}: {downloaded / 1e6:.1f} MB", flush=True)
                        started = time.monotonic()
                if expected and downloaded != int(expected):
                    raise OSError("Incomplete download")
            temporary.replace(destination)
            return
        except OSError:
            if attempt == 2:
                raise
    raise RuntimeError("Download failed")


def download(root, selected="all"):
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    for dataset, (repo, revision, files) in SOURCES.items():
        if selected not in {"all", dataset}:
            continue
        if (root / "prepared" / dataset / "manifest.json").exists():
            from baseline.data import load_data
            _, existing = load_data(root / "prepared" / dataset, dataset)
            if existing.get("download_provenance", {}).get("revision") != revision:
                raise ValueError("Existing prepared dataset has a different source revision")
            print(f"Already prepared and verified {dataset}: {existing['counts']}", flush=True)
            continue
        provenance, pools = [], {"train": [], "validation": []}
        for split, filename in files:
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
            local = root / "raw" / dataset / Path(filename).name
            print(f"Downloading {dataset} {split}: {local.name}", flush=True)
            fetch(url, local)
            hasher = hashlib.sha256()
            with local.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
            provenance.append({"url": url, "sha256": hasher.hexdigest(), "bytes": local.stat().st_size})
            # Iterate batches rather than expanding an entire corpus into Python objects.
            parquet = pq.ParquetFile(local)
            if dataset == "searchqa":
                allowed = set()
                for name in ["train", "val", "test"]:
                    allowed.update(str(r["id"]) for r in read_json(VENDOR / "data/searchqa_id_split" / name / "items.json"))
                for batch in parquet.iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        if str(row["key"]) in allowed:
                            row["metadata"] = {"public_source_split": split, "source_repo": repo, "revision": revision}
                            pools[split].append(row)
            else:
                # First read only IDs; choose before inspecting any question, answer or outcome.
                ids = parquet.read(columns=["id"]).column("id").to_pylist()
                pools[split].extend((str(item_id), str(local), index) for index, item_id in enumerate(ids))
        source_dir = root / "source_splits" / dataset
        source_dir.mkdir(parents=True, exist_ok=True)
        prepared = root / "prepared" / dataset
        if dataset == "searchqa":
            all_rows = pools["train"] + pools["validation"]
            if len({r["key"] for r in all_rows}) != len(all_rows):
                raise ValueError("Duplicate SearchQA IDs across public source splits")
            by_id = {r["key"]: r for r in all_rows}
            paths = {}
            for split, name in [("train", "train"), ("dev", "val"), ("test", "test")]:
                ids = [r["id"] for r in read_json(VENDOR / "data/searchqa_id_split" / name / "items.json")]
                missing = set(ids) - set(by_id)
                if missing:
                    raise ValueError(f"Missing {len(missing)} pinned {split} SearchQA IDs")
                paths[split] = source_dir / f"{split}.json"
                write_json(paths[split], [by_id[i] for i in ids])
            manifest = prepare(paths, prepared, dataset)
        else:
            import random
            paths = {"test": None}
            for split, source_split, count in [("train", "train", 100), ("dev", "validation", 30)]:
                chosen = random.Random(42).sample(sorted(pools[source_split]), count)
                wanted = {item[0] for item in chosen}
                selected_rows = {}
                for file in sorted({item[1] for item in chosen}):
                    for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                        for row in batch.to_pylist():
                            if row["id"] in wanted:
                                row["context"] = list(zip(row["context"]["title"], row["context"]["sentences"]))
                                row["metadata"] = {"public_source_split": source_split, "source_repo": repo,
                                                   "revision": revision, "course_equivalence": "unconfirmed"}
                                selected_rows[row["id"]] = row
                paths[split] = source_dir / f"{split}.json"
                write_json(paths[split], [selected_rows[item[0]] for item in chosen])
            manifest = prepare(paths, prepared, dataset)
            manifest["course_equivalence"] = "unconfirmed_public_distractor"
            manifest["test_pending_reason"] = "PDF specifies course-provided HotpotQA. Public test has no gold answers; supply course partitions before final evaluation."
        manifest["download_provenance"] = {"repo": repo, "revision": revision, "files": provenance}
        write_json(prepared / "manifest.json", manifest)
        print(f"Prepared {dataset}: {manifest['counts']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--dataset", choices=["all", "searchqa", "hotpotqa"], default="all")
    args = parser.parse_args()
    download(args.output, args.dataset)
