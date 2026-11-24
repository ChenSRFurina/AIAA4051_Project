"""Reproduce the 11.02 HotpotQA course split from pinned public distractor shards."""
import hashlib
import random
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from baseline.data import COURSE_HOTPOT, check_splits, load_data, normalize_item
from baseline.io import digest, read_json, write_json
from download_data import SOURCES, fetch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Baseline/data/raw/hotpotqa"
DATA = ROOT / "Baseline/data/prepared/hotpotqa_course"
RESULTS = ROOT / "results/hotpotqa_course"
SEED = 42
LEVELS = ("easy", "medium", "hard")
FILE_HASHES = {
    "train-00000-of-00002.parquet": "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    "train-00001-of-00002.parquet": "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
    "validation-00000-of-00001.parquet": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
}


def select_ids(pools, excluded):
    train, val = [], []
    rng = random.Random(SEED)
    for level in LEVELS:
        candidates = sorted(qid for qid, difficulty in pools["train"].items() if difficulty == level)
        block = rng.sample(candidates, 150)
        train.extend(block[:100])
        val.extend(block[100:])
    eligible = sorted(pools["validation"].keys() - set(excluded))
    test = rng.sample(eligible, 1000)
    return {"train_ids": train, "val_ids": val, "test_ids": test}


def prepare_course():
    repo, revision, shards = SOURCES["hotpotqa"]
    excluded_path = RESULTS / "excluded_old_dev_ids.json"
    historical = ROOT / "Baseline/data/prepared/hotpotqa/dev_ids.json"
    ids_archive = ROOT / "results/20261123/fixed_ids.json"
    candidates = []
    if excluded_path.exists():
        candidates.append(read_json(excluded_path))
    if historical.exists():
        candidates.append(read_json(historical))
    if ids_archive.exists():
        candidates.append(read_json(ids_archive)["datasets"]["hotpotqa"]["ids"]["dev"])
    if not candidates or len(set(candidates[0])) != 30 or any(set(c) != set(candidates[0]) for c in candidates):
        raise ValueError("The pinned lists of 30 previously seen dev IDs are missing or disagree")
    excluded = sorted(set(candidates[0]))
    RAW.mkdir(parents=True, exist_ok=True)
    pools = {"train": {}, "validation": {}}
    for partition, filename in shards:
        path = RAW / Path(filename).name
        fetch(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}", path)
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != FILE_HASHES[path.name]:
            raise ValueError(f"Source checksum mismatch for {path.name}")
        source = pq.ParquetFile(path)
        for row in source.read(columns=["id", "level"]).to_pylist():
            if row["id"] in pools[partition] or row["level"] not in LEVELS:
                raise ValueError("Unexpected source IDs/difficulty")
            pools[partition][row["id"]] = row["level"]
    if not set(excluded) <= pools["validation"].keys() or pools["train"].keys() & pools["validation"].keys():
        raise ValueError("Original source partitions or old dev exclusions failed verification")
    ids = select_ids(pools, excluded)
    keys = {"train": "train_ids", "dev": "val_ids", "test": "test_ids"}
    expected = {s: ids[k] for s, k in keys.items()}
    ids_path = RESULTS / "split_ids.json"
    if ids_path.exists() and read_json(ids_path) != ids:
        raise ValueError("The reproducible selection differs from the published split")
    if DATA.exists():
        splits, manifest = load_data(DATA, "hotpotqa")
        if any([r["id"] for r in splits[s]] != li for s, li in expected.items()):
            raise ValueError("Existing course split membership differs")
    else:
        wanted = set().union(*expected.values())
        rows = {}
        for _, filename in shards:
            source = pq.ParquetFile(RAW / Path(filename).name)
            # Exclude gold supporting facts and type even from the prepared records.
            for batch in source.iter_batches(batch_size=512, columns=["id", "question", "answer", "context", "level"]):
                for row in batch.to_pylist():
                    if row["id"] in wanted:
                        row["metadata"] = {"level": row["level"]}
                        rows[row["id"]] = normalize_item(row, "hotpotqa")
        splits = {s: [rows[i] for i in li] for s, li in expected.items()}
        duplicates = check_splits(splits, course_hotpot=True)
        if any(x["kind"] == "normalized_question_duplicate" for x in duplicates):
            raise ValueError("Duplicate question across selected partitions")
        manifest = {
            "protocol": "hotpotqa_distractor_course_11_02", "dataset": "hotpotqa", "demo": False,
            "seed": SEED, "source_repo": repo, "source_revision": revision,
            "source_version": "HotpotQA v1.1 train and v1 distractor dev", "source_sha256": FILE_HASHES,
            "source_counts": {k: len(v) for k, v in pools.items()}, "counts": COURSE_HOTPOT,
            "test_source": "official distractor development set", "test_pending": False,
            "excluded_old_dev_ids": excluded, "pilot_train_ids": ids["train_ids"][:10],
            "difficulty": {k: dict(Counter(r["metadata"]["level"] for r in v)) for k, v in splits.items()},
            "question_duplicates": duplicates,
            "sampling": "Python Random(42); easy/medium/hard in order; sample 150 sorted original train IDs per level, first 100 train, last 50 val; sample 1000 sorted official dev IDs after excluding old 30.",
        }
        DATA.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=DATA.parent) as temp:
            staged = Path(temp) / "prepared"
            for split, items in splits.items():
                write_json(staged / f"{split}.json", items)
                write_json(staged / f"{split}_ids.json", expected[split])
                manifest[f"{split}_hash"] = digest(items)
            write_json(staged / "manifest.json", manifest)
            load_data(staged, "hotpotqa")
            staged.rename(DATA)
    write_json(ids_path, ids)
    write_json(excluded_path, excluded)
    write_json(RESULTS / "split_manifest.json", manifest)
    print(f"Prepared {manifest['counts']}; excluded {len(excluded)} old dev IDs; test overlap {len(set(excluded) & set(ids['test_ids']))}", flush=True)


if __name__ == "__main__":
    prepare_course()

