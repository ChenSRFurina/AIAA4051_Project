"""Materialize supplied splits without changing their partition boundaries."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from . import VENDOR
from .io import digest, read_json, write_json

CAPS = {"train": 100, "dev": 30, "test": 50}


def raw_rows(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise ValueError("Expected a JSON array, {data: [...]} or JSONL")


def normalize_item(row, dataset):
    item_id = str(row.get("id", row.get("_id", row.get("key", ""))))
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", item_id) or item_id in {".", ".."} or ":" in item_id:
        raise ValueError(f"Unsafe dataset ID for prediction logs: {item_id!r}")
    question = str(row.get("question", "")).strip()
    answers = row.get("answers", row.get("answer"))
    if isinstance(answers, str):
        answers = [answers]
    if isinstance(answers, dict):
        answers = answers.get("text")
    if not item_id or not question or not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
        raise ValueError(f"Missing ID/question/scoring answers in record {item_id!r}")
    context = row.get("context", "")
    metadata = dict(row.get("metadata") or {})
    if dataset == "hotpotqa" and isinstance(context, dict) and "title" in context and "sentences" in context:
        context = list(zip(context["title"], context["sentences"]))
    if dataset == "hotpotqa" and isinstance(context, list):
        documents, sentences = [], []
        for title, content in context:
            if not isinstance(content, list):
                raise ValueError("HotpotQA context must contain [title, [sentences]]")
            documents.append(f"[DOC] {title}\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(content)))
            sentences.extend({"title": title, "sentence_index": i, "text": s} for i, s in enumerate(content))
        context = "\n".join(documents)
        metadata["sentences"] = sentences
    elif isinstance(context, list):
        context = "\n".join("[DOC] " + str(text) for text in context)
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"{item_id}: missing supplied context; retrieval tasks need an explicit adapter")
    # Do not use HotpotQA gold 'type' as observable task type.
    if "supporting_facts" in row:
        metadata["supporting_facts"] = row["supporting_facts"]
    return {"id": item_id, "dataset": dataset, "question": question, "context": context,
            "answers": answers, "metadata": metadata}


def check_splits(splits, *, test_pending=False):
    seen_ids, seen_questions, duplicates = {}, {}, []
    for split, rows in splits.items():
        if split == "test" and test_pending and not rows:
            continue
        if not rows or len(rows) > CAPS[split]:
            raise ValueError(f"{split}: expected 1..{CAPS[split]} records")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Repeated/cross-split ID: {row['id']}")
            seen_ids[row["id"]] = split
            normalized = " ".join(re.findall(r"\w+", row["question"].lower()))
            if normalized in seen_questions and seen_questions[normalized][0] != split:
                duplicates.append({"kind": "normalized_question_duplicate", "left": seen_questions[normalized], "right": [split, row["id"]]})
            seen_questions[normalized] = [split, row["id"]]
    # Record lexical near-duplicates without outcome-based filtering or resampling.
    groups = list(splits.items())
    for i, (left_split, left) in enumerate(groups):
        for right_split, right in groups[i + 1:]:
            for a in left:
                a_words = set(re.findall(r"\w+", a["question"].lower()))
                for b in right:
                    b_words = set(re.findall(r"\w+", b["question"].lower()))
                    similarity = len(a_words & b_words) / len(a_words | b_words) if a_words | b_words else 0
                    if .8 <= similarity < 1:
                        duplicates.append({"kind": "lexical_near_duplicate", "left": [left_split, a["id"]],
                                           "right": [right_split, b["id"]], "jaccard": similarity})
    return duplicates


def prepare(paths, output, dataset, seed=42, *, demo=False, sizes=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared destination must be empty; frozen IDs are never silently overwritten")
    sizes = sizes or CAPS
    splits, sources = {}, {}
    for split in CAPS:
        if split == "test" and not paths.get(split):
            splits[split], sources[split] = [], {"status": "course_test_pending"}
            continue
        rows = [normalize_item(row, dataset) for row in raw_rows(paths[split])]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in source {split}")
        if dataset == "searchqa" and not demo:
            name = "val" if split == "dev" else split
            allowed = {str(row["id"]) for row in read_json(VENDOR / "data" / "searchqa_id_split" / name / "items.json")}
            rows = [row for row in rows if row["id"] in allowed]
            if not rows:
                raise ValueError(f"{split}: no full records matching the pinned GraphSkillEvo SearchQA IDs")
        rows.sort(key=lambda row: row["id"])
        requested = sizes[split]
        if not 1 <= requested <= CAPS[split]:
            raise ValueError(f"Invalid {split} requested size")
        splits[split] = random.Random(seed).sample(rows, min(requested, len(rows)))
        import os
        sources[split] = {"path_relative_to_prepared": os.path.relpath(Path(paths[split]).resolve(), output.resolve()), "sha256": digest(Path(paths[split]).read_bytes()),
                          "available_in_partition": len(rows)}
    duplicate_report = check_splits(splits, test_pending=not splits["test"])
    manifest = {"dataset": dataset, "seed": seed, "demo": demo, "sources": sources,
                "counts": {k: len(v) for k, v in splits.items()}, "question_duplicates": duplicate_report,
                "near_duplicate_policy": "Normalized exact duplicates and cross-split token Jaccard>=0.8 are recorded without filtering. Semantic review remains manual.",
                "pilot_train_ids": [r["id"] for r in splits["train"][:10]], "test_pending": not splits["test"]}
    for split, rows in splits.items():
        write_json(output / f"{split}.json", rows)
        write_json(output / f"{split}_ids.json", [row["id"] for row in rows])
        manifest[f"{split}_hash"] = digest(rows)
    write_json(output / "manifest.json", manifest)
    return manifest


def load_data(folder, dataset, *, demo=False):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest["dataset"] != dataset or bool(manifest["demo"]) != bool(demo):
        raise ValueError("Dataset/demo provenance mismatch")
    splits = {split: read_json(folder / f"{split}.json") for split in CAPS}
    check_splits(splits, test_pending=manifest.get("test_pending", False))
    for split, rows in splits.items():
        if digest(rows) != manifest[f"{split}_hash"] or [r["id"] for r in rows] != read_json(folder / f"{split}_ids.json"):
            raise ValueError(f"Frozen {split} data/IDs changed")
    return splits, manifest
