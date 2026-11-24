"""Run all course HotpotQA conditions with a shared initialization and one joint test freeze."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Baseline"), str(ROOT / "Explore")]
from baseline.budget import TokenCounter
from baseline.config import load_config
from baseline.data import load_data
from baseline.evaluation import qa_prompts
from baseline.io import digest, read_json, write_json
from baseline.protocol import freeze, freeze_design, check_design, test_frozen
from baseline.trainer import CourseTrainer, tree_fingerprint
from explore.feedback import ContrastiveFeedback
from explore_v2.feedback import DiagnosticFeedback
from explore_c2c import ClueFeedback
from prepare_hotpotqa_course import DATA, RESULTS

ARMS = ("N", "I", "B", "C", "C2b", "C2c")
FACTORIES = {"C": ContrastiveFeedback, "C2b": DiagnosticFeedback, "C2c": ClueFeedback}
SESSION = ROOT / "Explore/runs/local/hotpotqa_course_20261124"
CONFIG = ROOT / "Baseline/configs/hotpotqa_course_local.json"


def seal(path, record):
    if path.exists():
        if read_json(path) != record:
            raise ValueError(f"Existing experiment seal differs: {path}")
    else:
        write_json(path, record)


def policy(config, manifest):
    files = [Path(__file__), ROOT / "Baseline/prepare_hotpotqa_course.py", ROOT / "Explore/explore_c2c.py",
             *sorted((ROOT / "Explore/explore_v2").glob("*.py"))]
    return {"conditions": list(ARMS), "primary": "B versus original C; C2b and C2c are additional comparisons",
            "selection": "three rounds; two parents; validation EM and upstream stable tie handling",
            "train": "one fixed permutation of 300 train IDs, shared batches of 100 without replacement",
            "inference": "full distractor contexts; single-turn answer generation; deterministic decoding; test after simultaneous freeze",
            "budget": config["budget"], "config_hash": digest(config), "data_hash": digest(manifest),
            "protocol_hash": tree_fingerprint(),
            "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in files}}


def preflight(config, splits):
    counter = TokenCounter(config["model"]["tokenizer_path"])
    limit = config["model"]["context_window"]
    lengths = {}
    for split, rows in splits.items():
        upper = []
        for item in rows:
            system, user, visible = qa_prompts(item, "", config=config, counter=counter)
            if visible != item["context"] or not visible.startswith("[DOC]"):
                raise ValueError("A supplied context was truncated or not serialized")
            upper.append(counter.count(system) + counter.count(user))
        lengths[split] = max(upper)
    worst = max(lengths.values()) + config["skill_max_tokens"] + 32 + config["qa_max_tokens"] + config["model"]["context_safety_tokens"]
    if worst > limit:
        raise ValueError(f"Full context cannot fit in the frozen window: {worst} > {limit}")
    return {"max_unskilled_input_tokens": lengths, "worst_case_with_skill_completion_and_reserve": worst,
            "context_window": limit, "tokenizer_sha256": counter.identity}


def run(session):
    session = session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG)
    splits, manifest = load_data(DATA, "hotpotqa")
    design = policy(config, manifest)
    seal(session / "policy.json", design)
    seal(session / "preflight.json", preflight(config, splits))
    init = session / "shared_init"
    if not (init / "summary.json").exists():
        print("Starting shared initialization", flush=True)
        CourseTrainer(config, DATA, init, condition="shared_initialization").initialize()
    for arm in ARMS:
        dest = session / arm
        if arm == "C":
            if not (session / "feedback_design.json").exists():
                freeze_design(session / "B", session / "feedback_design.json", config)
            check_design(session / "feedback_design.json", config)
        if not (dest / "summary.json").exists():
            print(f"Starting {arm}", flush=True)
            factory = FACTORIES.get(arm)
            feedback = factory("hotpotqa", [i["id"] for i in splits["train"]], dest, config) if factory else None
            trainer = CourseTrainer(config, DATA, dest, condition=arm, feedback=feedback)
            if factory:
                write_json(dest / "feedback_design.json", {"policy_hash": digest(design), "arm": arm})
            if arm in ("N", "I"):
                trainer.static()
            else:
                trainer.train_course(init)
        summary = read_json(dest / "summary.json")
        if summary["status"] not in {"completed", "static_complete"} or summary["dev"]["n"] != 150:
            raise ValueError(f"{arm} incomplete; inspect the preserved stage: {summary['status']}")
        if arm not in ("N", "I"):
            batches = [read_json(dest / "train_batches" / f"{i}.json")["ids"] for i in range(1, 4)]
            if set().union(*batches) != {i["id"] for i in splits["train"]} or sum(map(len, batches)) != 300 or summary["completed_rounds"] != 3:
                raise ValueError(f"{arm} missed training IDs/rounds")
        print(f"{arm} completed validation n={summary['dev']['n']}", flush=True)
    seal(session / "policy.json", policy(config, manifest))
    if not (session / "frozen.json").exists():
        freeze({arm: session / arm for arm in ARMS}, DATA, session / "frozen.json")
    frozen = read_json(session / "frozen.json")
    print(f"Frozen all six arms: {frozen['freeze_id']}", flush=True)
    if not (session / "test/results.json").exists():
        test_frozen(session / "frozen.json", session / "test")
    report = read_json(session / "test/results.json")
    if report["freeze_id"] != frozen["freeze_id"] or set(report["conditions"]) != set(ARMS) or any(
        r["n"] != 1000 or r["infrastructure_errors"] for r in report["conditions"].values()
    ):
        raise ValueError("Test coverage did not pass verification")
    print("Verified all 6 x 1000 test results", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=SESSION)
    run(parser.parse_args().session)

