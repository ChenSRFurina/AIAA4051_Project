"""Local SearchQA train/dev experiment with common initialization and fixed budgets."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Baseline"), str(ROOT / "Explore")]
from baseline.config import load_config
from baseline.io import read_json, write_json
from baseline.trainer import CourseTrainer, tree_fingerprint
from explore_v2.feedback import DiagnosticFeedback
from explore_c2c import ClueFeedback


def seal(config, session):
    sources = [*sorted((ROOT / "Explore/explore_v2").glob("*.py")), ROOT / "Explore/explore_c2c.py", Path(__file__)]
    return {"protocol": tree_fingerprint(), "config": config,
            "design": read_json(session / "design.json"),
            "files": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def run(session, arm):
    session = session.resolve()
    config = load_config(session / "config.json")
    policy = seal(config, session)
    if not (session / "policy.json").exists():
        write_json(session / "policy.json", policy)
    if read_json(session / "policy.json") != policy:
        raise ValueError("Experiment files or configuration differ from preregistration")
    data = ROOT / "Baseline/data/prepared/searchqa"
    if config["dataset"] != "searchqa" or config["backend"] != "local":
        raise ValueError("This comparison uses local SearchQA")
    factory = {"C2b": DiagnosticFeedback, "C2c": ClueFeedback}[arm]
    feedback = factory("searchqa", read_json(data / "train_ids.json"), session / arm, config)
    trainer = CourseTrainer(config, data, session / arm, condition=arm, feedback=feedback)
    result = trainer.train_course(session / "shared_init")
    if result["status"] != "completed" or result["completed_rounds"] != 3 or policy != seal(config, session):
        raise RuntimeError("Run incomplete or changed during execution")
    print(arm, result["dev"], result["cost"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("arm", choices=["C2b", "C2c"])
    args = parser.parse_args()
    run(args.session, args.arm)
