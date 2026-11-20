"""Run both API baselines sequentially with fresh outputs and per-stage logs."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_step(script, arguments, log_path):
    command = [sys.executable, "-u", str(ROOT / script), *map(str, arguments)]
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if returncode:
        raise RuntimeError(f"Stage failed with exit code {returncode}. See {log_path}")


def require_complete(folder, status):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != status:
        raise RuntimeError(f"Incomplete stage at {folder}: {summary['status']}")
    if status == "completed" and summary["completed_rounds"] != 3:
        raise RuntimeError(f"B did not complete three rounds at {folder}")
    if summary.get("dev", {}).get("infrastructure_errors", 0):
        raise RuntimeError(f"Unresolved dev infrastructure errors at {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="searchqa hotpotqa (default: both)")
    args = parser.parse_args()
    datasets = args.datasets or ["searchqa", "hotpotqa"]
    if any(dataset not in {"searchqa", "hotpotqa"} for dataset in datasets):
        parser.error("Datasets must be searchqa or hotpotqa")
    if len(set(datasets)) != len(datasets):
        parser.error("Each dataset may be specified only once")
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{os.getpid()}"
    for dataset in datasets:
        run = ROOT / "Baseline/runs/api" / dataset / run_tag
        run.mkdir(parents=True, exist_ok=False)
        (run / "execution_environment.json").write_text(json.dumps({
            "python": sys.executable,
            "api_source_ip": os.environ.get("PROJECT5_API_SOURCE_IP") or None,
            "network_note": "Optional process-local interface binding; model request payload is unchanged.",
            "started_local": datetime.now().astimezone().isoformat(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config = run / "config.json"
        shutil.copyfile(ROOT / "Baseline/configs" / f"{dataset}_api.json", config)
        data = ROOT / "Baseline/data/prepared" / dataset
        common = ["--config", config, "--data", data]
        print(f"\n[{dataset}] Output directory {run}", flush=True)
        stages = [
            ("download", "Baseline/download_data.py", ["--dataset", dataset], None),
            ("pilot", "Baseline/run.py", ["pilot", *common, "--output", run / "pilot"], None),
            ("shared_init", "Baseline/run.py",
             ["initialize", *common, "--output", run / "shared_init"], "shared_initialization_complete"),
            ("N", "Baseline/run.py",
             ["static", "--condition", "N", *common, "--output", run / "N"], "static_complete"),
            ("I", "Baseline/run.py",
             ["static", "--condition", "I", *common, "--output", run / "I"], "static_complete"),
            ("B", "Baseline/run.py",
             ["train", *common, "--initialization", run / "shared_init", "--output", run / "B"], "completed"),
        ]
        for name, script, arguments, status in stages:
            print(f"\n[{dataset}] Starting {name}", flush=True)
            run_step(script, arguments, run / f"{name}.log")
            if status is not None:
                require_complete(run / name, status)
        run_step("api/status.py", ["--run", run], run / "status.log")
        print(f"\n[{dataset}] Baseline completed at {run}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Baseline stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
