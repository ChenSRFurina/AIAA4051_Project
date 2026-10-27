from pathlib import Path
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

PROJECT = Path(r"F:\Document\HKUST\2026fall\NLP\Project\AIAA4051_Project")
output = PROJECT / "Baseline/runs/searchqa" / ("pilot-qwen35-q4-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
metrics = PROJECT / "local_model/measurements"
metrics.mkdir(parents=True, exist_ok=True)
done = threading.Event()
samples = []


def monitor():
    with (metrics / (output.name + "-gpu.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utc", "gpu_memory_used_mib", "gpu_memory_total_mib", "gpu_utilization_percent"])
        while not done.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if result.returncode == 0:
                row = [int(v.strip()) for v in result.stdout.strip().splitlines()[0].split(",")]
                samples.append(row)
                writer.writerow([datetime.now(timezone.utc).isoformat(), *row])
                stream.flush()
            done.wait(1)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
started = time.monotonic()
try:
    result = subprocess.run([sys.executable, "-X", "utf8", "Baseline/run.py", "pilot", "--config", "Baseline/configs/searchqa.json",
                             "--data", "Baseline/data/prepared/searchqa", "--output", str(output)],
                            cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
finally:
    done.set()
    thread.join(timeout=10)
output.mkdir(parents=True, exist_ok=True)
(output / "console.stdout.log").write_text(result.stdout, encoding="utf-8")
(output / "console.stderr.log").write_text(result.stderr, encoding="utf-8")
profile = {"pilot_output": str(output), "exit_code": result.returncode, "wall_seconds": time.monotonic() - started,
           "gpu": "RTX 4070 Ti SUPER", "gpu_peak_used_mib_including_desktop": max((s[0] for s in samples), default=None),
           "gpu_total_mib": samples[0][1] if samples else None, "sample_count": len(samples),
           "service_remains_running_until_explicit_stop": True}
if result.returncode == 0:
    summary = json.loads((output / "pilot_summary.json").read_text(encoding="utf-8"))
    if summary["demo"] or len(summary["pilot_train_ids"]) != 10:
        raise RuntimeError("Not a genuine 10-ID pilot")
    for name, evaluation in summary["conditions"].items():
        if evaluation["n"] != 10 or evaluation["infrastructure_errors"]:
            raise RuntimeError(f"Incomplete/failed evaluation for {name}: {evaluation}")
    profile["summary"] = summary
(output / "hardware_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
(PROJECT / "local_model/latest_pilot.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
print(result.stdout)
print(result.stderr)
print(json.dumps(profile, ensure_ascii=False, indent=2))
if result.returncode:
    raise SystemExit(result.returncode)
