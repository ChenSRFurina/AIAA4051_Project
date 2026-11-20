"""Course-size GraphSkillEvo reproduction, with an isolated feedback hook."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "GraphSkillEvo"
sys.path.insert(0, str(VENDOR))
UPSTREAM_COMMIT = "722a8a913b05dc9d1b4078f0bec604cde4dc7529"
