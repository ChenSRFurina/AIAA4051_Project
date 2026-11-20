"""Load credentials at runtime; never put them in experiment configurations."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def api_key(config):
    name = config["model"]["api_key_env"]
    if config.get("backend", "local") == "api":
        from dotenv import load_dotenv
        # Explicit path works from both entrypoints and from any working directory.
        # Existing environment variables take precedence over the private file.
        load_dotenv(REPO_ROOT / ".env", override=False, interpolate=False)
        key = os.environ.get(name)
        if not key or not key.strip():
            raise ValueError(f"Missing {name}; set it in the repository .env or process environment")
        return key.strip()
    return os.environ.get(name)
