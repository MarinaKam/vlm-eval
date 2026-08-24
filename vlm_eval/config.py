"""Central config. Everything machine- or company-specific comes from environment variables,
optionally loaded from a `.env` file in the repo root (KEY=VALUE lines, no shell quoting).

Variables:
  VLM_EVAL_SOURCE_REPO   path to the Django app the dataset is exported from (used by scripts/)
  VLM_EVAL_DATA_DIR      where the dataset lives          (default: <repo>/data)
  VLM_EVAL_RUNS_DIR      where run outputs are written    (default: <repo>/runs)
  VLM_EVAL_REPORT_DIR    where reports are rendered       (default: <repo>/reports)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _dir(var: str, default: Path) -> Path:
    return Path(os.path.expanduser(os.environ.get(var, str(default))))


DATA = _dir("VLM_EVAL_DATA_DIR", ROOT / "data")
RUNS = _dir("VLM_EVAL_RUNS_DIR", ROOT / "runs")
REPORTS = _dir("VLM_EVAL_REPORT_DIR", ROOT / "reports")


def source_repo() -> Path:
    """Path to the source Django app (for the export scripts). Required, no default."""
    val = os.environ.get("VLM_EVAL_SOURCE_REPO")
    if not val:
        raise SystemExit(
            "VLM_EVAL_SOURCE_REPO is not set. Put it in .env (see .env.example) — "
            "the path to the Django app you export the dataset from."
        )
    return Path(os.path.expanduser(val))
