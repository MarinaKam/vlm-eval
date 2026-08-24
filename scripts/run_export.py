"""Load the source app's .env (without shell interpretation) and run the dataset export via manage.py shell.

    <source-repo>/.venv/bin/python scripts/run_export.py

Needs VLM_EVAL_SOURCE_REPO (see .env.example). Prints which DB host is used (no password) before running.
READ-ONLY export.
"""

import os
import subprocess
import sys
import sys as _sys
from pathlib import Path
from urllib.parse import urlparse

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_eval.config import source_repo

SOURCE = source_repo()
EXPORT = Path(__file__).resolve().parent / "export_staging_dataset.py"
os.environ.setdefault("VLM_EVAL_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))

for line in (SOURCE / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.replace("export ", "").strip(), v.strip().strip('"').strip("'"))

u = urlparse(os.environ["DATABASE_URL"])
print(f"DB: host={u.hostname} db={u.path.lstrip('/')} user={u.username}", flush=True)
with EXPORT.open() as stdin:
    sys.exit(subprocess.call([str(SOURCE / ".venv/bin/python"), "manage.py", "shell"], cwd=SOURCE, stdin=stdin))
