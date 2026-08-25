"""Run a management command in the source app with its own .env loaded (no shell quoting problems).

    python scripts/run_source_manage.py <command> [args...]

Reads VLM_EVAL_SOURCE_REPO from .env, loads that app's .env into the environment, prints which DB is
in use (host only, no credentials), then execs `manage.py <command> ...` there.

Whether a given command writes anything is the command's business — this wrapper only supplies the
environment. Check what you are running before you run it.
"""

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_eval.config import source_repo

SOURCE = source_repo()

if len(sys.argv) < 2:
    raise SystemExit(__doc__)


def _read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.replace("export ", "").strip()] = v.strip().strip('"').strip("'")
    return out


# `--env-file <path>` replaces the source repo's .env entirely.
# `--db-from <path>` keeps the local .env and takes ONLY DATABASE_URL from the other file — the right
# choice for pointing at another deployment's database: local library paths (GDAL/GEOS) stay intact and
# none of that deployment's other secrets enter the process.
env_path, db_from = SOURCE / ".env", None
for flag in ("--env-file", "--db-from"):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        target = Path(sys.argv[i + 1]).expanduser().resolve()
        if flag == "--env-file":
            env_path = target
        else:
            db_from = target
        del sys.argv[i : i + 2]

for k, v in _read_env(env_path).items():
    os.environ.setdefault(k, v)
if db_from:
    os.environ["DATABASE_URL"] = _read_env(db_from)["DATABASE_URL"]
    print(f"DATABASE_URL taken from {db_from.name}; the rest of the environment stays local", flush=True)


# The command runs with cwd=SOURCE, so a relative path typed here would resolve there instead.
# Anything that looks like a path (contains a separator and points at an existing file or directory)
# is made absolute first. Bare words like "indoor" are left alone.
def _abs(arg: str) -> str:
    if arg.startswith("-") or "/" not in arg:
        return arg
    candidate = Path(arg)
    if candidate.exists() or candidate.parent.is_dir():
        return str(candidate.resolve())
    return arg


argv = [_abs(a) for a in sys.argv[1:]]

db = os.environ.get("DATABASE_URL")
if db:
    u = urlparse(db)
    print(f"DB: host={u.hostname} db={u.path.lstrip('/')} user={u.username}", flush=True)
# `--stdin <file>` feeds a script to `manage.py shell` without shell redirection, which would
# otherwise resolve the path against the source repo after the cwd change.
stdin_file = None
if "--stdin" in argv:
    i = argv.index("--stdin")
    stdin_file = Path(argv[i + 1]).resolve()
    argv = argv[:i] + argv[i + 2 :]

print(f"$ manage.py {' '.join(argv)}" + (f" < {stdin_file}" if stdin_file else ""), flush=True)

handle = stdin_file.open() if stdin_file else None
try:
    sys.exit(subprocess.call([str(SOURCE / ".venv/bin/python"), "manage.py", *argv], cwd=SOURCE, stdin=handle))
finally:
    if handle:
        handle.close()
