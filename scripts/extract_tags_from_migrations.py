"""TEMPLATE: fallback for data/tags.json when the database export is not available yet.

Reads seed question texts straight out of a Django application's migration files (read-only). The
database is the source of truth; this recovers the initial texts only, and writes them to
data/tags_from_migrations.json, which the harness reads when data/tags.json is absent.

This is not a generic tool. Which migration holds which constant is specific to one application, so the
SEED_LISTS and SEED_SINGLES tables below are placeholders: fill them in for your own migrations before
running. Settings:

    VLM_EVAL_SOURCE_REPO   path to the source application (see .env.example)
    VLM_EVAL_SOURCE_APP    label of the Django app whose migrations/ directory is read

Usage (needs Django importable, so use the source application's interpreter):

    $VLM_EVAL_SOURCE_REPO/.venv/bin/python scripts/extract_tags_from_migrations.py
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_eval.config import source_repo

OUT = Path(__file__).resolve().parent.parent / "data/tags_from_migrations.json"

# ---------------------------------------------------------------- ADAPT THIS
# (migration number prefix, constant name, category). The constant is a list of
# (slug, name, question_text) tuples; list position becomes the tag's `order`.
SEED_LISTS: list[tuple[str, str, str]] = [
    # ("0001", "SEED_QUESTIONS", "common"),
]
# (migration number prefix, constant name, slug, name, category, order). The constant is one question
# text, for tags added one at a time in later migrations.
SEED_SINGLES: list[tuple[str, str, str, str, str, int]] = [
    # ("0002", "BICYCLE_QUESTION", "bicycle", "Bicycle", "common", 100),
]


def _migrations_dir() -> Path:
    app = os.environ.get("VLM_EVAL_SOURCE_APP")
    if not app:
        raise SystemExit("VLM_EVAL_SOURCE_APP is not set — the label of the Django app whose migrations to read")
    return source_repo() / app.replace(".", "/") / "migrations"


def _load(migrations: Path, prefix: str):
    path = next(migrations.glob(f"{prefix}_*.py"), None)
    if path is None:
        raise SystemExit(f"no migration starting with {prefix}_ in {migrations}")
    spec = importlib.util.spec_from_file_location(prefix, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source_repo()))
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    if not SEED_LISTS and not SEED_SINGLES:
        raise SystemExit(
            "SEED_LISTS and SEED_SINGLES are empty. This script is a template: name the migrations and "
            "constants that hold your seed questions at the top of the file first."
        )
    migrations = _migrations_dir()
    tags = []
    for prefix, constant, category in SEED_LISTS:
        rows = getattr(_load(migrations, prefix), constant)
        for order, (slug, name, question) in enumerate(rows):
            tags.append({"slug": slug, "name": name, "question_text": question, "category": category, "order": order})
    for prefix, constant, slug, name, category, order in SEED_SINGLES:
        question = getattr(_load(migrations, prefix), constant)
        tags.append({"slug": slug, "name": name, "question_text": question, "category": category, "order": order})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
    by_cat = {}
    for t in tags:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    print(f"wrote {len(tags)} tags -> {OUT} {by_cat}")


if __name__ == "__main__":
    main()
