"""Fallback for data/tags.json when the DB export is not available yet.

Reads the seed question texts straight out of the source app's migrations (read-only).
The DB is the source of truth (admin-editable); this is the initial text only.

Usage (needs Django importable, so use the source app's venv):
    $VLM_EVAL_SOURCE_REPO/.venv/bin/python scripts/extract_tags_from_migrations.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_eval.config import source_repo  # noqa: E402

SOURCE = source_repo()
MIG = SOURCE / "computer_vision/migrations"
OUT = Path(__file__).resolve().parent.parent / "data/tags_from_migrations.json"


def _load(name: str):
    path = next(MIG.glob(f"{name}_*.py"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SOURCE))
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    m40 = _load("0040")
    tags = []
    for category, rows in (("indoor", m40.INDOOR_QUESTIONS), ("outdoor", m40.OUTDOOR_QUESTIONS),
                           ("common", m40.COMMON_QUESTIONS)):
        for order, (slug, name, question) in enumerate(rows):
            tags.append({"slug": slug, "name": name, "question_text": question, "category": category, "order": order})
    tags.append({"slug": "radiator", "name": "Radiator", "question_text": _load("0044").RADIATOR_QUESTION,
                 "category": "indoor", "order": 100})
    tags.append({"slug": "woods_view", "name": "Woods View", "question_text": _load("0047").WOODS_VIEW_QUESTION,
                 "category": "common", "order": 100})
    tags.append({"slug": "non_property_image", "name": "Non-property image",
                 "question_text": _load("0049").NON_PROPERTY_QUESTION, "category": "common", "order": 101})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tags, indent=2, ensure_ascii=False))
    by_cat = {}
    for t in tags:
        by_cat[t["category"]] = by_cat.get(t["category"], 0) + 1
    print(f"wrote {len(tags)} tags -> {OUT} {by_cat}")


if __name__ == "__main__":
    main()
