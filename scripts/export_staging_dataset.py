"""Export an evaluation dataset from the source app's DB (READ-ONLY). Easiest via scripts/run_export.py.

Or run manually from the source repo with DATABASE_URL pointing at the right DB:

    .venv/bin/python manage.py shell < <vlm-eval>/scripts/export_staging_dataset.py

Writes to ~/PycharmProjects/vlm-eval/data/:
  manifest.csv          image_id,url,s3_url,image_type,user_id,job_created_at
  reference_tags.jsonl  {"image_id", "image_type", "tags": {slug: conf>0}, "evaluable_slugs": [...]}
  reference_captions.jsonl  {"image_id", "captions": {slug: text}}
  properties.jsonl      {"property_job_id", "image_ids": [...], "property_summary", "architectural_style"}
  tags.json             ALL ClassificationTag rows (slug, name, question_text, category, order, is_active)
  prompts.json          caption Prompt rows + PromptTemplate rows + ProcessingConfig (chunk_size etc.)
  features.json         ImageProcessingFeature rows (slug, type, is_active)

Only SELECTs. Tunables are at the top.
"""

import csv
import functools
import json
import os
import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from computer_vision.models import (
    ClassificationTag,
    FeatureType,
    Image,
    ImageFeature,
    ImageProcessingFeature,
    ImageProcessingJob,
    ProcessingConfig,
    Prompt,
    PromptTemplate,
    PropertyProcessingJob,
)
from django.utils import timezone

json_dumps = functools.partial(json.dumps, default=str)
# When piped through `manage.py shell`, __file__ is not this script — the data dir must come from env.
_out = os.environ.get("VLM_EVAL_DATA_DIR")
if not _out:
    raise SystemExit("VLM_EVAL_DATA_DIR is not set — run this via scripts/run_export.py")
OUT_DIR = Path(os.path.expanduser(_out))
DAYS_BACK = int(os.environ.get("VLM_EVAL_DAYS_BACK", "180"))
TARGET_INDOOR = int(os.environ.get("VLM_EVAL_INDOOR", "650"))
TARGET_OUTDOOR = int(os.environ.get("VLM_EVAL_OUTDOOR", "350"))
MAX_PROPERTIES = int(os.environ.get("VLM_EVAL_PROPERTIES", "5"))
MIN_PROPERTY_IMAGES = 10
SEED = 7104

OUT_DIR.mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)

# ---------------------------------------------------------------- config / prompts
all_tags = list(
    ClassificationTag.objects.order_by("category", "order", "slug").values(
        "slug", "name", "question_text", "category", "order", "is_active"
    )
)
active_tags = [t for t in all_tags if t["is_active"]]
(OUT_DIR / "tags.json").write_text(json_dumps(all_tags, indent=2, ensure_ascii=False))
print(
    f"tags.json: {len(all_tags)} classification tags ({len(active_tags)} active, "
    f"{len(all_tags) - len(active_tags)} inactive)"
)

# Only prompts whose feature is active: the DB also holds retired and placeholder prompts (in one
# deployment two still contained Lorem ipsum), and running those would measure nothing.
_active_prompts = Prompt.objects.select_related("key").filter(key__is_active=True)
caption_prompts = {p["key__slug"]: p["text"] for p in _active_prompts.values("key__slug", "text")}
templates = {t["slug"]: t["text"] for t in PromptTemplate.objects.filter(is_active=True).values("slug", "text")}
# The opening line of the caption prompt belongs to the pipeline, not to the harness. Export it so the
# tool has no domain wording of its own; adjust the source if yours lives elsewhere.
templates.setdefault("caption_header", "You are a helpful SEO expert in the real estate area.")
configs = {
    c["key"]: {k: v for k, v in c.items() if k != "key" and v is not None}
    for c in ProcessingConfig.objects.filter(is_active=True).values(
        "key", "value_text", "value_int", "value_float", "value_bool", "value_json"
    )
}
(OUT_DIR / "prompts.json").write_text(
    json_dumps(
        {"caption_prompts": caption_prompts, "prompt_templates": templates, "processing_config": configs},
        indent=2,
        ensure_ascii=False,
    )
)
print(f"prompts.json: {len(caption_prompts)} caption prompts, {len(templates)} templates, {len(configs)} configs")

features = list(ImageProcessingFeature.objects.values("id", "slug", "type", "is_active", "show_in_classification"))
(OUT_DIR / "features.json").write_text(json_dumps(features, indent=2))
feature_by_id = {f["id"]: f for f in features}
classification_ids = {f["id"] for f in features if f["type"] == FeatureType.CLASSIFICATION and f["is_active"]}
caption_ids = {f["id"] for f in features if f["type"] == FeatureType.CAPTION}
type_slugs = {"indoor", "outdoor", "other", "split"}

# ---------------------------------------------------------------- candidate images
since = timezone.now() - timedelta(days=DAYS_BACK)
jobs = (
    ImageProcessingJob.objects.filter(status="completed", created_at__gte=since)
    .exclude(image__s3_url__isnull=True)
    .exclude(image__s3_url="")
    .values("image_id", "created_at", "user_id")
    .order_by("-created_at")
)
job_by_image = {}
for j in jobs:
    job_by_image.setdefault(j["image_id"], j)
image_ids = list(job_by_image)
print(f"candidates: {len(image_ids)} completed images in last {DAYS_BACK} days")

rows = ImageFeature.objects.filter(image_id__in=image_ids, confidence__gt=0).values(
    "image_id", "feature_id", "feature__slug", "confidence"
)
tags_by_image = defaultdict(dict)
image_type = {}
for r in rows:
    slug = r["feature__slug"]
    if slug in type_slugs:
        if slug in ("indoor", "outdoor") and (r["image_id"] not in image_type):
            image_type[r["image_id"]] = slug
        continue
    if r["feature_id"] in classification_ids:
        tags_by_image[r["image_id"]][slug] = r["confidence"]

# Fallback: accounts without the indoor/outdoor features stored -> infer type from the
# category of the positive tags (indoor-only tag => indoor, outdoor-only => outdoor).
cat_by_slug = {t["slug"]: t["category"] for t in active_tags}
for i in image_ids:
    if i in image_type:
        continue
    cats = {cat_by_slug.get(s) for s in tags_by_image.get(i, {})} - {None, "common"}
    if cats == {"indoor"}:
        image_type[i] = "indoor"
    elif cats == {"outdoor"}:
        image_type[i] = "outdoor"

# Images whose type we cannot infer are skipped (other/split/unknown).
typed = [i for i in image_ids if i in image_type]
print(
    f"typed candidates: {len(typed)} (indoor={sum(1 for i in typed if image_type[i] == 'indoor')}, "
    f"outdoor={sum(1 for i in typed if image_type[i] == 'outdoor')})"
)

# ---------------------------------------------------------------- greedy tag cover, then stratified fill
selected, covered = [], set()
all_slugs = {t["slug"] for t in active_tags}
remaining = set(typed)
while remaining:
    best = max(remaining, key=lambda i: len(set(tags_by_image.get(i, {})) & all_slugs - covered))
    gain = set(tags_by_image.get(best, {})) & all_slugs - covered
    if not gain:
        break
    selected.append(best)
    covered |= gain
    remaining.discard(best)
print(
    f"greedy cover: {len(selected)} images cover {len(covered)}/{len(all_slugs)} tags; "
    f"uncovered={sorted(all_slugs - covered)}"
)


def _fill(kind, target):
    have = [i for i in selected if image_type[i] == kind]
    pool = [i for i in typed if image_type[i] == kind and i not in selected]
    rng.shuffle(pool)
    need = max(0, target - len(have))
    selected.extend(pool[:need])


_fill("indoor", TARGET_INDOOR)
_fill("outdoor", TARGET_OUTDOOR)
print(f"selected: {len(selected)} images")

images = {i.id: i for i in Image.objects.filter(id__in=selected)}
with (OUT_DIR / "manifest.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image_id", "url", "s3_url", "image_type", "user_id", "job_created_at"])
    for i in selected:
        img = images[i]
        w.writerow(
            [
                str(i),
                img.url,
                img.s3_url,
                image_type[i],
                str(job_by_image[i]["user_id"]),
                job_by_image[i]["created_at"].isoformat(),
            ]
        )

# Only slugs the account had active are stored at all (conf 0 for negatives), so the set of
# stored classification slugs per image == the tags on which Gemini actually gave a verdict.
evaluable = defaultdict(set)
for r in ImageFeature.objects.filter(image_id__in=selected, feature_id__in=classification_ids).values(
    "image_id", "feature__slug"
):
    evaluable[r["image_id"]].add(r["feature__slug"])

with (OUT_DIR / "reference_tags.jsonl").open("w") as fh:
    for i in selected:
        fh.write(
            json_dumps(
                {
                    "image_id": str(i),
                    "image_type": image_type[i],
                    "tags": tags_by_image.get(i, {}),
                    "evaluable_slugs": sorted(evaluable.get(i, set()) - type_slugs),
                }
            )
            + "\n"
        )

cap_rows = ImageFeature.objects.filter(image_id__in=selected, feature_id__in=caption_ids).values(
    "image_id", "feature__slug", "literal_value"
)
caps = defaultdict(dict)
for r in cap_rows:
    if r["literal_value"]:
        caps[r["image_id"]][r["feature__slug"]] = r["literal_value"]
with (OUT_DIR / "reference_captions.jsonl").open("w") as fh:
    for i in selected:
        fh.write(json_dumps({"image_id": str(i), "captions": caps.get(i, {})}, ensure_ascii=False) + "\n")
print(f"captions: {sum(1 for i in selected if caps.get(i))} images have captions")

# ---------------------------------------------------------------- property jobs for multi-image summary
props = (
    PropertyProcessingJob.objects.filter(status="completed", created_at__gte=since)
    .exclude(property_summary__isnull=True)
    .exclude(property_summary="")
    .order_by("-created_at")
)
written = 0
with (OUT_DIR / "properties.jsonl").open("w") as fh:
    for p in props:
        img_ids = list(p.images.values_list("image_id", flat=True))
        if len(img_ids) < MIN_PROPERTY_IMAGES:
            continue
        urls = dict(Image.objects.filter(id__in=img_ids).values_list("id", "s3_url"))
        fh.write(
            json_dumps(
                {
                    "property_job_id": str(p.id),
                    "property_id": p.property_id,
                    "image_ids": [str(i) for i in img_ids],
                    "s3_urls": [urls.get(i) for i in img_ids],
                    "property_summary": p.property_summary,
                    "architectural_style": p.architectural_style,
                    "created_at": p.created_at.isoformat(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        written += 1
        if written >= MAX_PROPERTIES:
            break
print(f"properties.jsonl: {written} property jobs with >= {MIN_PROPERTY_IMAGES} images")
print(f"done -> {OUT_DIR}")
