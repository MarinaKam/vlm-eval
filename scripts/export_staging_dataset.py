"""TEMPLATE: export a sampled evaluation dataset from a Django application's database (read-only).

This is not a generic exporter. It is written against one particular Django schema and kept as a worked
example of how to produce the files the harness reads. To use it on your own application, set
VLM_EVAL_SOURCE_APP (and VLM_EVAL_SOURCE_MODELS if your class names differ) and adapt the queries to
your field names — or ignore it and write the files any other way; docs/datasets.md specifies them.

Easiest via `vlm-eval export` (scripts/run_export.py). To run it by hand, from the source application
with DATABASE_URL pointing at the database to read and VLM_EVAL_DATA_DIR set:

    .venv/bin/python manage.py shell < <vlm-eval>/scripts/export_staging_dataset.py

Writes to $VLM_EVAL_DATA_DIR:
  manifest.csv              image_id,url,s3_url,image_type,user_id,job_created_at
  reference_tags.jsonl      {"image_id", "image_type", "tags": {slug: conf>0}, "evaluable_slugs": [...]}
  reference_captions.jsonl  {"image_id", "captions": {slug: text}}
  properties.jsonl          one listing (a group of images) per line, for the multi-image summary task
  tags.json                 every tag row (slug, name, question_text, category, order, is_active)
  prompts.json              caption prompts + prompt templates + pipeline settings (chunk size etc.)
  features.json             feature rows (slug, type, is_active)

Only SELECTs. Tunables are environment variables, read just below.
"""

import csv
import functools
import importlib
import json
import os
import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

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

# ---------------------------------------------------------------- source schema (ADAPT THIS)
# Everything below is written against ONE Django schema. Two settings say where that schema lives; the
# queries still assume its field names and status values, so on a different schema edit the queries
# too, not only these names.
#
#   VLM_EVAL_SOURCE_APP     label of the Django app whose `models` module holds the classes below
#   VLM_EVAL_SOURCE_MODELS  optional `role=ClassName` pairs, comma-separated, for classes whose names
#                           differ from the placeholders in SOURCE_MODELS
_app = os.environ.get("VLM_EVAL_SOURCE_APP")
if not _app:
    raise SystemExit("VLM_EVAL_SOURCE_APP is not set — the label of the Django app that holds the source models")
SOURCE_MODELS = {
    "tag": "Tag",  # one yes/no question: slug, name, question_text, category, order, is_active
    "feature_kind": "FeatureKind",  # enum with CLASSIFICATION and CAPTION members
    "image": "Image",  # one stored image: id, url, s3_url
    "image_answer": "ImageAnswer",  # one stored answer per (image, feature): confidence, literal_value
    "feature": "Feature",  # what can be answered about an image: slug, type, is_active
    "image_job": "ImageJob",  # one processing job per image: status, created_at, user_id
    "setting": "PipelineSetting",  # key/value pipeline settings: key, value_int, value_json, ...
    "prompt": "Prompt",  # caption prompt texts, keyed by feature
    "prompt_template": "PromptTemplate",  # named prompt templates: slug, text, is_active
    "listing_job": "ListingJob",  # one listing (a group of images): its images, a summary text, status
}
SOURCE_MODELS.update(
    pair.strip().split("=", 1) for pair in os.environ.get("VLM_EVAL_SOURCE_MODELS", "").split(",") if "=" in pair
)
_source_models = importlib.import_module(f"{_app}.models")


def source(role: str):
    """The class playing `role` in the source app, or a message naming what to configure."""
    name = SOURCE_MODELS[role].strip()
    try:
        return getattr(_source_models, name)
    except AttributeError:
        raise SystemExit(
            f"{_app}.models has no {name!r} (role {role!r}). This script is a template for one schema: "
            f"set VLM_EVAL_SOURCE_MODELS={role}=<YourClass>, or adapt the queries."
        ) from None


Tag = source("tag")
FeatureKind = source("feature_kind")
Image = source("image")
ImageAnswer = source("image_answer")
Feature = source("feature")
ImageJob = source("image_job")
PipelineSetting = source("setting")
Prompt = source("prompt")
PromptTemplate = source("prompt_template")
ListingJob = source("listing_job")

# ---------------------------------------------------------------- config / prompts
all_tags = list(
    Tag.objects.order_by("category", "order", "slug").values(
        "slug", "name", "question_text", "category", "order", "is_active"
    )
)
active_tags = [t for t in all_tags if t["is_active"]]
(OUT_DIR / "tags.json").write_text(json_dumps(all_tags, indent=2, ensure_ascii=False))
print(
    f"tags.json: {len(all_tags)} classification tags ({len(active_tags)} active, "
    f"{len(all_tags) - len(active_tags)} inactive)"
)

# Only prompts whose feature is active: a database can also hold retired and placeholder prompts, and
# running those would measure nothing.
_active_prompts = Prompt.objects.select_related("key").filter(key__is_active=True)
caption_prompts = {p["key__slug"]: p["text"] for p in _active_prompts.values("key__slug", "text")}
templates = {t["slug"]: t["text"] for t in PromptTemplate.objects.filter(is_active=True).values("slug", "text")}
# The opening line of the caption prompt belongs to the pipeline, not to the harness. If the source
# application does not store it as a template, supply it through VLM_EVAL_CAPTION_HEADER so the export
# carries your pipeline's wording; without either, the harness falls back to a domain-free opening line.
_caption_header = os.environ.get("VLM_EVAL_CAPTION_HEADER")
if _caption_header:
    templates.setdefault("caption_header", _caption_header)
configs = {
    c["key"]: {k: v for k, v in c.items() if k != "key" and v is not None}
    for c in PipelineSetting.objects.filter(is_active=True).values(
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

features = list(Feature.objects.values("id", "slug", "type", "is_active", "show_in_classification"))
(OUT_DIR / "features.json").write_text(json_dumps(features, indent=2))
feature_by_id = {f["id"]: f for f in features}
classification_ids = {f["id"] for f in features if f["type"] == FeatureKind.CLASSIFICATION and f["is_active"]}
caption_ids = {f["id"] for f in features if f["type"] == FeatureKind.CAPTION}
type_slugs = {"indoor", "outdoor", "other", "split"}

# ---------------------------------------------------------------- candidate images
since = timezone.now() - timedelta(days=DAYS_BACK)
jobs = (
    ImageJob.objects.filter(status="completed", created_at__gte=since)
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

rows = ImageAnswer.objects.filter(image_id__in=image_ids, confidence__gt=0).values(
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

# Shuffle before writing. The selection above is built in blocks — tag-cover images first, then all
# indoor, then all outdoor — so any `--limit N` prefix of that order is not a sample of the dataset but
# a sample of one block. Measured on a real export: the first 280 rows were 98% indoor against 52% in
# the rest, and indoor images carry nearly twice as many questions. Order is not part of the data, so
# fixing it here costs nothing; a biased prefix costs a wrong conclusion.
rng.shuffle(selected)
print(f"selected: {len(selected)} images (order shuffled so any --limit prefix is representative)")

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
# stored classification slugs per image == the tags on which the reference actually gave a verdict.
evaluable = defaultdict(set)
for r in ImageAnswer.objects.filter(image_id__in=selected, feature_id__in=classification_ids).values(
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

cap_rows = ImageAnswer.objects.filter(image_id__in=selected, feature_id__in=caption_ids).values(
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

# ---------------------------------------------------------------- listings for the multi-image summary
props = (
    ListingJob.objects.filter(status="completed", created_at__gte=since)
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
print(f"properties.jsonl: {written} listings with >= {MIN_PROPERTY_IMAGES} images")
print(f"done -> {OUT_DIR}")
