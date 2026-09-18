"""Export the WHOLE CV dataset from the source app's DB (READ-ONLY), not the sampled 1000.

`export_staging_dataset.py` builds a balanced 1000-image benchmark: greedy tag cover, a 650/350 fill,
a 180-day window. This one does the opposite — every image the database has, with everything attached —
for standing up an independent copy of the corpus. Same file formats, so the harness reads either one.

    scripts/run_source_manage.py shell --db-from <staging.env> --stdin scripts/export_full_dataset.py

Writes to $VLM_EVAL_DATA_DIR (a fresh, empty directory is wise — this is the full corpus):
  manifest.csv          image_id,url,s3_url,image_type,size,is_processed,is_failed,user_id,created_at
  reference_tags.jsonl  {"image_id","image_type","tags":{slug:conf},"evaluable_slugs":[...]}
  reference_captions.jsonl  {"image_id","captions":{slug:text}}
  tags.json / prompts.json / features.json   the config catalogs, identical to the sampled export
  properties.jsonl      every completed property job with a summary (no cap)

Streams in chunks so 21k images and their ~1-2M feature rows never all sit in memory at once. Only
SELECTs; writes nothing back to the database.
"""

import csv
import functools
import json
import os
from collections import defaultdict
from pathlib import Path

from computer_vision.models import (
    ClassificationTag,
    FeatureType,
    Image,
    ImageFeature,
    ImageProcessingFeature,
    ProcessingConfig,
    Prompt,
    PromptTemplate,
    PropertyProcessingJob,
)

json_dumps = functools.partial(json.dumps, default=str)

_out = os.environ.get("VLM_EVAL_DATA_DIR")
if not _out:
    raise SystemExit("VLM_EVAL_DATA_DIR is not set — run this via scripts/run_source_manage.py")
OUT_DIR = Path(os.path.expanduser(_out))
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK = int(os.environ.get("VLM_EVAL_CHUNK", "2000"))

# ---------------------------------------------------------------- config catalogs (same as the sample)
all_tags = list(
    ClassificationTag.objects.order_by("category", "order", "slug").values(
        "slug", "name", "question_text", "category", "order", "is_active"
    )
)
active_tags = [t for t in all_tags if t["is_active"]]
(OUT_DIR / "tags.json").write_text(json_dumps(all_tags, indent=2, ensure_ascii=False))
print(f"tags.json: {len(all_tags)} classification tags ({len(active_tags)} active)")

_active_prompts = Prompt.objects.select_related("key").filter(key__is_active=True)
caption_prompts = {p["key__slug"]: p["text"] for p in _active_prompts.values("key__slug", "text")}
templates = {t["slug"]: t["text"] for t in PromptTemplate.objects.filter(is_active=True).values("slug", "text")}
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
classification_ids = {f["id"] for f in features if f["type"] == FeatureType.CLASSIFICATION and f["is_active"]}
caption_ids = {f["id"] for f in features if f["type"] == FeatureType.CAPTION}
cat_by_slug = {t["slug"]: t["category"] for t in active_tags}
type_slugs = {"indoor", "outdoor", "other", "split"}

# ---------------------------------------------------------------- every image, streamed in chunks
total = Image.objects.exclude(s3_url__isnull=True).exclude(s3_url="").count()
print(f"images with an s3_url: {total:,} — exporting all of them in chunks of {CHUNK}")

base = Image.objects.exclude(s3_url__isnull=True).exclude(s3_url="").order_by("id")
manifest = (OUT_DIR / "manifest.csv").open("w", newline="")
ref_tags = (OUT_DIR / "reference_tags.jsonl").open("w")
ref_caps = (OUT_DIR / "reference_captions.jsonl").open("w")
mw = csv.writer(manifest)
mw.writerow(["image_id", "url", "s3_url", "image_type", "size", "is_processed", "is_failed", "user_id", "created_at"])

written = with_tags = with_caps = 0
try:
    last_id = None
    while True:
        # Keyset paging by id, so the DB never sorts or offsets the whole 21k table per page.
        page = base.filter(id__gt=last_id) if last_id else base
        chunk = list(
            page.values("id", "url", "s3_url", "size", "is_processed", "is_failed", "user_id", "created_at")[:CHUNK]
        )
        if not chunk:
            break
        ids = [r["id"] for r in chunk]
        last_id = ids[-1]

        # Feature rows for just this chunk: tags (conf>0), the stored indoor/outdoor verdict, and captions.
        tags_by_image: dict = defaultdict(dict)
        evaluable: dict = defaultdict(set)
        caps: dict = defaultdict(dict)
        stored_type: dict = {}
        for r in ImageFeature.objects.filter(image_id__in=ids).values(
            "image_id", "feature_id", "feature__slug", "confidence", "literal_value"
        ):
            slug = r["feature__slug"]
            if slug in type_slugs:
                if slug in ("indoor", "outdoor") and r["image_id"] not in stored_type and r["confidence"]:
                    stored_type[r["image_id"]] = slug
                continue
            if r["feature_id"] in classification_ids:
                evaluable[r["image_id"]].add(slug)
                if r["confidence"] and r["confidence"] > 0:
                    tags_by_image[r["image_id"]][slug] = r["confidence"]
            elif r["feature_id"] in caption_ids and r["literal_value"]:
                caps[r["image_id"]][slug] = r["literal_value"]

        for r in chunk:
            iid = r["id"]
            itype = stored_type.get(iid)
            if not itype:  # fall back to tag categories, as the sampled export does
                cats = {cat_by_slug.get(s) for s in tags_by_image.get(iid, {})} - {None, "common"}
                itype = "indoor" if cats == {"indoor"} else "outdoor" if cats == {"outdoor"} else "unknown"
            mw.writerow(
                [
                    str(iid),
                    r["url"],
                    r["s3_url"],
                    itype,
                    r["size"],
                    r["is_processed"],
                    r["is_failed"],
                    str(r["user_id"]) if r["user_id"] else "",
                    r["created_at"].isoformat() if r["created_at"] else "",
                ]
            )
            ref_tags.write(
                json_dumps(
                    {
                        "image_id": str(iid),
                        "image_type": itype,
                        "tags": tags_by_image.get(iid, {}),
                        "evaluable_slugs": sorted(evaluable.get(iid, set()) - type_slugs),
                    }
                )
                + "\n"
            )
            ref_caps.write(json_dumps({"image_id": str(iid), "captions": caps.get(iid, {})}, ensure_ascii=False) + "\n")
            written += 1
            with_tags += bool(tags_by_image.get(iid))
            with_caps += bool(caps.get(iid))
        print(f"  {written:,}/{total:,}", flush=True)
finally:
    manifest.close()
    ref_tags.close()
    ref_caps.close()
print(f"manifest + references: {written:,} images ({with_tags:,} have tags, {with_caps:,} have captions)")

# ---------------------------------------------------------------- every property job with a summary
props = (
    PropertyProcessingJob.objects.filter(status="completed")
    .exclude(property_summary__isnull=True)
    .exclude(property_summary="")
    .order_by("-created_at")
)
written_props = 0
with (OUT_DIR / "properties.jsonl").open("w") as fh:
    for p in props.iterator(chunk_size=500):
        img_ids = list(p.images.values_list("image_id", flat=True))
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
        written_props += 1
print(f"properties.jsonl: {written_props:,} property jobs with a summary")
print(f"done -> {OUT_DIR}")
