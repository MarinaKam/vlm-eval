# Datasets

Nothing in the `vlm_eval` package knows about a particular database. The harness reads six files from
`data/` (or the directory named by `VLM_EVAL_DATA_DIR`), plus the images. `vlm-eval export` is one way
to produce them; writing them yourself works the same.

- [The six data files](#the-six-data-files)
- [Images](#images)
- [Optional files](#optional-files)
- [The export templates](#the-export-templates)
- [Privacy: what is safe to publish](#privacy-what-is-safe-to-publish)

## The six data files

### `manifest.csv`

Which images exist, one row per image.

```csv
image_id,url,s3_url,image_type,user_id,job_created_at
a1f043fd,https://example.com/photo.jpg,https://example.com/photo.jpg,outdoor,,2026-01-15T06:46:22+00:00
```

| Column | Used for |
|---|---|
| `image_id` | the file name under `data/images/` (`<image_id>.jpg`) and the key in every other file |
| `s3_url`, `url` | where `download` fetches the image: `s3_url` if it is set, otherwise `url` |
| `image_type` | which question set the image gets: `indoor` or `outdoor` |
| `user_id`, `job_created_at` | not read by the harness; the columns may be empty |

The harness reads `image_id`, `url`, `s3_url` and `image_type`; all four columns must be present.

### `tags.json`

The questions, exactly as your pipeline asks them.

```json
[
  {"slug": "bicycle", "name": "Bicycle",
   "question_text": "Is there a bicycle in the image?",
   "category": "common", "order": 0, "is_active": true}
]
```

`category` is `common`, `indoor` or `outdoor`. An image is asked the `common` questions plus those of
its own `image_type`, ordered by category, then `order`, then `slug`; that order decides which
questions share a batch. Entries with `"is_active": false` are skipped.

### `prompts.json`

Caption and summary prompts, and the settings your pipeline runs with.

```json
{
  "caption_prompts": {"base_caption": "Describe the image in one sentence."},
  "prompt_templates": {"caption_header": "...", "multi_image_summary": "..."},
  "processing_config": {
    "classification_chunk_size": {"value_int": 15},
    "individual_questions": {"value_json": ["bicycle"]},
    "image_optimization_enabled": {"value_bool": true},
    "image_optimization_max_dimension": {"value_int": 1536},
    "image_optimization_jpeg_quality": {"value_int": 85},
    "image_optimization_target_size_kb": {"value_int": 500}
  }
}
```

| Key | Meaning |
|---|---|
| `caption_prompts` | one instruction per caption key; the captions task asks for all of them in one call |
| `prompt_templates.caption_header` | the opening line of the caption prompt |
| `prompt_templates.multi_image_summary` | the prompt for the summary task |
| `processing_config.classification_chunk_size` | questions per call |
| `processing_config.individual_questions` | slugs that are asked in a call of their own |
| `processing_config.image_optimization_*` | how the images were prepared; recorded in each run's fingerprint |

Each `processing_config` value may be wrapped as shown (`value_int`, `value_float`, `value_bool`,
`value_json`, `value_text`) or given bare. The six settings are required: if one is missing, `run`
stops and names it unless `--allow-defaults` is passed. The other keys have fallbacks in the code: when
`caption_prompts` is empty, two generic caption prompts are used; when `caption_header` is missing, a
generic opening line is used; when `multi_image_summary` is missing, a built-in summary prompt is used.
That built-in prompt was written for one domain, so set your own.

### `reference_tags.jsonl`

What the candidate is compared with: one line per image.

```json
{"image_id": "a1f043fd", "image_type": "outdoor",
 "tags": {"bicycle": 0.87, "people": 0.83},
 "evaluable_slugs": ["bicycle", "people", "traffic_light"]}
```

`tags` maps a slug to the reference's confidence; a slug that is present counts as a yes, and the value
itself is not used for scoring. `evaluable_slugs` lists the questions the reference actually answered
for this image. A candidate is scored only on those: scoring it on a question the reference never
answered would turn every yes into a false positive. If `evaluable_slugs` is absent, the slugs in
`tags` are used.

### `reference_captions.jsonl`

The reference's free text, keyed by the same names as `caption_prompts`.

```json
{"image_id": "a1f043fd", "captions": {"base_caption": "A bicycle leaning against a brick wall."}}
```

### `properties.jsonl`

Listings for the summary task. A listing is a group of images that belong to one item; the task sends
all of a listing's images in one call and asks for one text. The file and field names are historical.

```json
{"property_job_id": "p1", "image_ids": ["a1f043fd", "b2e154ae"],
 "s3_urls": ["https://example.com/a.jpg", "https://example.com/b.jpg"],
 "property_summary": "The reference summary, if you have one."}
```

`property_job_id` and `image_ids` are required. `s3_urls`, parallel to `image_ids`, lets `download`
fetch images that are not in the manifest. `property_summary` is the reference text and is copied into
each summary row. At most the first 20 images of a listing are sent. If any of them is missing from
disk, the row is recorded as a failure and the model is not called.

### Older file names

`reference_tags.jsonl` and `reference_captions.jsonl` may instead be named `gemini_tags.jsonl` and
`gemini_captions.jsonl`, after the vendor early datasets were exported from. Both names are read, and
the `reference_*` name wins when both exist.

## Images

Put the images at `data/images/<image_id>.jpg`, or run `vlm-eval download`, which fetches each
manifest row and each listing image and stores the bytes exactly as served. Images are not re-encoded
on the way in, so what is on disk should already be what your pipeline sends to the reference.

## Optional files

| File | Used by | Content |
|---|---|---|
| `grounding_targets.json` | `run <model> grounding` (required for it) | `{"<slug>": "<phrase to locate>"}`; use slugs from `tags.json` so detections can be scored against the reference |
| `prototypes.json` | encoder strategies other than `question` and `name` | tag prompts for encoders; drafted by `vlm-eval prototypes` |
| `manual_labels.json` | `metrics`, reports | human verdicts, written by `vlm-eval review --decisions` |
| `economics.json` | `economics`, report projections | see [economics.md](economics.md) |
| `tags_from_migrations.json` | read when `tags.json` is absent | same format as `tags.json` |

## The export templates

The scripts that read a database are templates. They were written against one Django application's
schema and are kept as a worked example, not as a generic exporter. They read through the Django ORM,
and only read.

| Script | Behind | What it does |
|---|---|---|
| `scripts/export_staging_dataset.py` | `vlm-eval export` | a sampled, balanced dataset: every tag covered, then a fixed number of images per image type from a recent time window |
| `scripts/export_full_dataset.py` | run by hand | every image in the database, in the same file formats |
| `scripts/count_volume.py` | `vlm-eval volume` | monthly volume and burstiness, see [economics.md](economics.md) |
| `scripts/extract_tags_from_migrations.py` | run by hand | recovers seed question texts from migration files into `tags_from_migrations.json`; its lookup tables are empty placeholders to fill in |

Helper scripts that are not tied to a schema: `scripts/run_export.py` and
`scripts/run_source_manage.py` load the source application's `.env` and run a management command in
it; `scripts/make_cost_urls.py` builds the URL list for `vlm-eval cost`; `scripts/smoke_manifest.py`
builds a manifest from a folder of images; `scripts/build_prototypes.py` is behind `vlm-eval prototypes`.

### Settings

Set these in `.env` at the repository root (it is gitignored), or in the environment.

| Variable | Meaning |
|---|---|
| `VLM_EVAL_SOURCE_REPO` | path to the source Django application; its `.venv/bin/python`, `manage.py` and `.env` (with `DATABASE_URL`) are used |
| `VLM_EVAL_SOURCE_APP` | label of the Django app inside it whose `models` module the templates import |
| `VLM_EVAL_SOURCE_MODELS` | optional `role=ClassName` pairs, comma-separated, for classes named differently from the placeholders |
| `VLM_EVAL_CAPTION_HEADER` | optional opening line for the caption prompt, exported as `caption_header` when the application does not store one |
| `VLM_EVAL_DATA_DIR` | where the files are written (default `data/`) |
| `VLM_EVAL_DAYS_BACK` | sampled export: time window in days (default 180) |
| `VLM_EVAL_INDOOR`, `VLM_EVAL_OUTDOOR` | sampled export: target image count per image type (defaults 650 and 350) |
| `VLM_EVAL_PROPERTIES` | sampled export: number of listings to export (default 5) |
| `VLM_EVAL_CHUNK` | full export: rows per page (default 2000) |

The templates refer to the source models by role. The placeholder class names are:

| Role | Placeholder | What the template expects |
|---|---|---|
| `tag` | `Tag` | one question: `slug`, `name`, `question_text`, `category`, `order`, `is_active` |
| `feature` | `Feature` | something that can be answered about an image: `slug`, `type`, `is_active` |
| `feature_kind` | `FeatureKind` | an enum with `CLASSIFICATION` and `CAPTION` members |
| `image` | `Image` | a stored image: `id`, `url`, `s3_url` |
| `image_answer` | `ImageAnswer` | one stored answer per image and feature: `confidence`, `literal_value` |
| `image_job` | `ImageJob` | one processing job per image: `status`, `created_at`, `user_id` |
| `listing_job` | `ListingJob` | one listing: its images, a summary text, `status`, `created_at` |
| `setting` | `PipelineSetting` | key/value pipeline settings: `key`, `value_int`, `value_json`, ... |
| `prompt` | `Prompt` | caption prompt texts, keyed by feature |
| `prompt_template` | `PromptTemplate` | named templates: `slug`, `text`, `is_active` |

Renaming classes through `VLM_EVAL_SOURCE_MODELS` is enough only if your schema has the same shape. The
queries also assume field names, relations and status values (for example `status="completed"`), so on
a different schema you edit the queries. If that is more work than writing the six files directly,
write the files.

## Privacy: what is safe to publish

The code is generic; everything private stays in ignored places.

- `.env` holds your paths and tokens.
- `data/` holds your images, database exports, prompt texts and human verdicts.
- `runs/` and `reports/` are derived from them and contain image ids and model outputs.
- `models.local.json` and `encoders.local.json` hold your own presets.

All of these are in `.gitignore`, together with any `data-*/` directory, `manifest.csv`, `*.jsonl` and
`decisions_*.json` wherever they land. Before pushing, check that none of them is tracked:

```bash
git ls-files | grep -E "^(data|runs|reports)/|\.env$|models\.local\.json|decisions_.*\.json"
# must print nothing
```

Two things the ignore rules do not cover. Preset files that are tracked (`models.json`,
`encoders.json`) should contain only public model names. And the package has built-in fallback prompts
for datasets that carry none, so the texts of your own prompts belong in `data/prompts.json`, not in
code.
