# Backlog

Deferred deliberately, with the reason. Not a wish list.

## Generalise the database export beyond one Django schema

`scripts/export_staging_dataset.py`, `scripts/count_volume.py` and
`scripts/extract_tags_from_migrations.py` import concrete model classes (`ImageProcessingJob`,
`ClassificationTag`, …) from one application. Anyone on another schema — or another stack — has to
rewrite them.

**Why it is deferred:** an abstraction over "any backend's tag storage" would be guesswork, and a
half-generic exporter is more misleading than an honest template. The harness itself is already
decoupled: it reads six plain files from `data/`, documented in the README, and does not care how they
were produced.

**What would make it worth doing:** a second real schema to export from. Two concrete cases show which
parts actually vary; one case only produces a fake abstraction.

**Shape it might take:** a small mapping file (model name, field names, status values) consumed by a
generic exporter, with the current script kept as the reference implementation.

## Robustness to perturbation, not just repetition

`--repeats` measures whether a model answers the same way twice. At `temperature=0` that is guaranteed
by greedy decoding, so it proves the serving stack is reproducible, not that the model is stable.

The informative test is the same scene slightly changed: different crop, a few degrees of rotation,
heavier compression. That is where models diverge, and it predicts behaviour on the photos people
actually upload. The plumbing exists (`--repeats` plus a transform step in `dataset.optimize`).

## Rename the reference-model fields

`gemini_tags.jsonl`, `gemini_captions.jsonl`, the `gemini` argument in `metrics`/`review`, and the
`gemini_summary` field in summary rows all name one vendor in a tool meant for any of them.

**Why it is deferred:** renaming breaks existing run files and reports for a purely cosmetic gain. Worth
doing the first time someone evaluates against a different reference API — at that point the rename pays
for itself and a migration is justified anyway.
