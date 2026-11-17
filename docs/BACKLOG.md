# Roadmap and known gaps

Work that has been deferred on purpose, with the reason, and gaps that are known. This is not a wish
list.

## Deferred

### A generic dataset export

The scripts behind `vlm-eval export` and `vlm-eval volume` are templates written against one Django
schema ([datasets.md](datasets.md#the-export-templates)). Anyone on another schema or stack adapts them
or writes the data files directly.

An abstraction over "any application's tag storage" built from a single example would be guesswork, and
a half-generic exporter is more misleading than an honest template. The harness itself is already
decoupled: it reads plain files and does not care how they were produced. A second real schema would
show which parts actually vary; the likely shape is a small mapping file consumed by a generic
exporter, with the current script kept as the reference implementation.

### Robustness to perturbation

`--repeats` measures whether a model answers the same way twice. At temperature 0 that is close to
guaranteed by greedy decoding, so it shows that the serving stack is reproducible, not that the model
is stable.

The informative test is the same scene slightly changed: a different crop, a few degrees of rotation,
heavier compression. Beyond perturbations, an acceptance set should include look-alike classes and
small objects inside a larger scene, which are the cases where an image-text encoder used as a cheap
first pass is weakest.

### Neutral names for the reference

Some names still refer to the vendor the first dataset came from: the older `gemini_*.jsonl` file
names (still read), the `gemini_summary` field in summary rows, `gemini_correct_pct` in the review
metrics, and a few labels in the review page and the model report. Renaming them breaks existing run
files and metrics, so it is deferred until a migration is justified for another reason.

### Neutral names and categories in the data model

The file `properties.jsonl` and its `property_job_id` and `property_summary` fields, and the image
types `indoor` and `outdoor`, come from the first dataset. Generalising them (arbitrary category names,
a neutral file name for listings) changes file formats and run-file fingerprints, so it is deferred for
the same reason. The built-in fallback summary prompt is also specific to that first dataset; a
dataset should always carry its own prompt in `data/prompts.json`.

## Known gaps

- **vLLM has not been run against a live server.** The request format, schema-constrained output and
  logprob parsing are covered by mocks only ([usage.md](usage.md#checking-the-vllm-path)).
- **InternVL and PaliGemma backends have not been executed.** Their routing is tested.
- **API keys.** The built-in server backend sends a placeholder bearer token, and no command-line
  option sets a real one.
- **`sweep` and the dataset's batch size.** `sweep` runs its tagging stage with 15 questions per call
  instead of reading the batch size from the dataset, and it cannot pass `--allow-defaults`.
- **`report` with a relocated report directory.** When `VLM_EVAL_REPORT_DIR` points outside the
  repository and the model has no card file, `report` fails while printing the note about the missing
  card. Creating the card file avoids it.
- **`vlm-eval verify` is project-specific.** It checks the author's own published figures and is a
  pattern to copy, not a general check ([design.md](design.md#independent-verification)).
- **Cost estimates are simple.** They multiply throughput by an hourly price and do not model idle
  time beyond always-on capacity, autoscaling headroom or egress.
