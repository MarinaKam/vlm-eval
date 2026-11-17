# Usage

The workflow has three parts: build a dataset, run models against it, and turn the runs into reports.
Every step writes plain files and can be interrupted and resumed. Commands assume the virtual
environment is active (`source .venv/bin/activate`) and are run from the repository root.

- [Order of the steps](#order-of-the-steps)
- [1. Build the dataset](#1-build-the-dataset)
- [2. Run models](#2-run-models)
- [3. Metrics, review and reports](#3-metrics-review-and-reports)
- [4. Encoders](#4-encoders)
- [5. A paid API as the candidate](#5-a-paid-api-as-the-candidate)
- [What has been exercised](#what-has-been-exercised)

## Order of the steps

Each step reads what the previous one wrote. Where it can, a command that is run too early stops and
names the command to run first, but the order below is the one to follow.

```
export ─► download ─► run / sweep ─► metrics ─► report ─► compare
                           └─► review ─► metrics (again, to include the verdicts)
```

`vlm-eval status` prints what exists so far: the dataset files, the images on disk, the rows in every
run file, and the pipeline settings the harness will replay. `report` and `compare` print a note when a
run file is newer than the `metrics.json` they are reading.

## 1. Build the dataset

The harness reads its dataset from `data/` (or `VLM_EVAL_DATA_DIR`). The files and their formats are
specified in [datasets.md](datasets.md). There are two ways to produce them.

**From your own files.** Write the files by hand or with your own script. This is the route for any
stack that the export template does not match.

**From a Django application.**

```bash
cp .env.example .env      # set VLM_EVAL_SOURCE_REPO and VLM_EVAL_SOURCE_APP, see datasets.md
vlm-eval export
```

`export` runs a template script inside the source application. It prints the database host it is about
to read and issues only `SELECT`s. It is written against one schema; read
[datasets.md](datasets.md#the-export-templates) before pointing it at yours.

Then fetch the images:

```bash
vlm-eval download         # data/manifest.csv -> data/images/<image_id>.jpg
vlm-eval status
```

`download` stores the bytes exactly as served, so that every model sees the pixels the reference saw. It
also fetches the images of the listings in `data/properties.jsonl`, which are a separate set from the
manifest. A listing is a group of images that belong to one item; the summary task writes one text
per listing. A response that is not an image (an expired link returning an HTML page, for
example) is reported as a failed download instead of being saved. `--force` re-downloads files that
already exist.

To try the harness without any of this, `python scripts/smoke_manifest.py <folder> <indoor|outdoor> [limit]`
builds a manifest from a folder of images. The README quick start uses it.

### Pipeline settings come from the dataset

The batch size, the questions that are asked on their own, and the image settings are read from
`data/prompts.json` (`processing_config`), not from constants in the code. If a setting is missing,
`run` stops and names it. `--allow-defaults` accepts the built-in fallbacks and prints every value that
was guessed.

## 2. Run models

### Presets

`models.json` holds the connection details for each model, so commands name a preset:

```bash
vlm-eval run qwen3 tagging                # yes/no questions, in the pipeline's own batches
vlm-eval run qwen3 tagging --chunk 0      # every question in one call
vlm-eval run qwen3 tagging --repeats 3 --limit 100    # same images three times
vlm-eval run qwen3 captions
vlm-eval run qwen3 grounding              # bounding boxes; needs data/grounding_targets.json
vlm-eval run qwen3 summary                # one text per listing
```

Your own models go in `models.local.json` (copy `models.local.example.json`). That file is gitignored
and overrides same-named entries in `models.json`. `VLM_EVAL_MODELS` points at an additional preset
file kept elsewhere. A preset has these fields:

| Field | Meaning |
|---|---|
| `run_name` | folder under `runs/`, and the name used in reports |
| `served_name` | model name as the server exposes it |
| `base_url` | OpenAI-compatible endpoint (default `http://localhost:8000/v1`) |
| `flavor` | `vllm` (default) or `ollama` |
| `coords` | bounding-box convention for grounding: `norm1000` (default) or `abs` |
| `via` | `server` (default), `florence`, `internvl` or `paligemma` |
| `checkpoint` | Hugging Face id or local path, for the `via` values that load a checkpoint |
| `extra_output_tokens` | added to every task's output budget, for models that reason before answering |

`--served-name`, `--base-url`, `--flavor` and `--coords` override the preset on the command line. A
name that is not a preset is used as the run name directly, so a one-off server needs no preset:

```bash
vlm-eval run my-model tagging --served-name Qwen/Qwen3-VL-8B-Instruct \
    --base-url http://localhost:8000/v1 --flavor vllm
```

Other flags on `run`: `--limit N` (first N images), `--workers N` (concurrent requests),
`--no-logprobs`, `--chunk N` (questions per call; `0` means all in one call).

### Models without a server

These load a checkpoint in this process and need the `hf` extra (`uv pip install -e ".[dev,hf]"`):

```bash
vlm-eval florence captions --limit 300      # Florence-2: captions, grounding, tagging
vlm-eval hf internvl tagging --limit 150    # InternVL: all four tasks
vlm-eval hf paligemma tagging --limit 150   # PaliGemma 2: captions and tagging; gated, set HF_TOKEN
```

Florence-2 defaults to the `florence-community/Florence-2-large` checkpoint, InternVL to
`OpenGVLab/InternVL3_5-8B-HF`, PaliGemma to `google/paligemma2-3b-mix-448`; `--checkpoint` selects
another, `--device` forces a device, and `--model` sets the run folder name. A task that an architecture
cannot do is refused with an explanation: Florence-2 takes one image per pass, so it has no listing
summary; PaliGemma is single-image and single-turn, so it answers one question per call.

### Everything for one model

```bash
vlm-eval sweep qwen3                              # all tasks, default sample sizes
vlm-eval sweep qwen3 --tagging 1000 --captions 500
vlm-eval sweep qwen3 --captions 0 --grounding 0   # 0 skips a stage
vlm-eval sweep florence --via florence            # also: --via internvl, --via paligemma
```

For a served model the stages run cheapest first: summary, grounding, captions, all questions in one
call, tagging, and optionally a consistency stage (`--consistency N` runs N images three times).
Stopping early therefore leaves every capability measured on a smaller sample. The default sizes are
500 images for tagging, 100 for captions, 100 for grounding and 60 for the one-call comparison. `sweep`
finishes with `metrics`.

Two details of the current implementation: `sweep` runs its main tagging stage with 15 questions per
call rather than reading the batch size from the dataset, and it has no `--allow-defaults` flag. Use
`run` directly when either matters.

### Throughput

```bash
vlm-eval perf qwen3 --concurrency 4 --n 40
```

Runs the tagging task on `--n` images at the given concurrency and writes
`runs/<model>/perf_c<N>.json`. The machine and the concurrency are recorded with the result.

### Interrupting and resuming

Results are appended row by row. Re-running the same command continues where it stopped.

Resuming under different settings is refused. Each run file has a sidecar (`<file>.meta.json`) that
records what produced it: the rendered prompt and schema, a digest of the image bytes the task reads,
the model and its weights digest, the endpoint, the batch size and token budget, and a digest of the
code that builds requests and parses answers. If any of these change, the next run stops and lists the
differences; archive the file or use another run name. [design.md](design.md#provenance) explains the
reasoning and the special cases.

One consequence to know in advance: a plain vLLM or other OpenAI-compatible endpoint reports a model
name but no weights digest, so a fresh run works there, but resuming a file that already has rows is
refused. Ollama and in-process checkpoints report a digest and resume normally.

### Unfinished answers

A response with `finish_reason: "length"` is recorded as unknown for every answer it was meant to
contain, for all four tasks; the partial text is kept in the row but not parsed. Each row carries a
`completion` record with the number of calls, how many were truncated, how many failed, and a status of
`complete`, `truncated`, `failed` or `not_called`. Metrics read that record and report the share of
affected images per task.

## 3. Metrics, review and reports

```bash
vlm-eval metrics qwen3                # -> runs/<run_name>/metrics.json
```

Agreement with the reference overall and per tag (precision, recall, false-positive rate, unparsed
rate), consistency across repeats, latency, caption statistics, grounding statistics, completion per
task, and the provenance status of each run file.

### Judging the disagreements

```bash
vlm-eval review qwen3                              # writes reports/review/<run_name>.html
vlm-eval review qwen3 --decisions decisions_<run_name>.json
vlm-eval metrics qwen3                             # recompute, now including the verdicts
```

The reference is the current pipeline, not ground truth. Without this step the result is "the candidate
behaves differently"; with it, the result says who was right. The page shows a sample of disagreements
(`--per-tag`, default 5 per tag) with the image, the question and both answers. You mark each one
present, absent or unsure and download a decisions file. Verdicts are stored in
`data/manual_labels.json`, keyed by image and tag, so reviewing can be spread over several sittings.

### Reports

```bash
vlm-eval report qwen3                 # -> reports/<run_name>.md
vlm-eval compare qwen3 qwen2.5        # -> reports/comparison.md
vlm-eval pdf                          # -> reports/pdf/*.pdf, via headless Chrome
```

Facts that cannot be measured come from a card you write by hand, `reports/cards/<run_name>.json`:
`name`, `checkpoint`, `params`, `licence`, `serving`, `gpu`, `vram_gb`, `model_size_gb`,
`cold_start_s`, `verdict`, `verdict_reason`, and the free-text sections `hosting_md`, `finetune_md` and
`integration_md`. Without a card the report is still rendered with those fields blank. `compare` also
accepts a model that has a card but no run, and shows it with dashes and its verdict.

Speed and cost are reported per machine. The card names the machine the run was measured on, and may
add a projection for different hardware with an explicit source:

```json
{
  "measured_on": "laptop, Ollama, 4-bit build",
  "projection": {
    "hardware": "single cloud GPU",
    "images_per_hour": 2000,
    "usd_per_hour": 0.58,
    "source": "published benchmark, not measured here"
  }
}
```

The values above are placeholders. A cost per 1,000 images is derived only when a throughput and a
price are both known for the same machine. A projection may instead name an option from
`data/economics.json` (`"projection": {"option": "<name>"}`) so the price is written in one place.
Monthly totals and break-even volumes belong to [economics.md](economics.md).

## 4. Encoders

A generative model is asked questions and answers them. An image-text encoder (SigLIP 2, CLIP) cannot
be asked anything: it places the image and a text in one vector space and the two are compared. The
same dataset measures both. The commands from `embed` to `hybrid` load the checkpoint in this process
and need the `hf` extra; `embedding-report` only reads files.

```bash
vlm-eval prototypes                                     # draft data/prototypes.json, then edit it
vlm-eval embed siglip2-base                             # one vector per image, cached
vlm-eval similarity siglip2-base --strategy ensemble    # score the tags against the vectors
vlm-eval calibrate siglip2-base --strategy ensemble     # thresholds fitted out of fold
vlm-eval scale siglip2-base                             # what changes as the vocabulary grows
vlm-eval probe siglip2-base                             # a linear probe on the frozen vectors
vlm-eval hybrid siglip2-base                            # encoder first, a VLM for the borderline cases
vlm-eval embedding-report                               # -> reports/embedding-*.md
```

Checkpoints are named in `encoders.json` (`run_name`, `repo_id`, `licence`); your own go in
`encoders.local.json` or a file named by `VLM_EVAL_ENCODERS`. A bare Hugging Face repository id also
works in place of a preset. The text window, the embedding width and the trained decision boundary are
read from the checkpoint. The `smoke` preset writes to its own run folder, so
`vlm-eval embed smoke --limit 20` checks the chain without touching a real measurement;
`embedding-report` refuses to render it.

What differs from scoring a generative model:

- **A similarity is not an answer.** A threshold turns a score into a tag. `calibrate` fits thresholds
  under k-fold cross-validation grouped by image content (`--folds`, default 5), so no reported decision
  was judged by a threshold that had seen it. A tag with fewer than `--min-positives` positives
  (default 30) uses the global threshold instead of its own.
- **Some checkpoints ship a decision boundary and some do not.** SigLIP is trained with a sigmoid loss
  and carries its own scale and bias; `similarity` uses that boundary. CLIP scores are only meaningful
  relative to other candidates, so `similarity` records its answers as unknown until `calibrate` has run.
- **Tag definitions may not fit the text window.** A definition written for a model that follows
  instructions is often longer than an encoder reads, and the part that is cut is the end. Each command
  prints which definitions were truncated, and the list is recorded with the run.
- **Six ways to turn a tag into prompts** (`--strategy`): `question` (the question text verbatim),
  `name` (the tag name as a caption), `ensemble` (several paraphrases averaged into one vector),
  `prototypes-max` and `prototypes-mean` (several prompts kept apart and aggregated), and `pos-neg`
  (best positive minus best negative). All but the first two read `data/prototypes.json`.
- **Draft prompts are labelled.** `prototypes` derives prompts from tag names and marks each entry
  `"reviewed_by": "generator"`. Change it to `"human"` once you have read and edited the entry; the
  harness prints the breakdown and treats only `human` as reviewed. `--force` overwrites the file.
- **Duplicates are scored once.** Byte-identical images under different ids are collapsed to one for
  calibration, the probe, the hybrid curve and the reports. `--keep-duplicates` scores every row.
- **Two decision rules** (`similarity --rule`): `independent` thresholds each tag on its own; `softmax`
  normalises across the vocabulary. `scale` measures how each behaves as unrelated tags are added
  (`--sizes`, `--word-list`).
- **`hybrid` makes no API call.** It estimates how many calls remain if only the least confident
  decisions are deferred to a VLM, reading the deferred answers from the reference file.

## 5. A paid API as the candidate

The scoring is direction-neutral: candidates are compared with whatever is in the reference files. If
your pipeline runs a self-hosted model today and you are evaluating a hosted API, put your model's
answers in the reference files and treat the API as the candidate.

| You run today | You are evaluating | How |
|---|---|---|
| a paid API | open models you would host | reference files from the API; candidates through presets |
| a self-hosted model | a paid API | reference files from your model; the API as the candidate |
| model A | model B | reference files from A |

Practical points:

1. **Access.** The built-in server backend sends a placeholder bearer token and has no setting for an
   API key. An endpoint that requires a key needs a small backend class of its own
   ([design.md](design.md#adding-a-backend)), or a local proxy that adds the key.
2. **Rate limits.** A benchmark sends several calls per image. Check the provider's current limits
   against the number of images before starting a long run; free tiers are often too small.
3. **Data handling.** Check the provider's terms for how inputs are used before sending private images.
4. **Cost.** Measure the price per image on your own images before deciding; it depends on image size
   and the number of questions per call. See [economics.md](economics.md).
5. **Economics.** Add the API to `data/economics.json` as a `per_image` option and mark what you run
   today as `current`.

## What has been exercised

Not every path has been run against a real model. Tests and review do not catch a renamed field or a
server that behaves differently from its documentation, so this table says which is which.

| Path | Status | How to check it yourself |
|---|---|---|
| OpenAI-compatible server via Ollama | run end to end, all four tasks | — |
| Florence-2 via transformers | run end to end (captions, grounding, tagging) | — |
| Metrics, review, reports, economics | run on real data | — |
| Provenance gate and completion records | run end to end | start a run twice: the second reports everything as already done |
| SigLIP 2 and CLIP encoders | run end to end, all six strategies, calibrated | `vlm-eval embed smoke --limit 20`, then `similarity smoke` |
| Threshold calibration, probe, vocabulary growth | run on real scores; covered by tests | `pytest tests/test_calibration.py` |
| Dataset export | run against one Django schema only | on another schema it is a template, see [datasets.md](datasets.md) |
| vLLM server | not run; mocked in tests only | below |
| InternVL via transformers | not run; routing tested, backend not executed | `vlm-eval hf internvl captions --limit 2` (large download) |
| PaliGemma via transformers | not run; routing tested, backend not executed | accept the model licence, set `HF_TOKEN`, then `vlm-eval hf paligemma captions --limit 2` |

### Checking the vLLM path

vLLM is where schema-constrained decoding and token logprobs come from. Both are implemented from the
vLLM documentation and covered by mocks; neither has been run against a live server. vLLM needs an
NVIDIA GPU, so this cannot be checked on Apple silicon; [serving.md](serving.md) has a recipe for a
cloud VM. With a server running:

```bash
curl -s localhost:8000/health && vlm-eval run vllm-smoke tagging --limit 1 \
    --served-name <served-model-name> --base-url http://localhost:8000/v1 --flavor vllm
```

In `runs/vllm-smoke/tagging_chunk*.jsonl` the row should have `"errors": []`, an answer for every tag,
and a non-empty `confidence` map. An empty `confidence` map means logprobs did not come through: the
answers are still valid, but per-tag confidence is unavailable.
