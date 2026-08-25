# vlm-eval

A test bench for vision-language models. It takes the exact questions and prompts your production
pipeline sends to a paid API (Gemini, GPT-4V, …), asks the same questions to any open-weight model you
can run yourself, and tells you — in numbers — how far apart the answers are and whether self-hosting
would actually save money.

## The problem it solves

Say your product tags real-estate photos ("is there a kitchen island?", "is there a pool?" — dozens of
yes/no questions per image), writes captions, and summarises listings, all through a paid vision API.
Open models keep getting better and cheaper to host, so "should we switch?" comes up regularly. Poking a
demo is not an answer, and neither is a leaderboard: what matters is how a model behaves on *your*
images with *your* prompts, and what it would cost at *your* volume.

vlm-eval turns that into a repeatable experiment. Because it replays the production prompts verbatim, a
good score means "you could swap this in", not "it does well on some academic benchmark".

---

# Usage

The workflow has three parts: **build a dataset**, **run models against it**, **decide with numbers**.
Every step writes plain files, and every step can be interrupted and resumed.

## Order matters

Each step reads what the previous one wrote. Running them out of order does not fail loudly by itself —
it produces a confident report from nothing — so the tool refuses where it can and you should follow
this order regardless:

```
  export ──► download ──► run / sweep ──► metrics ──► report ────┐
     │                        │              │                   ├─► the markdown you send people
     │                        └─► review ────┘                   │
     │                                       └─► compare ────────┤
     ├─► volume ─┐                                               │
     └─► cost ───┴─► (fill data/economics.json) ─► economics ────┘
```

| command | needs | produces |
|---|---|---|
| `export` | access to your database | `data/manifest.csv`, tags, prompts, reference answers |
| `download` | `manifest.csv` | `data/images/` |
| `run` / `sweep` | images + tags + prompts | `runs/<model>/*.jsonl` |
| `metrics` | a run | `runs/<model>/metrics.json` |
| `review` | a run | HTML page → your verdicts → sharper metrics |
| `report` | `metrics.json` + `reports/cards/<model>.json` | `reports/<model>.md` |
| `compare` | `metrics.json` for each model | `reports/comparison.md` |
| `pdf` | any rendered report | `reports/pdf/*.pdf` — for attaching where Markdown is not read |
| `volume` | database access | images/month, busiest hour → into `economics.json` |
| `cost` | `manifest.csv` + your app's cost command | $/image → into `economics.json` |
| `economics` | **`volume` and `cost` done first** | `reports/economics.md` |

`vlm-eval status` shows where you are at any point. `economics` **refuses to run** while its inputs are
still the example values — the arithmetic is trivial, the measurements are the whole point, and a
plausible report built on placeholders is worse than no report. `--allow-unmeasured` overrides it and
stamps the warning into the file.

## 0. Setup (once)

```bash
uv venv --python 3.14 && uv pip install -e ".[dev]"   # add ".[hf]" for transformers-based models
cp .env.example .env                                   # edit: point VLM_EVAL_SOURCE_REPO at your app
source .venv/bin/activate                              # so `vlm-eval` works without the path prefix
.venv/bin/pytest                                       # 60 tests including end-to-end
```

`.env` holds everything machine-specific. It is gitignored, as are `data/`, `runs/` and `reports/`.

---

## Part 1 — Build the dataset

### Step 1.1 Export from your database

```bash
vlm-eval export
```

**Why:** the benchmark has to be your own images, and the reference answers have to be what your current
API actually said about them — you already paid for those. The script pulls both, plus the tag
questions, prompt texts and chunking config, so the harness asks models *exactly* what production asks.

**What you get** in `data/`: `manifest.csv` (which images, indoor/outdoor), `tags.json` (the questions),
`prompts.json` (caption/summary prompts), `reference_tags.jsonl` + `reference_captions.jsonl` (the
reference answers), `properties.jsonl` (image groups for multi-image summaries).

### Which side is the reference?

Whatever your pipeline runs **today**. A paid API, a model you already host, last quarter's checkpoint —
the tool has no opinion. Comparison, metrics, review and reports are direction-neutral: they score
candidates against the answers in `reference_tags.jsonl` and never ask where those came from.

| you run today | you are evaluating | works? |
|---|---|---|
| a paid API | open models you would host | yes — the case this was built for |
| a self-hosted model | a paid API | yes — put your model's answers in the reference file |
| model A | model B of the same class | yes |

`vlm-eval economics` is symmetric too: you list **options**, each billed either per image or per hour,
mark whichever you run today as `current`, and everything is compared against it. Moving off a paid API
and moving onto one are the same arithmetic read in opposite directions.

*Datasets exported earlier name these files `gemini_*.jsonl`, after the vendor they first came from.
Both names are read, the neutral one wins when both exist, and new exports write the neutral one — so
nothing needs renaming by hand.*

The script prints which database host it is about to read, and only reads. It is written for one
particular Django schema — on another stack, treat it as a template and produce the same files any way
you like. To try the harness without any export, `scripts/smoke_manifest.py <folder> indoor 5` builds a
minimal manifest from a folder of JPEGs.

### Step 1.2 Fetch the images

```bash
vlm-eval download
vlm-eval status      # what is measured so far, what is still missing
```

**Why:** every model must see the same pixels the reference API saw. Production uploads its images
already resized and compressed, so they are stored **byte-for-byte as served** — re-encoding them here
would add a second generation of JPEG loss and quietly change the comparison. (`reencode=True` exists
for a source that serves originals.) The command also fetches listing images for multi-image summaries,
which are a separate set from the sampled manifest.

`vlm-eval status` then prints the pipeline settings the harness will replay — batch size, which tags are
asked alone, image dimensions and quality — all read from the export, and flagged loudly if any had to
be guessed.

---

## Part 2 — Run models

Anything with an OpenAI-compatible API works: **Ollama** for a laptop, **vLLM** for a real GPU.

Models are named by **preset** — `models.json` holds the connection details, so commands stay short:

```bash
vlm-eval run qwen3 tagging              # the batched yes/no questions, production's own chunking
vlm-eval run qwen3 tagging --chunk 0    # every question in ONE call — see the cost section
vlm-eval run qwen3 tagging --repeats 3 --limit 100   # same image 3x: is the model consistent?
vlm-eval run qwen3 captions             # short + detailed descriptions
vlm-eval run qwen3 grounding            # bounding boxes for specific features
vlm-eval run qwen3 summary              # one description from all photos of a listing
```

### Your own models

Copy `models.local.example.json` to `models.local.json` and describe them there. That file is
gitignored, so your models never become changes to a tracked file and a `git pull` cannot take them
away; it also overrides same-named entries in `models.json`. `VLM_EVAL_MODELS` points at a third file
if you keep presets elsewhere (a shared one on a team machine, say).

```json
{
  "my-vllm-model": {"run_name": "qwen3-vl-8b-vllm", "served_name": "Qwen/Qwen3-VL-8B-Instruct",
                    "base_url": "http://localhost:8000/v1", "flavor": "vllm", "coords": "norm1000"},
  "my-finetune":   {"run_name": "internvl-ft-v3", "via": "internvl",
                    "checkpoint": "/path/to/your/checkpoint", "coords": "abs"}
}
```

A preset can carry `via` and `checkpoint`, so `vlm-eval sweep my-finetune` reaches a local checkpoint
without repeating those flags. Any field can be overridden with a flag, and a name that is not a preset
is used as-is — a one-off server needs no preset at all:

```bash
vlm-eval run my-model tagging --served-name Qwen/Qwen3-VL-8B-Instruct --base-url http://localhost:8000/v1 --flavor vllm
```

**A genuinely new architecture** — one that is neither an OpenAI-compatible API nor one of the
transformers backends already here — needs about thirty lines of code: a class with a `chat(images,
prompt, *, json_schema, max_tokens, temperature, logprobs) -> Response` method, next to the ones in
`vlm_eval/backends/`. Everything downstream works unchanged, because that is the only contract.

Useful flags: `--limit N` (subset), `--workers N` (concurrency, for a server), `--no-logprobs` (Ollama
doesn't expose them), `--coords abs|norm1000` (bbox convention differs per model family).

Models with no server run directly:

```bash
vlm-eval florence captions --limit 300      # Florence-2, task-token model
vlm-eval hf internvl tagging --limit 150    # ~17 GB download on first run
vlm-eval hf paligemma tagging --limit 150   # ~6 GB, gated repo: accept the licence, set HF_TOKEN
```

Florence-2 defaults to the `florence-community/*` port: the original `microsoft/*` repos ship custom
code that no longer runs on current transformers.

To run everything for one model, in one command:

```bash
vlm-eval sweep qwen3                             # all tasks, sensible sample sizes
vlm-eval sweep qwen3 --tagging 1000 --captions 500
vlm-eval sweep qwen3 --captions 0 --grounding 0  # skip stages with 0

vlm-eval sweep florence --via florence           # models with no server go through transformers
vlm-eval sweep internvl --via internvl
vlm-eval sweep paligemma --via paligemma
```

`--via` also knows what each architecture cannot do and skips it rather than failing: Florence-2 takes
one image at a time, so there is no property summary; PaliGemma is single-image and single-turn.

Stages run cheapest first — summary, grounding, captions, all-in-one-call, tagging — so stopping early
still leaves every capability measured; only the sample size shrinks. It finishes with `metrics` and
`status`.

**Interrupting is safe.** Results are appended row by row; re-running continues where it stopped.

**Resuming under different settings is refused, not merged.** `(image_id, repeat)` only proves the
work was *done*, never that it is still *valid*: raise a token budget and every old row still counts as
finished, the new rows land beside them, and one file quietly holds two experiments that average into a
single number. So each run file carries a sidecar recording everything that changes what an answer
means:

- **the request as actually rendered** — prompt text *and* JSON schema, so rewording the wrapper around
  unchanged questions counts as a different run;
- **the pixels the task actually reads** — a per-task digest of the image bytes, so replacing a
  listing-only image blocks the summary resume and leaves tagging resumable;
- **the structure** — tag categories and ordering (they decide chunk composition), each image's
  indoor/outdoor type (it decides the question set), which images belong to which listing in which
  order. Identical bytes and identical texts cannot vouch for any of these;
- **the machinery** — model, checkpoint, backend, batch size, token budget, logprobs, image encoding
  settings; the server route (`flavor@base_url` — two servers answering to one served name are two
  experiments); a digest of the answer-producing source (the runner, the task module, the backend —
  deliberately not a git SHA, so editing a report cannot refuse a resume but editing a parser must);
  for throughput runs, also the hardware and concurrency, without which the number is meaningless;
- **the weights themselves** — a served name like `qwen3-vl:8b` is a mutable tag: pull an update and
  the same name answers with a different model. Ollama's manifest digest and the HF commit are the
  immutable identities; a backend that cannot prove one is recorded as `unknown` and may run fresh
  files but never resume non-empty ones.

Any compatible model may be evaluated — your own Ollama build, any HF checkpoint, anything a server
exposes. Model identity does not restrict which model you can use; it prevents results from different
model weights from being silently combined in one run file. Updating a model between experiments is
fine — the updated weights are simply a new experiment, run under a new name or after archiving the old
file. The one backend caveat: a plain vLLM/OpenAI endpoint reports a model name but no weights digest,
so fresh runs work fully there while automatic resume of a non-empty file is refused.

Change any of it and the next run stops and names what changed; archive the file or pick another run
name. Every backend goes through the same gate — a served model, Florence-2, PaliGemma, a throughput
run — and a test walks the CLI's syntax tree to prove no path writes rows around it.

A file that existed before any of this was recorded is a third case, and stamping it with today's
settings would be the worst answer: its rows would look exactly as checked as rows that really were.
Instead it is labelled `legacy_unknown` permanently, with the count of unverified rows. The label prints
on every run that touches the file, travels into `metrics.json`, and appears in the report — publishing
it as a clean measurement stays a decision somebody makes on purpose, with the label in front of them.

**An answer the model did not finish counts as no answer.** A response with `finish_reason: "length"` —
common with reasoning models, which can spend the whole budget thinking and return nothing — records
unknown rather than parsing what arrived, for tags, captions, bounding boxes and summaries alike. The
tempting alternative is worse than it looks: a half-written JSON contributes real tags to accuracy, and
a truncated "found nothing" quietly agrees with the reference without ever seeing the image.

Each row carries a `completion` record (`calls`, `truncated`, `failed`) with a status of `complete` /
`truncated` / `failed` / `not_called` — an exception is not success. Metrics read that field, never the
wording of an error message, and report the share affected per task, so a run that measured less than
it looks says so itself.

---

## Part 3 — Turn runs into an answer

### Step 3.1 Compute metrics

```bash
vlm-eval metrics qwen3      # -> runs/<model>/metrics.json
```

Agreement with the reference per tag, precision/recall/false-positive rate, consistency across repeats,
latency, caption lengths, detection rates.

### Step 3.2 Judge the disagreements yourself

```bash
vlm-eval review qwen3                          # builds an HTML page
vlm-eval review qwen3 --decisions <file.json>  # merge your verdicts back in
```

**Why this matters more than any other step:** the reference is a paid API, not ground truth, and it is
wrong often. Without this you only learn "the candidate behaves differently"; with it you learn *who was
right*. In our own run the two systems were level in disputed cases (51% vs 49%) — a conclusion that the
raw agreement numbers pointed in the wrong direction on.

The page shows the image, the tag definition, and both answers; you pick present / absent / unsure and
download a decisions file. Verdicts are keyed to (image, tag), so you can review in several sittings.

### Step 3.3 Render reports

```bash
vlm-eval report qwen3               # reports/<model>.md
vlm-eval compare qwen3 qwen2.5      # reports/comparison.md
```

Capability and performance tables per model, plus one comparison table. Facts that cannot be measured
(licence, checkpoint, VRAM, hosting notes, verdict) come from `reports/cards/<model>.json`, which you
fill in by hand.

**Speed and cost are reported per machine, and never mixed.** A run on a laptop and a price for a cloud
GPU are numbers from different worlds; multiplying them once produced a figure eight times off in a
document that looked authoritative. So the card describes each world separately:

```json
{
  "measured_on": "Apple M4 Max, Ollama 4-bit",
  "projection": {
    "hardware": "NVIDIA L4 (GCP spot)",
    "images_per_hour": 2000,
    "usd_per_hour": 0.5832,
    "source": "published vLLM benchmarks — NOT measured by us"
  }
}
```

The report then shows measured throughput on the machine that produced the quality numbers, and — only
where a throughput and a price for the *same* machine are both known — a cost. A projection always
states where its throughput came from. Monthly totals, break-even volumes and fixed costs such as a
cluster fee stay in `economics.md`, which models always-on versus autoscaled capacity; a single
cost-per-hour figure cannot.

---

### Evaluating a paid API as the candidate

If your pipeline runs a local model today and you want to know whether a hosted one would be better,
everything above works unchanged — but you need working API access before the first run, and the
free tiers are usually too small for a benchmark. Concretely, for Gemini (checked 2026-08):

1. **Get a key.** Google AI Studio issues one without a credit card. The free tier allows commercial
   use, but Google may train on free-tier inputs and outputs — for client photos that alone is a reason
   to enable billing or use Vertex AI, which does not.
2. **Check the free tier will not stall the run.** It is rate-limited per minute *and* per day —
   Flash-Lite sits around 15 requests/minute and 1,000 requests/day. Our pipeline sends roughly 4 calls
   per image, so a free key covers about **250 images a day** and a 1,000-image benchmark takes four
   days. Fine for a smoke test, painful for the real thing.
3. **Enable billing if you want it to finish.** Expect to prepay a small amount (about $10 at the time
   of writing) or attach a billing account. Then measure what it actually costs — `vlm-eval cost` on
   60 images answers that for a couple of cents, before you commit to anything.
4. **Point the harness at it.** A hosted API is just another backend: add a preset with its
   OpenAI-compatible endpoint, or write a ~30-line backend class if it speaks its own protocol.
5. **Put its price in `economics.json`** as a `per_image` option, and mark your current local setup as
   `current`. The report then reads in the direction you actually care about.

Rate limits and free-tier sizes change; check the provider's current page rather than trusting this
list. The point that does not change: **measure the price on your own images before deciding**, because
cost per image depends on your image sizes and how many questions you ask per call, not on the
advertised per-token rate.

## Part 4 — The money question

Quality only decides *whether you can* switch. Whether you *should* is arithmetic, and it usually
surprises people: a paid API charges per image, a GPU charges per hour, and below a certain volume the
GPU idles most of the time and costs more.

### How much do you actually process?

```bash
vlm-eval volume
```

Images per month, busiest days and hours, and what fraction of hours have any work at all. That last
number decides whether a GPU can be scaled to zero between bursts.

The session is switched to read-only at the Postgres level first, so an accidental write fails with a
database error. Safe to point at production:

```bash
vlm-eval volume --db-from /path/to/prod.env
```

### Is the chunk size justified?

Production splits the questions into chunks and re-sends the image with each one, which is usually the
dominant cost. To find out what that costs and what one big call would cost instead:

```bash
vlm-eval cost --chunks 15 47 --images 60
```

`--chunks` is **questions per API call**, not a number of images (`--images` is that). The command
measures each size and then tells you what the cheaper one changed:

```
questions/call  API calls  input tokens    $/image
            15        4.0        10,639   0.001216
            47        2.0         5,621   0.000707  (-42%)

Does the answer change?
  15 vs 47: identical on 50% of images, 84.7% tag agreement (81 -> 76 tags)
      lost: air_conditioning x3, curtains x2, elevator x1
```

Halving the bill is not free if it drops tags. And part of any difference is simply the API answering
differently on a re-run — measure that baseline with the same size twice (`--chunks 15 --refresh`)
before blaming the batch size. Names your app's command via `VLM_EVAL_COST_COMMAND` in `.env`.

### Put it together: is switching worth it?

```bash
vlm-eval economics          # -> reports/economics.md
```

Reads `data/economics.json`: a list of **options**, each billed `per_image` or `per_hour`, with
whichever you use today marked `current`. It writes the whole argument — cost per year at each volume
scenario, what each alternative saves or costs against today, the volume at which that flips, and what
the busiest hour demands of per-hour capacity. Run it without the config and it prints a filled-in
example.

Per-hour options need `throughput_per_hour` (there is no other way to turn a monthly volume into hours,
and a made-up figure is worse than none) and may carry `fixed_monthly` — a cluster management fee, a
reserved disk, a persistent endpoint. That last one is easy to forget and, on a small workload, is often
most of the bill: a GKE cluster costs about **$73/month** in control-plane fees before a single pod
runs, which is why it is worth listing "a pod in the cluster we already pay for" and "a cluster stood up
for this" as two separate options.

The peak matters more than the average. A vendor absorbs a burst invisibly; your own hardware answers it
either by paying for idle capacity or by delaying the queue.

Do **not** assume the answer transfers between models: in our run the open model gave 98.9% identical
answers with all questions in one call, while the paid API dropped ~6% of its tags. Halving the bill
cost accuracy there. Measure it on the model you actually use.

---

## Helper scripts

| script | generic? | what it does |
|---|---|---|
| `run_source_manage.py` | ✅ fully | runs a management command in your app with its environment loaded; `--db-from` points at another deployment's database while keeping local library paths |
| `make_cost_urls.py` | ✅ fully | builds the URL list for token-cost measurement |
| `smoke_manifest.py` | ✅ fully | minimal dataset from a folder of JPEGs |
| `verify_published_figures.py` | ⚠️ project-specific | re-derives every number in `reports/` from the raw run files, using none of this package's code |
| `run_export.py` | ✅ wrapper | loads the app's `.env` without shell quoting problems and runs the export |
| `export_staging_dataset.py` | ⚠️ template | written against one Django schema — adapt to yours |
| `count_volume.py` | ⚠️ template | same: adapt the model and field names |
| `extract_tags_from_migrations.py` | ⚠️ template | fallback source for tag questions |

Model presets live in `models.json`; measured economics inputs in `data/economics.json`.

---

## Bring your own dataset

Nothing in `vlm_eval/` knows about any particular database. The harness reads six files from `data/`,
and `vlm-eval export` is only one way to produce them — a template written against one Django schema,
kept honest as a template rather than pretending to be generic. Write these six yourself and every
command works the same.

**`data/manifest.csv`** — which images exist. One row per image; `image_id` is the filename stem under
`data/images/`, `image_type` selects which question set the image gets.

```csv
image_id,url,s3_url,image_type,user_id,job_created_at
a1f043fd,https://example/photo.jpg,https://example/photo.jpg,indoor,,2026-08-13T06:46:22+00:00
```

**`data/tags.json`** — the questions, exactly as your pipeline asks them. `category` decides which
images get the question (`common` plus one of `indoor`/`outdoor`); `order` decides batch composition.

```json
[{"slug": "tennis_court", "name": "Tennis Court",
  "question_text": "Does this image contain a tennis court?",
  "category": "common", "order": 0, "is_active": true}]
```

**`data/prompts.json`** — caption and summary prompts, plus the settings your pipeline runs with. The
harness refuses to guess these: `classification_chunk_size` and `individual_questions` decide how
questions are batched, and the image settings decide what the model actually sees.

```json
{"caption_prompts": {"base_caption": "Describe this room in one sentence."},
 "prompt_templates": {"caption_header": "...", "multi_image_summary": "..."},
 "processing_config": {"classification_chunk_size": {"value_int": 15},
                       "individual_questions": {"value_json": ["utility_room"]},
                       "image_optimization_enabled": {"value_bool": true},
                       "image_optimization_max_dimension": {"value_int": 1536},
                       "image_optimization_jpeg_quality": {"value_int": 54},
                       "image_optimization_target_size_kb": {"value_int": 600}}}
```

**`data/reference_tags.jsonl`** — what you are comparing against: one line per image. `tags` maps a slug
to the reference's confidence (presence is what matters, the value is carried through). **`evaluable_slugs`
is the important field**: the slugs the reference actually judged on this image. Scoring a candidate on a
question the reference never answered manufactures false positives, so anything outside this list is
counted as not-comparable rather than wrong.

```json
{"image_id": "a1f043fd", "image_type": "indoor",
 "tags": {"kitchen": 0.87, "floorboards": 0.83},
 "evaluable_slugs": ["kitchen", "floorboards", "pool"]}
```

**`data/reference_captions.jsonl`** — the reference's free text, keyed by the same prompt names as
`caption_prompts`.

```json
{"image_id": "a1f043fd", "captions": {"base_caption": "A bright kitchen with wooden floors."}}
```

**`data/properties.jsonl`** — image groups for multi-image summaries. Only `property_job_id` and
`image_ids` are required; `property_summary` is the reference answer if you have one.

```json
{"property_job_id": "p1", "image_ids": ["a1f043fd", "b2e154ae"],
 "property_summary": "A three-bedroom terraced house..."}
```

Then put the images in `data/images/<image_id>.jpg` (or run `vlm-eval download`, which fetches `s3_url`
from the manifest and encodes them to the settings above), and `vlm-eval status` will tell you what is
still missing.

`reference_*.jsonl` may also be named `gemini_*.jsonl` — the older names are still read, and the
rename is deferred rather than done because it would break existing run files (`docs/BACKLOG.md`).

## The settings come from your export, never from this code

The claim is "your questions, your batches, your pixels". That only holds if the numbers are read
rather than assumed, so `data/prompts.json` carries production's own `processing_config` — batch size,
which tags are asked on their own, image dimensions, JPEG quality, target size — and the harness applies
those. A constant in this repository that happens to match production today is not evidence: change the
value in production and a hardcoded harness keeps measuring the old thing without saying so.

If the export is missing a setting, the run **stops** and tells you to re-export. `--allow-defaults`
overrides that knowingly, and then every guessed value is printed.

The same rule applies to domain content. Nothing about real estate lives in the code:

| what | where it comes from |
|---|---|
| tag questions | `data/tags.json` |
| caption prompts and their opening line | `data/prompts.json` (`caption_header`) |
| property-summary prompt | `data/prompts.json` (`multi_image_summary`) |
| what to localise | `data/grounding_targets.json` |
| batching, image settings | `data/prompts.json` (`processing_config`) |

Run the harness against someone else's `data/` and it asks their questions, not ours.

## Why the design is the way it is

- **The reference is your current API, not human labels.** Labelling 1000 images properly takes a team;
  exporting what you already paid for takes a minute. The trade-off is that "agreement" means "behaves
  like today", not "is correct" — which is what the review step is for.
- **Everything is a plain JSONL file.** Every number in a report traces back to the raw model output
  that produced it. No hidden state, no database.
- **Resume everywhere.** Runs take hours, laptops sleep, servers restart.
- **Real confidence where possible.** Served through vLLM, the harness reads token logprobs and turns
  them into an actual per-tag probability — something many API pipelines fake with a constant.
- **Refuse to measure nothing.** If a listing's images are missing from disk, the summary task records a
  failure instead of calling the model: given no images at all, a model will still write a fluent,
  entirely invented property description.

## Changing the tool

Three questions close every change; they are cheap and they are what actually goes wrong:

1. **Which branch?** — `git branch --show-current`. Never `main`. If the work builds on a branch that
   is not merged yet, branch from *that*, not from `main`.
2. **Does the README still describe reality?** — a new command, a changed flag or default, a deleted
   script: documentation is part of the change, not a follow-up.
3. **Do the end-to-end tests cover it?** — `tests/test_e2e.py` drives the real chain against a stub
   backend. A path exercised only by unit tests is untested where it matters. If you delete a script,
   prove its capability still exists (`sweep --via` exists because deleting `run_hf_models.sh` had
   quietly dropped three models).

Then:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check vlm_eval tests scripts
.venv/bin/ruff format --check vlm_eval tests scripts
```

Commands that reach a model or a database are verified by running them, not by their `--help`. Say
which ones were not run and why — "a 17 GB download" is a reason, "should work" is not.

And when a report has been written, re-derive it:

```bash
python scripts/verify_published_figures.py
```

It recomputes every published number straight from `runs/` and `data/` with an independent
implementation — importing nothing from this package, so a bug in `metrics.py` cannot confirm itself. A
MISMATCH means either the report is wrong or the data has moved since it was written; both matter
before someone forwards the document. It has already caught a report drifting from its data once.

## Privacy: what is safe to publish

The code is generic; everything private stays in ignored places. Before pushing:

```bash
git ls-files | grep -E "^(data|runs|reports)/|\.env$|models\.local\.json|decisions_.*\.json"
# must print nothing
```

`.env` holds your paths and tokens. `data/` holds client images, database exports and your prompt texts.
`runs/` and `reports/` are derived from them. Tag questions and prompts never appear in code.

## What has actually been exercised

Not every path here has been run against a real model. Code review and unit tests catch a lot, but they
do not catch a wrong header, a renamed field or a server that answers differently than its docs — so
this table says plainly which is which. Verify a row yourself before trusting a long run through it.

| path | status | how to check it yourself |
|---|---|---|
| OpenAI-compatible server via **Ollama** | run end to end, all four tasks | — |
| **Florence-2** via transformers | run end to end (captions, grounding, tagging) | — |
| Metrics, review, reports, economics | run on real data; every published figure re-derived independently | `python scripts/verify_published_figures.py` |
| Provenance gate + completion records | run end to end: a full sweep (765 images, 5 tasks) wrote verified sidecars and caught a wrong model variant | start any run twice, second must say `already done`; change `extra_output_tokens`, it must refuse |
| Dataset export | run against one Django schema only | on another schema it is a template — write the six files yourself, see [Bring your own dataset](#bring-your-own-dataset) |
| **vLLM** server | **not run** — mocked in tests only | needs an NVIDIA GPU; see below |
| **InternVL** via transformers | **not run** — routing tested, backend not executed | `vlm-eval hf internvl captions --limit 2` (~17 GB download on first run) |
| **PaliGemma** via transformers | **not run** — same | accept the Gemma licence, `export HF_TOKEN=…`, then `vlm-eval hf paligemma captions --limit 2` (~6 GB) |

### Checking the vLLM path

This is the gap worth closing first, because vLLM is where two things this tool advertises actually
come from: a response schema enforced by the decoder (malformed JSON becomes impossible, rather than
rare) and token logprobs (a real per-tag probability instead of a constant). Both are written from the
documentation and covered by mocks; neither has met a live server.

**It cannot be checked on Apple silicon.** The `vllm/vllm-openai` image is CUDA and x86-64; Docker
Desktop on a Mac has no GPU passthrough, so `--gpus all` has nothing to attach to. You need a machine
with an NVIDIA GPU — a cloud VM (`docs/INFRA.md` has a recipe) or any rented box. Ollama is the
Apple-silicon path and is fully exercised.

On that GPU machine, in **two terminals** — the server runs in the foreground:

```bash
# terminal 1 — the server, stays running
docker run --rm --gpus all -p 8000:8000 -v /opt/hf:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen2.5-VL-7B-Instruct --served-model-name qwen2.5-vl \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 20}'
```

```bash
# terminal 2 — wait for it to answer, then send one image
curl -s localhost:8000/health && vlm-eval run vllm-smoke tagging --limit 1 \
    --served-name qwen2.5-vl --base-url http://localhost:8000/v1 --flavor vllm
```

In `runs/vllm-smoke/tagging_chunk*.jsonl` the row should have `errors: []`, an answer for every tag, and
a non-empty `confidence` map — that last one is the proof logprobs came through. If `confidence` is
empty the numbers are still valid, but per-tag confidence is not available and the report should say so.

## Limitations, honestly

- Latency measured on a laptop is not production latency; `docs/INFRA.md` has a recipe for re-running
  the timing part on a cloud GPU. Quality numbers do not change.
- Ollama does not expose logprobs, so per-tag confidence is unavailable there; vLLM does, and the
  harness reads it, but that path has not been run against a live server. Ollama *does* enforce a
  response schema (>= 0.5 honours `response_format`), and the runs bear it out: zero unparsed answers
  across 30,912 and 15,136 tag decisions. An earlier version of this file blamed the serving stack for
  a small share of malformed JSON; that was wrong — the unparsed answers in question turned out to be
  truncation, and the wrong explanation survived here longer than in the report it came from.
- **A model tag is not a model.** Ollama's short tags can resolve to a variant you did not intend:
  `qwen3-vl:8b` is the *Thinking* checkpoint, whose Modelfile has no `$.Think` branch, so `think:false`
  is silently ignored and reasoning cannot be turned off. Measured on that build, 65% of free-text
  captions returned nothing but reasoning and it ran 4x slower than the instruct build for worse
  tagging. Name the variant explicitly; the fingerprint records the weights digest so the mistake
  cannot survive unnoticed into a second run, but only naming it right avoids the first one.
- **Free-text generation is where 4-bit builds break.** Structured tasks in the same sweep truncated
  zero times across 500 tagging images, 60 all-in-one-call images and 300 grounding calls, while 28% of
  free-text captions fell into a synonym repetition loop and generated until the token cap. A bigger
  budget cannot end an unbounded loop. Whether sampling defaults contribute is untested.
- Consistency at `temperature=0` is guaranteed by greedy decoding, so a perfect score proves
  reproducibility, not robustness. The informative version of that test is perturbation (different crop,
  rotation, compression), which this harness does not yet do.
- Cost estimates are "GPU-hours at measured throughput × hourly price"; real bills add idle time,
  autoscaling headroom and egress.

## License

MIT — see [LICENSE](LICENSE).

## Author

[Marie Kam](https://github.com/MarinaKam)
