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

## 0. Setup (once)

```bash
uv venv --python 3.14 && uv pip install -e ".[dev]"   # add ".[hf]" for transformers-based models
cp .env.example .env                                   # edit: point VLM_EVAL_SOURCE_REPO at your app
source .venv/bin/activate                              # so `vlm-eval` works without the path prefix
.venv/bin/pytest                                       # 47 tests including end-to-end
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
`prompts.json` (caption/summary prompts), `gemini_tags.jsonl` + `gemini_captions.jsonl` (the reference
answers), `properties.jsonl` (image groups for multi-image summaries).

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

Add a preset to `models.json` (`run_name`, `served_name`, `flavor`, `base_url`, `coords`) and it works
everywhere. Any field can still be overridden with a flag, and a name that is not a preset is used as-is
— so a one-off vLLM server needs no preset:

```bash
vlm-eval run my-model tagging --served-name Qwen/Qwen3-VL-8B-Instruct --base-url http://localhost:8000/v1 --flavor vllm
```

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

---

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

Reads `data/economics.json` — the numbers you measured above (cost per image, GPU price and throughput,
volume scenarios, busiest hour) — and writes the whole argument: cost per scenario, the break-even
volume, a comparison of **hosting options** (dedicated VM, autoscaled VM, a pod in a cluster you already
run), what the busiest hour demands of self-hosted capacity, and a verdict. Run it without the config and
it prints a filled-in example to start from.

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
| `run_export.py` | ✅ wrapper | loads the app's `.env` without shell quoting problems and runs the export |
| `export_staging_dataset.py` | ⚠️ template | written against one Django schema — adapt to yours |
| `count_volume.py` | ⚠️ template | same: adapt the model and field names |
| `extract_tags_from_migrations.py` | ⚠️ template | fallback source for tag questions |

Model presets live in `models.json`; measured economics inputs in `data/economics.json`.

---

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

## Privacy: what is safe to publish

The code is generic; everything private stays in ignored places. Before pushing:

```bash
git ls-files | grep -E "^(data|runs|reports)/|\.env$|decisions_.*\.json"   # must print nothing
```

`.env` holds your paths and tokens. `data/` holds client images, database exports and your prompt texts.
`runs/` and `reports/` are derived from them. Tag questions and prompts never appear in code.

## Limitations, honestly

- Latency measured on a laptop is not production latency; `docs/INFRA.md` has a recipe for re-running
  the timing part on a cloud GPU. Quality numbers do not change.
- Ollama does not expose logprobs, and cannot enforce a response schema — a small share of malformed
  JSON there is a property of the serving stack, not of the model.
- Consistency at `temperature=0` is guaranteed by greedy decoding, so a perfect score proves
  reproducibility, not robustness. The informative version of that test is perturbation (different crop,
  rotation, compression), which this harness does not yet do.
- Cost estimates are "GPU-hours at measured throughput × hourly price"; real bills add idle time,
  autoscaling headroom and egress.

## License

MIT — see [LICENSE](LICENSE).

## Author

[Marie Kam](https://github.com/MarinaKam)
