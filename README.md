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
uv venv --python 3.12 && uv pip install -e ".[dev]"   # add ".[hf]" for transformers-based models
cp .env.example .env                                   # edit: point VLM_EVAL_SOURCE_REPO at your app
.venv/bin/pytest                                       # 29 tests, all should pass
```

`.env` holds everything machine-specific. It is gitignored, as are `data/`, `runs/` and `reports/`.

---

## Part 1 — Build the dataset

### Step 1.1 Export from your database

```bash
.venv/bin/python scripts/run_export.py
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
.venv/bin/vlm-eval download
```

**Why:** models are fed local files, resized and compressed exactly the way production does it, so every
model sees the same pixels the reference API saw. Also fetches the listing images needed for
multi-image summaries — those are a separate set from the sampled manifest.

---

## Part 2 — Run models

Anything with an OpenAI-compatible API works: **Ollama** for a laptop, **vLLM** for a real GPU.

```bash
# tagging: the batched yes/no questions, in production's own chunking
.venv/bin/vlm-eval run --model qwen3-vl-8b --served-name qwen3-vl:8b \
    --base-url http://localhost:11434/v1 --flavor ollama --task tagging

# same, but ask every question in ONE call — see "is the chunk size justified?" below
.venv/bin/vlm-eval run ... --task tagging --chunk 0

# same image three times — does the model answer consistently?
.venv/bin/vlm-eval run ... --task tagging --repeats 3 --limit 100

.venv/bin/vlm-eval run ... --task captions     # short + detailed descriptions
.venv/bin/vlm-eval run ... --task grounding    # bounding boxes for specific features
.venv/bin/vlm-eval run ... --task summary      # one description from all photos of a listing
```

Useful flags: `--limit N` (subset), `--workers N` (concurrency, for a server), `--no-logprobs` (Ollama
doesn't expose them), `--coords abs|norm1000` (bbox convention differs per model family).

Models with no server run directly:

```bash
.venv/bin/vlm-eval florence --task captions --limit 300          # Florence-2, task-token model
.venv/bin/vlm-eval hf --backend internvl  --task tagging --limit 150
.venv/bin/vlm-eval hf --backend paligemma --task tagging --limit 150   # gated repo: needs HF_TOKEN
```

Batch scripts for a full sweep: `scripts/run_all_local.sh` (Ollama models),
`scripts/run_hf_models.sh` (transformers models), `scripts/finish_minimal.sh` (a short top-up when you
need every capability measured but not at full sample size).

**Interrupting is safe.** Results are appended row by row; re-running continues where it stopped.

---

## Part 3 — Turn runs into an answer

### Step 3.1 Compute metrics

```bash
.venv/bin/vlm-eval metrics --model qwen3-vl-8b     # -> runs/<model>/metrics.json
```

Agreement with the reference per tag, precision/recall/false-positive rate, consistency across repeats,
latency, caption lengths, detection rates.

### Step 3.2 Judge the disagreements yourself

```bash
.venv/bin/vlm-eval review --model qwen3-vl-8b                         # builds an HTML page
.venv/bin/vlm-eval review --model qwen3-vl-8b --decisions <file.json> # merge your verdicts
```

**Why this matters more than any other step:** the reference is a paid API, not ground truth, and it is
wrong often. Without this you only learn "the candidate behaves differently"; with it you learn *who was
right*. In our own run the two systems were level in disputed cases (51% vs 49%) — a conclusion that the
raw agreement numbers pointed in the wrong direction on.

The page shows the image, the tag definition, and both answers; you pick present / absent / unsure and
download a decisions file. Verdicts are keyed to (image, tag), so you can review in several sittings.

### Step 3.3 Render reports

```bash
.venv/bin/vlm-eval report  --model qwen3-vl-8b       # reports/<model>.md
.venv/bin/vlm-eval compare --models a b c            # reports/comparison.md
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
.venv/bin/python scripts/run_source_manage.py shell --stdin scripts/count_volume.py
```

Images per month, busiest days and hours, and what fraction of hours have any work at all. That last
number decides whether a GPU can be scaled to zero between bursts.

The session is switched to read-only at the Postgres level first, so an accidental write fails with a
database error. Safe to point at production:

```bash
.venv/bin/python scripts/run_source_manage.py shell --db-from /path/to/prod.env --stdin scripts/count_volume.py
```

### Is the chunk size justified?

Production splits the questions into chunks and re-sends the image with each one, which is usually the
dominant cost. To find out what that costs and what one big call would cost instead:

```bash
.venv/bin/python scripts/make_cost_urls.py --type indoor --limit 60
# then run your app's token-measuring command at different chunk sizes and diff the tag sets
```

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

---

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
