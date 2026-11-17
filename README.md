# vlm-eval

A test bench for vision-language models. It replays the exact prompts a production pipeline sends to a
paid vision API against any open-weight model you can run yourself, and reports how far the answers
agree, how stable they are, how long they take, and what hosting would cost at your volume.

## The problem

Suppose a product sends images to a paid vision API: a batch of yes/no tagging questions per image
("is there a bicycle in the image?", "is the photo taken outdoors?"), a couple of captions, and a
summary written from a listing (a group of images that belong to one item). Open models keep improving
and getting cheaper to host, so the question "should we switch?" comes back regularly.

A demo does not answer it, and neither does a public leaderboard. What matters is how a candidate
behaves on your images with your prompts, and what it costs at your volume. vlm-eval turns that into a
repeatable experiment: it asks a candidate model the same questions, in the same batches, on the same
image bytes, and scores the answers against what the current pipeline said.

## How it works

Every step reads plain files written by the previous one and can be interrupted and resumed.

```
 dataset                    runs                        reports

 export ─► download ─► run / sweep ─► metrics ─► report ─► compare
                            │             ▲
                            └─► review ───┘       (human verdicts on disagreements)

 volume ─┐
 cost ───┴─► data/economics.json ─► economics     (what each hosting option costs)

 prototypes ─► embed ─► similarity ─► calibrate ─► scale / probe / hybrid ─► embedding-report
                                                  (image-text encoders, scored on the same dataset)
```

- **Dataset.** Six files in `data/` describe the images, the questions, the prompts and the reference
  answers. `vlm-eval export` is one way to produce them; you can also write them yourself
  ([docs/datasets.md](docs/datasets.md)).
- **Runs.** A model is reached through an OpenAI-compatible server (Ollama, vLLM) or loaded in-process
  through `transformers`. Each task appends one JSON line per image to `runs/<model>/`.
- **Reports.** Metrics are computed from the run files, and Markdown reports are rendered from the
  metrics. Every figure traces back to a raw model output on disk.

## Requirements

- Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/) (the commands below use it).
- A model to evaluate: an OpenAI-compatible server such as [Ollama](https://ollama.com) or
  [vLLM](https://docs.vllm.ai), or a Hugging Face checkpoint run in-process (needs the `hf` extra).
- Chrome, Chromium or Edge, only for `vlm-eval pdf`.

## Install

```bash
git clone https://github.com/MarinaKam/vlm-eval.git && cd vlm-eval
uv venv --python 3.12
uv pip install -e ".[dev]"        # add ".[dev,hf]" to run models or encoders in this process
source .venv/bin/activate         # so `vlm-eval` is on the PATH
pytest -q                         # offline: backends are mocked, no weights or dataset needed
```

The base install covers everything that talks to a server, plus metrics and reports. The `hf` extra adds
torch and transformers and is needed by the commands that load a checkpoint in this process: `florence`,
`hf`, `sweep --via ...`, and the encoder commands from `embed` through `hybrid`.

## Quick start

This runs one task end to end on a handful of your own images, without a database export. It assumes
Ollama is serving the `qwen2.5` preset from [`models.json`](models.json) (`ollama pull qwen2.5vl:7b`).

```bash
# 1. A minimal dataset: five images from a folder, plus the questions to ask.
#    This writes data/manifest.csv and data/images/, replacing an existing manifest.
python scripts/smoke_manifest.py ~/Pictures/sample outdoor 5

cat > data/tags.json <<'EOF'
[
  {"slug": "bicycle", "name": "Bicycle", "question_text": "Is there a bicycle in the image?",
   "category": "common", "order": 0, "is_active": true},
  {"slug": "people", "name": "People", "question_text": "Are there people in the image?",
   "category": "common", "order": 1, "is_active": true}
]
EOF

# 2. Check what the harness sees.
vlm-eval status

# 3. Ask the model. --allow-defaults is needed here because this dataset carries no pipeline
#    settings; a real export does, and then the flag is not used.
vlm-eval run qwen2.5 tagging --allow-defaults

# 4. Turn the run into numbers and a report.
vlm-eval metrics qwen2.5          # -> runs/qwen2.5vl-7b-ollama/metrics.json
vlm-eval report qwen2.5           # -> reports/qwen2.5vl-7b-ollama.md
```

A dataset built this way has no reference answers, so the report shows latency, completion and
provenance but no agreement figures. For those, supply the reference files
([docs/datasets.md](docs/datasets.md)) and follow the full workflow in [docs/usage.md](docs/usage.md).

## Commands

Models are named by preset (`models.json`, `encoders.json`), so commands stay short. `vlm-eval status`
shows what has been measured and what is missing at any point.

| Command | Needs | Produces |
|---|---|---|
| `export` | a source application (template, see [datasets](docs/datasets.md)) | the dataset files in `data/` |
| `download` | `data/manifest.csv` | `data/images/` |
| `run <model> <task>` | images, tags, prompts | `runs/<model>/<task>*.jsonl` (`tagging`, `captions`, `grounding`, `summary`) |
| `sweep <model>` | the same | every supported task, cheapest first, then `metrics` |
| `florence`, `hf` | the same, plus the `hf` extra | the same run files, from a checkpoint loaded in-process |
| `perf <model>` | images, tags | `runs/<model>/perf_c<N>.json`, throughput at a given concurrency |
| `metrics <model>` | a run | `runs/<model>/metrics.json` |
| `review <model>` | a tagging run, reference tags | an HTML page for judging disagreements; merges verdicts back |
| `report <model>` | `metrics.json` | `reports/<model>.md` |
| `compare <models...>` | `metrics.json` or a card per model | `reports/comparison.md` |
| `prototypes` | `data/tags.json` | `data/prototypes.json`, draft tag prompts for encoders |
| `embed`, `similarity`, `calibrate` | images, an encoder checkpoint | vectors, scores and out-of-fold thresholds |
| `scale`, `probe`, `hybrid` | a scored (and calibrated) encoder run | `scale.json`, `probe.json`, `hybrid.json` |
| `embedding-report` | the encoder files above | `reports/embedding-*.md` |
| `volume` | a source application (template) | images per month and how bursty the traffic is |
| `cost` | `manifest.csv`, a cost command in the source application | measured price per image by batch size |
| `economics` | `data/economics.json` | `reports/economics.md` |
| `pdf` | rendered reports | `reports/pdf/*.pdf` |
| `verify` | the author's own runs | re-derives the author's published figures; see [design](docs/design.md) |

## What the reports contain

Per model: capability and performance tables, per-tag agreement with the reference, latency, and
consistency across repeats. Across models: one comparison table. A few properties are deliberate:

- **Agreement is not accuracy.** The reference is whatever your pipeline answers today, not ground
  truth. Every report states this, and says whether disagreements were adjudicated by a person.
  `vlm-eval review` exists to find out who was right when the two disagree.
- **Rates carry their sample size.** Per-tag tables include `n`; the comparison table puts the number
  of images and the share of unparsed answers next to every percentage.
- **Unfinished answers are not parsed.** A response cut off by the token budget, or a call that raised
  an error, is recorded as unknown. Unknown answers are left out of agreement, precision and recall,
  and the share of affected images is reported per task.
- **Only comparable questions are scored.** A candidate is scored on the questions the reference
  actually answered for that image (`evaluable_slugs`), not on every question it was asked.
- **Provenance is printed.** A run file whose settings were never recorded is labelled as such in the
  metrics and in the report.
- **Speed and cost are stated per machine.** A cost is derived only when a throughput and a price are
  known for the same hardware.

## Documentation

- [docs/usage.md](docs/usage.md) — the full workflow, including encoders and a paid API as the candidate
- [docs/datasets.md](docs/datasets.md) — the data files and their formats, the export templates, privacy
- [docs/economics.md](docs/economics.md) — volume, batch size, cost, and the decision to switch
- [docs/design.md](docs/design.md) — why it is built this way, provenance, and how to extend it
- [docs/serving.md](docs/serving.md) — serving a model with vLLM on a cloud GPU
- [docs/BACKLOG.md](docs/BACKLOG.md) — roadmap and known gaps

## Limitations

- **The dataset export is a template.** The scripts behind `export` and `volume` are written against
  one Django schema. On anything else, adapt them or write the data files yourself.
- **Not every backend has been run against a live model.** Ollama, Florence-2, and the SigLIP 2 and
  CLIP encoders have been run end to end. The vLLM path (schema-constrained decoding and token
  logprobs) is covered by mocked tests only; InternVL and PaliGemma have routing tests but the backends
  have not been executed. [docs/usage.md](docs/usage.md) lists how to check each.
- **Two image categories are built in.** Questions are selected by an image type of `indoor` or
  `outdoor` plus a `common` set; other category schemes need a code change.
- **Per-tag confidence needs logprobs.** vLLM exposes them; Ollama does not, so confidence is empty there.
- **Hosted APIs that require a key are not wired up.** The built-in server backend sends a placeholder
  bearer token, so a keyed endpoint needs a small backend of its own ([design](docs/design.md)).
- **Consistency at temperature 0 shows reproducibility, not robustness.** Repeats use the same bytes;
  perturbed inputs (crop, rotation, compression) are not generated yet.
- **Laptop latency is not production latency.** Quality figures carry over between machines; timing
  does not. Cost estimates are throughput times an hourly price and leave out idle time, autoscaling
  headroom and egress.
- **A model tag is not a model.** Short tags on a model server can resolve to a variant you did not
  intend, for example a reasoning build that spends its token budget before answering. Name the variant
  explicitly; the run fingerprint records the weights digest where the backend can report one.

## License

MIT — see [LICENSE](LICENSE).

## Author

[Marie Kam](https://github.com/MarinaKam)
