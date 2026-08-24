# vlm-eval

A test bench for vision-language models. It takes the exact questions and prompts your production
pipeline sends to a paid API (Gemini), asks the same questions to any open-weight model you can run
yourself, and tells you — in numbers — how far apart the answers are and what self-hosting would cost.

## The problem it solves

Say your product tags real-estate photos ("is there a kitchen island?", "is there a pool?" — dozens of
yes/no questions per image), writes captions, and summarises listings, all through a paid vision API.
Open models (Qwen-VL, Florence-2, InternVL, PaliGemma…) keep getting better and cheaper to host. The
question "should we switch?" comes up regularly, and answering it by poking a demo is not evidence.

vlm-eval turns that question into a repeatable experiment:

1. **Export** a dataset from your own DB: real images plus everything the current API said about them
   (tags, captions, summaries). The current API's answers become the reference.
2. **Run** any candidate model over the same images with *the same* questions, prompts, batching and
   JSON output format your production code uses — not a synthetic benchmark.
3. **Compare**: per-tag agreement, false positives, answer stability across repeated runs, speed,
   and a hosting-cost estimate per 1K/10K/100K/1M images.
4. **Report**: one markdown file per model (capability + performance tables) and one comparison table.

Because the harness mirrors production exactly, a good score means "you could swap this model in",
not "it does well on some academic benchmark".

## What's inside

```
vlm_eval/
  tasks/       the four jobs, copied faithfully from production:
               tagging (batched yes/no questions, strict JSON), captions,
               multi-image property summary, object grounding (bounding boxes)
  backends/    how to talk to models: any OpenAI-compatible server (vLLM, Ollama),
               Florence-2 (task-token model), InternVL / PaliGemma via transformers
  runner.py    runs a task over the dataset; resumable — kill it, rerun, it continues
  metrics.py   agreement/FPR/recall vs the reference, consistency, latency stats
  review.py    HTML page to eyeball cases where model and reference disagree,
               so a human decides who was right — the reference API is not infallible
  report.py    renders the markdown tables
scripts/       dataset export from your Django app's DB, batch run scripts
```

## Setup

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"   # add ".[hf]" for transformers-based models
cp .env.example .env                                   # then edit: point VLM_EVAL_SOURCE_REPO at your app
.venv/bin/pytest                                       # should pass before you trust anything else
```

## Typical session

```bash
# 1. Build the dataset (read-only against your DB; prints which DB it's about to use)
$VLM_EVAL_SOURCE_REPO/.venv/bin/python scripts/run_export.py

# 2. Fetch the images locally (resized/compressed the same way production does)
.venv/bin/vlm-eval download

# 3. Run a model. Anything with an OpenAI-compatible API works — Ollama, vLLM, etc.
.venv/bin/vlm-eval run --model qwen3-vl-8b --served-name qwen3-vl:8b \
    --base-url http://localhost:11434/v1 --flavor ollama --task tagging

# ... also: --task captions | grounding | summary,  --repeats 3 (stability),  --chunk 0 (all
# questions in one call),  --limit N (subset). Interrupt any time; rerun resumes.

# 4. Numbers and reports
.venv/bin/vlm-eval metrics --model qwen3-vl-8b
.venv/bin/vlm-eval report  --model qwen3-vl-8b        # needs reports/cards/<model>.json (facts you fill in)
.venv/bin/vlm-eval compare --models qwen3-vl-8b qwen2.5-vl-7b florence-2-large

# 5. Optional but recommended: judge the disagreements yourself
.venv/bin/vlm-eval review --model qwen3-vl-8b          # builds an HTML page, you click through it
.venv/bin/vlm-eval review --model qwen3-vl-8b --decisions decisions_qwen3-vl-8b.json
```

Models with no server (Florence-2, InternVL, PaliGemma) run directly:

```bash
.venv/bin/vlm-eval florence --task captions --limit 300
.venv/bin/vlm-eval hf --backend internvl  --task tagging --limit 150
.venv/bin/vlm-eval hf --backend paligemma --task tagging --limit 150   # gated repo: HF_TOKEN needed
```

## Why the design is the way it is

- **The reference is your current API, not human labels.** Labelling 1000 images properly takes a
  team; exporting what you already paid for takes a minute. The trade-off: where model and reference
  disagree, you don't know who's right — that's what the `review` step is for. Treat "agreement" as
  "would behave like what you have today", not as ground truth.
- **Everything is a plain JSONL file on disk.** Every metric in a report can be traced back to the
  raw model output that produced it. No hidden state, no DB.
- **Resume everywhere.** Runs take hours; laptops sleep; servers restart. Every step picks up where
  it left off.
- **Confidence for free.** When served through vLLM, the harness reads token logprobs and turns them
  into a real per-tag probability — something many API pipelines fake with a constant.

## Privacy: what is safe to publish

The repo is designed so that the *code* is generic and everything private stays in ignored places.
Before pushing anywhere, verify:

1. `.env` is not tracked — it holds your paths (and optionally an HF token).
2. `data/` is not tracked — it contains client images, DB exports and your prompt texts.
3. `runs/` and `reports/` are not tracked — they are derived from that private data.

Check with:

```bash
git ls-files | grep -E "^(data|runs|reports)/|\.env$"   # must print nothing
```

The tag questions and prompt texts live only in your DB and in `data/` — they never appear in code.

## Limitations, honestly

- Latency measured on a laptop is not production latency; use `docs/INFRA.md` to rerun the timing
  part on the GPU you'd actually deploy (the quality numbers don't change).
- Ollama doesn't expose logprobs, so confidence comes only from vLLM-served runs.
- Cost estimates are "GPU-hours at measured throughput × the card's hourly price" — real bills add
  idle time, autoscaling headroom and egress.
