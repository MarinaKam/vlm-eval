# Design

Why the harness is built the way it is, how it keeps results from being mixed or misread, and how to
extend it.

- [Principles](#principles)
- [Settings come from the dataset](#settings-come-from-the-dataset)
- [Provenance](#provenance)
- [The order guard](#the-order-guard)
- [Completion records](#completion-records)
- [Independent verification](#independent-verification)
- [Adding a backend](#adding-a-backend)
- [Adding a task](#adding-a-task)
- [Working on the code](#working-on-the-code)

## Principles

- **The reference is the current pipeline, not human labels.** Labelling a thousand images properly is
  a project of its own; exporting what the pipeline already answered is not. The trade-off is that
  agreement means "behaves like today", not "is correct". The review step exists to close that gap on
  the cases where the two disagree.
- **Everything is a plain file.** Runs are JSONL, metrics are JSON, reports are Markdown. Every number
  in a report traces back to the raw model output that produced it. There is no database and no hidden
  state.
- **Everything resumes.** Runs take hours, laptops sleep and servers restart. Rows are appended one at
  a time, and a re-run continues from the last finished row.
- **Real confidence where it is available.** When the server returns token logprobs, the harness reads
  the probability of `true` for each answer instead of reporting a constant.
- **Refuse to measure nothing.** If a listing's images are missing from disk, the summary task records
  a failure instead of calling the model, because a model given no images still writes a fluent text.
  Grounding refuses to run without targets for the same reason.
- **One dispatcher.** Every backend goes through the same `run_task` function for the tasks they share,
  so a served model and an in-process checkpoint are asked the same questions with the same settings.

## Settings come from the dataset

The claim is "your questions, your batches, your pixels". It only holds if the numbers are read and not
assumed, so `data/prompts.json` carries the pipeline's own `processing_config` and
`vlm_eval/pipeline_config.py` applies it. A constant in this repository that happens to match your
pipeline today is not evidence: change the value in the pipeline, and a hardcoded harness would keep
measuring the old configuration without saying so.

If a setting is missing from the dataset, `run` stops and names it. `--allow-defaults` accepts the
fallbacks, and every guessed value is printed. `vlm-eval status` shows the settings that will be
applied and which of them were guessed.

Domain content follows the same rule:

| What | Where it comes from |
|---|---|
| tag questions | `data/tags.json` |
| caption prompts and their opening line | `data/prompts.json` (`caption_prompts`, `caption_header`) |
| summary prompt | `data/prompts.json` (`multi_image_summary`) |
| what to localise | `data/grounding_targets.json` |
| batching and image settings | `data/prompts.json` (`processing_config`) |

Two things remain in the code. The package carries fallback prompts for datasets that provide none,
and the fallback summary prompt was written for one domain. And question selection assumes the image
types `indoor` and `outdoor` plus a `common` category. Both are listed in the
[backlog](BACKLOG.md).

## Provenance

Resume keys on `(image_id, repeat)`. That proves the work was done, not that it is still valid: raise a
token budget or change a prompt, and every old row still counts as finished, the new rows land beside
them, and one file holds two experiments that average into a single number.

So every run file has a sidecar, `<file>.meta.json`, holding a fingerprint
(`vlm_eval/provenance.py`, `RunFingerprint`) of everything that changes what an answer means:

- **The request as rendered.** A digest of the prompt text and the JSON schema, so rewording the
  wrapper around unchanged questions counts as a different run.
- **The pixels the task reads.** A digest of the image bytes on disk, computed per task: replacing an
  image that only the summary task reads blocks resuming the summary and leaves tagging resumable.
- **The structure.** Tag categories and order (they decide batch composition), each image's type (it
  decides the question set), and which images belong to which listing in which order.
- **The machinery.** Served name, checkpoint, backend class, batch size, token budget, logprobs,
  coordinate convention and image settings; the route (`flavor@base_url`, or `in-process/<device>`);
  and a digest of the source files that build requests and parse answers. It is deliberately not a git
  commit: editing a report template must not block a resume, editing a parser must.
- **The weights.** A served name is a mutable tag; pulling an update makes the same name answer with a
  different model. Ollama's manifest digest and a Hugging Face commit are immutable identities. A
  backend that cannot report one is recorded as `unknown`.

What the gate does (`provenance.check`):

| Situation | Result |
|---|---|
| no run file yet | the fingerprint is written; status `verified` |
| same fingerprint, weights identity known | the run resumes |
| same fingerprint, weights identity unknown, file has rows | refused: the name may now point at other weights |
| different fingerprint | refused, with a list of what changed; archive the file or use another run name |
| different fingerprint, but the command rewrites the whole file | allowed; the change is printed and recorded |
| run file with no sidecar | allowed, and labelled `legacy_unknown` permanently, with the count of unverified rows |

The last two rows are the special cases. Rescoring cached vectors replaces every row, so there is
nothing for two configurations to mix in. And a file that predates the sidecar is not stamped with
today's settings, because its rows would then look as checked as rows that really were; the label is
printed on every run that touches the file, travels into `metrics.json`, and appears in the report.

Model identity does not restrict which model you may evaluate. It prevents results from different
weights from being combined in one file. Updating a model between experiments is a new experiment,
under a new run name or after archiving the old file.

A test parses the CLI module and checks that no command writes run rows without going through this
gate.

## The order guard

Running the steps out of order does not fail by itself; it produces a report from whatever happens to
be on disk. `vlm_eval/preconditions.py` covers the two ways that goes wrong.

**Running too early.** A command checks for what it needs and names the command that creates it:
`run` needs the manifest, images on disk and tag questions; `metrics` and `review` need a run;
`review` needs reference answers; `report` needs `metrics.json`; `compare` needs metrics or a card for
every model; `cost` needs the manifest.

**Running on stale results.** `report` and `compare` compare the age of `metrics.json` with the run
files and print a note when a run file is newer. Staleness is reported, not corrected, because the
older result is sometimes the one that was wanted.

`economics` has its own guard and refuses inputs that were never measured
([economics.md](economics.md#refusing-placeholder-inputs)). `embedding-report` refuses to render the
`smoke` preset, because that would overwrite the comparison files with a throwaway run.

## Completion records

An answer the model did not finish counts as no answer. A response that ends with
`finish_reason: "length"` is recorded as unknown and its partial text is not parsed. The alternative is
worse than it looks: half-written JSON would contribute real tags to precision and recall, and a
truncated "found nothing" would agree with the reference without the model having looked at the image.
Reasoning models make this common, because they can spend the whole budget before writing an answer;
`extra_output_tokens` in a preset raises the budget for such a model without changing the prompt.

Each row stores `completion: {calls, truncated, failed, status}`. A call that raised an exception is
`failed`, not `complete`. Metrics read this field, never the wording of an error message, and report
the share of affected images per task.

## Independent verification

`vlm-eval verify` runs `scripts/verify_published_figures.py`, which recomputes the figures from the
author's own reports straight from `runs/` and `data/`, using only the standard library and nothing
from the `vlm_eval` package, so a bug in the metrics code cannot confirm itself. It runs as a separate
process for the same reason.

The script is project-specific: the expected values in it are the author's results and it will report
mismatches or skipped sections on anyone else's data. Treat it as a pattern. If you publish figures
from this tool, keep a second, independent derivation of them and edit the expected values only when
you regenerate the report.

## Adding a backend

A backend is a class with a `name` attribute and one method (`vlm_eval/backends/base.py`):

```python
def chat(self, images: list[bytes], prompt: str, *, json_schema: dict | None = None,
         max_tokens: int = 1024, temperature: float = 0.0, logprobs: bool = False) -> Response: ...
```

`Response` carries `text`, `latency_s`, `usage`, optional `logprobs`, `finish_reason`,
`reasoning_chars` and `raw`. Set `finish_reason` to `"length"` when the answer was cut off, so
truncation is detected.

For the provenance gate, a backend can expose `weights_digest` (an immutable identity of the weights),
`checkpoint`, and either `base_url` and `flavor` or `device`. Without a weights identity, fresh runs
work and resuming a non-empty file is refused.

The existing backends are `openai_compat.py` (any OpenAI-compatible server), `hf_chat.py` (InternVL and
PaliGemma), `florence_hf.py` (Florence-2) and `encoder.py` (image-text encoders). To make a new
backend reachable from the command line, add a command or a `--via` value in `vlm_eval/cli.py` next to
the existing ones, declare which tasks it supports in `BACKEND_TASKS`, and call `run_task` so that it
goes through the shared dispatcher and the gate. Keep torch and transformers imports inside functions:
a test checks that nothing in the package imports them at module level, so the base install keeps
working.

A hosted API that needs a key is such a case. `OpenAICompatBackend` accepts an `api_key` argument, but
the command line constructs it with the default placeholder, so a keyed endpoint needs its own wiring.

## Adding a task

A task is a module in `vlm_eval/tasks/` that renders a prompt, defines a JSON schema and parses the
answer, plus three pieces elsewhere:

1. a `run_<task>_one` function and an output path in `vlm_eval/runner.py`, returning a row with a
   `completion` record;
2. a branch in `run_task` in `vlm_eval/cli.py` that builds the fingerprint from the rendered prompt and
   schema and calls `provenance.check` before writing;
3. a metrics function in `vlm_eval/metrics.py`, a section in `cmd_metrics`, and a section in
   `vlm_eval/report.py`.

Add the task name to the `run` command's choices and to `BACKEND_TASKS` for each backend that supports
it.

## Working on the code

```bash
pytest -q
ruff check vlm_eval tests scripts
ruff format --check vlm_eval tests scripts
```

CI runs the same three on Python 3.12 and 3.14. The suite is offline: backends are mocked and no
weights or dataset are needed. `tests/test_e2e.py` drives the command chain against a stub backend; a
path covered only by unit tests is untested where it matters, so new commands belong there too.

Commands that reach a model or a database are verified by running them, not by their `--help`. When a
path has not been run, say so; [usage.md](usage.md#what-has-been-exercised) keeps that list.
Documentation is part of a change: a new command, flag or default is not finished until these files
describe it.
