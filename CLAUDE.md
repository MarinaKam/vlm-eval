# Working rules for this repository

## Branches

**Never work on `main`.** Before the first edit of any task, create a branch:

```bash
git checkout -b <type>/<short-name>     # feat/ fix/ docs/ test/ chore/
```

This holds even for a one-line change. `main` is protected (CI must pass, no force-push), so work that
lands there directly cannot be reviewed or reverted cleanly.

One branch per task, not per file: incidental fixes discovered along the way belong in the same branch,
not in a new one.

**Check where you are before starting, and branch from the right place.** `git branch --show-current`
first. If the new work builds on an unmerged branch, branch from *that*, not from `main` — otherwise
stashing across the two produces conflicts for no reason. If uncommitted work is already in the tree,
move it with the branch rather than leaving it behind.

Claude creates the branch. Claude does **not** commit, push, merge, rebase, tag, force-checkout over
someone else's work, or delete branches — those are prepared as text for the user to run.

## Before handing work back

```bash
.venv/bin/pytest -q
.venv/bin/ruff check vlm_eval tests scripts
.venv/bin/ruff format --check vlm_eval tests scripts
```

All three must pass. "Should work" is not a report — run it and quote the output.

New behaviour needs a test that fails without the change. End-to-end paths belong in
`tests/test_e2e.py`, which drives the real chain (run → metrics → review → reports) against a stub
backend.

Every change ends with three questions, answered explicitly rather than assumed:

1. **Which branch is this on?** — `git branch --show-current`, and is it the right base?
2. **Does the README still describe reality?** — new command, changed flag, changed default, removed
   script: the README is part of the change, not a follow-up.
3. **Do the end-to-end tests cover what changed?** — a new path that only unit tests touch is untested
   where it matters. If a shell script or command is deleted, prove its capability still exists
   somewhere (`sweep --via` exists because deleting `run_hf_models.sh` had silently dropped three
   models).

Commands that talk to a model or a database are verified by **running them**, not by their `--help`.
Say plainly which ones were not run and why (a 17 GB download is a reason; "should work" is not).

The README carries a table of what has and has not been exercised against a real model. **Update it in
the same change** that runs a path for the first time, or that adds one. A public repository that
implies more testing than happened is worse than one that admits the gap.

After writing or regenerating a report, re-derive it:

```bash
python scripts/verify_published_figures.py
```

It recomputes every published number from `runs/` and `data/` importing nothing from this package. A
MISMATCH means the report is wrong or the data moved on — it has already caught the latter once.

## Never commit

`.env`, `data/`, `runs/`, `reports/`, `decisions_*.json` — client images, database exports, prompt texts
and everything derived from them. All are gitignored; verify before pushing:

```bash
git ls-files | grep -E "^(data|runs|reports)/|\.env$|decisions_.*\.json"   # must print nothing
```

The repository is public. Code stays generic: no company names, ticket ids, internal hostnames, or
absolute paths from anyone's machine. Anything environment-specific belongs in `.env`
(see `.env.example`).

## Commands

If checking something takes a raw shell command more than once, it belongs in the CLI. `vlm-eval status`
exists because the alternative was `wc -l` on run files.

Keep commands short: model presets live in `models.json`, so `vlm-eval run qwen3 tagging` — not five
flags. Flags override presets; a name that is not a preset is used as-is.

Argument names must be unambiguous. `vlm-eval cost 47` was rewritten to `--chunks 47 --images 60`
because "47" read as a number of photos.

## Measurement discipline

This is an evaluation tool; a wrong number is worse than no number.

- **Every code path that writes run rows goes through `provenance.check()`.** No exceptions for "small"
  backends — a test walks the CLI's syntax tree and fails on any `run_over_items` call without a check.
- **The fingerprint's composition is frozen while runs are in flight.** Adding a field changes every
  digest and refuses every resume, so batch such changes between sweeps, never during one.
- **Truncation and failure are read from the `completion` record**, never by matching error-message
  text. If a new task variant writes rows, it writes `completion_record(...)` too.
- **`legacy_unknown` is permanent.** Nothing may promote a pre-provenance file to verified; if a clean
  measurement is needed, the file is archived and recomputed.

- Every figure in a report traces back to a file in `runs/` or `data/`. If it was computed ad hoc, it
  belongs in the code instead — that is why `vlm-eval economics` exists.
- Compare like with like. Scoring a model against tags the reference never judged manufactures false
  positives; two chunk sizes measured on different image sets are not comparable.
- Separate signal from noise before concluding. The reference API answers differently on a re-run
  (~4% of tag decisions), so any difference below that is not evidence.
- State what was not measured. "Not verified end-to-end" is an acceptable report; silence is not.
