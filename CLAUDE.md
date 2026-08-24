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

Claude creates the branch. Claude does **not** commit, push, merge, rebase or tag — those are prepared
as text for the user to run.

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

- Every figure in a report traces back to a file in `runs/` or `data/`. If it was computed ad hoc, it
  belongs in the code instead — that is why `vlm-eval economics` exists.
- Compare like with like. Scoring a model against tags the reference never judged manufactures false
  positives; two chunk sizes measured on different image sets are not comparable.
- Separate signal from noise before concluding. The reference API answers differently on a re-run
  (~4% of tag decisions), so any difference below that is not evidence.
- State what was not measured. "Not verified end-to-end" is an acceptable report; silence is not.
