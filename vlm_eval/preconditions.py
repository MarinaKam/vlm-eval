"""What each command needs before it can say anything true, and whether what it has is still current.

Two failure modes this prevents:

* **Running too early** — a raw `FileNotFoundError` three frames deep tells you a file is missing but
  not which command creates it. Every check here names the command to run instead.
* **Running on stale results** — worse, because it looks like it worked. Metrics computed before the
  last batch of images finished are not wrong-looking, they are simply out of date, and a report built
  on them is quietly answering an older question. Staleness is reported, not corrected: re-running is
  the user's call, and sometimes the older number is the one they meant.
"""

import sys
from pathlib import Path


def fail(problem: str, fix: str) -> None:
    sys.exit(f"{problem}\n  -> {fix}")


def need_dataset(data: Path) -> None:
    """Images and questions have to exist before any model can be asked anything."""
    if not (data / "manifest.csv").exists():
        fail("No dataset yet.", "vlm-eval export     (then `vlm-eval download`)")
    images = data / "images"
    if not images.exists() or not any(images.glob("*.jpg")):
        fail("The manifest is there but no images are on disk.", "vlm-eval download")
    if not (data / "tags.json").exists() and not (data / "tags_from_migrations.json").exists():
        fail("No tag questions — there is nothing to ask the model.", "vlm-eval export")


def need_reference(data: Path) -> None:
    """Scoring needs something to score against."""
    if not (data / "gemini_tags.jsonl").exists():
        fail(
            "No reference answers, so agreement cannot be computed.",
            "vlm-eval export     (the export captures what your current API answered)",
        )


def need_run(runs: Path, model: str, what: str = "tagging") -> Path:
    """A task has to have been run before its numbers exist."""
    folder = runs / model
    found = sorted(folder.glob(f"{what}*.jsonl")) if folder.exists() else []
    if not found:
        fail(f"No {what} run for '{model}'.", f"vlm-eval run {model} {what}     (or `vlm-eval sweep {model}`)")
    return found[0]


def need_metrics(runs: Path, model: str) -> Path:
    path = runs / model / "metrics.json"
    if not path.exists():
        fail(f"No metrics for '{model}' yet.", f"vlm-eval metrics {model}")
    return path


def stale(target: Path, sources: list[Path]) -> list[str]:
    """Source files newer than the thing derived from them."""
    if not target.exists():
        return []
    t = target.stat().st_mtime
    return [s.name for s in sources if s.exists() and s.stat().st_mtime > t]


def warn_if_stale(target: Path, sources: list[Path], fix: str) -> bool:
    """Say so, loudly, and carry on — the older result may still be the one that was wanted."""
    outdated = stale(target, sources)
    if outdated:
        print(
            f"NOTE: {target.name} is older than {', '.join(sorted(outdated))} — showing the previous "
            f"result.\n      Refresh it with: {fix}",
            flush=True,
        )
    return bool(outdated)
