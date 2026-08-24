"""Run tasks over the dataset against a backend, one JSONL row per (image, repeat); resumable."""

import functools
import json
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends.base import Backend, Response
from .config import RUNS
from .dataset import Item, image_size, load_jsonl
from .tasks import captions, grounding, summary, tagging


def truncation_error(r: Response, budget: int) -> str | None:
    """An answer cut off by the token budget, reported rather than parsed as a refusal.

    A reasoning model can spend the entire budget thinking and return empty content with
    `finish_reason="length"`. Parsed naively that is a model with no opinion; it is really a model
    that never got to speak, and the difference decides whether a number means anything.
    """
    if r.finish_reason != "length":
        return None
    detail = f"answer cut off at the {budget}-token budget"
    if r.reasoning_chars:
        detail += f"; the model spent it reasoning ({r.reasoning_chars} characters of it)"
    if not r.text.strip():
        detail += " and returned nothing — raise the budget for this model"
    return detail


COMPLETE, TRUNCATED, FAILED, NOT_CALLED = "complete", "truncated", "failed", "not_called"


def completion_record(calls: int, truncated: int, failed: int = 0) -> dict[str, Any]:
    """Whether the model got to finish, as a field rather than a phrase inside an error string.

    Metrics used to detect a cut-off answer by searching the error text for "cut off at the". That
    works until somebody rewords the message, and then truncation silently reads as zero — the exact
    failure this record exists to make visible.

    A call that died on an exception is `failed`, not `ok`: both leave no answer, and a status that
    says otherwise while `errors` is non-empty is the kind of small lie a summary table repeats.
    """
    if not calls:
        status = NOT_CALLED
    elif truncated:
        status = TRUNCATED
    elif failed:
        status = FAILED
    else:
        status = COMPLETE
    return {"calls": calls, "truncated": truncated, "failed": failed, "status": status}


@dataclass
class RunConfig:
    model: str
    chunk_size: int = 15
    # Which tags production asks on their own. Comes from the export; a tag slug from someone's
    # database has no business being a default in this code.
    individual: list[str] = field(default_factory=list)
    repeats: int = 1
    workers: int = 1
    logprobs: bool = True
    coords: str = "norm1000"  # grounding coordinate convention of the model family
    limit: int | None = None
    # Added to every task's output budget. Production's budgets were sized for a model that answers
    # directly; a reasoning model spends tokens thinking before it writes anything, and how many
    # depends on how much input there is — an 11-image summary needs far more than a single image.
    # Raising this does not change the prompt or the task, it only lets the model finish.
    extra_output_tokens: int = 0


def _out(model: str, task: str, tag: str = "") -> Path:
    p = RUNS / model
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{task}{('_' + tag) if tag else ''}.jsonl"


def _done_keys(path: Path) -> set[tuple]:
    return {(r["image_id"], r.get("repeat", 0)) for r in load_jsonl(path)}


def _append(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _usage_sum(responses: list[Response]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in responses:
        for k, v in r.usage.items():
            out[k] = out.get(k, 0) + v
    return out


# ------------------------------------------------------------------ per-image task functions


def _read_image(item: Item) -> bytes:
    """Raises with the image id in the message, so a bad file is identifiable in the run log."""
    try:
        return item.path.read_bytes()
    except OSError as exc:
        raise OSError(f"cannot read {item.path.name}: {exc}") from exc


def run_tagging_one(backend: Backend, item: Item, tags: list[dict], cfg: RunConfig) -> dict[str, Any]:
    img = _read_image(item)
    questions = tagging.questions_for(item.image_type, tags)
    chunks = tagging.chunk_questions(questions, cfg.chunk_size, cfg.individual)
    answers, confidence, raw, responses, errors = {}, {}, [], [], []
    n_cut = n_failed = 0
    t0 = time.perf_counter()
    for chunk in chunks:
        try:
            r = backend.chat(
                [img],
                tagging.prompt_text(chunk),
                json_schema=tagging.boolean_schema(chunk),
                max_tokens=3000 + cfg.extra_output_tokens,
                temperature=0.0,
                logprobs=cfg.logprobs,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            n_failed += 1
            answers.update({s: None for s in chunk})
            continue
        responses.append(r)
        raw.append(r.text)
        cut = truncation_error(r, 3000 + cfg.extra_output_tokens)
        if cut:
            # An answer the model did not finish is not a smaller answer, it is no answer. Parsing what
            # arrived would let a half-written JSON contribute real tags to accuracy and recall, and the
            # tags it happened to reach are not a sample of anything.
            errors.append(cut)
            n_cut += 1
            answers.update({s: None for s in chunk})
            continue
        answers.update(tagging.parse_answers(r.text, chunk))
        if r.logprobs:
            confidence.update(tagging.confidence_from_logprobs(r.logprobs, chunk))
    return {
        "image_id": item.image_id,
        "image_type": item.image_type,
        "chunk_size": cfg.chunk_size,
        "n_questions": len(questions),
        "n_calls": len(chunks),
        "answers": answers,
        "confidence": confidence,
        "latency_s": round(time.perf_counter() - t0, 3),
        "call_latencies_s": [round(r.latency_s, 3) for r in responses],
        "usage": _usage_sum(responses),
        "completion": completion_record(len(chunks), n_cut, n_failed),
        "raw": raw,
        "errors": errors,
    }


def run_captions_one(
    backend: Backend,
    item: Item,
    prompts: dict[str, str],
    cfg: RunConfig,
    templates: dict[str, str] | None = None,
) -> dict[str, Any]:
    img = _read_image(item)
    t0 = time.perf_counter()
    n_cut = n_failed = 0
    try:
        r = backend.chat(
            [img],
            captions.prompt_text(prompts, templates),
            json_schema=captions.schema(prompts),
            max_tokens=captions.MAX_TOKENS + cfg.extra_output_tokens,
            temperature=captions.TEMPERATURE,
        )
        cut = truncation_error(r, captions.MAX_TOKENS + cfg.extra_output_tokens)
        n_cut = 1 if cut else 0
        parsed = {k: None for k in prompts} if cut else captions.parse(r.text, prompts)
        raw, usage, err = r.text, r.usage, [cut] if cut else []
    except Exception as exc:
        n_failed = 1
        parsed, raw, usage, err = {k: None for k in prompts}, "", {}, [f"{type(exc).__name__}: {exc}"]
    return {
        "image_id": item.image_id,
        "image_type": item.image_type,
        "captions": parsed,
        "latency_s": round(time.perf_counter() - t0, 3),
        "usage": usage,
        "completion": completion_record(1, n_cut, n_failed),
        "raw": raw,
        "errors": err,
    }


def run_grounding_one(backend: Backend, item: Item, targets: dict[str, str], cfg: RunConfig) -> dict[str, Any]:
    if not targets:
        raise ValueError(
            "no detection targets — grounding would produce empty rows that look like a model finding "
            "nothing. Populate data/grounding_targets.json."
        )
    img = _read_image(item)
    w, h = image_size(img)
    out, raw, errors, lat = {}, {}, [], {}
    n_cut = n_failed = 0
    for label, desc in targets.items():
        t0 = time.perf_counter()
        try:
            r = backend.chat(
                [img],
                grounding.prompt_text(label, desc),
                json_schema=grounding.SCHEMA,
                max_tokens=512 + cfg.extra_output_tokens,
                temperature=0.0,
            )
            raw[label] = r.text
            cut = truncation_error(r, 512 + cfg.extra_output_tokens)
            if cut:
                errors.append(f"{label}: {cut}")
                n_cut += 1
                out[label] = None  # unknown, not "found nothing"
            else:
                out[label] = grounding.parse(r.text, coords=cfg.coords, width=w, height=h)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            n_failed += 1
            out[label] = None
        lat[label] = round(time.perf_counter() - t0, 3)
    return {
        "image_id": item.image_id,
        "image_type": item.image_type,
        "width": w,
        "height": h,
        "detections": out,
        "latency_s": lat,
        "completion": completion_record(len(targets), n_cut, n_failed),
        "raw": raw,
        "errors": errors,
    }


def run_summary_one(
    backend: Backend, prop: dict, prompt: str, images_dir: Path, extra_output_tokens: int = 0
) -> dict[str, Any]:
    # Cap first, then check what exists. Filtering before capping let image 21 slide in to replace a
    # missing one among the first 20 — the count came out right, so nothing failed, and the model was
    # quietly shown a different listing than production would send.
    wanted = prop["image_ids"][: summary.MAX_IMAGES]
    paths = [images_dir / f"{i}.jpg" for i in wanted]
    imgs = [p.read_bytes() for p in paths if p.exists()]
    expected = len(wanted)
    if len(imgs) < expected:
        # A model handed fewer images than the listing has still writes a fluent description — one
        # given no images at all invented 130 words of it. Scoring that would measure nothing, so the
        # row is recorded as a failure instead. Run `vlm-eval download` to fetch the listing images.
        return {
            "property_job_id": prop["property_job_id"],
            "image_ids": wanted,
            "n_images": len(imgs),
            "n_expected": expected,
            "summary": None,
            "gemini_summary": prop.get("property_summary"),
            "latency_s": 0.0,
            "usage": {},
            "completion": completion_record(0, 0),
            "raw": "",
            "errors": [f"only {len(imgs)}/{expected} listing images present on disk — run download"],
        }
    t0 = time.perf_counter()
    n_cut = n_failed = 0
    try:
        r = backend.chat(
            imgs,
            prompt,
            json_schema=summary.SCHEMA,
            max_tokens=summary.MAX_TOKENS + extra_output_tokens,
            temperature=summary.TEMPERATURE,
        )
        cut = truncation_error(r, summary.MAX_TOKENS + extra_output_tokens)
        n_cut = 1 if cut else 0
        text = None if cut else summary.parse(r.text)
        raw, usage, err = r.text, r.usage, [cut] if cut else []
    except Exception as exc:
        n_failed = 1
        text, raw, usage, err = None, "", {}, [f"{type(exc).__name__}: {exc}"]
    return {
        "property_job_id": prop["property_job_id"],
        "image_ids": wanted,
        "n_images": len(imgs),
        "summary": text,
        "gemini_summary": prop.get("property_summary"),
        "latency_s": round(time.perf_counter() - t0, 3),
        "usage": usage,
        "completion": completion_record(1, n_cut, n_failed),
        "raw": raw,
        "errors": err,
    }


# ------------------------------------------------------------------ driver


def run_over_items(
    items: Iterable[Item],
    fn: Callable[[Item], dict],
    out_path: Path,
    *,
    repeats: int,
    workers: int,
    limit: int | None = None,
    log: Callable[[str], None] = functools.partial(print, flush=True),
) -> int:
    items = list(items)[:limit] if limit else list(items)
    done = _done_keys(out_path)
    todo = [(it, rep) for rep in range(repeats) for it in items if (it.image_id, rep) not in done]
    log(f"{out_path.name}: {len(items)} items x {repeats} repeats, {len(todo)} to do, {len(done)} already done")
    t_start, n, failed = time.perf_counter(), 0, 0

    def work(pair):
        it, rep = pair
        try:
            row = fn(it)
        except Exception as exc:
            # A corrupt file, a missing image, an unexpected shape in one row: record it and carry on.
            # Ending a four-hour run because of one bad image loses the run, not the bad image — and the
            # row below is what tells you afterwards which images were skipped and why.
            row = {
                "image_id": it.image_id,
                "image_type": it.image_type,
                "answers": {},
                "errors": [f"{type(exc).__name__}: {exc}"],
                "latency_s": 0.0,
            }
        row["repeat"] = rep
        row["model_ts"] = time.time()
        return row

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work, p) for p in todo]
            for fut in as_completed(futures):
                row = fut.result()
                _append(out_path, row)
                n += 1
                failed += bool(row.get("errors"))
                if n % 10 == 0 or n == len(todo):
                    el = max(time.perf_counter() - t_start, 1e-9)
                    log(
                        f"  {n}/{len(todo)} done, {el:.0f}s elapsed, "
                        f"{n / el * 3600:.0f} items/hour at workers={workers}"
                    )
    except KeyboardInterrupt:
        # Stopping a long run is a normal thing to do, not a crash. Every finished row is already on
        # disk and re-running resumes from here, so say that instead of printing a stack trace.
        interrupted = True
        log(f"\n  stopped by you after {n} item(s) — everything finished is saved; re-run to continue")
    if interrupted:
        raise SystemExit(130)
    if failed:
        log(f'  {failed} of {n} item(s) recorded an error — grep the run file for "errors" to see which')
    return n


def tagging_out(model: str, chunk_size: int) -> Path:
    return _out(model, "tagging", f"chunk{chunk_size if chunk_size > 0 else 'all'}")


def captions_out(model: str) -> Path:
    return _out(model, "captions")


def grounding_out(model: str) -> Path:
    return _out(model, "grounding")


def summary_out(model: str) -> Path:
    return _out(model, "summary")
