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


@dataclass
class RunConfig:
    model: str
    chunk_size: int = 15
    individual: list[str] = field(default_factory=lambda: ["utility_room"])
    repeats: int = 1
    workers: int = 1
    logprobs: bool = True
    coords: str = "norm1000"  # grounding coordinate convention of the model family
    limit: int | None = None


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


def run_tagging_one(backend: Backend, item: Item, tags: list[dict], cfg: RunConfig) -> dict[str, Any]:
    img = item.path.read_bytes()
    questions = tagging.questions_for(item.image_type, tags)
    chunks = tagging.chunk_questions(questions, cfg.chunk_size, cfg.individual)
    answers, confidence, raw, responses, errors = {}, {}, [], [], []
    t0 = time.perf_counter()
    for chunk in chunks:
        try:
            r = backend.chat(
                [img],
                tagging.prompt_text(chunk),
                json_schema=tagging.boolean_schema(chunk),
                max_tokens=3000,
                temperature=0.0,
                logprobs=cfg.logprobs,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            answers.update({s: None for s in chunk})
            continue
        responses.append(r)
        raw.append(r.text)
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
        "raw": raw,
        "errors": errors,
    }


def run_captions_one(backend: Backend, item: Item, prompts: dict[str, str], cfg: RunConfig) -> dict[str, Any]:
    img = item.path.read_bytes()
    t0 = time.perf_counter()
    try:
        r = backend.chat(
            [img],
            captions.prompt_text(prompts),
            json_schema=captions.schema(prompts),
            max_tokens=captions.MAX_TOKENS,
            temperature=captions.TEMPERATURE,
        )
        parsed, raw, usage, err = captions.parse(r.text, prompts), r.text, r.usage, []
    except Exception as exc:
        parsed, raw, usage, err = {k: None for k in prompts}, "", {}, [f"{type(exc).__name__}: {exc}"]
    return {
        "image_id": item.image_id,
        "image_type": item.image_type,
        "captions": parsed,
        "latency_s": round(time.perf_counter() - t0, 3),
        "usage": usage,
        "raw": raw,
        "errors": err,
    }


def run_grounding_one(backend: Backend, item: Item, targets: dict[str, str], cfg: RunConfig) -> dict[str, Any]:
    img = item.path.read_bytes()
    w, h = image_size(img)
    out, raw, errors, lat = {}, {}, [], {}
    for label, desc in targets.items():
        t0 = time.perf_counter()
        try:
            r = backend.chat(
                [img], grounding.prompt_text(label, desc), json_schema=grounding.SCHEMA, max_tokens=512, temperature=0.0
            )
            out[label] = grounding.parse(r.text, coords=cfg.coords, width=w, height=h)
            raw[label] = r.text
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            out[label] = None
        lat[label] = round(time.perf_counter() - t0, 3)
    return {
        "image_id": item.image_id,
        "image_type": item.image_type,
        "width": w,
        "height": h,
        "detections": out,
        "latency_s": lat,
        "raw": raw,
        "errors": errors,
    }


def run_summary_one(backend: Backend, prop: dict, prompt: str, images_dir: Path) -> dict[str, Any]:
    paths = [images_dir / f"{i}.jpg" for i in prop["image_ids"]]
    imgs = [p.read_bytes() for p in paths if p.exists()][: summary.MAX_IMAGES]
    t0 = time.perf_counter()
    try:
        r = backend.chat(
            imgs, prompt, json_schema=summary.SCHEMA, max_tokens=summary.MAX_TOKENS, temperature=summary.TEMPERATURE
        )
        text, raw, usage, err = summary.parse(r.text), r.text, r.usage, []
    except Exception as exc:
        text, raw, usage, err = None, "", {}, [f"{type(exc).__name__}: {exc}"]
    return {
        "property_job_id": prop["property_job_id"],
        "image_ids": prop["image_ids"][: summary.MAX_IMAGES],
        "n_images": len(imgs),
        "summary": text,
        "gemini_summary": prop.get("property_summary"),
        "latency_s": round(time.perf_counter() - t0, 3),
        "usage": usage,
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
    t_start, n = time.perf_counter(), 0

    def work(pair):
        it, rep = pair
        row = fn(it)
        row["repeat"] = rep
        row["model_ts"] = time.time()
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, p) for p in todo]
        for fut in as_completed(futures):
            row = fut.result()
            _append(out_path, row)
            n += 1
            if n % 10 == 0 or n == len(todo):
                el = time.perf_counter() - t_start
                log(f"  {n}/{len(todo)} done, {el:.0f}s elapsed, {n / el * 3600:.0f} items/hour at workers={workers}")
    return n


def tagging_out(model: str, chunk_size: int) -> Path:
    return _out(model, "tagging", f"chunk{chunk_size if chunk_size > 0 else 'all'}")


def captions_out(model: str) -> Path:
    return _out(model, "captions")


def grounding_out(model: str) -> Path:
    return _out(model, "grounding")


def summary_out(model: str) -> Path:
    return _out(model, "summary")
