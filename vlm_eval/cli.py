"""vlm-eval CLI. Models are named by preset (see models.json), so commands stay short:

  vlm-eval status                       what is measured so far, what is missing
  vlm-eval download                     fetch the images named in the manifest
  vlm-eval run qwen3 tagging            run one task (also: captions | grounding | summary)
  vlm-eval sweep qwen3                  run every task for a model, cheapest first
  vlm-eval run qwen3 tagging --chunk 0  every question in one call
  vlm-eval metrics qwen3                compute metrics       -> runs/<model>/metrics.json
  vlm-eval review qwen3                 judge the disagreements yourself
  vlm-eval report qwen3                 per-model report      -> reports/<model>.md
  vlm-eval compare qwen3 qwen2.5        comparison table      -> reports/comparison.md
  vlm-eval economics                    self-host vs API      -> reports/economics.md

Measuring the inputs for that last one:

  vlm-eval export                       build the dataset from your database
  vlm-eval volume [--db-from prod.env]  images per month, busiest hour
  vlm-eval cost --chunks 15 47          API cost per image at each batch size, and what changes

Any preset field can be overridden with a flag (--served-name, --base-url, --flavor, --coords), and a
name that is not a preset is used as-is.
"""

import argparse
import json
import sys
import time

from . import dataset, metrics, pipeline_config, report, review, runner
from .backends.openai_compat import OpenAICompatBackend
from .config import REPORTS
from .tasks import captions, grounding, summary


def _presets() -> dict:
    """Every known model, later files overriding earlier ones (see config.model_presets)."""
    from .config import model_presets

    merged: dict[str, dict] = {}
    for f in model_presets():
        merged.update({k: v for k, v in json.loads(f.read_text()).items() if not k.startswith("_")})
    return merged


def _resolve(a) -> None:
    """Fill in model connection details from a preset, unless the flag was given explicitly."""
    preset = _presets().get(getattr(a, "model", None) or "", {})
    a.model = preset.get("run_name", a.model)
    for field, default in (
        ("served_name", None),
        ("base_url", "http://localhost:8000/v1"),
        ("flavor", "vllm"),
        ("coords", "norm1000"),
        # A preset can also say how the model is reached at all, so `sweep <name>` works for a
        # transformers-backed checkpoint without repeating --via and --checkpoint every time.
        ("via", "server"),
        ("checkpoint", None),
    ):
        if getattr(a, field, None) in (None, "server") and field in preset:
            setattr(a, field, preset[field])
        elif getattr(a, field, None) is None:
            setattr(a, field, default)


def _backend(a) -> OpenAICompatBackend:
    be = OpenAICompatBackend(a.base_url, a.served_name or a.model, flavor=a.flavor)
    if not be.health():
        sys.exit(f"backend not reachable at {a.base_url} (GET /models failed)")
    return be


def cmd_download(a) -> None:
    # Property images are a separate set from the per-image manifest: the multi-image summary task
    # needs every photo of a listing, and most of them are not in the sampled 1000.
    items = dataset.load_manifest() + dataset.property_items()
    done, failed = dataset.download_all(items, force=a.force)
    print(f"downloaded {done}, failed {len(failed)}, total {len(items)}; failed ids: {failed[:20]}")


def cmd_run(a) -> None:
    _resolve(a)
    be = _backend(a)
    prompts = dataset.load_prompts()
    pipe = pipeline_config.load(prompts)
    if not getattr(a, "allow_defaults", False):
        pipe.strict()
    elif pipe.defaulted:
        print(f"WARNING: guessing {', '.join(pipe.defaulted)} — these may not match production", flush=True)
    # `--chunk` is an experiment knob; without it the harness uses production's own batch size.
    chunk = a.chunk if a.chunk is not None else pipe.chunk_size
    cfg = runner.RunConfig(
        model=a.model,
        chunk_size=chunk,
        individual=pipe.individual_questions,
        repeats=a.repeats,
        workers=a.workers,
        logprobs=not a.no_logprobs,
        coords=a.coords,
        limit=a.limit,
    )
    items = [it for it in dataset.load_manifest() if it.path.exists()]
    if a.task == "tagging":
        tags = dataset.load_tags()
        runner.run_over_items(
            items,
            lambda it: runner.run_tagging_one(be, it, tags, cfg),
            runner.tagging_out(a.model, chunk),
            repeats=a.repeats,
            workers=a.workers,
            limit=a.limit,
        )
    elif a.task == "captions":
        cp = prompts.get("caption_prompts") or captions.DEFAULT_PROMPTS
        runner.run_over_items(
            items,
            lambda it: runner.run_captions_one(be, it, cp, cfg, prompts.get("prompt_templates")),
            runner.captions_out(a.model),
            repeats=1,
            workers=a.workers,
            limit=a.limit,
        )
    elif a.task == "grounding":
        targets = grounding.load_targets()
        runner.run_over_items(
            items,
            lambda it: runner.run_grounding_one(be, it, targets, cfg),
            runner.grounding_out(a.model),
            repeats=1,
            workers=a.workers,
            limit=a.limit,
        )
    elif a.task == "summary":
        prompt = (prompts.get("prompt_templates") or {}).get("multi_image_summary") or summary.DEFAULT_PROMPT
        props = dataset.load_jsonl(dataset.DATA / "properties.jsonl")
        out = runner.summary_out(a.model)
        done = {r["property_job_id"] for r in dataset.load_jsonl(out)}
        for p in props:
            if p["property_job_id"] in done:
                continue
            row = runner.run_summary_one(be, p, prompt, dataset.IMAGES)
            runner._append(out, row)
            print(
                f"property {p['property_job_id']}: {row['n_images']} images, {row['latency_s']}s, "
                f"{len((row['summary'] or '').split())} words"
            )


def cmd_perf(a) -> None:
    _resolve(a)
    be = _backend(a)
    tags = dataset.load_tags()
    cfg = runner.RunConfig(model=a.model, chunk_size=15, logprobs=False)
    items = [it for it in dataset.load_manifest() if it.path.exists()][: a.n]
    out = runner.RUNS / a.model / f"perf_c{a.concurrency}.jsonl"
    if out.exists():
        out.unlink()
    t0 = time.perf_counter()
    n = runner.run_over_items(
        items, lambda it: runner.run_tagging_one(be, it, tags, cfg), out, repeats=1, workers=a.concurrency
    )
    el = time.perf_counter() - t0
    res = {
        "model": a.model,
        "concurrency": a.concurrency,
        "n_images": n,
        "elapsed_s": round(el, 1),
        "images_per_hour_measured": round(n / el * 3600) if el else None,
    }
    (runner.RUNS / a.model / f"perf_c{a.concurrency}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def _primary_tagging_run(model: str):
    """The tagging run made with production's batch size — found, not assumed.

    Hardcoding 15 here was the same mistake as hardcoding it in the runner: change the batch size in
    production and the metrics would quietly read a file that no longer exists.
    """
    pipe = pipeline_config.load()
    preferred = runner.tagging_out(model, pipe.chunk_size)
    if preferred.exists():
        return preferred
    others = sorted(p for p in (runner.RUNS / model).glob("tagging_chunk*.jsonl") if not p.stem.endswith("chunkall"))
    return others[0] if others else preferred


def cmd_metrics(a) -> None:
    _resolve(a)
    gem = dataset.gemini_tags_by_image()
    d = runner.RUNS / a.model
    out: dict = {"model": a.model, "tagging": {}, "captions": {}, "grounding": {}, "summary": {}, "perf": {}}
    t15 = dataset.load_jsonl(_primary_tagging_run(a.model))
    tall = dataset.load_jsonl(runner.tagging_out(a.model, 0))
    if t15:
        out["tagging"]["agreement"] = metrics.tagging_agreement(t15, gem)
        out["tagging"]["consistency"] = metrics.tagging_consistency(t15)
        out["tagging"]["latency"] = metrics.latency_stats([r["latency_s"] for r in t15 if r.get("repeat", 0) == 0])
        out["tagging"]["errors"] = sum(1 for r in t15 if r.get("errors"))
        out["tagging"]["manual"] = review.manual_agreement(t15, gem)
    if tall:
        out["tagging"]["agreement_chunk_all"] = metrics.tagging_agreement(tall, gem)
        out["tagging"]["latency_chunk_all"] = metrics.latency_stats([r["latency_s"] for r in tall])
        if t15:
            out["tagging"]["chunk_all_agreement_pct"] = metrics.tagging_chunk_comparison(t15, tall)["agreement_pct"]
    caps = dataset.load_jsonl(runner.captions_out(a.model))
    if caps:
        out["captions"] = {
            **metrics.caption_stats(caps),
            "latency": metrics.latency_stats([r["latency_s"] for r in caps]),
        }
    gr = dataset.load_jsonl(runner.grounding_out(a.model))
    if gr:
        out["grounding"] = metrics.grounding_stats(gr, gem)
    sm = dataset.load_jsonl(runner.summary_out(a.model))
    if sm:
        out["summary"] = {
            "n": len(sm),
            "ok": sum(1 for r in sm if r.get("summary")),
            "mean_words": round(sum(len((r.get("summary") or "").split()) for r in sm) / len(sm), 1),
            "latency": metrics.latency_stats([r["latency_s"] for r in sm]),
        }
    perfs = sorted(d.glob("perf_c*.json"))
    if perfs:
        best = max((json.loads(p.read_text()) for p in perfs), key=lambda r: r.get("images_per_hour_measured") or 0)
        out["perf"] = best
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {k: v for k, v in out.items() if k != "tagging"}
            | {
                "tagging_overall": out["tagging"].get("agreement", {}).get("overall"),
                "tagging_consistency": out["tagging"].get("consistency"),
                "tagging_latency": out["tagging"].get("latency"),
            },
            indent=2,
        )
    )


def cmd_florence(a) -> None:
    """Florence-2 via transformers (no server). Writes the same run files as `run` under runs/<model>/."""
    from PIL import Image as PILImage

    from .backends.florence_hf import FlorenceBackend
    from .tasks import tagging

    be = FlorenceBackend(a.checkpoint, device=a.device)
    model = a.model or be.name
    items = [it for it in dataset.load_manifest() if it.path.exists()]
    tags = dataset.load_tags()

    def _img(it):
        return PILImage.open(it.path).convert("RGB")

    if a.task == "captions":

        def fn(it):
            return {
                "image_id": it.image_id,
                "image_type": it.image_type,
                **be.captions(_img(it)),
                "usage": {},
                "raw": "",
                "errors": [],
            }

        out = runner.captions_out(model)
    elif a.task == "grounding":

        def fn(it):
            img = _img(it)
            dets, lat = {}, {}
            for label, desc in grounding.TARGETS.items():
                dets[label], lat[label] = be.ovd(img, desc.split(" (")[0])
            return {
                "image_id": it.image_id,
                "image_type": it.image_type,
                "width": img.width,
                "height": img.height,
                "detections": dets,
                "latency_s": lat,
                "raw": {},
                "errors": [],
            }

        out = runner.grounding_out(model)
    else:  # tagging via open-vocabulary detection

        def fn(it):
            q = tagging.questions_for(it.image_type, tags)
            qt = [t for t in tags if t["slug"] in q]
            r = be.tagging_via_ovd(_img(it), qt)
            return {
                "image_id": it.image_id,
                "image_type": it.image_type,
                "chunk_size": 1,
                "n_questions": len(q),
                **r,
                "call_latencies_s": [],
                "usage": {},
                "errors": [],
            }

        out = runner.tagging_out(model, 15)  # stored under chunk15 so metrics/report pick it up
    runner.run_over_items(items, fn, out, repeats=a.repeats, workers=1, limit=a.limit)


def cmd_review(a) -> None:
    _resolve(a)
    gem = dataset.gemini_tags_by_image()
    rows = dataset.load_jsonl(_primary_tagging_run(a.model))
    if a.decisions:
        print(review.apply_decisions([dataset.ROOT / f for f in a.decisions]))
        print(review.manual_agreement(rows, gem))
        return
    cases = review.sample_by_tag(review.disagreements(rows, gem, dataset.load_tags()), per_tag=a.per_tag)
    out = review.build_review_html(a.model, cases, REPORTS / "review" / f"{a.model}.html")
    print(
        f"{len(cases)} cases -> open {out} in a browser, then: vlm-eval review {a.model} --decisions <downloaded json>"
    )


def cmd_hf(a) -> None:
    """InternVL3.5 / PaliGemma2 via transformers locally. Same run-file layout as `run`."""
    from .tasks import tagging

    items = [it for it in dataset.load_manifest() if it.path.exists()]
    tags = dataset.load_tags()
    prompts = dataset.load_prompts()
    if a.backend == "internvl":
        from .backends.hf_chat import InternVLBackend

        be = InternVLBackend(a.checkpoint or "OpenGVLab/InternVL3_5-8B-HF", device=a.device)
        model = a.model or be.name
        cfg = runner.RunConfig(
            model=model,
            chunk_size=a.chunk,
            repeats=a.repeats,
            workers=1,
            logprobs=False,
            coords="norm1000",
            limit=a.limit,
        )
        if a.task == "tagging":
            runner.run_over_items(
                items,
                lambda it: runner.run_tagging_one(be, it, tags, cfg),
                runner.tagging_out(model, a.chunk),
                repeats=a.repeats,
                workers=1,
                limit=a.limit,
            )
        elif a.task == "captions":
            cp = prompts.get("caption_prompts") or captions.DEFAULT_PROMPTS
            runner.run_over_items(
                items,
                lambda it: runner.run_captions_one(be, it, cp, cfg),
                runner.captions_out(model),
                repeats=1,
                workers=1,
                limit=a.limit,
            )
        elif a.task == "grounding":
            runner.run_over_items(
                items,
                lambda it: runner.run_grounding_one(be, it, grounding.TARGETS, cfg),
                runner.grounding_out(model),
                repeats=1,
                workers=1,
                limit=a.limit,
            )
        elif a.task == "summary":
            prompt = (prompts.get("prompt_templates") or {}).get("multi_image_summary") or summary.DEFAULT_PROMPT
            props = dataset.load_jsonl(dataset.DATA / "properties.jsonl")
            out = runner.summary_out(model)
            done = {r["property_job_id"] for r in dataset.load_jsonl(out)}
            for pr in props:
                if pr["property_job_id"] not in done:
                    runner._append(out, runner.run_summary_one(be, pr, prompt, dataset.IMAGES))
    else:  # paligemma
        from .backends.hf_chat import PaliGemmaBackend

        be = PaliGemmaBackend(a.checkpoint or "google/paligemma2-3b-mix-448", device=a.device)
        model = a.model or be.name
        if a.task == "tagging":

            def fn(it):
                q = tagging.questions_for(it.image_type, tags)
                row = be.tagging_rows(it.path.read_bytes(), q)
                return {
                    "image_id": it.image_id,
                    "image_type": it.image_type,
                    "chunk_size": 1,
                    "n_questions": len(q),
                    **row,
                    "call_latencies_s": [],
                    "usage": {},
                    "errors": [],
                }

            runner.run_over_items(items, fn, runner.tagging_out(model, 15), repeats=a.repeats, workers=1, limit=a.limit)
        elif a.task == "captions":

            def fn(it):
                img = it.path.read_bytes()
                r1 = be.chat([img], "caption en", max_tokens=64)
                r2 = be.chat([img], "describe en", max_tokens=256)
                return {
                    "image_id": it.image_id,
                    "image_type": it.image_type,
                    "captions": {"base_caption": r1.text, "detailed_caption": r2.text},
                    "latency_s": round(r1.latency_s + r2.latency_s, 3),
                    "usage": {},
                    "raw": "",
                    "errors": [],
                }

            runner.run_over_items(items, fn, runner.captions_out(model), repeats=1, workers=1, limit=a.limit)
        else:
            sys.exit("paligemma supports tasks: tagging, captions (grounding via 'detect' TBD; no multi-image)")
    subprocess_metrics = argparse.Namespace(model=model)
    cmd_metrics(subprocess_metrics)


def _script(name: str, *args: str) -> int:
    """Run one of the scripts/ helpers with this interpreter."""
    import subprocess

    return subprocess.call([sys.executable, str(dataset.ROOT / "scripts" / name), *args])


def cmd_export(a) -> None:
    """Build the dataset from the source application's database (read-only)."""
    sys.exit(_script("run_export.py"))


def cmd_volume(a) -> None:
    """How many images per month actually go through, and how bursty the traffic is."""
    args = ["shell", "--stdin", str(dataset.ROOT / "scripts" / "count_volume.py")]
    if a.db_from:
        args = ["shell", "--db-from", a.db_from, "--stdin", str(dataset.ROOT / "scripts" / "count_volume.py")]
    sys.exit(_script("run_source_manage.py", *args))


def cmd_sweep(a) -> None:
    """Run every task for one model, cheapest first.

    Ordered so that stopping early still leaves every capability measured — only the sample size for
    the expensive tasks shrinks. Everything resumes, so re-running picks up where it stopped.
    """
    import argparse as _argparse

    # Models without a server (Florence-2, InternVL, PaliGemma) go through their own commands, and
    # not all of them do every task: Florence-2 has no multi-image input, PaliGemma is single-turn.
    _resolve(a)
    if a.via != "server":
        _sweep_local(a)
        return

    stages = [
        ("summary", {"task": "summary", "limit": None}),
        ("grounding", {"task": "grounding", "limit": a.grounding}),
        ("captions", {"task": "captions", "limit": a.captions}),
        ("all questions in one call", {"task": "tagging", "limit": a.chunk_all, "chunk": 0}),
        ("tagging", {"task": "tagging", "limit": a.tagging}),
        ("consistency", {"task": "tagging", "limit": a.consistency, "repeats": 3}),
    ]
    for label, opts in stages:
        if opts.get("limit") == 0:
            print(f"--- skipping {label}")
            continue
        n = opts.get("limit")
        print(f"\n=== {a.model}: {label}" + (f" ({n} images)" if n else "") + " ===", flush=True)
        run_args = _argparse.Namespace(
            model=a.model,
            served_name=a.served_name,
            base_url=a.base_url,
            flavor=a.flavor,
            coords=a.coords,
            task=opts["task"],
            chunk=opts.get("chunk", 15),
            repeats=opts.get("repeats", 1),
            workers=a.workers,
            limit=opts.get("limit"),
            no_logprobs=a.no_logprobs,
        )
        cmd_run(run_args)

    cmd_metrics(_argparse.Namespace(model=a.model))
    cmd_status(_argparse.Namespace())


def _sweep_local(a) -> None:
    """Full run for a transformers-backed model, skipping the tasks its architecture cannot do."""
    import argparse as _argparse

    if a.via == "florence":
        supported = [("captions", a.captions), ("grounding", a.grounding), ("tagging", a.tagging)]
        run = cmd_florence
        base = {
            "checkpoint": a.checkpoint or "florence-community/Florence-2-large",
            "model": a.model if a.model != a.via else None,
            "device": a.device,
            "repeats": 1,
        }
        note = "Florence-2 takes one image at a time, so there is no property summary."
    else:
        supported = [("captions", a.captions), ("tagging", a.tagging)]
        if a.via == "internvl":
            supported = [("summary", None), ("grounding", a.grounding), *supported]
            note = ""
        else:
            note = "PaliGemma is single-image and single-turn: no summary, and one call per tag."
        run = cmd_hf
        base = {
            "backend": a.via,
            "checkpoint": a.checkpoint,
            "model": a.model if a.model != a.via else None,
            "device": a.device,
            "chunk": 15,
            "repeats": 1,
        }

    if note:
        print(note, flush=True)
    model_name = None
    for task, limit in supported:
        if limit == 0:
            print(f"--- skipping {task}")
            continue
        print(f"\n=== {a.via}: {task}" + (f" ({limit} images)" if limit else "") + " ===", flush=True)
        ns = _argparse.Namespace(task=task, limit=limit, **base)
        run(ns)
        model_name = ns.model or model_name
    if model_name:
        cmd_metrics(_argparse.Namespace(model=model_name))


def cmd_status(a) -> None:
    """What has been measured so far, and what is missing — so the answer never needs `wc -l`."""
    import csv

    data, runs = dataset.DATA, runner.RUNS
    print(f"data: {data}")
    manifest = data / "manifest.csv"
    if manifest.exists():
        items = dataset.load_manifest()
        on_disk = sum(1 for it in items if it.path.exists())
        kinds: dict[str, int] = {}
        for it in items:
            kinds[it.image_type] = kinds.get(it.image_type, 0) + 1
        print(f"  manifest        {len(items):>6} images ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})")
        print(f"  downloaded      {on_disk:>6}" + ("" if on_disk == len(items) else "   <- run `vlm-eval download`"))
    else:
        print("  manifest        missing   <- run `vlm-eval export`, then `vlm-eval download`")

    for name, label in (("tags.json", "tag questions"), ("prompts.json", "prompts")):
        f = data / name
        n = len(json.loads(f.read_text())) if f.exists() else 0
        print(f"  {label:<15} {n:>6}" if f.exists() else f"  {label:<15} missing")
    for name, label in (
        ("gemini_tags.jsonl", "reference tags"),
        ("gemini_captions.jsonl", "reference captions"),
        ("properties.jsonl", "listings"),
    ):
        f = data / name
        print(f"  {label:<15} {len(dataset.load_jsonl(f)):>6}" if f.exists() else f"  {label:<15} missing")
    labels = data / "manual_labels.json"
    if labels.exists():
        print(f"  {'human verdicts':<15} {len(json.loads(labels.read_text())):>6}")

    cost = sorted(data.glob("cost_chunk*.csv"))
    if cost:
        print("\ncost measurements:")
        for f in cost:
            with f.open() as fh:
                rows = list(csv.DictReader(fh))
            chunk = f.stem.replace("cost_chunk", "")
            print(f"  {chunk:>4} questions/call  {len(rows):>4} images")

    print(f"\nruns: {runs}")
    if runs.exists():
        for model_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            parts = [f"{f.stem} {len(dataset.load_jsonl(f))}" for f in sorted(model_dir.glob("*.jsonl"))]
            has_metrics = " +metrics" if (model_dir / "metrics.json").exists() else ""
            print(f"  {model_dir.name}: {', '.join(parts) if parts else 'empty'}{has_metrics}")
    else:
        print("  none yet")

    targets = data / "grounding_targets.json"
    print(
        f"  {'grounding targets':<15} "
        + (f"{len(json.loads(targets.read_text())):>6}" if targets.exists() else "missing")
    )

    # The settings the harness will actually apply — shown so a mismatch with production is visible
    # rather than discovered three hours into a run.
    print("\npipeline settings replayed from the export:")
    print(pipeline_config.load().describe())

    econ = data / "economics.json"
    print(f"\neconomics config: {'present' if econ.exists() else 'missing (see `vlm-eval economics`)'}")


def _read_cost_csv(path):
    """(tags per image, mean cost, mean calls, mean prompt tokens) from a cost-measurement CSV."""
    import csv

    tags, cost, calls, prompt = {}, [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = r.get("image_id") or r["url"]
            tags[key] = {t.strip() for t in (r.get("classification_tags") or "").split(";") if t.strip()}
            cost.append(float(r["cost_2_5"]))
            calls.append(int(r["n_calls"]))
            prompt.append(int(r["prompt_tokens"]))
    n = max(len(cost), 1)
    return tags, sum(cost) / n, sum(calls) / n, sum(prompt) / n


def cmd_cost(a) -> None:
    """Measure what an image costs on the API, at one or more batch sizes, and diff the answers.

    Splitting the questions into chunks re-sends the image with every chunk, which is usually the
    dominant cost — but bigger batches can change what the model answers. Both halves have to be
    measured together, or the saving looks free when it is not.
    """
    import os

    command = os.environ.get("VLM_EVAL_COST_COMMAND")
    if not command:
        sys.exit("VLM_EVAL_COST_COMMAND is not set (see .env.example) — name your app's cost command there")

    urls = dataset.DATA / f"cost_urls_{a.type}.txt"
    if not urls.exists() or a.refresh:
        _script("make_cost_urls.py", "--type", a.type, "--limit", str(a.images))

    results = {}
    for chunk in a.chunks:
        out = dataset.DATA / f"cost_chunk{chunk}.csv"
        if out.exists() and not a.refresh:
            print(f"chunk {chunk}: reusing {out.name} (--refresh to measure again)")
        else:
            print(f"\n=== measuring {a.images} images with {chunk} questions per call ===", flush=True)
            code = _script(
                "run_source_manage.py",
                command,
                "--urls-file",
                str(urls),
                "--out",
                str(out),
                "--no-gpu",
                "--assume-type",
                a.type,
                "--chunk-size",
                str(chunk),
            )
            if code != 0:
                sys.exit(code)
        results[chunk] = _read_cost_csv(out)

    print(f"\n{'questions/call':>15} {'API calls':>10} {'input tokens':>13} {'$/image':>10}")
    print("-" * 52)
    base = results[a.chunks[0]][1]
    for chunk, (_, cost, calls, prompt) in results.items():
        delta = f"  ({100 * (cost / base - 1):+.0f}%)" if chunk != a.chunks[0] else ""
        print(f"{chunk:>15} {calls:>10.1f} {prompt:>13,.0f} {cost:>10.6f}{delta}")

    if len(a.chunks) > 1:
        first = a.chunks[0]
        print("\nDoes the answer change?")
        for chunk in a.chunks[1:]:
            d = metrics.tagset_agreement(results[first][0], results[chunk][0])
            print(
                f"  {first} vs {chunk}: identical on {d['identical_pct']}% of images, "
                f"{d['jaccard_pct']}% tag agreement ({d['tags_first']} -> {d['tags_second']} tags)"
            )
            if d["lost"]:
                print(f"      lost:   {', '.join(f'{t} x{n}' for t, n in d['lost'])}")
            if d["gained"]:
                print(f"      gained: {', '.join(f'{t} x{n}' for t, n in d['gained'])}")
        print(
            "\nNote: some of that difference is the API answering differently on a re-run, not the batch"
            "\nsize. Measure that baseline with the same size twice: vlm-eval cost --chunks 15 --refresh"
        )


def cmd_economics(a) -> None:
    from .economics import Inputs, render

    path = dataset.DATA / "economics.json" if a.config is None else dataset.ROOT / a.config
    if not path.exists():
        sys.exit(
            f"{path} not found. Create it with the numbers you measured, for example:\n"
            + json.dumps(
                {
                    "api_cost_per_image": 0.001446,
                    "api_cost_per_image_optimized": 0.000707,
                    "gpu_usd_per_hour": 0.5832,
                    "gpu_images_per_hour": 2000,
                    "gpu_name": "L4 (spot)",
                    "peak_hour_images": 12404,
                    "busy_hours_pct": 8.6,
                    "scenarios": [["last 3 months", 23466], ["current pace", 61072]],
                },
                indent=2,
            )
        )
    from .economics import Hosting

    cfg = json.loads(path.read_text())
    cfg["scenarios"] = [tuple(x) for x in cfg.get("scenarios", [])]
    cfg["hosting"] = [Hosting(**h) for h in cfg.get("hosting", [])]
    note = cfg.pop("note", "")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "economics.md"
    out.write_text(render(Inputs(**cfg), currency_note=note))
    print(f"wrote {out}")


def _card(model: str) -> dict:
    p = REPORTS / "cards" / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else {"model": model, "name": model}


def cmd_report(a) -> None:
    _resolve(a)
    m = json.loads((runner.RUNS / a.model / "metrics.json").read_text())
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{a.model}.md"
    out.write_text(report.render_model(_card(a.model), m))
    print(f"wrote {out}")


def cmd_compare(a) -> None:
    a.models = [_presets().get(m, {}).get("run_name", m) for m in a.models]
    cards = [_card(m) for m in a.models]
    ms = [json.loads((runner.RUNS / m / "metrics.json").read_text()) for m in a.models]
    out = REPORTS / "comparison.md"
    out.write_text(report.render_comparison(cards, ms))
    print(f"wrote {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vlm-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_download)

    def conn(sp):
        """Connection overrides — normally supplied by the preset."""
        sp.add_argument("--served-name", default=None, help="model name as the server exposes it")
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--flavor", choices=["vllm", "ollama"], default=None)

    s = sub.add_parser("run", help="run a task against a model")
    s.add_argument("model", help="preset name from models.json, or a run folder name")
    s.add_argument("task", choices=["tagging", "captions", "grounding", "summary"])
    conn(s)
    s.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="questions per call; default is production's own setting, 0 = all in one",
    )
    s.add_argument(
        "--allow-defaults",
        action="store_true",
        help="run even if pipeline settings are missing from the export (they get guessed)",
    )
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-logprobs", action="store_true")
    s.add_argument("--coords", choices=["abs", "norm1000"], default=None)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="run every task for one model, cheapest first, resumable")
    s.add_argument("model", help="preset name from models.json, or a run folder name")
    conn(s)
    s.add_argument("--tagging", type=int, default=500, help="images for the main tagging run (0 = skip)")
    s.add_argument("--captions", type=int, default=100)
    s.add_argument("--grounding", type=int, default=100)
    s.add_argument("--chunk-all", type=int, default=60, help="images for the all-in-one-call comparison")
    s.add_argument("--consistency", type=int, default=0, help="images to run 3x (0 = skip)")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--no-logprobs", action="store_true")
    s.add_argument("--coords", choices=["abs", "norm1000"], default=None)
    s.add_argument(
        "--via",
        choices=["server", "florence", "internvl", "paligemma"],
        default="server",
        help="how to reach the model: an OpenAI-compatible server (default) or transformers",
    )
    s.add_argument("--checkpoint", default=None, help="for --via florence|internvl|paligemma")
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("perf", help="throughput at a given concurrency")
    s.add_argument("model")
    conn(s)
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--n", type=int, default=40)
    s.set_defaults(fn=cmd_perf)

    s = sub.add_parser("florence", help="Florence-2 via transformers (no server)")
    s.add_argument("task", choices=["tagging", "captions", "grounding"])
    s.add_argument("--checkpoint", default="florence-community/Florence-2-large")
    s.add_argument("--model", default=None, help="run folder name (default: checkpoint basename)")
    s.add_argument("--device", default=None)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_florence)

    s = sub.add_parser("hf", help="InternVL / PaliGemma via transformers (no server)")
    s.add_argument("backend", choices=["internvl", "paligemma"])
    s.add_argument("task", choices=["tagging", "captions", "grounding", "summary"])
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--chunk", type=int, default=15)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_hf)

    s = sub.add_parser("review", help="judge model-vs-reference disagreements by eye")
    s.add_argument("model")
    s.add_argument("--per-tag", type=int, default=5)
    s.add_argument("--decisions", nargs="*", help="decision files downloaded from the review page")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("metrics", help="compute metrics from the run files")
    s.add_argument("model")
    s.set_defaults(fn=cmd_metrics)

    s = sub.add_parser("report", help="render the per-model report")
    s.add_argument("model")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("compare", help="render the comparison table")
    s.add_argument("models", nargs="+")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("status", help="what is measured so far and what is missing")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("export", help="build the dataset from the source app's database")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("volume", help="images per month and how bursty the traffic is")
    s.add_argument("--db-from", default=None, help="take DATABASE_URL from this env file (e.g. production)")
    s.set_defaults(fn=cmd_volume)

    s = sub.add_parser("cost", help="API cost per image at one or more batch sizes, and what it changes")
    s.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[15],
        help="questions per API call to try, e.g. --chunks 15 47 (NOT a number of images)",
    )
    s.add_argument("--images", type=int, default=60, help="how many images to measure on")
    s.add_argument("--type", choices=["indoor", "outdoor"], default="indoor")
    s.add_argument("--refresh", action="store_true", help="re-measure even if a result file exists")
    s.set_defaults(fn=cmd_cost)

    s = sub.add_parser("economics", help="self-host vs pay-per-call, from measured inputs")
    s.add_argument("--config", default=None, help="default: <data>/economics.json")
    s.set_defaults(fn=cmd_economics)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
