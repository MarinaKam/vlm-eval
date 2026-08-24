"""vlm-eval CLI.

  vlm-eval download                                  # fetch data/images from manifest (optimized like production)
  vlm-eval run --model qwen2.5-vl-7b --base-url http://localhost:8000/v1 --task tagging [--chunk 15|0] [--repeats 3]
  vlm-eval run ... --task captions|grounding|summary
  vlm-eval perf --model ... --concurrency 4 --n 40      # throughput at concurrency (tagging, chunk 15)
  vlm-eval metrics --model qwen2.5-vl-7b               # -> runs/<model>/metrics.json
  vlm-eval report --model qwen2.5-vl-7b                # -> reports/<model>.md (needs cards/<model>.json)
  vlm-eval compare --models a b c                      # -> reports/comparison.md
"""
import argparse
import json
import sys
import time

from . import dataset, metrics, report, review, runner
from .backends.openai_compat import OpenAICompatBackend
from .tasks import captions, grounding, summary

from .config import REPORTS  # noqa: E402


def _backend(a) -> OpenAICompatBackend:
    be = OpenAICompatBackend(a.base_url, a.served_name or a.model, flavor=a.flavor)
    if not be.health():
        sys.exit(f"backend not reachable at {a.base_url} (GET /models failed)")
    return be


def cmd_download(a) -> None:
    items = dataset.load_manifest()
    done, failed = dataset.download_all(items, force=a.force)
    print(f"downloaded {done}, failed {len(failed)}, total {len(items)}; failed ids: {failed[:20]}")


def cmd_run(a) -> None:
    be = _backend(a)
    cfg = runner.RunConfig(model=a.model, chunk_size=a.chunk, repeats=a.repeats, workers=a.workers,
                           logprobs=not a.no_logprobs, coords=a.coords, limit=a.limit)
    items = [it for it in dataset.load_manifest() if it.path.exists()]
    prompts = dataset.load_prompts()
    if a.task == "tagging":
        tags = dataset.load_tags()
        runner.run_over_items(items, lambda it: runner.run_tagging_one(be, it, tags, cfg),
                              runner.tagging_out(a.model, a.chunk), repeats=a.repeats, workers=a.workers, limit=a.limit)
    elif a.task == "captions":
        cp = prompts.get("caption_prompts") or captions.DEFAULT_PROMPTS
        runner.run_over_items(items, lambda it: runner.run_captions_one(be, it, cp, cfg),
                              runner.captions_out(a.model), repeats=1, workers=a.workers, limit=a.limit)
    elif a.task == "grounding":
        runner.run_over_items(items, lambda it: runner.run_grounding_one(be, it, grounding.TARGETS, cfg),
                              runner.grounding_out(a.model), repeats=1, workers=a.workers, limit=a.limit)
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
            print(f"property {p['property_job_id']}: {row['n_images']} images, {row['latency_s']}s, "
                  f"{len((row['summary'] or '').split())} words")


def cmd_perf(a) -> None:
    be = _backend(a)
    tags = dataset.load_tags()
    cfg = runner.RunConfig(model=a.model, chunk_size=15, logprobs=False)
    items = [it for it in dataset.load_manifest() if it.path.exists()][: a.n]
    out = runner.RUNS / a.model / f"perf_c{a.concurrency}.jsonl"
    if out.exists():
        out.unlink()
    t0 = time.perf_counter()
    n = runner.run_over_items(items, lambda it: runner.run_tagging_one(be, it, tags, cfg), out,
                              repeats=1, workers=a.concurrency)
    el = time.perf_counter() - t0
    res = {"model": a.model, "concurrency": a.concurrency, "n_images": n, "elapsed_s": round(el, 1),
           "images_per_hour_measured": round(n / el * 3600) if el else None}
    (runner.RUNS / a.model / f"perf_c{a.concurrency}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def cmd_metrics(a) -> None:
    gem = dataset.gemini_tags_by_image()
    d = runner.RUNS / a.model
    out: dict = {"model": a.model, "tagging": {}, "captions": {}, "grounding": {}, "summary": {}, "perf": {}}
    t15 = dataset.load_jsonl(runner.tagging_out(a.model, 15))
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
        out["captions"] = {**metrics.caption_stats(caps),
                           "latency": metrics.latency_stats([r["latency_s"] for r in caps])}
    gr = dataset.load_jsonl(runner.grounding_out(a.model))
    if gr:
        out["grounding"] = metrics.grounding_stats(gr, gem)
    sm = dataset.load_jsonl(runner.summary_out(a.model))
    if sm:
        out["summary"] = {"n": len(sm), "ok": sum(1 for r in sm if r.get("summary")),
                          "mean_words": round(sum(len((r.get("summary") or "").split()) for r in sm) / len(sm), 1),
                          "latency": metrics.latency_stats([r["latency_s"] for r in sm])}
    perfs = sorted(d.glob("perf_c*.json"))
    if perfs:
        best = max((json.loads(p.read_text()) for p in perfs), key=lambda r: r.get("images_per_hour_measured") or 0)
        out["perf"] = best
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "tagging"} | {
        "tagging_overall": out["tagging"].get("agreement", {}).get("overall"),
        "tagging_consistency": out["tagging"].get("consistency"),
        "tagging_latency": out["tagging"].get("latency")}, indent=2))


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
            return {"image_id": it.image_id, "image_type": it.image_type, **be.captions(_img(it)),
                    "usage": {}, "raw": "", "errors": []}
        out = runner.captions_out(model)
    elif a.task == "grounding":
        def fn(it):
            img = _img(it)
            dets, lat = {}, {}
            for label, desc in grounding.TARGETS.items():
                dets[label], lat[label] = be.ovd(img, desc.split(" (")[0])
            return {"image_id": it.image_id, "image_type": it.image_type, "width": img.width, "height": img.height,
                    "detections": dets, "latency_s": lat, "raw": {}, "errors": []}
        out = runner.grounding_out(model)
    else:  # tagging via open-vocabulary detection
        def fn(it):
            q = tagging.questions_for(it.image_type, tags)
            qt = [t for t in tags if t["slug"] in q]
            r = be.tagging_via_ovd(_img(it), qt)
            return {"image_id": it.image_id, "image_type": it.image_type, "chunk_size": 1,
                    "n_questions": len(q), **r, "call_latencies_s": [], "usage": {}, "errors": []}
        out = runner.tagging_out(model, 15)  # stored under chunk15 so metrics/report pick it up
    runner.run_over_items(items, fn, out, repeats=a.repeats, workers=1, limit=a.limit)


def cmd_review(a) -> None:
    gem = dataset.gemini_tags_by_image()
    rows = dataset.load_jsonl(runner.tagging_out(a.model, 15))
    if a.decisions:
        print(review.apply_decisions([dataset.ROOT / f for f in a.decisions]))
        print(review.manual_agreement(rows, gem))
        return
    cases = review.sample_by_tag(review.disagreements(rows, gem, dataset.load_tags()), per_tag=a.per_tag)
    out = review.build_review_html(a.model, cases, REPORTS / "review" / f"{a.model}.html")
    print(f"{len(cases)} cases -> open {out} in a browser, then: vlm-eval review --model {a.model} "
          f"--decisions <downloaded json>")


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
        cfg = runner.RunConfig(model=model, chunk_size=a.chunk, repeats=a.repeats, workers=1,
                               logprobs=False, coords="norm1000", limit=a.limit)
        if a.task == "tagging":
            runner.run_over_items(items, lambda it: runner.run_tagging_one(be, it, tags, cfg),
                                  runner.tagging_out(model, a.chunk), repeats=a.repeats, workers=1, limit=a.limit)
        elif a.task == "captions":
            cp = prompts.get("caption_prompts") or captions.DEFAULT_PROMPTS
            runner.run_over_items(items, lambda it: runner.run_captions_one(be, it, cp, cfg),
                                  runner.captions_out(model), repeats=1, workers=1, limit=a.limit)
        elif a.task == "grounding":
            runner.run_over_items(items, lambda it: runner.run_grounding_one(be, it, grounding.TARGETS, cfg),
                                  runner.grounding_out(model), repeats=1, workers=1, limit=a.limit)
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
                return {"image_id": it.image_id, "image_type": it.image_type, "chunk_size": 1,
                        "n_questions": len(q), **row, "call_latencies_s": [], "usage": {}, "errors": []}
            runner.run_over_items(items, fn, runner.tagging_out(model, 15),
                                  repeats=a.repeats, workers=1, limit=a.limit)
        elif a.task == "captions":
            def fn(it):
                img = it.path.read_bytes()
                r1 = be.chat([img], "caption en", max_tokens=64)
                r2 = be.chat([img], "describe en", max_tokens=256)
                return {"image_id": it.image_id, "image_type": it.image_type,
                        "captions": {"base_caption": r1.text, "detailed_caption": r2.text},
                        "latency_s": round(r1.latency_s + r2.latency_s, 3), "usage": {}, "raw": "", "errors": []}
            runner.run_over_items(items, fn, runner.captions_out(model), repeats=1, workers=1, limit=a.limit)
        else:
            sys.exit("paligemma supports tasks: tagging, captions (grounding via 'detect' TBD; no multi-image)")
    subprocess_metrics = argparse.Namespace(model=model)
    cmd_metrics(subprocess_metrics)


def _card(model: str) -> dict:
    p = REPORTS / "cards" / f"{model}.json"
    return json.loads(p.read_text()) if p.exists() else {"model": model, "name": model}


def cmd_report(a) -> None:
    m = json.loads((runner.RUNS / a.model / "metrics.json").read_text())
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{a.model}.md"
    out.write_text(report.render_model(_card(a.model), m))
    print(f"wrote {out}")


def cmd_compare(a) -> None:
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

    def common(sp):
        sp.add_argument("--model", required=True, help="run folder name, e.g. qwen2.5-vl-7b")
        sp.add_argument("--served-name", default=None, help="model name as served (default: --model)")
        sp.add_argument("--base-url", default="http://localhost:8000/v1")
        sp.add_argument("--flavor", choices=["vllm", "ollama"], default="vllm")

    s = sub.add_parser("run")
    common(s)
    s.add_argument("--task", choices=["tagging", "captions", "grounding", "summary"], required=True)
    s.add_argument("--chunk", type=int, default=15, help="questions per call, 0 = all in one")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-logprobs", action="store_true")
    s.add_argument("--coords", choices=["abs", "norm1000"], default="norm1000")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("perf")
    common(s)
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--n", type=int, default=40)
    s.set_defaults(fn=cmd_perf)

    s = sub.add_parser("florence")
    s.add_argument("--task", choices=["tagging", "captions", "grounding"], required=True)
    s.add_argument("--checkpoint", default="microsoft/Florence-2-large")
    s.add_argument("--model", default=None, help="run folder name (default: checkpoint basename)")
    s.add_argument("--device", default=None)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_florence)

    s = sub.add_parser("review")
    s.add_argument("--model", required=True)
    s.add_argument("--per-tag", type=int, default=5)
    s.add_argument("--decisions", nargs="*", help="decision JSON files downloaded from the review page")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("hf")
    s.add_argument("--backend", choices=["internvl", "paligemma"], required=True)
    s.add_argument("--task", choices=["tagging", "captions", "grounding", "summary"], required=True)
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--chunk", type=int, default=15)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_hf)

    s = sub.add_parser("metrics")
    s.add_argument("--model", required=True)
    s.set_defaults(fn=cmd_metrics)
    s = sub.add_parser("report")
    s.add_argument("--model", required=True)
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("compare")
    s.add_argument("--models", nargs="+", required=True)
    s.set_defaults(fn=cmd_compare)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
