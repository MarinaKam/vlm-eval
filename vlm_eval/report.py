"""Render per-model markdown reports (capability + performance tables).

`card` = hand-maintained facts (reports/cards/<model>.json: checkpoint, params, licence, GPU, VRAM,
size, cold start, hosting notes, verdict). `m` = computed metrics (cli `metrics` output).
"""
from typing import Any


def _v(x: Any, suffix: str = "") -> str:
    return "—" if x is None else f"{x}{suffix}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cost_per_1k(images_per_hour: float | None, gpu_usd_per_hour: float | None) -> float | None:
    if not images_per_hour or gpu_usd_per_hour is None:
        return None
    return round(gpu_usd_per_hour / images_per_hour * 1000, 4)


def monthly_cost(images_per_month: int, images_per_hour: float | None, gpu_usd_per_hour: float | None,
                 *, min_hours: float = 0.0) -> float | None:
    """GPU-hours needed at measured throughput x price. min_hours models an always-on instance (730 h)."""
    if not images_per_hour or gpu_usd_per_hour is None:
        return None
    hours = max(images_per_month / images_per_hour, min_hours)
    return round(hours * gpu_usd_per_hour, 2)


def render_model(card: dict, m: dict) -> str:
    tag = m.get("tagging", {})
    over = tag.get("agreement", {}).get("overall", {})
    perf = m.get("perf", {})
    cons = tag.get("consistency", {})
    iph = perf.get("images_per_hour_measured") or tag.get("latency", {}).get("images_per_hour_serial")
    price = card.get("gpu_usd_per_hour")
    c1k = cost_per_1k(iph, price)
    iph_note = (f"measured at concurrency {perf.get('concurrency')}" if perf.get("images_per_hour_measured")
                else "derived, serial")
    cap_rows = [
        ["Image tagging (boolean VQA, Gemini prompts)",
         card.get("cap_tagging", _v(over.get("accuracy"), "% agreement")), card.get("cap_tagging_notes", "")],
        ["Multi-tag inference (15 / all-in-one)",
         card.get("cap_multitag", _v(tag.get("chunk_all_agreement_pct"), "% agreement 15 vs all")),
         card.get("cap_multitag_notes", "")],
        ["Captioning (base / detailed)", card.get("cap_caption", ""), card.get("cap_caption_notes", "")],
        ["Multi-image understanding", card.get("cap_multiimage", ""), card.get("cap_multiimage_notes", "")],
        ["Property summarisation", card.get("cap_summary", ""), card.get("cap_summary_notes", "")],
        ["Detection / grounding", card.get("cap_detection", ""), card.get("cap_detection_notes", "")],
        ["Confidence scores", card.get("cap_confidence", ""), card.get("cap_confidence_notes", "")],
    ]
    perf_rows = [
        ["GPU", _v(card.get("gpu"))],
        ["VRAM (peak, measured)", _v(perf.get("vram_peak_gb") or card.get("vram_gb"), " GB")],
        ["Model size (weights)", _v(card.get("model_size_gb"), " GB")],
        ["Cold start (container up → /health)", _v(card.get("cold_start_s"), " s")],
        ["Latency / image (tagging, all chunks, serial)", _v(tag.get("latency", {}).get("mean_s"), " s mean")
         + f", p95 {_v(tag.get('latency', {}).get('p95_s'), ' s')}"],
        ["Images / hour", f"{_v(iph)} ({iph_note})"],
        ["Est. hosting cost / 1K images", _v(c1k, " USD") + f" @ {_v(price, ' USD/h')}"],
        ["Est. hosting cost / 10K images / month", _v(monthly_cost(10_000, iph, price), " USD")],
        ["Est. hosting cost / 100K images / month", _v(monthly_cost(100_000, iph, price), " USD")],
        ["Est. hosting cost / 1M images / month", _v(monthly_cost(1_000_000, iph, price), " USD")],
        ["Tagging agreement w/ Gemini (accuracy)", _v(over.get("accuracy"), "%")],
        ["Tagging false-positive rate vs Gemini", _v(over.get("fpr"), "%")],
        ["Tagging recall vs Gemini", _v(over.get("recall"), "%")],
        ["Unparsed answers", _v(over.get("unparsed_rate"), "%")],
        ["Consistency (3 repeats): identical / mean Jaccard",
         f"{_v(cons.get('identical_pct'), '%')} / {_v(cons.get('mean_jaccard'))}"],
    ]
    parts = [
        f"# {card.get('name', card.get('model'))}",
        "",
        "## Model",
        _table(["Field", "Value"], [
            ["Model name", card.get("name")], ["Checkpoint / version", card.get("checkpoint")],
            ["Parameter count", card.get("params")], ["Licence", card.get("licence")],
            ["Serving", card.get("serving")],
        ]),
        "",
        "## Capability results",
        _table(["Capability", "Result", "Notes"], cap_rows),
        "",
        "## Performance results",
        _table(["Metric", "Result"], perf_rows),
        "",
        "## Hosting requirements",
        card.get("hosting_md", "_TBD_"),
        "",
        "## Fine-tuning / customisation",
        card.get("finetune_md", "_TBD_"),
        "",
        "## Integration complexity",
        card.get("integration_md", "_TBD_"),
        "",
        "## Recommendation",
        f"**{card.get('verdict', 'TBD')}** — {card.get('verdict_reason', '')}",
        "",
    ]
    if tag.get("agreement", {}).get("per_tag"):
        rows = [[s, c["n"], c["tp"], c["fp"], c["fn"], c["tn"], _v(c["precision"]), _v(c["recall"]), _v(c["fpr"])]
                for s, c in tag["agreement"]["per_tag"].items()]
        parts += ["## Per-tag agreement with Gemini (repeat 0)",
                  _table(["tag", "n", "TP", "FP", "FN", "TN", "precision %", "recall %", "FPR %"], rows), ""]
    return "\n".join(parts)


def render_comparison(cards: list[dict], ms: list[dict]) -> str:
    headers = ["Model", "Params", "Licence", "Tag agreement %", "FPR %", "Recall %", "Identical (3x) %",
               "Latency/img s", "Img/h", "$/1K img", "Verdict"]
    rows = []
    for card, m in zip(cards, ms):
        tag = m.get("tagging", {})
        over = tag.get("agreement", {}).get("overall", {})
        iph = m.get("perf", {}).get("images_per_hour_measured") or tag.get("latency", {}).get("images_per_hour_serial")
        rows.append([card.get("name"), card.get("params"), card.get("licence"), _v(over.get("accuracy")),
                     _v(over.get("fpr")), _v(over.get("recall")), _v(tag.get("consistency", {}).get("identical_pct")),
                     _v(tag.get("latency", {}).get("mean_s")), _v(iph),
                     _v(cost_per_1k(iph, card.get("gpu_usd_per_hour"))), card.get("verdict", "TBD")])
    return "# Model comparison\n\n" + _table(headers, rows) + "\n"
