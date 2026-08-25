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


def monthly_cost(
    images_per_month: int, images_per_hour: float | None, gpu_usd_per_hour: float | None, *, min_hours: float = 0.0
) -> float | None:
    """GPU-hours needed at measured throughput x price. min_hours models an always-on instance (730 h)."""
    if not images_per_hour or gpu_usd_per_hour is None:
        return None
    hours = max(images_per_month / images_per_hour, min_hours)
    return round(hours * gpu_usd_per_hour, 2)


def _world(
    label: str, hardware: str | None, iph: float | None, usd_per_hour: float | None, *, source: str = "", note: str = ""
) -> list[str]:
    """One coherent set of numbers: a machine, its throughput, and — only if both are known — its cost."""
    if not hardware and not iph:
        return []
    lines = [f"**{label}**" + (f" — {hardware}" if hardware else ""), ""]
    rows = [["Throughput", f"{_v(iph)} images/hour" + (f" ({source})" if source else "")]]
    if usd_per_hour is not None:
        rows.append(["Price of that machine", f"${usd_per_hour:.4f}/hour"])
        rows.append(["Cost per 1 000 images", _v(cost_per_1k(iph, usd_per_hour), " USD")])
    else:
        rows.append(
            [
                "Cost",
                "not derived — no hourly price for this machine, and a price from a *different* "
                "machine would not describe anything",
            ]
        )
    lines.append(_table(["Metric", "Value"], rows))
    if note:
        lines += ["", note]
    return lines


def resolve_projection(card: dict, options: list | None = None) -> dict:
    """A card may name an option from the economics config instead of repeating its numbers.

    Two copies of a price are two chances to disagree, and a stale copy in a per-model report is
    exactly how a cost figure ends up eight times off in a document somebody forwards.
    """
    proj = dict(card.get("projection") or {})
    ref = proj.pop("option", None)
    if not ref:
        return proj
    match = next((o for o in (options or []) if o.name == ref), None)
    if match is None:
        known = ", ".join(o.name for o in (options or [])) or "none loaded"
        raise SystemExit(
            f"card projection references the option {ref!r}, which is not in the economics config (known: {known})"
        )
    proj.setdefault("hardware", match.name)
    proj.setdefault("usd_per_hour", match.price)
    proj.setdefault("images_per_hour", match.throughput_per_hour)
    proj.setdefault("source", f"from economics.json option {ref!r}")
    return proj


def _completion_line(m: dict, tagging_trunc: dict) -> str:
    """Truncation for every task, not only tagging.

    A caption run that spends its whole budget reasoning comes back empty, and without this it reads as
    a model with nothing to say about the picture.
    """
    per_task = m.get("completion") or {}
    if not per_task:
        # Metrics written before completion was recorded per task. Say so — the fallback used to render
        # its missing values as "— images (—)", which reads as a number nobody bothered to fill in.
        if tagging_trunc.get("images_affected") is None:
            return "not recorded — these metrics predate per-task completion tracking"
        return (
            f"{tagging_trunc['images_affected']} images ({_v(tagging_trunc.get('pct'), '%')}) in tagging "
            "— recorded as unknown, not parsed"
        )
    parts = [
        f"{task} {t.get('images_affected', 0)} ({_v(t.get('pct'), '%')})"
        for task, t in per_task.items()
        if t.get("images_affected")
    ]
    if not parts:
        return "none — every call finished within its token budget"
    return ", ".join(parts) + " — recorded as unknown, not parsed"


def _provenance_line(m: dict) -> str:
    """Whether these files can be shown as a clean measurement, stated in the report itself.

    A file whose settings were never recorded is not disqualified — it is labelled, so the label has to
    travel with the numbers instead of living in somebody's memory of how the run was started.
    """
    per_task = m.get("provenance") or {}
    if not per_task:
        return "not recorded"
    unclean = {task: d for task, d in per_task.items() if d.get("status") != "verified"}
    if not unclean:
        return "verified — every run file records the settings that produced it"
    return (
        "; ".join(
            f"{task}: {d.get('status')}" + (f" ({d['unverified_rows']} rows)" if d.get("unverified_rows") else "")
            for task, d in unclean.items()
        )
        + " — settings asserted, not verified"
    )


def render_model(card: dict, m: dict, *, options: list | None = None) -> str:
    tag = m.get("tagging", {})
    over = tag.get("agreement", {}).get("overall", {})
    perf = m.get("perf", {})
    cons = tag.get("consistency", {})
    iph = perf.get("images_per_hour_measured") or tag.get("latency", {}).get("images_per_hour_serial")
    iph_note = (
        f"measured at concurrency {perf.get('concurrency')}"
        if perf.get("images_per_hour_measured")
        else "derived from serial latency"
    )
    # Two worlds, never multiplied together: what this run actually did, and what the hardware you
    # would deploy on is projected to do. A projection is only shown when the card supplies both its
    # throughput and its price, and it always says where the throughput came from.
    projection = resolve_projection(card, options)
    cap_rows = [
        [
            "Image tagging (boolean VQA, Gemini prompts)",
            card.get("cap_tagging", _v(over.get("accuracy"), "% agreement")),
            card.get("cap_tagging_notes", ""),
        ],
        [
            "Multi-tag inference (15 / all-in-one)",
            card.get("cap_multitag", _v(tag.get("chunk_all_agreement_pct"), "% agreement 15 vs all")),
            card.get("cap_multitag_notes", ""),
        ],
        ["Captioning (base / detailed)", card.get("cap_caption", ""), card.get("cap_caption_notes", "")],
        ["Multi-image understanding", card.get("cap_multiimage", ""), card.get("cap_multiimage_notes", "")],
        ["Property summarisation", card.get("cap_summary", ""), card.get("cap_summary_notes", "")],
        ["Detection / grounding", card.get("cap_detection", ""), card.get("cap_detection_notes", "")],
        ["Confidence scores", card.get("cap_confidence", ""), card.get("cap_confidence_notes", "")],
    ]
    comp = tag.get("agreement", {}).get("composition", {})
    trunc = tag.get("agreement", {}).get("truncation", {})
    comp_text = ", ".join(f"{k} {v}" for k, v in (comp.get("by_type") or {}).items()) or "—"
    perf_rows = [
        ["Sample the tagging numbers describe", f"{comp.get('n_images', '—')} images ({comp_text})"],
        ["How it was served", card.get("serving_caveat") or card.get("measured_on", "—")],
        ["Answers the model never finished", _completion_line(m, trunc)],
        ["Provenance of the run files", _provenance_line(m)],
        ["GPU", _v(card.get("gpu"))],
        ["VRAM (peak, measured)", _v(perf.get("vram_peak_gb") or card.get("vram_gb"), " GB")],
        ["Model size (weights)", _v(card.get("model_size_gb"), " GB")],
        ["Cold start (container up → /health)", _v(card.get("cold_start_s"), " s")],
        [
            "Latency / image (tagging, all chunks, serial)",
            _v(tag.get("latency", {}).get("mean_s"), " s mean")
            + f", p95 {_v(tag.get('latency', {}).get('p95_s'), ' s')}",
        ],
        ["Tagging agreement with the reference", _v(over.get("accuracy"), "%")],
        ["Tagging false-positive rate (claimed a tag the reference did not)", _v(over.get("fpr"), "%")],
        ["Tagging recall (found the tags the reference has)", _v(over.get("recall"), "%")],
        ["Unparsed answers", _v(over.get("unparsed_rate"), "%")],
        [
            "Consistency (3 repeats): identical / mean Jaccard",
            f"{_v(cons.get('identical_pct'), '%')} / {_v(cons.get('mean_jaccard'))}",
        ],
    ]
    parts = [
        f"# {card.get('name', card.get('model'))}",
        "",
        "## Model",
        _table(
            ["Field", "Value"],
            [
                ["Model name", card.get("name")],
                ["Checkpoint / version", card.get("checkpoint")],
                ["Parameter count", card.get("params")],
                ["Licence", card.get("licence")],
                ["Serving", card.get("serving")],
            ],
        ),
        "",
        "## Capability results",
        _table(["Capability", "Result", "Notes"], cap_rows),
        "",
        "## Performance results",
        _table(["Metric", "Result"], perf_rows),
        "",
        "### Speed and cost, by machine",
        "",
        *_world(
            "Measured on this run",
            card.get("measured_on"),
            iph,
            card.get("measured_usd_per_hour"),
            source=iph_note,
            note=card.get("measured_note", ""),
        ),
        "",
        *_world(
            "Projected for deployment",
            projection.get("hardware"),
            projection.get("images_per_hour"),
            projection.get("usd_per_hour"),
            source=projection.get("source", "projection, not measured here"),
            note=projection.get("note", ""),
        ),
        "",
        "Monthly totals, break-even volumes and fixed costs such as a cluster fee live in "
        "`economics.md` — that report models always-on versus autoscaled capacity, which a single "
        "cost-per-hour figure cannot.",
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
        rows = [
            [s, c["n"], c["tp"], c["fp"], c["fn"], c["tn"], _v(c["precision"]), _v(c["recall"]), _v(c["fpr"])]
            for s, c in tag["agreement"]["per_tag"].items()
        ]
        parts += [
            "## Per-tag agreement with the reference (repeat 0)",
            _reference_note(m),
            "",
            _table(["tag", "n", "TP", "FP", "FN", "TN", "precision %", "recall %", "FPR %"], rows),
            "",
            "A dash is not a missing measurement: precision is undefined where the model claimed the "
            "tag nowhere, and recall where the reference has it nowhere in this sample. `n` is how many "
            "images the tag was comparable on.",
            "",
        ]
    return "\n".join(parts)


def _reference_note(m: dict) -> str:
    """What every agreement figure in this report does and does not mean.

    The single most quotable mistake this report could invite is reading "agreement" as "accuracy".
    They differ whenever the reference is wrong, and on the disputed cases we adjudicated by hand the
    reference was wrong about as often as the model — so the note carries the adjudication numbers
    when they exist, and says plainly that they are missing when they do not.
    """
    lines = [
        "Every number here is **agreement with the current pipeline's answers**, not accuracy against",
        "the truth. One image-and-tag pair is one yes/no question. **TP** the reference says yes and so",
        "does the model; **FP** the model claims a tag the reference does not have — it invented a",
        "feature; **FN** the reference has the tag and the model missed it; **TN** both say no.",
        "**Recall** is the share of the reference's tags the model found; **precision** is the share of",
        "the model's claims the reference confirms.",
        "",
        "A disagreement therefore does not say who is wrong.",
    ]
    manual = (m.get("tagging") or {}).get("manual") or {}
    n = manual.get("n") or manual.get("verdicts")
    if n:
        model_right = manual.get("model_correct_pct")
        ref_right = manual.get("gemini_correct_pct")
        lines.append(
            f"On the {n} disputed pairs a human adjudicated, the model was right "
            f"{_v(model_right, '%')} of the time and the reference {_v(ref_right, '%')} — they fail in "
            "opposite directions rather than one being better, so recall below 100% is partly the "
            "model missing real features and partly the reference tagging things that are not there."
        )
    else:
        lines.append(
            "No disagreements have been adjudicated by hand for this model, so how much of the gap is "
            "the model's error and how much is the reference's is **unmeasured**."
        )
    return "\n".join(lines)


def render_comparison(cards: list[dict], ms: list[dict]) -> str:
    headers = [
        "Model",
        "Params",
        "Licence",
        "Tag agreement %",
        "FPR %",
        "Recall %",
        "Identical (3x) %",
        "Latency/img s",
        "Img/h (serial)",
        "$/1K img",
        "Verdict",
    ]
    rows = []
    for card, m in zip(cards, ms):
        tag = m.get("tagging", {})
        over = tag.get("agreement", {}).get("overall", {})
        # Serial throughput for every row, even where a concurrent measurement exists. Mixing the two
        # put 429 images/hour (measured at concurrency 2) beside 145 (derived serially) and read as a
        # 3x difference between models that are 1.85x apart. The concurrent figure belongs in the
        # per-model report, where it can say what concurrency produced it.
        iph = tag.get("latency", {}).get("images_per_hour_serial")
        rows.append(
            [
                card.get("name"),
                card.get("params"),
                card.get("licence"),
                _v(over.get("accuracy")),
                _v(over.get("fpr")),
                _v(over.get("recall")),
                _v(tag.get("consistency", {}).get("identical_pct")),
                _v(tag.get("latency", {}).get("mean_s")),
                _v(iph),
                _v(cost_per_1k(iph, card.get("gpu_usd_per_hour"))),
                card.get("verdict", "TBD"),
            ]
        )
    return "# Model comparison\n\n" + _table(headers, rows) + "\n"
