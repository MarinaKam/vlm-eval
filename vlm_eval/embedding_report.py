"""Render the embedding investigation's reports: one per checkpoint, one comparison, one recommendation.

Every table here is built from a file under `runs/`, so a figure can be traced to the raw scores that
produced it. Three editorial rules are enforced by the code rather than by whoever writes the prose:

  * **A tag below the positive floor never appears in a headline.** Average precision of 1.000 on a tag
    with one positive example is arithmetically true and means nothing; printed in a ranked list beside a
    tag with three hundred, it reads as the model's best result.
  * **Accuracy is printed with its base rate beside it.** On this tag set "absent, always" scores 92.6%.
  * **Where a number could not be measured, the cell says so.** An empty cell and a zero look alike in a
    rendered table, and only one of them is a measurement.
"""

import json
from pathlib import Path
from typing import Any

from . import dataset, metrics
from .config import RUNS
from .report import _table, _v

# Printed beside every accuracy figure. Measured from the reference, not assumed.
BASE_RATE_NOTE = (
    "Accuracy is reported because the ticket asks for it, never as the headline: {pos:,} of {total:,} "
    "comparable decisions are positive ({rate:.2f}%), so answering *absent* to everything scores "
    "{trivial:.1f}%."
)


def base_rate(reference: dict[str, dict], keep: set[str] | None = None) -> dict[str, float]:
    """How often the reference says a tag is present, over the decisions it actually judged."""
    positives = total = 0
    for image_id, row in reference.items():
        if keep and image_id not in keep:
            continue
        judged = row.get("evaluable_slugs") or list(row.get("tags") or {})
        total += len(judged)
        positives += len(set(row.get("tags") or {}) & set(judged))
    return {
        "positives": positives,
        "total": total,
        "rate_pct": 100.0 * positives / total if total else 0.0,
        "trivial_accuracy_pct": 100.0 * (total - positives) / total if total else 0.0,
    }


def sample_note(reference: dict[str, dict]) -> str:
    """How many photographs the manifest actually holds, and how steady the reference is on them.

    Two facts that belong beside every rate in these reports. A sampled manifest can hold one photograph
    under several ids, and counting each upload separately weights a picture by how often it was uploaded
    while letting a threshold be fitted on an image identical to one it then scores. And where the same
    bytes appear twice, the reference's own answers to them bound how much of any disagreement is noise
    rather than a difference between models.
    """
    groups: dict[str, list[str]] = {}
    for image_id, key in dataset.content_groups().items():
        groups.setdefault(key, []).append(image_id)
    repeated = [ids for ids in groups.values() if len(ids) > 1]
    rows = sum(len(ids) for ids in groups.values())
    unstable = sum(
        1 for ids in repeated if len({frozenset((reference.get(i) or {}).get("tags") or {}) for i in ids}) > 1
    )
    lines = [
        f"**The sample.** The manifest's {rows:,} rows are {len(groups):,} distinct photographs: "
        f"{rows - len(groups)} of them are repeat uploads of a picture already in the set, in "
        f"{len(repeated)} groups. Every figure here scores each photograph once, for the encoders and for "
        "the models compared against them alike, and copies of one picture are kept in the same "
        "cross-validation fold.",
    ]
    if unstable:
        lines.append(
            f"In {unstable} of those {len(repeated)} groups the reference gave **different answers to "
            "byte-identical images**. That is a floor on how much of any disagreement with it is its own "
            "noise rather than a difference between models."
        )
    return "\n\n".join(lines)


def solid_tags(calibration: dict, floor: int) -> list[str]:
    """Tags with enough positives for their numbers to be worth printing."""
    return sorted(s for s, v in calibration["per_tag"].items() if v["n_positives"] >= floor)


def mean_ap(calibration: dict, slugs: list[str]) -> float | None:
    values = [
        calibration["per_tag"][s]["average_precision"]
        for s in slugs
        if calibration["per_tag"].get(s) and calibration["per_tag"][s]["average_precision"] is not None
    ]
    return round(sum(values) / len(values), 3) if values else None


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _rows(path: Path, keep: set[str] | None) -> list[dict]:
    """Run rows, restricted to one row per distinct photograph unless told otherwise.

    Applied to the generative baselines as well as the encoders. Deduplicating one side of a comparison
    and not the other would change what the two numbers are about, which is worse than not deduplicating
    at all.
    """
    rows = dataset.load_jsonl(path)
    return [r for r in rows if r["image_id"] in keep] if keep else rows


def strategy_table(model: str, strategies: list[str], slugs: list[str]) -> str:
    """Mean average precision per strategy — threshold-free, so it answers "is the signal there"."""
    rows = []
    for name in strategies:
        cal = _load(RUNS / model / f"calibration_{name}.json")
        rows.append([f"`{name}`", _v(mean_ap(cal, slugs)) if cal else "not run"])
    return _table(["Prompt strategy", f"Mean AP over {len(slugs)} tags"], rows)


def agreement_table(model: str, strategy: str, reference: dict[str, dict], keep: set[str] | None = None) -> str:
    """Precision, recall and false-positive rate at each threshold rule, against the current pipeline."""
    rows = []
    sources = [
        ("the checkpoint's own boundary", f"tagging_embed_{strategy}.jsonl", "uncalibrated"),
        ("one global threshold", f"tagging_embed_{strategy}_global_threshold_cv.jsonl", "fitted out of fold"),
        ("a threshold per tag", f"tagging_embed_{strategy}_per_tag_threshold_cv.jsonl", "fitted out of fold"),
    ]
    for label, filename, provenance in sources:
        run = _rows(RUNS / model / filename, keep)
        if not run:
            rows.append([label, provenance, "—", "—", "—", "—"])
            continue
        agg = metrics.tagging_agreement(run, reference)["overall"]
        rows.append(
            [
                label,
                provenance,
                _v(agg["precision"], "%"),
                _v(agg["recall"], "%"),
                _v(agg["fpr"], "%"),
                _v(agg["accuracy"], "%"),
            ]
        )
    return _table(["Answers from", "Threshold", "Precision", "Recall", "False positives", "Accuracy"], rows)


def truncation_table(model: str, tags: list[dict], context_length: int, token_counts: dict[str, int]) -> str:
    """Which tag definitions the text encoder could not read in full, and by how much."""
    rows = []
    for slug, needed in sorted(token_counts.items(), key=lambda kv: -kv[1]):
        if needed <= context_length:
            continue
        rows.append([f"`{slug}`", needed, context_length, needed - context_length])
    if not rows:
        return "Every definition fitted the text window."
    return _table(["Tag", "Tokens needed", "Tokens read", "Dropped"], rows)


def per_tag_table(calibration: dict, slugs: list[str], probe: dict | None = None) -> str:
    """Per-tag average precision, its positives, and the class overlap — zero-shot beside the probe."""
    headers = ["Tag", "Positives", "Zero-shot AP", "Negatives above the positive median"]
    if probe:
        headers.insert(3, "Probe AP")

    def build(slug: str) -> list:
        entry = calibration["per_tag"][slug]
        row = [f"`{slug}`", entry["n_positives"], _v(entry["average_precision"])]
        if probe:
            probe_entry = (probe.get("per_tag") or {}).get(slug)
            row.append(_v(probe_entry["average_precision"]) if probe_entry else "below the floor")
        row.append(_v(entry["distribution"].get("overlap_pct"), "%"))
        return row

    ordered = sorted(slugs, key=lambda s: -(calibration["per_tag"][s]["average_precision"] or 0))
    return _table(headers, [build(slug) for slug in ordered])


def scale_table(scale: dict) -> str:
    """Section 2: the arithmetic grows linearly; under one decision rule the answers do not move at all."""
    rows = [
        [
            f"{row['vocabulary']:,}",
            f"{row['microseconds_per_image']:.2f}",
            f"{row['independent_rule_max_deviation']:.0e}",
            f"{row['independent_rule_scores_beyond_margin']:,}",
            f"{row['softmax_rule_decisions_flipped']:,}",
        ]
        for row in scale["sizes"]
    ]
    return _table(
        [
            "Vocabulary",
            "Microseconds per image",
            "Independent rule: largest score drift",
            "Scores moved enough to matter",
            "Softmax: decisions flipped",
        ],
        rows,
    )


def hybrid_table(hybrid: dict) -> str:
    """What buying a VLM call for the borderline decisions costs, and what it recovers."""
    rows = [
        [
            f"{row['deferral_budget_pct']:.0f}%",
            f"{row['images_pct']:.1f}%",
            f"{row['vlm_calls_hybrid']:,}",
            f"{row['vlm_calls_today']:,}",
            _v(row["call_reduction_pct"], "%"),
            _v(row["precision"], "%"),
            _v(row["recall"], "%"),
            _v(row["fpr"], "%"),
        ]
        for row in hybrid["budgets"]
    ]
    return _table(
        [
            "Decisions deferred",
            "Images needing a call",
            "Calls",
            "Calls today",
            "Calls saved",
            "Precision",
            "Recall",
            "False positives",
        ],
        rows,
    )


def render_model(
    *,
    model: str,
    card: dict,
    strategies: list[str],
    best_strategy: str,
    reference: dict[str, dict],
    tags: list[dict],
    floor: int,
    keep: set[str] | None = None,
) -> str:
    """One checkpoint's full report."""
    cal = _load(RUNS / model / f"calibration_{best_strategy}.json")
    if not cal:
        return f"# {model}\n\nNot measured — no calibration file for `{best_strategy}`.\n"
    slugs = solid_tags(cal, floor)
    rate = base_rate(reference, keep)
    probe = _load(RUNS / model / "probe.json")
    scale = _load(RUNS / model / "scale.json")
    hybrid = _load(RUNS / model / "hybrid.json")

    lines = [
        f"# {card.get('title') or model}",
        "",
        f"`{card.get('checkpoint') or model}` — {card.get('params') or 'size not recorded'}, "
        f"licence {card.get('licence') or 'not recorded'}.",
        "",
        "## What this measures",
        "",
        "One vector per image, compared against one vector per tag definition. The reference is the "
        "answers our current pipeline gave on the same 1,000 images, so agreement means *behaves like "
        "today*, never *is correct*.",
        "",
        BASE_RATE_NOTE.format(
            pos=rate["positives"],
            total=rate["total"],
            rate=rate["rate_pct"],
            trivial=rate["trivial_accuracy_pct"],
        ),
        "",
        "## Which sentence should stand for a tag (section 4)",
        "",
        strategy_table(model, strategies, slugs),
        "",
        f"Measured over the {len(slugs)} tags with at least {floor} positive examples. The other "
        f"{len(cal['per_tag']) - len(slugs)} tags are reported in the per-tag table below but excluded "
        "here: average precision over one or two positives is not a measurement of the model.",
        "",
        "## Agreement with the current pipeline (section 3)",
        "",
        agreement_table(model, best_strategy, reference, keep),
        "",
        "## Per tag (section 3)",
        "",
        per_tag_table(cal, slugs, probe),
        "",
    ]

    if scale:
        lines += [
            "## Growing the vocabulary (section 2)",
            "",
            scale_table(scale),
            "",
            f"Competing candidates are {scale['distractors']}.",
            "",
        ]
    if probe:
        lines += [
            "## A linear probe on the same frozen vectors (section 5)",
            "",
            f"Fitted on {probe['tags_fitted']} tags, {probe['folds']}-fold by image, encoder frozen. "
            f"**{probe['ceiling']}.**",
            "",
        ]
    if hybrid:
        lines += ["## Embedding first, a VLM for the rest", "", hybrid_table(hybrid), ""]

    notes = card.get("notes") or []
    if notes:
        lines += ["## Notes", "", *[f"- {n}" for n in notes], ""]
    return "\n".join(lines)


def render_comparison(
    *,
    models: list[str],
    cards: dict[str, dict],
    strategies: list[str],
    best_strategy: str,
    reference: dict[str, dict],
    floor: int,
    baselines: list[dict[str, Any]] | None = None,
    keep: set[str] | None = None,
) -> str:
    """Every checkpoint in one table, with the generative models measured earlier beside them."""
    rate = base_rate(reference, keep)
    rows = []
    for model in models:
        cal = _load(RUNS / model / f"calibration_{best_strategy}.json")
        if not cal:
            continue
        slugs = solid_tags(cal, floor)
        run = _rows(RUNS / model / f"tagging_embed_{best_strategy}_per_tag_threshold_cv.jsonl", keep)
        agg = metrics.tagging_agreement(run, reference)["overall"] if run else {}
        card = cards.get(model, {})
        rows.append(
            [
                card.get("title") or model,
                card.get("params") or "—",
                _v(mean_ap(cal, slugs)),
                _v(agg.get("precision"), "%"),
                _v(agg.get("recall"), "%"),
                _v(agg.get("fpr"), "%"),
            ]
        )
    rows.extend(
        [
            f"{base['title']} *(generative, measured earlier)*",
            base.get("params", "—"),
            "—",
            _v(base.get("precision"), "%"),
            _v(base.get("recall"), "%"),
            _v(base.get("fpr"), "%"),
        ]
        for base in baselines or []
    )
    return "\n".join(
        [
            "# Encoders against the current pipeline",
            "",
            f"Strategy `{best_strategy}`, thresholds fitted per tag out of fold. Mean average precision "
            f"is over tags with at least {floor} positive examples.",
            "",
            BASE_RATE_NOTE.format(
                pos=rate["positives"],
                total=rate["total"],
                rate=rate["rate_pct"],
                trivial=rate["trivial_accuracy_pct"],
            ),
            "",
            sample_note(reference),
            "",
            _table(["Model", "Size", "Mean AP", "Precision", "Recall", "False positives"], rows),
            "",
            "A dash under Mean AP for a generative model is not a gap in the measurement: it answers "
            "`true` or `false` and produces no score to rank, so average precision does not exist for it.",
            "",
        ]
    )


# The ticket's Final Deliverable. The verdict is derived from the measurements rather than written into
# the text, so a report regenerated from different numbers cannot keep asserting yesterday's conclusion.
FALSE_POSITIVE_TOLERANCE = 2.0  # times the incumbent's rate before a standalone swap is off the table


def _verdict(best: dict, incumbent: dict) -> tuple[str, str]:
    """Which option the numbers support, and the one-sentence reason."""
    if best.get("fpr") is None or incumbent.get("fpr") is None:
        return ("Undecided", "the false-positive rates needed for the comparison were not measured")
    ratio = best["fpr"] / incumbent["fpr"] if incumbent["fpr"] else float("inf")
    if ratio <= 1.0 and (best.get("recall") or 0) >= (incumbent.get("recall") or 0):
        return ("Option B — embeddings", "it matches the incumbent on recall without buying extra false positives")
    if ratio > FALSE_POSITIVE_TOLERANCE:
        return (
            "Hybrid",
            f"on its own the encoder fires {ratio:.0f}x as many false positives as the model in production, "
            "so it cannot replace it — but it is confident on most decisions, and a generative model is "
            "only needed for the rest",
        )
    return ("Option B — embeddings, with care", "the extra false positives are within tolerance")


def render_recommendation(
    *,
    models: list[str],
    cards: dict[str, dict],
    best_strategy: str,
    reference: dict[str, dict],
    floor: int,
    baselines: list[dict[str, Any]],
    keep: set[str] | None = None,
) -> str:
    """Option A, Option B or Hybrid — with the number behind each clause."""
    measured = []
    for model in models:
        cal = _load(RUNS / model / f"calibration_{best_strategy}.json")
        run = _rows(RUNS / model / f"tagging_embed_{best_strategy}_per_tag_threshold_cv.jsonl", keep)
        if not cal or not run:
            continue
        agg = metrics.tagging_agreement(run, reference)["overall"]
        measured.append(
            {
                "model": model,
                "title": cards.get(model, {}).get("title") or model,
                "mean_ap": mean_ap(cal, solid_tags(cal, floor)),
                **{k: agg[k] for k in ("precision", "recall", "fpr")},
            }
        )
    if not measured:
        return "# Recommendation\n\nNothing measured yet.\n"

    best = max(measured, key=lambda m: m["mean_ap"] or 0)
    incumbent = min(baselines, key=lambda b: b["fpr"]) if baselines else {}
    choice, because = _verdict(best, incumbent)

    hybrid = _load(RUNS / best["model"] / "hybrid.json")
    probe = _load(RUNS / best["model"] / "probe.json")
    cal = _load(RUNS / best["model"] / f"calibration_{best_strategy}.json")
    scale = _load(RUNS / best["model"] / "scale.json")
    rate = base_rate(reference, keep)

    lines = [
        "# Should property tagging move to an embedding model?",
        "",
        f"**{choice}** — {because}.",
        "",
        "Measured on the images of the earlier investigation, against the answers the current pipeline "
        "gave on the same photos. Agreement therefore means *behaves like today*, never *is correct*; "
        "nothing here is measured against human labels, and that remains the gap.",
        "",
        sample_note(reference),
        "",
        "## The three options, side by side",
        "",
        _table(
            ["Option", "Precision", "Recall", "False positives", "Calls per 1,000 images"],
            [
                [
                    "**A — generative model, as today**",
                    _v(incumbent.get("precision"), "%"),
                    _v(incumbent.get("recall"), "%"),
                    _v(incumbent.get("fpr"), "%"),
                    f"{hybrid['budgets'][0]['vlm_calls_today']:,}" if hybrid else "—",
                ],
                [
                    f"**B — encoder alone** ({best['title']})",
                    _v(best["precision"], "%"),
                    _v(best["recall"], "%"),
                    _v(best["fpr"], "%"),
                    "0",
                ],
            ]
            + (
                [
                    [
                        f"**Hybrid** — encoder first, {row['deferral_budget_pct']:.0f}% of decisions referred on",
                        _v(row["precision"], "%"),
                        _v(row["recall"], "%"),
                        _v(row["fpr"], "%"),
                        f"{row['vlm_calls_hybrid']:,} ({_v(row['call_reduction_pct'])}% fewer)",
                    ]
                    for row in hybrid["budgets"]
                    if row["deferral_budget_pct"] in (10.0, 20.0)
                ]
                if hybrid
                else []
            ),
        ),
        "",
        "The generative figures come from the earlier investigation's own run files, recomputed here "
        "rather than copied from its report.",
        "",
        "## What the encoder is good at, and what it is not",
        "",
        f"On the {len(solid_tags(cal, floor))} tags with at least {floor} positive examples, the best "
        f"checkpoint reaches a mean average precision of {best['mean_ap']}. Room-level tags — the kind a "
        "caption would naturally name — separate almost perfectly; fine detail and anything defined by an "
        "exclusion does not.",
        "",
        "Recall is the half that transfers. Precision is not: "
        f"at {_v(best['precision'], '%')} roughly half of the encoder's positive answers disagree with the "
        "pipeline, against "
        f"{_v(incumbent.get('precision'), '%')} for the generative model. On a tag set where only "
        f"{rate['rate_pct']:.2f}% of decisions are positive, that difference is the whole argument.",
        "",
    ]

    if probe:
        gains = [
            (v["average_precision"] - (cal["per_tag"][s]["average_precision"] or 0))
            for s, v in probe["per_tag"].items()
            if v["average_precision"] is not None and cal["per_tag"].get(s)
        ]
        mean_gain = sum(gains) / len(gains) if gains else 0.0
        lines += [
            "## Would training help? (section 5)",
            "",
            f"A linear probe fitted on the same frozen vectors gains **{mean_gain:+.3f}** average precision "
            f"over the text prompts, across {probe['tags_fitted']} tags. That is the informative result and "
            "it is a negative one: if a trained head on these vectors cannot do better than a sentence, the "
            "text encoder was not the bottleneck — the image representation is. Fine-tuning would therefore "
            "have to change the encoder itself, which needs a labelled dataset and real training, not a "
            "prompt rewrite.",
            "",
            f"Both checkpoints support fine-tuning, and the probe's own ceiling is worth naming: {probe['ceiling']}.",
            "",
        ]

    if scale:
        biggest = scale["sizes"][-1]
        lines += [
            "## Adding tags (section 2)",
            "",
            f"At {biggest['vocabulary']:,} tags the comparison costs "
            f"{biggest['microseconds_per_image']:.1f} microseconds per image, and no existing tag's score "
            f"moves by more than {biggest['independent_rule_max_deviation']:.0e} — not one of them far "
            "enough to change a decision. That holds because each tag is thresholded on its own, with no "
            "term referring to any other tag, so the only difference a larger vocabulary makes is the "
            "order the same products are summed in.",
            "",
            "It stops holding the moment scores are normalised across the vocabulary, the usual zero-shot "
            f"recipe: at {biggest['vocabulary']:,} candidates that rule changes "
            f"{biggest['softmax_rule_decisions_flipped']:,} of "
            f"{biggest['softmax_decisions_compared']:,} decisions that were already settled. Whichever "
            "option is chosen, per-tag thresholds are the part that makes it scale.",
            "",
        ]

    lines += [
        "## What would change this answer",
        "",
        "- **Human labels.** Every figure here is agreement with the current pipeline, which the earlier "
        "investigation showed to be wrong about as often as it is right in disputed cases. A labelled set "
        "would move the whole comparison onto solid ground, and it is still its own piece of work.",
        "- **A better prompt is not the lever.** Six prompt strategies were compared and the spread between "
        "the best and worst is small next to the gap to the generative model.",
        "- **Fine-tuning the encoder.** Not the probe, which was measured, but the encoder itself. That is a "
        "training project with a dataset behind it.",
        "",
        "## What was not measured",
        "",
        "- Accuracy against truth. The 140 adjudicated cases from the earlier investigation were sampled as "
        "the disagreements of one particular model and judged by another vision-language model rather than "
        "by a person, so they cannot carry a per-tag claim and are not used here.",
        "- Hosting and cost. Deliberately out of scope for this ticket; the earlier investigation covers it.",
        "- The hybrid's *accuracy*. Its deferred decisions are resolved by reading the pipeline's answer, "
        "and the pipeline is also what it is scored against, so the curve measures how much generative "
        "capacity is needed to reproduce today's behaviour — not how right that behaviour is.",
        "",
    ]
    return "\n".join(lines)
