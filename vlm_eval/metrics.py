"""Metrics over run JSONL rows. Pure functions, no I/O."""

import statistics
from collections import Counter, defaultdict
from typing import Any


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def tagging_agreement(rows: list[dict], gemini: dict[int, dict], *, repeat: int = 0) -> dict[str, Any]:
    """Model vs Gemini pseudo-GT on the tags Gemini actually judged (evaluable_slugs).

    Gemini positive = slug in gemini["tags"]; model positive = answers[slug] is True.
    Returns overall + per-tag TP/FP/FN/TN, accuracy, precision, recall, FPR, and unparsed rate.
    """
    per_tag: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "null": 0})
    n_images = 0
    for row in rows:
        if row.get("repeat", 0) != repeat:
            continue
        g = gemini.get(row["image_id"])
        if not g:
            continue
        n_images += 1
        g_pos = set(g.get("tags", {}))
        evaluable = set(g.get("evaluable_slugs") or g_pos) & set(row["answers"])
        for slug in evaluable:
            ans = row["answers"].get(slug)
            c = per_tag[slug]
            if ans is None:
                c["null"] += 1
                continue
            gp = slug in g_pos
            if ans and gp:
                c["tp"] += 1
            elif ans and not gp:
                c["fp"] += 1
            elif not ans and gp:
                c["fn"] += 1
            else:
                c["tn"] += 1
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "null": 0}
    for c in per_tag.values():
        for k in tot:
            tot[k] += c[k]

    def derived(c: dict[str, int]) -> dict[str, Any]:
        judged = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        return {
            **c,
            "n": judged + c["null"],
            "accuracy": _pct(c["tp"] + c["tn"], judged),
            "precision": _pct(c["tp"], c["tp"] + c["fp"]),
            "recall": _pct(c["tp"], c["tp"] + c["fn"]),
            "fpr": _pct(c["fp"], c["fp"] + c["tn"]),
            "unparsed_rate": _pct(c["null"], judged + c["null"]),
        }

    return {
        "n_images": n_images,
        "composition": composition(rows, repeat=repeat),
        "truncation": truncation(rows, repeat=repeat),
        "overall": derived(tot),
        "per_tag": {slug: derived(c) for slug, c in sorted(per_tag.items())},
    }


def was_truncated(row: dict) -> bool:
    """Did the model run out of budget on this row.

    Prefers the `completion` record written by the runner. Rows produced before that record existed
    are read from their error text — the only evidence they carry — and counted separately, because a
    row with no record and no phrase is not proof the model finished, only that nothing was written
    down.
    """
    record = row.get("completion")
    if isinstance(record, dict):
        return bool(record.get("truncated"))
    return any("cut off at the" in e for e in (row.get("errors") or []))


def truncation(rows: list[dict], *, repeat: int = 0) -> dict[str, Any]:
    """How much of the run the model never got to finish.

    A truncated call contributes no answers — they are recorded as unknown — so this does not distort
    accuracy. It does bound how much the run actually measured, which belongs next to the result rather
    than in an error string nobody reads.
    """
    considered = [r for r in rows if r.get("repeat", 0) == repeat]
    cut = [r for r in considered if was_truncated(r)]
    unrecorded = [r for r in considered if not isinstance(r.get("completion"), dict)]
    note = "answers from truncated calls are recorded as unknown, never parsed" if cut else ""
    if unrecorded:
        note = (note + "; " if note else "") + (
            f"{len(unrecorded)} row(s) predate the completion record — read from their error text, "
            "which is weaker evidence"
        )
    return {
        "images_affected": len(cut),
        "pct": _pct(len(cut), len(considered)),
        "rows_without_record": len(unrecorded),
        "note": note,
    }


def composition(rows: list[dict], *, repeat: int = 0) -> dict[str, Any]:
    """What the sample was made of — reported next to every result, never left implied.

    Indoor and outdoor images are asked different numbers of questions, so a subset that is mostly one
    of them shifts recall, latency and cost together. A run on a `--limit` prefix of a manifest whose
    order was built in blocks was once 98% indoor while the dataset was 63% — the numbers were right
    and described something other than the dataset.
    """
    kinds: Counter = Counter(r.get("image_type", "unknown") for r in rows if r.get("repeat", 0) == repeat)
    total = sum(kinds.values())
    return {
        "n_images": total,
        "by_type": dict(kinds.most_common()),
        "share_pct": {k: _pct(v, total) for k, v in kinds.most_common()},
    }


def tagging_consistency(rows: list[dict]) -> dict[str, Any]:
    """Across repeats of the same image: mean Jaccard of positive sets and % images with identical answers."""
    by_img: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_img[r["image_id"]].append(r)
    jaccards, identical, n = [], 0, 0
    for reps in by_img.values():
        if len(reps) < 2:
            continue
        n += 1
        sets = [frozenset(s for s, v in r["answers"].items() if v) for r in reps]
        ans = [tuple(sorted(r["answers"].items())) for r in reps]
        identical += int(all(a == ans[0] for a in ans))
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                jaccards.append(1.0 if not u else len(sets[i] & sets[j]) / len(u))
    return {
        "n_images_with_repeats": n,
        "mean_jaccard": round(statistics.mean(jaccards), 3) if jaccards else None,
        "identical_pct": _pct(identical, n),
    }


def tagging_chunk_comparison(rows_a: list[dict], rows_b: list[dict]) -> dict[str, Any]:
    """Answer agreement between two chunk sizes (e.g. 15 vs all) on the same images, repeat 0."""
    a = {r["image_id"]: r for r in rows_a if r.get("repeat", 0) == 0}
    b = {r["image_id"]: r for r in rows_b if r.get("repeat", 0) == 0}
    same = diff = null_b = 0
    for img in a.keys() & b.keys():
        for slug, va in a[img]["answers"].items():
            vb = b[img]["answers"].get(slug)
            if vb is None:
                null_b += 1
            elif va == vb:
                same += 1
            else:
                diff += 1
    return {"n_images": len(a.keys() & b.keys()), "agreement_pct": _pct(same, same + diff), "null_in_b": null_b}


def latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    vs = sorted(values)
    mean = statistics.mean(vs)
    return {
        "n": len(vs),
        "mean_s": round(mean, 2),
        "median_s": round(statistics.median(vs), 2),
        "p95_s": round(vs[min(len(vs) - 1, int(0.95 * len(vs)))], 2),
        # A stubbed or cached backend can report zero elapsed time; there is no throughput to derive
        # from that, and it must not take the whole metrics run down.
        "images_per_hour_serial": round(3600 / mean, 0) if mean > 0 else None,
    }


def caption_stats(rows: list[dict]) -> dict[str, Any]:
    per_key: dict[str, list[int]] = defaultdict(list)
    empty = total = 0
    for r in rows:
        for k, v in (r.get("captions") or {}).items():
            total += 1
            if not v:
                empty += 1
            else:
                per_key[k].append(len(v.split()))
    return {
        "n_images": len(rows),
        "empty_pct": _pct(empty, total),
        "mean_words": {k: round(statistics.mean(v), 1) for k, v in per_key.items() if v},
        "truncation": truncation(rows),
        "errors": sum(1 for r in rows if r.get("errors")),
    }


def grounding_stats(rows: list[dict], gemini: dict[str, dict]) -> dict[str, Any]:
    """Detection rate per target, scored only where the reference actually has a verdict.

    A target is comparable on an image only if the matching classification tag was asked there
    (`evaluable_slugs`). Targets with no corresponding tag at all, and image types that never get that
    question (a radiator is only asked indoors), are counted as `not_comparable` instead of being
    silently scored as absences — otherwise every detection of an unknown object reads as a false
    positive.
    """
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "pos_img": 0,
            "pos_detected": 0,
            "neg_img": 0,
            "neg_detected": 0,
            "not_comparable": 0,
            "not_comparable_detected": 0,
        }
    )
    for r in rows:
        g = gemini.get(r["image_id"]) or {}
        evaluable = set(g.get("evaluable_slugs") or [])
        tags = g.get("tags", {})
        for label, dets in (r.get("detections") or {}).items():
            if dets is None:
                continue
            c = out[label]
            if label not in evaluable:
                c["not_comparable"] += 1
                c["not_comparable_detected"] += int(bool(dets))
                continue
            key = "pos" if label in tags else "neg"
            c[f"{key}_img"] += 1
            c[f"{key}_detected"] += int(bool(dets))
    return {
        label: {
            **c,
            "recall_vs_reference": _pct(c["pos_detected"], c["pos_img"]),
            "fp_rate_vs_reference": _pct(c["neg_detected"], c["neg_img"]),
        }
        for label, c in out.items()
    }


def tagset_agreement(a: dict[str, set[str]], b: dict[str, set[str]]) -> dict[str, Any]:
    """Compare two runs' positive-tag sets over the images they share.

    Used to answer "does asking the questions in bigger batches change the answers" — and, when both
    runs used the *same* settings, "does the API even answer the same way twice".
    """
    common = set(a) & set(b)
    if not common:
        return {"n_images": 0}
    identical = sum(1 for k in common if a[k] == b[k])
    inter = sum(len(a[k] & b[k]) for k in common)
    union = sum(len(a[k] | b[k]) for k in common)
    lost: Counter = Counter()
    gained: Counter = Counter()
    for k in common:
        lost.update(a[k] - b[k])
        gained.update(b[k] - a[k])
    return {
        "n_images": len(common),
        "identical_pct": _pct(identical, len(common)),
        "jaccard_pct": _pct(inter, union),
        "tags_first": sum(len(a[k]) for k in common),
        "tags_second": sum(len(b[k]) for k in common),
        "lost": lost.most_common(8),
        "gained": gained.most_common(8),
    }
