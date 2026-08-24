"""Metrics over run JSONL rows. Pure functions, no I/O."""
import statistics
from collections import defaultdict
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

    return {"n_images": n_images, "overall": derived(tot),
            "per_tag": {slug: derived(c) for slug, c in sorted(per_tag.items())}}


def tagging_consistency(rows: list[dict]) -> dict[str, Any]:
    """Across repeats of the same image: mean Jaccard of positive sets and % images with identical answers."""
    by_img: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_img[r["image_id"]].append(r)
    jaccards, identical, n = [], 0, 0
    for img, reps in by_img.items():
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
    return {"n_images_with_repeats": n, "mean_jaccard": round(statistics.mean(jaccards), 3) if jaccards else None,
            "identical_pct": _pct(identical, n)}


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
    return {"n": len(vs), "mean_s": round(statistics.mean(vs), 2), "median_s": round(statistics.median(vs), 2),
            "p95_s": round(vs[min(len(vs) - 1, int(0.95 * len(vs)))], 2),
            "images_per_hour_serial": round(3600 / statistics.mean(vs), 0)}


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
    return {"n_images": len(rows), "empty_pct": _pct(empty, total),
            "mean_words": {k: round(statistics.mean(v), 1) for k, v in per_key.items() if v},
            "errors": sum(1 for r in rows if r.get("errors"))}


def grounding_stats(rows: list[dict], gemini: dict[int, dict]) -> dict[str, Any]:
    """Detection rate per target on images where Gemini tagged it present vs absent (sanity of localisation)."""
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pos_img": 0, "pos_detected": 0, "neg_img": 0, "neg_detected": 0})
    for r in rows:
        g = gemini.get(r["image_id"], {}).get("tags", {})
        for label, dets in (r.get("detections") or {}).items():
            if dets is None:
                continue
            key = "pos" if label in g else "neg"
            out[label][f"{key}_img"] += 1
            out[label][f"{key}_detected"] += int(bool(dets))
    return {label: {**c, "recall_vs_gemini": _pct(c["pos_detected"], c["pos_img"]),
                    "fp_rate_vs_gemini": _pct(c["neg_detected"], c["neg_img"])} for label, c in out.items()}
