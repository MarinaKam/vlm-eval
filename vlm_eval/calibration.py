"""Turning similarity scores into yes/no answers, and measuring that honestly. Pure functions, no I/O.

A generative model answers `true`; an encoder answers `0.2413`. Somewhere a threshold has to turn the
number into a tag, and where that threshold comes from decides whether the reported quality means
anything:

  * **Fit it on the same decisions you report on and the number is inflated.** One threshold per tag is
    sixty free parameters fitted to the very images being scored. So thresholds are fitted under
    k-fold cross-validation by image: each fold is scored with a threshold that never saw it.
  * **A tag with one positive example cannot have its own threshold.** Fitting one anyway produces a
    number with a confidence interval wider than the scale it sits on. Tags below a stated floor fall
    back to the global threshold, and the report says which got their own.
  * **Average precision is the threshold-free answer.** It says whether the signal is in the embedding
    at all, separately from whether a threshold was chosen well, so the two failures cannot be confused.

Accuracy is deliberately not the headline anywhere here: on a tag set where 7% of decisions are
positive, answering "absent" to everything scores 93%.
"""

import hashlib
import statistics
from collections import defaultdict
from typing import Any, NamedTuple

# A per-tag threshold needs enough positives to be a measurement rather than a coin flip. 30 is a
# judgement, not a law, which is why it is a parameter and why the report names how many tags cleared it.
MIN_POSITIVES_FOR_OWN_THRESHOLD = 30


class Decision(NamedTuple):
    """One comparable (image, tag) pair: what the encoder scored, and what the reference said."""

    image_id: str
    slug: str
    score: float
    reference_positive: bool


def fold_of(image_id: str, *, folds: int, seed: int) -> int:
    """Which fold an image belongs to — by digest, so the split cannot depend on iteration order.

    Splitting by *image* rather than by decision matters: two tags on the same photo are not
    independent samples, and letting them land in different folds leaks the image across the split.
    """
    h = hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()
    return int(h[:8], 16) % folds


def average_precision(decisions: list[Decision]) -> float | None:
    """Area under the precision-recall curve, tie-aware. None when the reference has no positives.

    Equal scores are credited as a group rather than in whatever order they were listed: ranking ties
    arbitrarily inflates AP, and with a coarse scoring rule ties are not rare.
    """
    ordered = sorted(decisions, key=lambda d: -d.score)
    n_pos = sum(d.reference_positive for d in ordered)
    if not n_pos:
        return None
    total = seen = hits = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j].score == ordered[i].score:
            j += 1
        group = ordered[i:j]
        seen += len(group)
        hits += sum(d.reference_positive for d in group)
        total += sum(d.reference_positive for d in group) * (hits / seen)
        i = j
    return round(total / n_pos, 4)


def _counts(decisions: list[Decision], threshold: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for d in decisions:
        predicted = d.score >= threshold
        if predicted and d.reference_positive:
            tp += 1
        elif predicted:
            fp += 1
        elif d.reference_positive:
            fn += 1
    return tp, fp, fn


def f1_at(decisions: list[Decision], threshold: float) -> float:
    tp, fp, fn = _counts(decisions, threshold)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def candidate_thresholds(decisions: list[Decision]) -> list[float]:
    """Midpoints between consecutive distinct scores, plus one above every score.

    Midpoints rather than the scores themselves: a threshold equal to an observed score makes the
    result depend on whether the comparison is `>=` or `>`, which is not a property of the data.
    """
    scores = sorted({d.score for d in decisions})
    if not scores:
        return []
    mids = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    return [scores[0] - 1e-6, *mids, scores[-1] + 1e-6]


def f1_scan(decisions: list[Decision]) -> tuple[float | None, float]:
    """Reference implementation: evaluate every candidate threshold from scratch.

    Obviously correct and quadratic. Kept because `best_threshold` must agree with it — a test drives
    both over random data, including heavy score ties, so the fast version cannot drift unnoticed.
    """
    if not any(d.reference_positive for d in decisions):
        return None, 0.0
    best, best_f1 = None, -1.0
    for t in candidate_thresholds(decisions):
        f1 = f1_at(decisions, t)
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best, round(best_f1, 4)


def best_threshold(decisions: list[Decision]) -> tuple[float | None, float]:
    """The threshold maximising F1, and that F1. (None, 0.0) when there is nothing to fit on.

    One pass instead of one pass per candidate. Walking the scores from low to high, the predicted-positive
    set only shrinks, so the true-positive and predicted counts at every candidate come from running
    totals and F1 is `2*tp / (predicted + positives)`. On the 30,912 decisions of one strategy this is
    0.012s against 36s for the scan above — the difference between a calibration that runs and one that
    times out at five folds times six strategies times four checkpoints.

    Candidates, their order, and the first-wins tie-break are identical to `f1_scan`, so the two agree
    exactly rather than approximately.
    """
    total_pos = sum(d.reference_positive for d in decisions)
    if not total_pos:
        return None, 0.0

    ascending = sorted(decisions, key=lambda d: d.score)
    scores: list[float] = []
    sizes: list[int] = []
    positives: list[int] = []
    for d in ascending:
        if not scores or d.score != scores[-1]:
            scores.append(d.score)
            sizes.append(0)
            positives.append(0)
        sizes[-1] += 1
        positives[-1] += d.reference_positive

    # Items and positives scoring at or above each distinct value, accumulated from the top.
    n_at_or_above, pos_at_or_above = [0] * len(scores), [0] * len(scores)
    seen = seen_pos = 0
    for j in range(len(scores) - 1, -1, -1):
        seen += sizes[j]
        seen_pos += positives[j]
        n_at_or_above[j], pos_at_or_above[j] = seen, seen_pos

    # Below the lowest score everything is predicted positive; above the highest, nothing is. In between,
    # a midpoint predicts exactly the scores at or above the upper of the two values it sits between.
    candidates = [(scores[0] - 1e-6, len(decisions), total_pos)]
    candidates += [
        ((scores[j - 1] + scores[j]) / 2, n_at_or_above[j], pos_at_or_above[j]) for j in range(1, len(scores))
    ]
    candidates.append((scores[-1] + 1e-6, 0, 0))

    best, best_f1 = None, -1.0
    for threshold, n_predicted, tp in candidates:
        denominator = n_predicted + total_pos
        f1 = (2 * tp / denominator) if denominator else 0.0
        if f1 > best_f1:
            best, best_f1 = threshold, f1
    return best, round(best_f1, 4)


def score_distribution(decisions: list[Decision]) -> dict[str, Any]:
    """Where the reference's positives and negatives sit on the score scale, and how far apart.

    The ticket asks for "coefficient score distributions"; this is the form that answers it. `overlap`
    is the share of negatives scoring at or above the median positive — a separation measure that needs
    no threshold and degrades gracefully when one class is tiny.
    """
    pos = [d.score for d in decisions if d.reference_positive]
    neg = [d.score for d in decisions if not d.reference_positive]

    def summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "mean": round(statistics.mean(ordered), 4),
            "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
            "median": round(statistics.median(ordered), 4),
            "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
        }

    out: dict[str, Any] = {"positive": summary(pos), "negative": summary(neg)}
    if pos and neg:
        cut = statistics.median(pos)
        out["overlap_pct"] = round(100.0 * sum(s >= cut for s in neg) / len(neg), 1)
        out["median_gap"] = round(cut - statistics.median(neg), 4)
    return out


def _by_slug(decisions: list[Decision]) -> dict[str, list[Decision]]:
    grouped: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        grouped[d.slug].append(d)
    return dict(grouped)


def fit_thresholds(
    decisions: list[Decision], *, min_positives: int = MIN_POSITIVES_FOR_OWN_THRESHOLD
) -> dict[str, Any]:
    """One global threshold, plus a per-tag threshold for every tag with enough positives to earn one.

    Both are reported because the gap between them is itself a finding: cosine similarity against
    different text prompts is not on a shared scale, so a single global threshold can be far worse than
    per-tag ones — which is the ticket's "per-tag thresholds (if relevant)" question, answered with a
    number instead of a guess.
    """
    global_threshold, global_f1 = best_threshold(decisions)
    per_tag: dict[str, dict[str, Any]] = {}
    for slug, rows in sorted(_by_slug(decisions).items()):
        n_pos = sum(d.reference_positive for d in rows)
        if n_pos < min_positives:
            per_tag[slug] = {"threshold": global_threshold, "own": False, "n_positives": n_pos}
            continue
        t, f1 = best_threshold(rows)
        per_tag[slug] = {"threshold": t, "own": True, "n_positives": n_pos, "f1_fitted": f1}
    return {
        "global": {"threshold": global_threshold, "f1_fitted": global_f1},
        "per_tag": per_tag,
        "min_positives": min_positives,
        "tags_with_own_threshold": sum(1 for v in per_tag.values() if v["own"]),
    }


def cross_validated(
    decisions: list[Decision],
    *,
    folds: int = 5,
    seed: int = 7104,
    min_positives: int = MIN_POSITIVES_FOR_OWN_THRESHOLD,
) -> dict[str, Any]:
    """Thresholds fitted on every fold but the one they are applied to.

    The predictions this returns are the only ones fit to publish: each decision was thresholded by a
    rule fitted without it. `per_tag` carries the threshold-free average precision over all decisions,
    which needs no split because it never chooses a threshold.
    """
    assigned = {image_id: fold_of(image_id, folds=folds, seed=seed) for image_id in {d.image_id for d in decisions}}
    predictions_global: dict[str, dict[str, bool | None]] = defaultdict(dict)
    predictions_per_tag: dict[str, dict[str, bool | None]] = defaultdict(dict)
    fitted_per_fold: list[dict[str, Any]] = []

    for fold in range(folds):
        train = [d for d in decisions if assigned[d.image_id] != fold]
        test = [d for d in decisions if assigned[d.image_id] == fold]
        if not train or not test:
            continue
        fit = fit_thresholds(train, min_positives=min_positives)
        fitted_per_fold.append({"fold": fold, "n_train": len(train), "n_test": len(test), **fit["global"]})
        gt = fit["global"]["threshold"]
        for d in test:
            own = fit["per_tag"].get(d.slug, {}).get("threshold")
            # A tag absent from the training fold has no threshold at all; calling it absent would be a
            # measurement of the split rather than of the model, so it stays unanswered.
            predictions_global[d.image_id][d.slug] = (d.score >= gt) if gt is not None else None
            predictions_per_tag[d.image_id][d.slug] = (d.score >= own) if own is not None else None

    per_tag: dict[str, Any] = {}
    for slug, rows in sorted(_by_slug(decisions).items()):
        per_tag[slug] = {
            "n": len(rows),
            "n_positives": sum(d.reference_positive for d in rows),
            "average_precision": average_precision(rows),
            "distribution": score_distribution(rows),
        }
    return {
        "folds": folds,
        "seed": seed,
        "min_positives": min_positives,
        "n_decisions": len(decisions),
        "n_images": len(assigned),
        "fitted_per_fold": fitted_per_fold,
        "whole_set_fit": fit_thresholds(decisions, min_positives=min_positives),
        "per_tag": per_tag,
        "predictions": {"global_threshold": dict(predictions_global), "per_tag_threshold": dict(predictions_per_tag)},
    }
