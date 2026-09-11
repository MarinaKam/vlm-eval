"""A linear probe on frozen embeddings: the cheapest form of fine-tuning, and its ceiling.

Zero-shot asks whether the text encoder can *name* a feature the image encoder already sees. Those are
two different questions, and the gap between them is where most of the disappointment lives: a tag can be
perfectly represented in the image embedding while no English sentence lands near it. A probe answers the
second question on its own — one logistic regression per tag over the image vectors, the encoder frozen,
no gradient reaching it.

**The ceiling is the pipeline, by construction.** The labels are what the current pipeline answered, so a
probe can only learn to imitate it and can never be measured as better than it. What a probe *can* show is
whether the signal exists at all: if zero-shot is weak and the probe is strong, the information was in the
embedding the whole time and the text side was the bottleneck. That distinction changes which option the
recommendation should pick, which is why it is worth the thirty lines.

Same folds as the threshold calibration, so a probe number and a zero-shot number are measured on the same
split and can sit in one table.
"""

from typing import Any

import numpy as np

from .calibration import MIN_POSITIVES_FOR_OWN_THRESHOLD, Decision, average_precision, fold_of

# Enough to converge on a few hundred examples of a 768- to 1152-dimensional vector, and small enough that
# sixty tags times five folds stays a few seconds.
MAX_ITERATIONS = 200
L2 = 1.0


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = L2) -> tuple[np.ndarray, float]:
    """Logistic regression by L-BFGS in torch, which is already a dependency of every encoder run.

    Class weighting is not optional here: at a 7% base rate an unweighted fit converges on "never" and
    reports a loss that looks healthy. Positives are weighted by the inverse of their frequency so the
    probe is asked to separate the classes rather than to count them.
    """
    import torch

    features = torch.from_numpy(x.astype(np.float32))
    target = torch.from_numpy(y.astype(np.float32))
    n_pos = float(target.sum())
    weight = torch.where(target > 0, (len(target) - n_pos) / max(n_pos, 1.0), 1.0)

    w = torch.zeros(features.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([w, b], max_iter=MAX_ITERATIONS, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=weight)

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(features @ w + b, target) + l2 * (w @ w)
        loss.backward()
        return loss

    optimiser.step(closure)
    return w.detach().numpy(), float(b.detach()[0])


def cross_validated_probe(
    *,
    image_ids: list[str],
    vectors: np.ndarray,
    decisions: list[Decision],
    folds: int = 5,
    seed: int = 7104,
    min_positives: int = MIN_POSITIVES_FOR_OWN_THRESHOLD,
    group_of: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Out-of-fold probe scores per tag, reported as average precision so they compare with zero-shot.

    Only tags above the positive floor are fitted. A probe for a tag with one positive example is a line
    through one point; reporting its average precision beside a real one would invite exactly the
    comparison it cannot support.
    """
    index = {image_id: i for i, image_id in enumerate(image_ids)}
    by_slug: dict[str, list[Decision]] = {}
    for d in decisions:
        if d.image_id in index:
            by_slug.setdefault(d.slug, []).append(d)

    per_tag: dict[str, Any] = {}
    skipped: list[str] = []
    for slug, rows in sorted(by_slug.items()):
        n_pos = sum(r.reference_positive for r in rows)
        if n_pos < min_positives:
            skipped.append(slug)
            continue
        scored: list[Decision] = []
        # Same grouping as the threshold calibration: a photograph present under two ids must not be
        # trained on and tested against at once.
        keys = group_of or {}
        for fold in range(folds):
            train = [r for r in rows if fold_of(keys.get(r.image_id, r.image_id), folds=folds, seed=seed) != fold]
            test = [r for r in rows if fold_of(keys.get(r.image_id, r.image_id), folds=folds, seed=seed) == fold]
            if not test or not any(r.reference_positive for r in train):
                continue
            x = vectors[[index[r.image_id] for r in train]]
            y = np.asarray([r.reference_positive for r in train], dtype=np.float32)
            w, b = _fit_logistic(x, y)
            scored.extend(
                Decision(r.image_id, slug, float(vectors[index[r.image_id]] @ w + b), r.reference_positive)
                for r in test
            )
        per_tag[slug] = {
            "n": len(scored),
            "n_positives": n_pos,
            "average_precision": average_precision(scored),
        }
    return {
        "folds": folds,
        "seed": seed,
        "min_positives": min_positives,
        "dim": int(vectors.shape[1]),
        "tags_fitted": len(per_tag),
        "tags_below_floor": skipped,
        "ceiling": "labels are the current pipeline's answers, so a probe can imitate it and never beat it",
        "per_tag": per_tag,
    }
