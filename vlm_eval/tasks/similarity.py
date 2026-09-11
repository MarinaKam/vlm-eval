"""Tag prompts as vectors, and the ways one image vector can be turned into sixty scores.

The production call asks sixty questions and reads sixty booleans. An encoder can only answer "how
close is this image to this sentence", so two choices stand between a vector and a tag, and the ticket
asks about both:

  * **what sentence stands for a tag** (section 4). The production question text is one candidate and
    the obvious one, but it is written for a model that reads instructions — an encoder reads captions.
    So: the question verbatim, the tag name as a caption, several paraphrases averaged into one
    prototype, several kept as separate prototypes and aggregated per image, and positives scored
    against negatives.
  * **how the scores become decisions** (section 2). Thresholding each tag on its own makes the tags
    independent: adding a thousand more tags cannot move an existing tag's score, because nothing in the
    arithmetic refers to the other tags. Normalising across the vocabulary instead — the classic
    zero-shot softmax — makes every score a function of every other candidate, so growing the vocabulary
    changes answers that were already settled. That is the real scalability question; the matrix
    multiply was never the bottleneck.

Averaging prototype *vectors* and averaging their *similarities* are different operations and are kept
as different strategies, because the ticket asks about both and they do not agree.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Strategy:
    """One way of turning a tag into prompts and prompts into a score."""

    name: str
    source: str  # "question" | "name" | "prototypes"
    aggregate: str  # "single" | "embedding_mean" | "max" | "mean"
    use_negatives: bool = False
    description: str = ""


STRATEGIES: dict[str, Strategy] = {
    "question": Strategy(
        "question",
        "question",
        "single",
        description="the production question text verbatim — the ticket's literal premise",
    ),
    "name": Strategy(
        "name",
        "name",
        "single",
        description="the tag name as a short caption, the form these encoders were trained on",
    ),
    "ensemble": Strategy(
        "ensemble",
        "prototypes",
        "embedding_mean",
        description="several paraphrases averaged into one prototype vector",
    ),
    "prototypes-max": Strategy(
        "prototypes-max",
        "prototypes",
        "max",
        description="several prototypes kept apart; the best-matching one wins",
    ),
    "prototypes-mean": Strategy(
        "prototypes-mean",
        "prototypes",
        "mean",
        description="several prototypes kept apart; their similarities averaged",
    ),
    "pos-neg": Strategy(
        "pos-neg",
        "prototypes",
        "max",
        use_negatives=True,
        description="best positive minus best negative, so a look-alike subtracts instead of scoring",
    ),
}

DECISION_RULES = ("independent", "softmax")


@dataclass(frozen=True)
class PrototypeSet:
    """The encoded prompts for one strategy, and which rows belong to which tag."""

    matrix: np.ndarray  # (n_rows, dim), L2-normalised
    texts: list[str]
    positives: dict[str, list[int]]
    negatives: dict[str, list[int]]
    token_counts: list[int]
    context_length: int

    @property
    def slugs(self) -> list[str]:
        return list(self.positives)

    def truncated_texts(self) -> list[tuple[str, int]]:
        """(prompt, tokens needed) for every prompt the encoder could not read in full."""
        return [(self.texts[i], n) for i, n in enumerate(self.token_counts) if n > self.context_length]


def texts_for(tag: dict, prototypes: dict[str, Any], strategy: Strategy) -> tuple[list[str], list[str]]:
    """The positive and negative prompts a strategy uses for one tag.

    Falls back to the tag name when a prototype file has nothing for a slug: a missing entry must not
    silently score the tag as absent on every image, which is what an empty prompt list would do.
    """
    slug = tag["slug"]
    if strategy.source == "question":
        return [tag["question_text"]], []
    if strategy.source == "name":
        return [f"a photo of {tag['name'].lower()}"], []
    entry = prototypes.get(slug) or {}
    positives = list(entry.get("positive") or []) or [f"a photo of {tag['name'].lower()}"]
    negatives = list(entry.get("negative") or []) if strategy.use_negatives else []
    if strategy.aggregate == "single":
        positives = positives[:1]
    return positives, negatives


def build_prototype_set(encoder, tags: list[dict], prototypes: dict[str, Any], strategy: Strategy) -> PrototypeSet:
    """Encode every prompt once, then lay out which rows each tag scores against.

    For `embedding_mean` the averaging happens here, so that downstream scoring is the same arithmetic
    for every strategy: one similarity row per image, indexed by tag.
    """
    texts: list[str] = []
    positives: dict[str, list[int]] = {}
    negatives: dict[str, list[int]] = {}
    for tag in tags:
        pos, neg = texts_for(tag, prototypes, strategy)
        positives[tag["slug"]] = [len(texts) + i for i in range(len(pos))]
        texts.extend(pos)
        negatives[tag["slug"]] = [len(texts) + i for i in range(len(neg))]
        texts.extend(neg)

    encoded = encoder.encode_texts(texts)
    matrix, counts = encoded.vectors, encoded.token_counts

    if strategy.aggregate == "embedding_mean":
        if strategy.use_negatives:
            # Averaging collapses each tag to one row, which has nowhere to put a negative. Failing here
            # beats dropping them silently and reporting the result as if they had been scored.
            raise ValueError(f"strategy {strategy.name!r} averages prototypes and cannot use negatives")
        rows, new_pos, new_labels, kept_counts = [], {}, [], []
        for slug, idx in positives.items():
            mean = matrix[idx].mean(axis=0)
            rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
            new_pos[slug] = [len(rows) - 1]
            new_labels.append(f"mean of {len(idx)} prototype(s) for {slug}")
            # The averaged prototype is as truncated as the worst prompt that went into it.
            kept_counts.append(max(counts[i] for i in idx))
        matrix = np.asarray(rows, dtype=np.float32)
        positives = new_pos
        negatives = {slug: [] for slug in new_pos}  # a shared list here would alias across every tag
        counts, texts = kept_counts, new_labels

    return PrototypeSet(
        matrix=matrix,
        texts=texts,
        positives=positives,
        negatives=negatives,
        token_counts=counts,
        context_length=encoded.context_length,
    )


def _pool(values: np.ndarray, how: str) -> float:
    if values.size == 0:
        return 0.0
    return float(values.max()) if how == "max" else float(values.mean())


def raw_scores(similarities: np.ndarray, prototype_set: PrototypeSet, strategy: Strategy) -> dict[str, float]:
    """One score per tag from one image's similarity row. Cosine scale, before any decision rule."""
    how = "max" if strategy.aggregate in ("max", "single", "embedding_mean") else "mean"
    out: dict[str, float] = {}
    for slug, idx in prototype_set.positives.items():
        score = _pool(similarities[idx], how)
        if strategy.use_negatives:
            against = prototype_set.negatives.get(slug) or []
            if against:
                score -= _pool(similarities[against], "max")
        out[slug] = score
    return out


def softmax_scores(scores: dict[str, float], *, temperature: float = 0.01) -> dict[str, float]:
    """Normalise across the whole vocabulary — the classic zero-shot rule, and the one that does not scale.

    Kept so the claim in section 2 is measured rather than argued: every value here depends on every
    other tag in the dictionary, so the same image scored against a larger vocabulary answers differently.
    """
    slugs = list(scores)
    raw = np.asarray([scores[s] for s in slugs], dtype=np.float64) / temperature
    exp = np.exp(raw - raw.max())
    total = exp.sum()
    return {s: float(v / total) for s, v in zip(slugs, exp)}


def decide(scores: dict[str, float], thresholds: dict[str, float | None]) -> dict[str, bool | None]:
    """Apply one threshold per tag. A tag with no threshold is unanswered, never a confident absence."""
    out: dict[str, bool | None] = {}
    for slug, score in scores.items():
        t = thresholds.get(slug)
        out[slug] = None if t is None else bool(score >= t)
    return out


def prompt_fit(prototype_set: PrototypeSet, tags: list[dict]) -> dict[str, Any]:
    """How much of each tag definition the encoder could actually read.

    This is reported beside the quality numbers rather than as a footnote: a tag whose definition was
    cut in half was never given the question production asks, so a low score for it measures the prompt,
    not the model.
    """
    limit = prototype_set.context_length
    per_tag = {}
    for tag in tags:
        idx = prototype_set.positives.get(tag["slug"]) or []
        needed = max((prototype_set.token_counts[i] for i in idx), default=0)
        per_tag[tag["slug"]] = {
            "tokens_needed": needed,
            "tokens_read": min(needed, limit),
            "truncated": needed > limit,
        }
    cut = [s for s, v in per_tag.items() if v["truncated"]]
    return {
        "context_length": limit,
        "n_prompts": len(prototype_set.texts),
        "n_tags_truncated": len(cut),
        "tags_truncated": sorted(cut),
        "per_tag": per_tag,
    }
