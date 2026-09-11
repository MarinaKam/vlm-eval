"""Driving an encoder over the dataset: encode once, then score each strategy against the cached vectors.

The split is the point. Encoding a thousand images is the only expensive step and it does not depend on
which prompts a tag gets, so it happens once per checkpoint and is reused by every strategy and every
threshold. Scoring is a matrix multiply over cached vectors, which is what makes it affordable to compare
six prompt strategies and two decision rules instead of guessing which one to try.

Three kinds of file come out, and each is a separate experiment in the provenance sense:

  * `embeddings.jsonl` — one vector per image, with the latency it took.
  * `tagging_embed_<strategy>.jsonl` — answers at a threshold that was not fitted on this data: the
    checkpoint's own trained boundary where it has one. Uncalibrated, and labelled as such.
  * `tagging_embed_<strategy>_cv.jsonl` — answers at thresholds fitted under cross-validation, so no
    decision was made by a rule that had seen it.

Rows carry the same fields a tagging run writes, so `metrics`, `review`, `report` and `compare` read
them without knowing an encoder produced them.
"""

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import calibration, dataset, runner
from .config import RUNS
from .tasks import similarity, tagging


def _run_dir(model: str) -> Path:
    """Created on the way out, like `runner._out` — the provenance sidecar is written before any row."""
    path = RUNS / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def embeddings_path(model: str) -> Path:
    return _run_dir(model) / "embeddings.jsonl"


def scores_path(model: str, strategy: str, *, calibrated: bool = False) -> Path:
    suffix = "_cv" if calibrated else ""
    return _run_dir(model) / f"tagging_embed_{strategy}{suffix}.jsonl"


def calibration_path(model: str, strategy: str) -> Path:
    return _run_dir(model) / f"calibration_{strategy}.json"


# ------------------------------------------------------------------ encoding


def embed_one(encoder, item: dataset.Item) -> dict[str, Any]:
    """One image to one vector. Latency is per image on purpose — production pays it per image too."""
    data = runner._read_image(item)
    t0 = time.perf_counter()
    vector = encoder.encode_images([data])[0]
    elapsed = time.perf_counter() - t0
    return {
        "image_id": item.image_id,
        "image_type": item.image_type,
        "dim": int(vector.shape[0]),
        # Six decimals: the cosine similarities these feed are reported to four, so the rounding is
        # two orders below anything published, and the file stays readable.
        "vector": [round(float(v), 6) for v in vector],
        "latency_s": round(elapsed, 4),
        "completion": runner.completion_record(1, 0),
        "errors": [],
    }


def load_vectors(model: str) -> tuple[list[str], dict[str, str], np.ndarray]:
    """The cached image vectors as a matrix, with the image ids and types that index it."""
    rows = dataset.load_jsonl(embeddings_path(model))
    usable = [r for r in rows if r.get("vector") and r.get("repeat", 0) == 0]
    if not usable:
        raise SystemExit(
            f"no image vectors in {embeddings_path(model)} — run `vlm-eval embed {model}` first.\n"
            "Scoring against an empty cache would report a model that found nothing."
        )
    ids = [str(r["image_id"]) for r in usable]
    types = {str(r["image_id"]): r.get("image_type", "") for r in usable}
    matrix = np.asarray([r["vector"] for r in usable], dtype=np.float32)
    return ids, types, matrix


def load_prototypes(path: Path | None = None) -> dict[str, Any]:
    """The prompt file, or nothing — strategies that only need the tag itself still work without it."""
    path = path or dataset.DATA / "prototypes.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("tags", payload)


def review_state(prototypes: dict[str, Any]) -> dict[str, list[str]]:
    """Who last touched each tag's prompts, grouped.

    Three states rather than a boolean, because they are three different kinds of evidence: `generator`
    prompts were derived mechanically from the tag name, `agent` prompts were rewritten against the tag
    definition and the measured confusions of the current pipeline, and `human` means a person has read
    them. A report that called the middle one "reviewed" would be claiming a check nobody performed.
    """
    grouped: dict[str, list[str]] = {}
    for slug, entry in prototypes.items():
        state = entry.get("reviewed_by") or ("generator" if entry.get("review") else "human")
        grouped.setdefault(state, []).append(slug)
    return {state: sorted(slugs) for state, slugs in grouped.items()}


def unreviewed(prototypes: dict[str, Any]) -> list[str]:
    """Tags no person has read the prompts for — anything not marked `human`."""
    states = review_state(prototypes)
    return sorted(slug for state, slugs in states.items() if state != "human" for slug in slugs)


# ------------------------------------------------------------------ scoring


def evaluable_for(reference: dict[str, dict], image_id: str, tags: list[dict], image_type: str) -> set[str]:
    """The tags comparable on this image: the question set production would ask, and the reference judged.

    Both halves matter. Scoring a tag the reference never judged manufactures a false positive, and
    scoring an indoor-only tag on an outdoor photo asks a question production never asks. The first half
    reuses the production question selection the tagging task already mirrors.
    """
    asked = set(tagging.questions_for(image_type, tags))
    judged = (reference.get(image_id) or {}).get("evaluable_slugs")
    return asked & set(judged) if judged else asked


def score_rows(
    *,
    model: str,
    tags: list[dict],
    prototype_set: similarity.PrototypeSet,
    strategy: similarity.Strategy,
    thresholds: dict[str, float | None] | None,
    reference: dict[str, dict],
    threshold_source: str,
    rule: str = "independent",
) -> list[dict[str, Any]]:
    """Turn cached vectors into rows shaped exactly like a tagging run's.

    `thresholds` of None means no defensible boundary exists yet — every answer is recorded as unknown
    rather than guessed, which keeps an uncalibrated CLIP run from reading as a model that found nothing.
    """
    ids, types, images = load_vectors(model)
    # Apple's Accelerate BLAS leaves the floating-point status flags set after a matmul, and NumPy 1.26
    # reports them as divide-by-zero / overflow / invalid even though nothing of the sort happened: the
    # result here was checked against the same product in float64 and agrees to 2e-07, which is float32
    # round-off. The flags are ignored and the property they claim to report is asserted instead, so a
    # genuine non-finite value still stops the run instead of reaching a report.
    with np.errstate(all="ignore"):
        sims = images @ prototype_set.matrix.T
    if not np.isfinite(sims).all():
        raise SystemExit(
            f"{(~np.isfinite(sims)).sum()} non-finite similarity score(s) for {model} — a vector in the "
            "cache or a prompt embedding is corrupt. Re-run `embed` rather than scoring this."
        )
    rows = []
    for i, image_id in enumerate(ids):
        image_type = types[image_id]
        scores = similarity.raw_scores(sims[i], prototype_set, strategy)
        if rule == "softmax":
            scores = similarity.softmax_scores(scores)
        comparable = evaluable_for(reference, image_id, tags, image_type)
        scores = {slug: v for slug, v in scores.items() if slug in comparable}
        answers = similarity.decide(scores, thresholds) if thresholds is not None else dict.fromkeys(scores, None)
        rows.append(
            {
                "image_id": image_id,
                "image_type": image_type,
                "strategy": strategy.name,
                "decision_rule": rule,
                "threshold_source": threshold_source,
                "n_questions": len(scores),
                "n_calls": 1,
                "answers": answers,
                "scores": {slug: round(v, 6) for slug, v in scores.items()},
                "latency_s": 0.0,
                "completion": runner.completion_record(1, 0),
                "errors": [],
                "repeat": 0,
            }
        )
    return rows


def decisions_from_rows(rows: list[dict], reference: dict[str, dict]) -> list[calibration.Decision]:
    """Every comparable (image, tag) score paired with what the reference said about it."""
    out = []
    for row in rows:
        positives = set((reference.get(row["image_id"]) or {}).get("tags") or {})
        for slug, score in (row.get("scores") or {}).items():
            out.append(calibration.Decision(row["image_id"], slug, float(score), slug in positives))
    return out


def apply_predictions(rows: list[dict], predictions: dict[str, dict[str, bool | None]]) -> list[dict]:
    """Rebuild rows with out-of-fold answers, keeping every score so the numbers stay traceable."""
    rebuilt = []
    for row in rows:
        answers = predictions.get(row["image_id"], {})
        rebuilt.append({**row, "answers": {s: answers.get(s) for s in row["scores"]}})
    return rebuilt


# ------------------------------------------------------------------ vocabulary growth (section 2)

# Generic English nouns, not real-estate words. Deliberately conservative: semantically near distractors
# ("sofa", "washbasin") would crowd the softmax harder, so whatever degradation generic words cause is a
# floor on the real effect, never an overstatement of it.
WORD_LIST = Path("/usr/share/dict/words")


def distractor_prompts(n: int, *, seed: int = 7104, word_list: Path | None = None) -> list[str]:
    """`n` competing candidates for the vocabulary to grow into, from a fixed-seed word sample."""
    import random

    path = word_list or WORD_LIST
    if not path.exists():
        raise SystemExit(
            f"no word list at {path} — the vocabulary-growth test needs one to build competing candidates. "
            "Pass --word-list to point at any newline-separated file."
        )
    words = sorted({w.strip().lower() for w in path.read_text().splitlines() if w.strip().isalpha() and len(w) > 3})
    if len(words) < n:
        # Returning fewer would report a vocabulary size that was never built, which is worse than stopping.
        raise SystemExit(
            f"{path} yields only {len(words)} usable word(s); {n} are needed for this vocabulary size. "
            "Pass a larger --word-list or drop the larger sizes."
        )
    rng = random.Random(seed)
    rng.shuffle(words)
    return [f"a photo of a {w}" for w in words[:n]]


def _aggregate_per_slug(sims: np.ndarray, prototype_set: similarity.PrototypeSet, how: str) -> np.ndarray:
    """(images, prototypes) -> (images, tags), pooling each tag's prompt columns the way the strategy does."""
    columns = []
    for slug in prototype_set.slugs:
        idx = prototype_set.positives[slug]
        block = sims[:, idx]
        columns.append(block.max(axis=1) if how == "max" else block.mean(axis=1))
    return np.stack(columns, axis=1)


def vocabulary_growth(
    *,
    encoder,
    model: str,
    tags: list[dict],
    prototype_set: similarity.PrototypeSet,
    strategy: similarity.Strategy,
    sizes: list[int],
    word_list: Path | None = None,
    temperature: float = 0.01,
) -> dict[str, Any]:
    """What grows with the tag vocabulary: the arithmetic, and — under one rule only — the answers.

    Two questions the ticket runs together. The cost of comparing one image vector against N tag vectors
    is a matrix multiply, measured here at each N. Whether the *answers* change is a property of the
    decision rule rather than of N: thresholding each tag independently has no term referring to any other
    tag, so a tag's score is defined identically however large the vocabulary gets, and the only thing a
    wider matrix changes is the order the same products are summed in. That is reported as a drift, not as
    bit-identity — bit-identity held on one machine and not on another, which made it a property of the
    BLAS build rather than of the method. Normalising across the vocabulary is the opposite case: every
    score becomes a function of all the others, and the count of flipped decisions is what that costs.

    Both halves are vectorised. Built with Python dictionaries this took minutes per checkpoint, which is
    how a measurement quietly stops being run.
    """
    import time

    ids, _types, images = load_vectors(model)
    how = "max" if strategy.aggregate in ("max", "single", "embedding_mean") else "mean"
    own = prototype_set.matrix
    n_tags = len(prototype_set.slugs)

    with np.errstate(all="ignore"):  # see the note in score_rows
        baseline_sims = images @ own.T
    baseline_agg = _aggregate_per_slug(baseline_sims, prototype_set, how)
    baseline_soft = _softmax_rows(baseline_agg, temperature)
    baseline_decision = baseline_soft >= 1.0 / n_tags

    out = []
    for size in sorted(sizes):
        extra = max(size - own.shape[0], 0)
        padded = own
        if extra:
            vectors = encoder.encode_texts(distractor_prompts(extra, word_list=word_list)).vectors
            padded = np.concatenate([own, vectors])

        best = None
        for _ in range(3):  # three passes, keep the fastest: the others also measured other processes
            with np.errstate(all="ignore"):
                t0 = time.perf_counter()
                sims = images @ padded.T
                elapsed = time.perf_counter() - t0
            best = elapsed if best is None else min(best, elapsed)

        ours = _aggregate_per_slug(sims[:, : own.shape[0]], prototype_set, how)
        # Not bit-identity. Each tag's score is defined without reference to any other tag, so its *value*
        # cannot depend on the vocabulary — but the product is computed by a BLAS kernel whose blocking
        # depends on the matrix width, so the summation order changes and the last bits with it. Measured
        # bit-identical on one machine and not on another, which makes bit-identity a property of the
        # library rather than of the method. What is portable is the size of the wobble, and whether it is
        # anywhere near large enough to move a decision.
        deviation = float(np.max(np.abs(ours - baseline_agg))) if ours.size else 0.0
        moved = int((np.abs(ours - baseline_agg) > DECISION_SAFE_MARGIN).sum())

        # Under normalisation the distractors are candidates too, so they take probability mass from ours.
        candidates = np.concatenate([ours, sims[:, own.shape[0] :]], axis=1) if extra else ours
        soft = _softmax_rows(candidates, temperature)[:, :n_tags]
        flipped = int(((soft >= 1.0 / candidates.shape[1]) != baseline_decision).sum()) if extra else 0

        out.append(
            {
                "vocabulary": int(padded.shape[0]),
                "seconds_for_1000_images": round(best, 5),
                "microseconds_per_image": round(best / len(ids) * 1e6, 2),
                "independent_rule_max_deviation": deviation,
                "independent_rule_scores_beyond_margin": moved,
                "independent_rule_bit_identical": bool(np.array_equal(ours, baseline_agg)),
                "softmax_rule_decisions_flipped": flipped,
                "softmax_decisions_compared": int(baseline_decision.size),
            }
        )
    return {
        "n_tags": n_tags,
        "n_images": len(ids),
        "temperature": temperature,
        "distractors": "generic English nouns, fixed seed — near distractors would crowd softmax harder",
        "sizes": out,
    }


def _softmax_rows(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Row-wise softmax, shifted by the row maximum so a large score cannot overflow the exponential."""
    scaled = scores.astype(np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ the hybrid option

# Deferral budgets to measure, as a share of all comparable decisions.
HYBRID_BUDGETS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)

# A score this far from its threshold is not going to be moved across it by float32 round-off. Cosine
# similarities here span roughly 0.25, so a millionth is five orders below anything a threshold sits on.
DECISION_SAFE_MARGIN = 1e-6


def _margin_ranked(rows: list[dict], thresholds: dict[str, Any]) -> list[tuple[float, str, str]]:
    """Every decision ranked by how close it sits to its tag's threshold, least confident first.

    The distance is divided by the spread of that tag's own scores. Without it the ranking is not
    comparable across tags — a tag whose scores all sit inside a narrow band would look uniformly
    confident and never be deferred, while a wide-ranging tag would absorb the whole budget.
    """
    import statistics

    by_slug: dict[str, list[float]] = {}
    for row in rows:
        for slug, score in (row.get("scores") or {}).items():
            by_slug.setdefault(slug, []).append(score)
    spread = {
        slug: (statistics.pstdev(values) or 1e-9) if len(values) > 1 else 1e-9 for slug, values in by_slug.items()
    }

    ranked = []
    for row in rows:
        for slug, score in (row.get("scores") or {}).items():
            t = (thresholds.get(slug) or {}).get("threshold")
            if t is None:
                continue
            ranked.append((abs(score - t) / spread[slug], row["image_id"], slug))
    ranked.sort()
    return ranked


def hybrid_curve(
    *,
    rows: list[dict],
    thresholds: dict[str, Any],
    reference: dict[str, dict],
    calls_today: dict[str, int],
    chunk_size: int,
    budgets: tuple[float, ...] = HYBRID_BUDGETS,
) -> dict[str, Any]:
    """What the third option in the ticket actually costs: keep the confident scores, ask a VLM the rest.

    Measurable with no API call at all, because the reference file already holds the pipeline's answer for
    every tag on every image — deferring a decision means reading it from there.

    The saving is counted in **calls, not decisions**, under production's own chunking: an image whose
    borderline tags fit in one batch costs one call, an image with none costs nothing, and an image with
    more than a batch costs more than one. That is why the deferral rate and the cost reduction are not
    the same number.

    One thing these figures cannot say. A deferred decision is resolved by reading the pipeline's own
    answer, and the pipeline is also what the result is scored against — so deferring *guarantees*
    agreement on those decisions. The curve therefore measures how much generative capacity has to be
    bought to reproduce today's behaviour. It is not a measurement of accuracy, and at a large deferral
    budget most of the apparent quality is the answer key being copied back in.
    """
    ranked = _margin_ranked(rows, thresholds)
    total = len(ranked)
    out = []
    for budget in budgets:
        cut = int(round(budget * total))
        deferred = {(image_id, slug) for _, image_id, slug in ranked[:cut]}
        images_touched = {image_id for image_id, _ in deferred}

        merged = []
        per_image_deferred: dict[str, int] = {}
        for row in rows:
            answers = {}
            pipeline_positive = set((reference.get(row["image_id"]) or {}).get("tags") or {})
            for slug in row.get("scores") or {}:
                if (row["image_id"], slug) in deferred:
                    answers[slug] = slug in pipeline_positive
                    per_image_deferred[row["image_id"]] = per_image_deferred.get(row["image_id"], 0) + 1
                else:
                    answers[slug] = row["answers"].get(slug)
            merged.append({**row, "answers": answers})

        calls = sum(calls_today.get(r["image_id"], 0) for r in rows)
        # One call per batch of questions, the same chunking production uses — not one call per image.
        # An image with forty borderline tags needs three calls, and pretending otherwise would report a
        # saving nobody could collect.
        hybrid_calls = sum(math.ceil(n / chunk_size) for n in per_image_deferred.values())
        out.append(
            {
                "deferral_budget_pct": round(100 * budget, 1),
                "decisions_deferred": len(deferred),
                "images_needing_a_call": len(images_touched),
                "images_pct": round(100 * len(images_touched) / max(len(rows), 1), 1),
                "vlm_calls_today": calls,
                "vlm_calls_hybrid": hybrid_calls,
                "call_reduction_pct": round(100 * (1 - hybrid_calls / calls), 1) if calls else None,
                "rows": merged,
            }
        )
    return {"n_decisions": total, "budgets": out}


def calls_per_image(tags: list[dict], rows: list[dict], chunk_size: int, individual: list[str]) -> dict[str, int]:
    """How many calls production makes for each image today — its own chunking, not an estimate."""
    out = {}
    for row in rows:
        questions = tagging.questions_for(row.get("image_type", ""), tags)
        out[row["image_id"]] = len(tagging.chunk_questions(questions, chunk_size, individual))
    return out


def write_rows(path: Path, rows: list[dict]) -> int:
    """Replace the file's contents. Scoring is deterministic over a fixed cache, so there is nothing to
    resume and appending would double every row on a re-run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
