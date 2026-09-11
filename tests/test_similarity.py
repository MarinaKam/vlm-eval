import hashlib

import numpy as np
import pytest

from vlm_eval import embedding_run
from vlm_eval.backends.encoder import EncodedTexts
from vlm_eval.tasks import similarity


class StubEncoder:
    """Deterministic unit vectors per text, so a prototype's identity is its wording."""

    def __init__(self, context_length: int = 16, dim: int = 8):
        self.context_length = context_length
        self.dim = dim

    def token_count(self, text: str) -> int:
        return len(text.split())

    def encode_texts(self, texts: list[str], *, batch_size: int = 64) -> EncodedTexts:
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return EncodedTexts(
            vectors=np.asarray(rows, dtype=np.float32),
            token_counts=[self.token_count(t) for t in texts],
            context_length=self.context_length,
        )


TAGS = [
    {"slug": "kitchen", "name": "Kitchen", "category": "indoor", "order": 0, "question_text": "Is there a kitchen?"},
    {"slug": "pool", "name": "Pool", "category": "common", "order": 1, "question_text": "Is there a pool?"},
]
PROTOTYPES = {
    "kitchen": {"positive": ["a kitchen", "a fitted kitchen", "a kitchen with cabinets"], "negative": ["a bathroom"]},
    "pool": {"positive": ["a swimming pool"], "negative": ["a pool in a painting", "a jacuzzi"]},
}


# ---------------------------------------------------------------- which sentence stands for a tag


def test_the_question_strategy_uses_the_production_text_verbatim():
    pos, neg = similarity.texts_for(TAGS[0], PROTOTYPES, similarity.STRATEGIES["question"])
    assert pos == ["Is there a kitchen?"]
    assert neg == []


def test_the_name_strategy_builds_a_caption_from_the_tag_name():
    pos, _ = similarity.texts_for(TAGS[0], PROTOTYPES, similarity.STRATEGIES["name"])
    assert pos == ["a photo of kitchen"]


def test_a_tag_missing_from_the_prototype_file_falls_back_to_its_name():
    """An empty prompt list would score the tag absent on every image — a silent zero, not a gap."""
    pos, _ = similarity.texts_for(TAGS[0], {}, similarity.STRATEGIES["prototypes-max"])
    assert pos == ["a photo of kitchen"]


def test_only_the_negative_strategies_load_negatives():
    _, without = similarity.texts_for(TAGS[1], PROTOTYPES, similarity.STRATEGIES["prototypes-max"])
    _, with_neg = similarity.texts_for(TAGS[1], PROTOTYPES, similarity.STRATEGIES["pos-neg"])
    assert without == []
    assert with_neg == ["a pool in a painting", "a jacuzzi"]


# ---------------------------------------------------------------- building the prototype matrix


def test_every_prompt_gets_its_own_row_unless_the_strategy_averages_them():
    enc = StubEncoder()
    kept = similarity.build_prototype_set(enc, TAGS, PROTOTYPES, similarity.STRATEGIES["prototypes-max"])
    assert kept.matrix.shape[0] == 4  # three kitchen prompts, one pool
    assert len(kept.positives["kitchen"]) == 3

    averaged = similarity.build_prototype_set(enc, TAGS, PROTOTYPES, similarity.STRATEGIES["ensemble"])
    assert averaged.matrix.shape[0] == 2  # one row per tag
    assert len(averaged.positives["kitchen"]) == 1


def test_an_averaged_prototype_is_still_a_unit_vector():
    averaged = similarity.build_prototype_set(StubEncoder(), TAGS, PROTOTYPES, similarity.STRATEGIES["ensemble"])
    norms = np.linalg.norm(averaged.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_an_averaged_prototype_inherits_the_worst_truncation_of_its_inputs():
    """Averaging hides which prompt was cut, so the row has to carry the longest one's cost."""
    enc = StubEncoder(context_length=2)
    averaged = similarity.build_prototype_set(enc, TAGS, PROTOTYPES, similarity.STRATEGIES["ensemble"])
    # "a kitchen with cabinets" is four words, the longest of the three kitchen prompts.
    assert averaged.token_counts[0] == 4


def test_averaging_refuses_to_silently_drop_negatives():
    broken = similarity.Strategy("broken", "prototypes", "embedding_mean", use_negatives=True)
    with pytest.raises(ValueError, match="cannot use negatives"):
        similarity.build_prototype_set(StubEncoder(), TAGS, PROTOTYPES, broken)


# ---------------------------------------------------------------- scoring


def _sims(values: dict[str, float], prototype_set) -> np.ndarray:
    """A similarity row built by naming the value for each prompt index."""
    row = np.zeros(prototype_set.matrix.shape[0], dtype=np.float32)
    for slug, idx in prototype_set.positives.items():
        for i in idx:
            row[i] = values.get(slug, 0.0)
    for slug, idx in prototype_set.negatives.items():
        for i in idx:
            row[i] = values.get(f"{slug}:neg", 0.0)
    return row


def test_max_aggregation_takes_the_best_prototype():
    ps = similarity.build_prototype_set(StubEncoder(), TAGS, PROTOTYPES, similarity.STRATEGIES["prototypes-max"])
    row = np.zeros(ps.matrix.shape[0], dtype=np.float32)
    for rank, i in enumerate(ps.positives["kitchen"]):
        row[i] = [0.1, 0.7, 0.3][rank]
    scores = similarity.raw_scores(row, ps, similarity.STRATEGIES["prototypes-max"])
    assert scores["kitchen"] == pytest.approx(0.7)


def test_mean_aggregation_averages_the_prototypes():
    ps = similarity.build_prototype_set(StubEncoder(), TAGS, PROTOTYPES, similarity.STRATEGIES["prototypes-mean"])
    row = np.zeros(ps.matrix.shape[0], dtype=np.float32)
    for rank, i in enumerate(ps.positives["kitchen"]):
        row[i] = [0.1, 0.7, 0.4][rank]
    scores = similarity.raw_scores(row, ps, similarity.STRATEGIES["prototypes-mean"])
    assert scores["kitchen"] == pytest.approx(0.4, abs=1e-6)


def test_the_negative_strategy_subtracts_the_closest_look_alike():
    ps = similarity.build_prototype_set(StubEncoder(), TAGS, PROTOTYPES, similarity.STRATEGIES["pos-neg"])
    row = _sims({"pool": 0.6, "pool:neg": 0.5}, ps)
    scores = similarity.raw_scores(row, ps, similarity.STRATEGIES["pos-neg"])
    assert scores["pool"] == pytest.approx(0.1, abs=1e-6)


def test_independent_scores_ignore_every_column_outside_the_tag():
    """The guarantee section 2 rests on: a larger vocabulary cannot move an existing tag's score.

    Scores are read from the columns belonging to the tag, so appending a thousand candidates appends a
    thousand columns nothing reads.
    """
    ps = similarity.build_prototype_set(StubEncoder(), TAGS, PROTOTYPES, similarity.STRATEGIES["prototypes-max"])
    row = _sims({"kitchen": 0.5, "pool": 0.2}, ps)
    before = similarity.raw_scores(row, ps, similarity.STRATEGIES["prototypes-max"])
    padded = np.concatenate([row, np.full(1000, 0.99, dtype=np.float32)])
    after = similarity.raw_scores(padded, ps, similarity.STRATEGIES["prototypes-max"])
    assert before == after


def test_the_softmax_rule_makes_every_score_depend_on_the_others():
    """Which is exactly why it does not scale: the same image answers differently in a bigger dictionary."""
    two = similarity.softmax_scores({"kitchen": 0.5, "pool": 0.2})
    three = similarity.softmax_scores({"kitchen": 0.5, "pool": 0.2, "garden": 0.5})
    assert sum(two.values()) == pytest.approx(1.0)
    assert sum(three.values()) == pytest.approx(1.0)
    assert three["kitchen"] < two["kitchen"]


# ---------------------------------------------------------------- decisions and prompt fit


def test_a_tag_without_a_threshold_is_unanswered_not_absent():
    answers = similarity.decide({"kitchen": 0.5, "pool": 0.1}, {"kitchen": 0.4, "pool": None})
    assert answers == {"kitchen": True, "pool": None}


def test_prompt_fit_names_the_tags_the_encoder_could_not_read_in_full():
    enc = StubEncoder(context_length=3)
    ps = similarity.build_prototype_set(enc, TAGS, PROTOTYPES, similarity.STRATEGIES["question"])
    fit = similarity.prompt_fit(ps, TAGS)
    # "Is there a kitchen?" is four words against a three-token window.
    assert fit["n_tags_truncated"] == 2
    assert fit["per_tag"]["kitchen"]["tokens_needed"] == 4
    assert fit["per_tag"]["kitchen"]["tokens_read"] == 3


def test_prompt_fit_reports_nothing_truncated_when_everything_fits():
    ps = similarity.build_prototype_set(StubEncoder(context_length=64), TAGS, PROTOTYPES, similarity.STRATEGIES["name"])
    assert similarity.prompt_fit(ps, TAGS)["n_tags_truncated"] == 0


# ---------------------------------------------------------------- the hybrid's cost model


def test_calls_per_image_mirrors_production_chunking():
    """Five calls for an indoor photo, two for an outdoor one — the hybrid's saving is measured against these."""
    tags = (
        [{"slug": f"c{i}", "name": "", "category": "common", "order": i, "question_text": "?"} for i in range(18)]
        + [{"slug": f"i{i}", "name": "", "category": "indoor", "order": i, "question_text": "?"} for i in range(34)]
        + [{"slug": f"o{i}", "name": "", "category": "outdoor", "order": i, "question_text": "?"} for i in range(8)]
    )
    rows = [{"image_id": "a", "image_type": "indoor"}, {"image_id": "b", "image_type": "outdoor"}]
    calls = embedding_run.calls_per_image(tags, rows, 15, ["i0"])
    assert calls == {"a": 5, "b": 2}


def test_deferral_ranks_by_distance_to_the_threshold_relative_to_each_tags_spread():
    """Unnormalised, a tag whose scores barely move would never be deferred however unsure it is."""
    rows = [{"image_id": f"img{i}", "scores": {"wide": 0.5 + i, "narrow": 0.5 + i / 1000}} for i in range(6)]
    ranked = embedding_run._margin_ranked(rows, {"wide": {"threshold": 3.0}, "narrow": {"threshold": 0.5025}})
    deferred_tags = {slug for _, _, slug in ranked[:4]}
    assert deferred_tags == {"wide", "narrow"}


# ---------------------------------------------------------------- vocabulary growth (section 2)


class StubVocabEncoder(StubEncoder):
    """Encodes distractors the same deterministic way, so the growth test needs no checkpoint."""


def test_vocabulary_growth_leaves_independent_scores_untouched_and_moves_softmax_ones(tmp_path, monkeypatch):
    """The measured form of section 2's claim, on synthetic vectors so it runs in a unit test."""
    words = tmp_path / "words"
    # Letters only: the word list filter drops anything with a digit in it.
    words.write_text("\n".join("".join(chr(97 + (i // 26**k) % 26) for k in range(4)) for i in range(400)))

    enc = StubVocabEncoder(context_length=64, dim=8)
    ps = similarity.build_prototype_set(enc, TAGS, PROTOTYPES, similarity.STRATEGIES["prototypes-max"])

    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(12, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    monkeypatch.setattr(
        embedding_run,
        "load_vectors",
        lambda model: ([f"img{i}" for i in range(12)], {f"img{i}": "indoor" for i in range(12)}, vectors),
    )

    result = embedding_run.vocabulary_growth(
        encoder=enc,
        model="stub",
        tags=TAGS,
        prototype_set=ps,
        strategy=similarity.STRATEGIES["prototypes-max"],
        sizes=[4, 50, 200],
        word_list=words,
    )
    by_size = {row["vocabulary"]: row for row in result["sizes"]}
    assert all(row["independent_rule_scores_identical"] for row in result["sizes"])
    # Nothing was added at the smallest size, so nothing can have flipped there.
    assert by_size[4]["softmax_rule_decisions_flipped"] == 0
    # Competing candidates take probability mass, so the normalised rule changes its mind.
    assert by_size[200]["softmax_rule_decisions_flipped"] > 0


def test_vocabulary_growth_refuses_to_invent_a_word_list(tmp_path):
    with pytest.raises(SystemExit, match="word list"):
        embedding_run.distractor_prompts(10, word_list=tmp_path / "nope")


def test_a_short_word_list_stops_the_run_instead_of_shrinking_the_vocabulary(tmp_path):
    """Silently returning five candidates for a thousand would publish a size that was never built."""
    short = tmp_path / "few"
    short.write_text("alpha\nbravo\ncharlie\n")
    with pytest.raises(SystemExit, match="only 3 usable word"):
        embedding_run.distractor_prompts(100, word_list=short)


def test_the_hybrid_counts_calls_by_production_chunking_not_one_per_image():
    """An image with more borderline tags than fit in a batch costs more than one call.

    Counting one per image made the saving look better than anyone could collect: at a generous deferral
    budget an indoor photo can have forty borderline tags, which is three calls, not one.
    """
    scores = {f"t{i}": 0.5 for i in range(40)}
    rows = [{"image_id": "a", "image_type": "indoor", "scores": scores, "answers": dict.fromkeys(scores, False)}]
    thresholds = {slug: {"threshold": 0.5} for slug in scores}
    result = embedding_run.hybrid_curve(
        rows=rows,
        thresholds=thresholds,
        reference={"a": {"tags": {}, "evaluable_slugs": list(scores)}},
        calls_today={"a": 5},
        chunk_size=15,
        budgets=(1.0,),
    )
    row = result["budgets"][0]
    assert row["decisions_deferred"] == 40
    assert row["images_needing_a_call"] == 1
    assert row["vlm_calls_hybrid"] == 3  # ceil(40 / 15)


def test_an_image_with_nothing_borderline_costs_no_call_at_all():
    scores = {"t0": 0.9, "t1": 0.1}
    rows = [{"image_id": "a", "image_type": "indoor", "scores": scores, "answers": {"t0": True, "t1": False}}]
    result = embedding_run.hybrid_curve(
        rows=rows,
        thresholds={"t0": {"threshold": 0.5}, "t1": {"threshold": 0.5}},
        reference={"a": {"tags": {"t0": 0.8}, "evaluable_slugs": ["t0", "t1"]}},
        calls_today={"a": 5},
        chunk_size=15,
        budgets=(0.0,),
    )
    row = result["budgets"][0]
    assert row["vlm_calls_hybrid"] == 0
    assert row["call_reduction_pct"] == 100.0
