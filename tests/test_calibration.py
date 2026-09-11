from vlm_eval.calibration import (
    Decision,
    average_precision,
    best_threshold,
    cross_validated,
    f1_scan,
    fit_thresholds,
    fold_of,
    score_distribution,
)


def decisions(pairs, slug="kitchen", prefix="img"):
    """(score, reference_positive) pairs -> Decisions, one image each."""
    return [Decision(f"{prefix}{i}", slug, score, bool(pos)) for i, (score, pos) in enumerate(pairs)]


# ---------------------------------------------------------------- average precision


def test_average_precision_is_one_when_every_positive_outranks_every_negative():
    assert average_precision(decisions([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)])) == 1.0


def test_average_precision_is_none_without_positives():
    assert average_precision(decisions([(0.9, 0), (0.1, 0)])) is None


def test_average_precision_credits_ties_as_a_group_not_in_listed_order():
    """All scores equal: the ranking carries no information, so AP must equal the base rate.

    Breaking ties in listed order would score the same data 1.0 or 0.33 depending on input order.
    """
    tied = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
    assert average_precision(decisions(tied)) == 0.5
    assert average_precision(decisions(list(reversed(tied)))) == 0.5


def test_average_precision_drops_when_the_ranking_is_inverted():
    good = average_precision(decisions([(0.9, 1), (0.5, 1), (0.1, 0)]))
    bad = average_precision(decisions([(0.9, 0), (0.5, 1), (0.1, 1)]))
    assert good is not None and bad is not None and bad < good


# ---------------------------------------------------------------- thresholds


def test_best_threshold_lands_between_the_two_clusters():
    t, f1 = best_threshold(decisions([(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)]))
    assert t is not None and 0.2 < t < 0.8
    assert f1 == 1.0


def test_best_threshold_has_nothing_to_fit_without_positives():
    assert best_threshold(decisions([(0.1, 0), (0.4, 0)])) == (None, 0.0)


def test_a_threshold_is_never_equal_to_an_observed_score():
    """Otherwise the answer depends on `>=` versus `>`, which is not a property of the data."""
    rows = decisions([(0.1, 0), (0.5, 1), (0.9, 1)])
    t, _ = best_threshold(rows)
    assert t not in {d.score for d in rows}


# ---------------------------------------------------------------- per-tag floor


def _mixed_tags():
    """`floorboards` has plenty of positives; `sauna` has two. Only the first may earn a threshold."""
    return [
        *(Decision(f"i{i}", "floorboards", 0.9 if i < 20 else 0.1, i < 20) for i in range(40)),
        *(Decision(f"i{i}", "sauna", 0.9 if i < 2 else 0.1, i < 2) for i in range(40)),
    ]


def test_only_tags_above_the_floor_get_their_own_threshold():
    fit = fit_thresholds(_mixed_tags(), min_positives=30)
    assert fit["per_tag"]["floorboards"]["own"] is False  # 20 positives, floor is 30
    assert fit["per_tag"]["sauna"]["own"] is False
    assert fit["tags_with_own_threshold"] == 0

    generous = fit_thresholds(_mixed_tags(), min_positives=10)
    assert generous["per_tag"]["floorboards"]["own"] is True
    assert generous["per_tag"]["sauna"]["own"] is False
    assert generous["tags_with_own_threshold"] == 1


def test_a_tag_below_the_floor_falls_back_to_the_global_threshold():
    fit = fit_thresholds(_mixed_tags(), min_positives=10)
    assert fit["per_tag"]["sauna"]["threshold"] == fit["global"]["threshold"]


# ---------------------------------------------------------------- folds


def test_fold_assignment_is_deterministic_and_independent_of_order():
    ids = [f"img{i}" for i in range(200)]
    once = [fold_of(i, folds=5, seed=1) for i in ids]
    again = [fold_of(i, folds=5, seed=1) for i in reversed(ids)]
    assert once == list(reversed(again))


def test_every_fold_gets_some_images():
    counts = [0] * 5
    for i in range(500):
        counts[fold_of(f"img{i}", folds=5, seed=7104)] += 1
    assert all(c > 50 for c in counts)


def test_the_same_image_is_never_split_across_folds():
    """Two tags on one photo are not independent samples; letting them land apart leaks the image."""
    assert fold_of("abc", folds=5, seed=7104) == fold_of("abc", folds=5, seed=7104)


# ---------------------------------------------------------------- cross validation


def test_cross_validated_thresholds_cannot_score_the_fold_they_were_fitted_on():
    """Each half is separable, but at scales an order of magnitude apart.

    Fitted on everything, one threshold cannot split both halves; fitted per fold and applied to the
    other, it must get some of them wrong. A perfect result here would prove the split leaked.
    """
    rows = [
        *(Decision(f"low{i}", "kitchen", 0.9 if i < 30 else 0.1, i < 30) for i in range(60)),
        *(Decision(f"high{i}", "kitchen", 90.0 if i < 30 else 10.0, i < 30) for i in range(60)),
    ]
    out = cross_validated(rows, folds=2, seed=3, min_positives=10)
    predicted = out["predictions"]["per_tag_threshold"]
    truth = {d.image_id: d.reference_positive for d in rows}
    wrong = sum(1 for image, answers in predicted.items() if answers["kitchen"] != truth[image])
    assert wrong > 0


def test_cross_validated_reports_average_precision_over_every_decision():
    rows = decisions([(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)])
    out = cross_validated(rows, folds=2, seed=1, min_positives=1)
    assert out["per_tag"]["kitchen"]["average_precision"] == 1.0
    assert out["n_decisions"] == 4
    assert out["n_images"] == 4


def test_cross_validated_answers_nothing_for_a_tag_absent_from_the_training_fold():
    """An unanswerable decision stays unanswered — calling it absent would measure the split.

    `sauna` appears on exactly one image. When that image is the test fold, no training row mentions
    the tag, so there is no threshold to apply and the answer must be `None` rather than `False`.
    """
    rows = [
        Decision("lonely", "sauna", 0.7, True),
        *(Decision(f"k{i}", "kitchen", 0.9 if i < 20 else 0.1, i < 20) for i in range(40)),
    ]
    out = cross_validated(rows, folds=2, seed=11, min_positives=1)

    assert out["predictions"]["per_tag_threshold"]["lonely"]["sauna"] is None
    # The global threshold was still fitted on the kitchen rows, so kitchen stays answerable.
    assert all(
        answers["kitchen"] is not None
        for image, answers in out["predictions"]["per_tag_threshold"].items()
        if "kitchen" in answers
    )


# ---------------------------------------------------------------- distributions


def test_score_distribution_separates_the_two_classes():
    dist = score_distribution(decisions([(0.9, 1), (0.85, 1), (0.2, 0), (0.1, 0)]))
    assert dist["positive"]["n"] == 2
    assert dist["negative"]["n"] == 2
    assert dist["overlap_pct"] == 0.0
    assert dist["median_gap"] > 0


def test_score_distribution_reports_overlap_when_the_classes_mix():
    dist = score_distribution(decisions([(0.6, 1), (0.4, 1), (0.7, 0), (0.3, 0)]))
    assert dist["overlap_pct"] == 50.0


def test_score_distribution_survives_one_empty_class():
    dist = score_distribution(decisions([(0.6, 0), (0.4, 0)]))
    assert dist["positive"] == {"n": 0}
    assert "overlap_pct" not in dist


# ---------------------------------------------------------------- fast path agrees with the slow one


def test_the_one_pass_sweep_agrees_with_the_exhaustive_scan():
    """`best_threshold` is a one-pass rewrite of `f1_scan`; the scan is the oracle it must match.

    Random scores with deliberate repeats, because ties are where a sweep and a rescan diverge: a cut
    placed inside a group of equal scores is not a threshold any comparison could produce.
    """
    import random

    rng = random.Random(11)
    for _ in range(200):
        rows = [
            Decision(
                f"i{i}",
                "t",
                # Half continuous, half drawn from four repeated values, so groups are common.
                round(rng.random() if rng.random() < 0.5 else rng.choice([0.1, 0.2, 0.3, 0.3]), 3),
                rng.random() < 0.3,
            )
            for i in range(rng.randint(1, 80))
        ]
        assert best_threshold(rows) == f1_scan(rows)


def test_the_sweep_handles_a_single_decision_and_an_all_positive_set():
    assert best_threshold([Decision("a", "t", 0.4, True)]) == f1_scan([Decision("a", "t", 0.4, True)])
    both = [Decision("a", "t", 0.4, True), Decision("b", "t", 0.4, True)]
    assert best_threshold(both) == f1_scan(both)


# ---------------------------------------------------------------- duplicate photographs


def test_copies_of_one_photograph_share_a_fold():
    """Two ids holding identical pixels are one sample, however many times it was uploaded.

    Without the grouping a threshold is fitted on an image that is byte-identical to one it then scores,
    which is the training set marking its own exam.
    """
    rows = [
        *(Decision(f"copy{i}", "kitchen", 0.9, True) for i in range(4)),
        *(Decision(f"other{i}", "kitchen", 0.1, False) for i in range(20)),
    ]
    groups = {f"copy{i}": "same-photo" for i in range(4)}
    out = cross_validated(rows, folds=5, seed=7104, min_positives=1, group_of=groups)
    assert out["split_on"] == "image content"

    placed = {
        image_id: fold_of(groups.get(image_id, image_id), folds=5, seed=7104) for image_id in {d.image_id for d in rows}
    }
    assert len({placed[f"copy{i}"] for i in range(4)}) == 1


def test_the_split_records_how_many_distinct_images_it_actually_had():
    rows = [
        *(Decision(f"copy{i}", "kitchen", 0.9, True) for i in range(3)),
        *(Decision(f"other{i}", "kitchen", 0.1, False) for i in range(7)),
    ]
    groups = {f"copy{i}": "same-photo" for i in range(3)}
    out = cross_validated(rows, folds=2, seed=1, min_positives=1, group_of=groups)
    assert out["n_images"] == 10
    assert out["n_distinct_images"] == 8


def test_without_groups_the_split_says_so():
    rows = decisions([(0.9, 1), (0.1, 0)])
    out = cross_validated(rows, folds=2, seed=1, min_positives=1)
    assert out["split_on"] == "image id"
