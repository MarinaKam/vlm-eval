import pytest

from vlm_eval import metrics

GEMINI = {
    1: {"image_id": 1, "tags": {"kitchen": 0.8}, "evaluable_slugs": ["kitchen", "pool", "garden"]},
    2: {"image_id": 2, "tags": {"pool": 0.8}, "evaluable_slugs": ["kitchen", "pool"]},
}


def test_tagging_agreement_counts_only_evaluable_and_answered():
    rows = [
        {"image_id": 1, "repeat": 0, "answers": {"kitchen": True, "pool": True, "garden": None, "extra": True}},
        {"image_id": 2, "repeat": 0, "answers": {"kitchen": False, "pool": False}},
        {"image_id": 2, "repeat": 1, "answers": {"kitchen": True, "pool": True}},  # other repeat ignored
    ]
    m = metrics.tagging_agreement(rows, GEMINI)
    assert m["n_images"] == 2
    o = m["overall"]
    assert (o["tp"], o["fp"], o["fn"], o["tn"], o["null"]) == (1, 1, 1, 1, 1)
    assert o["accuracy"] == 50.0 and o["fpr"] == 50.0 and o["recall"] == 50.0
    assert m["per_tag"]["garden"]["unparsed_rate"] == 100.0
    assert "extra" not in m["per_tag"]


def test_consistency_jaccard_and_identical():
    rows = [
        {"image_id": 1, "repeat": 0, "answers": {"a": True, "b": False}},
        {"image_id": 1, "repeat": 1, "answers": {"a": True, "b": False}},
        {"image_id": 2, "repeat": 0, "answers": {"a": True, "b": True}},
        {"image_id": 2, "repeat": 1, "answers": {"a": True, "b": False}},
        {"image_id": 3, "repeat": 0, "answers": {"a": True}},  # single repeat, skipped
    ]
    c = metrics.tagging_consistency(rows)
    assert c["n_images_with_repeats"] == 2
    assert c["identical_pct"] == 50.0
    assert c["mean_jaccard"] == 0.75


def test_chunk_comparison():
    a = [{"image_id": 1, "repeat": 0, "answers": {"x": True, "y": False, "z": True}}]
    b = [{"image_id": 1, "repeat": 0, "answers": {"x": True, "y": True, "z": None}}]
    r = metrics.tagging_chunk_comparison(a, b)
    assert r == {"n_images": 1, "agreement_pct": 50.0, "null_in_b": 1}


def test_latency_stats():
    s = metrics.latency_stats([1.0, 2.0, 3.0])
    assert s["mean_s"] == 2.0 and s["images_per_hour_serial"] == 1800


def test_caption_stats():
    rows = [{"image_id": 1, "captions": {"base_caption": "a b c", "detailed_caption": None}, "errors": []}]
    s = metrics.caption_stats(rows)
    assert s["empty_pct"] == 50.0 and s["mean_words"] == {"base_caption": 3.0}


def test_grounding_stats_scores_only_comparable_images():
    rows = [
        {"image_id": 1, "detections": {"kitchen": [{"bbox": [0, 0, 1, 1]}]}},
        {"image_id": 2, "detections": {"kitchen": []}},
    ]
    g = metrics.grounding_stats(rows, GEMINI)
    assert g["kitchen"]["recall_vs_reference"] == 100.0
    assert g["kitchen"]["fp_rate_vs_reference"] == 0.0


def test_grounding_stats_does_not_score_targets_without_a_tag():
    # `fireplace` is not among the evaluable slugs -> detections are not judged either way.
    rows = [{"image_id": 1, "detections": {"fireplace": [{"bbox": [0, 0, 1, 1]}]}}]
    g = metrics.grounding_stats(rows, GEMINI)
    assert g["fireplace"]["not_comparable"] == 1
    assert g["fireplace"]["not_comparable_detected"] == 1
    assert g["fireplace"]["recall_vs_reference"] is None
    assert g["fireplace"]["fp_rate_vs_reference"] is None


def test_options_are_symmetric_and_include_fixed_costs():
    """Per-image and per-hour options are compared the same way whichever one is in use today, and a
    fixed monthly charge (a cluster fee, a reserved disk) counts."""
    from vlm_eval.economics import Inputs, Option, crossover, render

    api = Option(name="API", kind="per_image", price=0.001)
    autoscaled = Option(name="GPU autoscaled", kind="per_hour", price=1.0, throughput_per_hour=1000)
    always = Option(name="GPU always on", kind="per_hour", price=1.0, throughput_per_hour=1000, always_on=True)
    cluster = Option(
        name="GKE pod",
        kind="per_hour",
        price=1.0,
        throughput_per_hour=2000,  # cheaper per image than `api`, but carries a monthly fee
        fixed_monthly=73.0,
        fixed_note="cluster fee",
    )

    assert api.monthly_cost(10_000) == 10.0  # 10k * $0.001
    assert autoscaled.monthly_cost(10_000) == 10.0  # 10 GPU-hours at $1
    assert always.monthly_cost(10_000) == 730.0  # time, not volume
    assert always.monthly_cost(10) == 730.0
    assert cluster.monthly_cost(10_000) == 78.0  # 5 GPU-hours at 2000/h + the fixed fee

    # Same billing basis -> no crossover; the ratio just holds.
    assert crossover(api, Option(name="B", kind="per_image", price=0.002)) is None
    # Per-image vs autoscaled: both scale with volume, so growing into it changes nothing.
    assert crossover(api, autoscaled) is None
    # Per-image vs always-on: they meet where the fixed monthly cost is covered.
    assert crossover(api, always) == 730_000
    # A fixed monthly fee creates a crossover even for an autoscaled option: below it the fee dominates.
    assert crossover(api, cluster) == 146_000
    # Parallel lines (same effective per-image rate, different fixed cost) never meet.
    parallel = Option(name="same rate", kind="per_hour", price=1.0, throughput_per_hour=1000, fixed_monthly=73.0)
    assert crossover(api, parallel) is None

    # Direction does not change the arithmetic, only the wording.
    from_api = render(Inputs(options=[api, always], current="API", scenarios=[("now", 100_000)]))
    from_gpu = render(Inputs(options=[api, always], current="GPU always on", scenarios=[("now", 100_000)]))
    assert "Today: **API**" in from_api and "Today: **GPU always on**" in from_gpu
    assert "costs an extra $7,560/year" in from_api  # always-on is dearer at 100k
    assert "saves $7,560/year" in from_gpu  # ...which is the same fact, reversed


def test_a_per_hour_option_without_throughput_is_refused():
    """Without it there is no way to turn a volume into hours, so any cost would be invented."""
    from vlm_eval.economics import Option

    with pytest.raises(ValueError) as e:
        Option(name="mystery GPU", kind="per_hour", price=1.0)
    assert "throughput_per_hour" in str(e.value)


def test_current_option_must_exist():
    from vlm_eval.economics import Inputs, Option

    with pytest.raises(ValueError) as e:
        Inputs(options=[Option(name="A", kind="per_image", price=1.0)], current="B")
    assert "not one of the options" in str(e.value)
