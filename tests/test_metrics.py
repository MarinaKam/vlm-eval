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


def test_economics_breakeven_and_peak():
    from vlm_eval.economics import Inputs, break_even_images, monthly_gpu_cost, peak_analysis, render

    inp = Inputs(
        api_cost_per_image=0.001,
        api_cost_per_image_optimized=0.0005,
        gpu_usd_per_hour=1.0,
        gpu_images_per_hour=1000,
        peak_hour_images=5000,
        busy_hours_pct=10.0,
        scenarios=[("small", 10_000), ("big", 100_000)],
    )
    # 10k images at 1000/h = 10 GPU-hours at $1
    assert monthly_gpu_cost(10_000, inp) == 10.0
    # a GPU that never sleeps costs 730 -> break-even at 730/0.001
    assert break_even_images(inp) == 730_000
    # never charge more than running non-stop
    assert monthly_gpu_cost(10_000_000, inp) == 730.0

    peak = peak_analysis(inp)
    assert peak["hours_for_one_gpu"] == 5.0
    assert peak["gpus_to_absorb_in_one_hour"] == 5

    md = render(inp, currency_note="test note")
    assert "Break-even for a GPU running non-stop: 730,000" in md
    assert "5 GPUs at once" in md
    assert "Not yet." in md  # 100k/month is below break-even
    assert "test note" in md


def test_economics_verdict_flips_past_break_even():
    from vlm_eval.economics import Inputs, render

    inp = Inputs(
        api_cost_per_image=0.01, gpu_usd_per_hour=1.0, gpu_images_per_hour=1000, scenarios=[("huge", 1_000_000)]
    )
    assert "Worth building." in render(inp)
