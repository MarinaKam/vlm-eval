from vlm_eval import report


def test_cost_helpers():
    assert report.cost_per_1k(1000, 0.9) == 0.9
    assert report.monthly_cost(100_000, 1000, 0.9) == 90.0
    assert report.monthly_cost(1000, 1000, 0.9, min_hours=730) == 657.0
    assert report.cost_per_1k(None, 0.9) is None


def test_render_model_with_empty_and_full_metrics():
    card = {"model": "m", "name": "Model M", "gpu_usd_per_hour": 1.0, "verdict": "Not recommended"}
    md = report.render_model(card, {})
    assert "# Model M" in md and "| Capability | Result | Notes |" in md and "Not recommended" in md
    m = {
        "tagging": {
            "agreement": {
                "overall": {"accuracy": 91.0, "fpr": 4.0, "recall": 80.0, "unparsed_rate": 0.0},
                "per_tag": {
                    "pool": {
                        "n": 3,
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "tn": 2,
                        "precision": 100.0,
                        "recall": 100.0,
                        "fpr": 0.0,
                    }
                },
            },
            "consistency": {"identical_pct": 95.0, "mean_jaccard": 0.98},
            "latency": {"mean_s": 3.6, "p95_s": 5.0, "images_per_hour_serial": 1000},
        },
        "perf": {"images_per_hour_measured": 2000, "concurrency": 4},
    }
    md = report.render_model(card, m)
    assert "| Est. hosting cost / 1K images | 0.5 USD @ 1.0 USD/h |" in md
    assert "2000 (measured at concurrency 4)" in md
    assert "| pool | 3 | 1 | 0 | 0 | 2 | 100.0 | 100.0 | 0.0 |" in md
    cmp_md = report.render_comparison([card], [m])
    assert "| Model M |" in cmp_md
