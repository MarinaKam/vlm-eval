from vlm_eval import report


def test_cost_needs_a_throughput_and_a_price_from_the_same_machine():
    assert report.cost_per_1k(1000, 0.9) == 0.9
    assert report.cost_per_1k(None, 0.9) is None  # no throughput -> no figure
    assert report.cost_per_1k(1000, None) is None  # no price for *that* machine -> no figure


def test_render_model_with_empty_and_full_metrics():
    card = {"model": "m", "name": "Model M", "verdict": "Not recommended"}
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
    assert "| pool | 3 | 1 | 0 | 0 | 2 | 100.0 | 100.0 | 0.0 |" in md
    assert "2000 images/hour (measured at concurrency 4)" in md
    assert "| Model M |" in report.render_comparison([card], [m])


def test_the_two_worlds_are_reported_separately_and_never_multiplied():
    """Throughput from a laptop times a cloud GPU's hourly price is not a cost — it once produced a
    figure eight times off in a document that looked authoritative. Each machine gets its own block."""
    m = {"tagging": {"latency": {"mean_s": 8.4, "images_per_hour_serial": 429}}}
    card = {
        "model": "m",
        "name": "Model M",
        "measured_on": "Apple M4 Max, Ollama 4-bit",
        "projection": {
            "hardware": "NVIDIA L4 (GCP spot)",
            "images_per_hour": 2000,
            "usd_per_hour": 0.5832,
            "source": "published vLLM benchmark, not measured here",
        },
    }
    md = report.render_model(card, m)

    assert "**Measured on this run** — Apple M4 Max, Ollama 4-bit" in md
    assert "429 images/hour" in md
    # No hourly price for a laptop, so no cost is invented for it.
    assert "no hourly price for this machine" in md

    assert "**Projected for deployment** — NVIDIA L4 (GCP spot)" in md
    assert "2000 images/hour (published vLLM benchmark, not measured here)" in md
    assert "0.2916 USD" in md  # 0.5832 / 2000 * 1000, both numbers from the L4

    # The wrong pairing must not appear anywhere: laptop throughput at the L4 price would be $1.36.
    assert "1.3594" not in md
    # Monthly totals and fixed costs belong to the economics report, which models always-on capacity.
    assert "economics.md" in md


def test_a_projection_without_a_price_reports_speed_only():
    card = {"model": "m", "name": "M", "projection": {"hardware": "some GPU", "images_per_hour": 500}}
    md = report.render_model(card, {})
    assert "**Projected for deployment** — some GPU" in md
    assert "500 images/hour" in md
    assert "no hourly price for this machine" in md
