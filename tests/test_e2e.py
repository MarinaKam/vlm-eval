"""End-to-end: dataset on disk -> run -> metrics -> review -> reports, with nothing mocked but the model.

Each step consumes what the previous one wrote, so a break anywhere in the chain fails here rather than
three hours into a real run. The backend is a stub that answers like a real one; everything else — file
layout, resume, parsing, metric maths, report rendering — is the production code path.

A second test drives the same chain through the CLI, so the argument wiring is covered too.
"""

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from vlm_eval import metrics, report, review, runner
from vlm_eval.backends.base import Response
from vlm_eval.dataset import Item
from vlm_eval.tasks import tagging

TAGS = [
    {"slug": "pool", "name": "Pool", "question_text": "Is there a pool?", "category": "common", "order": 0},
    {"slug": "kitchen", "name": "Kitchen", "question_text": "Is this a kitchen?", "category": "indoor", "order": 0},
    {"slug": "garden", "name": "Garden", "question_text": "Is there a garden?", "category": "outdoor", "order": 0},
]


class StubBackend:
    """Answers true for whatever `positive` lists, and records what it was asked."""

    name = "stub"

    def __init__(self, positive: set[str], *, fail_on: str | None = None):
        self.positive = positive
        self.fail_on = fail_on
        self.calls: list[str] = []

    def chat(self, images, prompt, *, json_schema=None, max_tokens=1024, temperature=0.0, logprobs=False):
        self.calls.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("backend exploded")
        asked = json.loads(prompt) if prompt.startswith("{") else {}
        if asked:
            body = {slug: slug in self.positive for slug in asked}
        else:  # captions / summary / grounding all take a JSON schema with known keys
            props = (json_schema or {}).get("properties", {})
            body = {k: ("A room with a window. It is bright and airy." if k != "detections" else []) for k in props}
        return Response(text=json.dumps(body), latency_s=0.01, usage={"prompt_tokens": 10})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A miniature dataset: two images, reference answers, tags — laid out exactly like the real one."""
    data, runs, reports = tmp_path / "data", tmp_path / "runs", tmp_path / "reports"
    images = data / "images"
    images.mkdir(parents=True)
    runs.mkdir()

    items = []
    for image_id, kind in (("img-a", "indoor"), ("img-b", "outdoor")):
        Image.new("RGB", (32, 32), "white").save(images / f"{image_id}.jpg")
        items.append(Item(image_id, "http://x/a.jpg", "http://x/a.jpg", kind))

    with (data / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "url", "s3_url", "image_type", "user_id", "job_created_at"])
        for it in items:
            w.writerow([it.image_id, it.url, it.s3_url, it.image_type, "", ""])

    (data / "tags.json").write_text(json.dumps(TAGS))
    # The export carries production's own settings; the harness must use these, not its own constants.
    (data / "prompts.json").write_text(
        json.dumps(
            {
                "caption_prompts": {"base_caption": "short"},
                "prompt_templates": {"caption_header": "You are a describing machine."},
                "processing_config": {
                    "classification_chunk_size": {"value_int": 2},
                    "individual_questions": {"value_json": ["garden"]},
                    "image_optimization_enabled": {"value_bool": True},
                    "image_optimization_max_dimension": {"value_int": 800},
                    "image_optimization_jpeg_quality": {"value_int": 54},
                    "image_optimization_target_size_kb": {"value_int": 600},
                },
            }
        )
    )
    (data / "grounding_targets.json").write_text(json.dumps({"pool": "swimming pool"}))
    # Reference: pool present on both; kitchen judged on img-a and absent
    with (data / "gemini_tags.jsonl").open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "image_id": "img-a",
                    "image_type": "indoor",
                    "tags": {"pool": 0.8},
                    "evaluable_slugs": ["pool", "kitchen"],
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "image_id": "img-b",
                    "image_type": "outdoor",
                    "tags": {"pool": 0.8},
                    "evaluable_slugs": ["pool", "garden"],
                }
            )
            + "\n"
        )

    monkeypatch.setattr("vlm_eval.dataset.DATA", data)
    monkeypatch.setattr("vlm_eval.dataset.IMAGES", images)
    monkeypatch.setattr("vlm_eval.runner.RUNS", runs)
    monkeypatch.setattr("vlm_eval.review.DATA", data)
    monkeypatch.setattr("vlm_eval.review.IMAGES", images)
    return {"data": data, "runs": runs, "reports": reports, "items": items}


def test_full_chain_from_run_to_report(workspace, monkeypatch):
    from vlm_eval import dataset

    items = workspace["items"]
    # The model finds the pool on both, and wrongly claims a kitchen where the reference says none.
    be = StubBackend({"pool", "kitchen"})
    cfg = runner.RunConfig(model="stub", chunk_size=15, individual=[])

    out = runner.tagging_out("stub", 15)
    n = runner.run_over_items(
        items, lambda it: runner.run_tagging_one(be, it, TAGS, cfg), out, repeats=1, workers=1, log=lambda _: None
    )
    assert n == 2
    assert out.exists()

    # Resume: a second pass must do nothing at all.
    assert (
        runner.run_over_items(
            items, lambda it: runner.run_tagging_one(be, it, TAGS, cfg), out, repeats=1, workers=1, log=lambda _: None
        )
        == 0
    )

    rows = dataset.load_jsonl(out)
    assert len(rows) == 2
    # indoor asks common+indoor, outdoor asks common+outdoor
    assert set(rows[0]["answers"]) | set(rows[1]["answers"]) == {"pool", "kitchen", "garden"}

    gem = dataset.gemini_tags_by_image()
    agree = metrics.tagging_agreement(rows, gem)
    o = agree["overall"]
    assert o["tp"] == 2  # pool found on both
    assert o["fp"] == 1  # kitchen claimed, reference says no
    assert o["null"] == 0
    assert agree["per_tag"]["pool"]["recall"] == 100.0

    # The disagreement must surface in the review page, and a verdict must score both sides.
    cases = review.disagreements(rows, gem, TAGS)
    assert [(c["slug"], c["model"], c["gemini"]) for c in cases] == [("kitchen", True, False)]
    page = review.build_review_html("stub", cases, workspace["reports"] / "review" / "stub.html")
    assert "Download decisions" in page.read_text()

    decisions = workspace["data"] / "d.json"
    decisions.write_text(json.dumps([{"image_id": "img-a", "slug": "kitchen", "truth": False}]))
    store = workspace["data"] / "manual_labels.json"
    assert review.apply_decisions([decisions], store)["added"] == 1
    judged = review.manual_agreement(rows, gem, store)
    assert judged == {"n": 1, "model_correct_pct": 0.0, "gemini_correct_pct": 100.0}

    # Reports render from the numbers, with no placeholders left behind.
    m = {
        "tagging": {
            "agreement": agree,
            "consistency": metrics.tagging_consistency(rows),
            "latency": metrics.latency_stats([r["latency_s"] for r in rows]),
        }
    }
    card = {"model": "stub", "name": "Stub", "gpu_usd_per_hour": 1.0, "verdict": "Not recommended"}
    md = report.render_model(card, m)
    assert "# Stub" in md and "Not recommended" in md and "| pool |" in md
    assert report.render_comparison([card], [m]).count("| Stub |") == 1


def test_backend_failure_is_recorded_not_swallowed(workspace):
    """A model that breaks mid-run must leave a row saying so, not a silently empty answer."""
    be = StubBackend({"pool"}, fail_on="kitchen")
    cfg = runner.RunConfig(model="broken", chunk_size=15, individual=[])
    row = runner.run_tagging_one(be, workspace["items"][0], TAGS, cfg)
    assert row["errors"] and "backend exploded" in row["errors"][0]
    assert row["answers"]["kitchen"] is None  # unanswered, not "false"


def test_chunking_matches_the_reference_pipeline():
    """The whole point is asking production's questions in production's batches."""
    q = tagging.questions_for("indoor", TAGS)
    assert list(q) == ["pool", "kitchen"]  # common first, then the category
    assert tagging.chunk_questions(q, 15, []) == [q]  # fits in one chunk
    assert len(tagging.chunk_questions(q, 1, [])) == 2  # split by size
    assert tagging.chunk_questions(q, 15, ["kitchen"]) == [{"pool": q["pool"]}, {"kitchen": q["kitchen"]}]


def test_cli_wiring_end_to_end(workspace, monkeypatch, capsys):
    """Same chain, driven through the CLI: presets, positional args, file outputs."""
    from vlm_eval import cli

    monkeypatch.setattr(cli, "REPORTS", workspace["reports"])
    monkeypatch.setattr(
        cli,
        "_presets",
        lambda: {
            "stub": {
                "run_name": "stub",
                "served_name": "s",
                "flavor": "ollama",
                "base_url": "http://x/v1",
                "coords": "abs",
            }
        },
    )
    monkeypatch.setattr(cli, "_backend", lambda a: StubBackend({"pool"}))

    cli.main(["run", "stub", "tagging", "--no-logprobs"])
    # Production batches by 2 in this fixture; the harness must follow the export, not its own default.
    assert runner.tagging_out("stub", 2).exists()
    assert not runner.tagging_out("stub", 15).exists()

    cli.main(["metrics", "stub"])
    saved = json.loads((workspace["runs"] / "stub" / "metrics.json").read_text())
    assert saved["tagging"]["agreement"]["overall"]["tp"] == 2

    (workspace["reports"] / "cards").mkdir(parents=True)
    (workspace["reports"] / "cards" / "stub.json").write_text(json.dumps({"model": "stub", "name": "Stub"}))
    cli.main(["report", "stub"])
    assert (workspace["reports"] / "stub.md").exists()

    cli.main(["compare", "stub"])
    assert "| Stub |" in (workspace["reports"] / "comparison.md").read_text()


def test_economics_report_renders_from_config(tmp_path, monkeypatch, capsys):
    """The money argument is generated, not hand-written."""
    from vlm_eval import cli

    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    monkeypatch.setattr("vlm_eval.dataset.DATA", data)
    monkeypatch.setattr(cli, "REPORTS", reports)
    (data / "economics.json").write_text(
        json.dumps(
            {
                "api_cost_per_image": 0.001,
                "gpu_usd_per_hour": 1.0,
                "gpu_images_per_hour": 1000,
                "peak_hour_images": 4000,
                "busy_hours_pct": 9.0,
                "weights_gb": 10,
                "scenarios": [["now", 20_000]],
                "hosting": [
                    {"name": "always on", "usd_per_hour": 1.0, "always_on": True},
                    {"name": "autoscaled", "usd_per_hour": 1.0, "cold_start_min": 5},
                ],
            }
        )
    )
    cli.main(["economics"])
    md = (reports / "economics.md").read_text()
    assert "Break-even for a GPU running non-stop: 730,000" in md
    assert "4 GPUs at once" in md  # 4000 peak / 1000 per hour
    assert "| always on | $1.0000 | $8,760/yr |" in md
    assert "| autoscaled | $1.0000 | $240/yr |" in md
    assert "$0.20/month" in md  # 10 GB of weights in object storage
    assert "Not yet." in md


def test_missing_dataset_fails_with_an_instruction(tmp_path, monkeypatch):
    """A missing manifest should say what to run, not raise FileNotFoundError three frames deep."""
    from vlm_eval import dataset

    with pytest.raises(SystemExit) as e:
        dataset.load_manifest(tmp_path / "nope.csv")
    assert "export_staging_dataset.py" in str(e.value)


def test_summary_refuses_a_partial_listing(workspace):
    """Given fewer images than the listing has, record a failure — a model would invent the rest."""
    prop = {"property_job_id": "p", "image_ids": ["img-a", "missing-1", "missing-2"], "property_summary": "ref"}

    class Explode:
        def chat(self, *a, **k):
            raise AssertionError("must not reach the model")

    row = runner.run_summary_one(Explode(), prop, "prompt", Path(workspace["data"] / "images"))
    assert row["summary"] is None and row["n_images"] == 1 and row["n_expected"] == 3


def test_cost_comparison_reads_csvs_and_diffs_answers(tmp_path, monkeypatch):
    """`vlm-eval cost --chunks A B` must compare answers, not just prices: a cheaper batch size that
    changes the tags is not a free saving."""
    from vlm_eval import metrics
    from vlm_eval.cli import _read_cost_csv

    header = "image_id,url,classification_tags,n_calls,prompt_tokens,candidate_tokens,total_tokens,cost_2_5\n"
    a = tmp_path / "a.csv"
    a.write_text(header + "i1,u,pool; kitchen,4,1000,50,1050,0.002\ni2,u,garden,4,1000,50,1050,0.002\n")
    b = tmp_path / "b.csv"
    b.write_text(header + "i1,u,pool,2,500,50,550,0.001\ni2,u,garden,2,500,50,550,0.001\n")

    tags_a, cost_a, calls_a, prompt_a = _read_cost_csv(a)
    tags_b, cost_b, calls_b, _ = _read_cost_csv(b)
    assert tags_a["i1"] == {"pool", "kitchen"}  # "; " separated, whitespace stripped
    assert (cost_a, calls_a, prompt_a) == (0.002, 4.0, 1000.0)
    assert cost_b == 0.001 and calls_b == 2.0

    d = metrics.tagset_agreement(tags_a, tags_b)
    assert d["identical_pct"] == 50.0  # i1 changed, i2 did not
    assert d["lost"] == [("kitchen", 1)]
    assert d["gained"] == []
    assert (d["tags_first"], d["tags_second"]) == (3, 2)


def test_tagset_agreement_on_no_overlap():
    from vlm_eval import metrics

    assert metrics.tagset_agreement({"a": {"x"}}, {"b": {"x"}}) == {"n_images": 0}


def test_sweep_runs_every_task_for_any_model(workspace, monkeypatch):
    """`sweep` is the generic full run: same stages for any preset, and skips what you set to 0."""
    import argparse

    from vlm_eval import cli

    called = []
    monkeypatch.setattr(cli, "cmd_run", lambda a: called.append((a.task, a.limit, a.chunk, a.repeats)))
    monkeypatch.setattr(cli, "cmd_metrics", lambda a: called.append(("metrics", a.model, None, None)))
    monkeypatch.setattr(cli, "cmd_status", lambda a: None)

    cli.cmd_sweep(
        argparse.Namespace(
            model="any-model",
            served_name=None,
            base_url=None,
            flavor=None,
            coords=None,
            tagging=500,
            captions=100,
            grounding=100,
            chunk_all=60,
            consistency=0,
            workers=1,
            no_logprobs=True,
        )
    )

    tasks = [c[0] for c in called]
    # cheapest first, so an interrupted sweep still covers every capability
    assert tasks == ["summary", "grounding", "captions", "tagging", "tagging", "metrics"]
    assert called[3][2] == 0  # the all-in-one-call stage asks every question at once
    assert called[4][1] == 500  # the main tagging run honours --tagging
    assert called[-1][1] == "any-model"


def test_sweep_skips_stages_set_to_zero(workspace, monkeypatch):
    import argparse

    from vlm_eval import cli

    called = []
    monkeypatch.setattr(cli, "cmd_run", lambda a: called.append(a.task))
    monkeypatch.setattr(cli, "cmd_metrics", lambda a: None)
    monkeypatch.setattr(cli, "cmd_status", lambda a: None)

    cli.cmd_sweep(
        argparse.Namespace(
            model="m",
            served_name=None,
            base_url=None,
            flavor=None,
            coords=None,
            tagging=0,
            captions=0,
            grounding=0,
            chunk_all=0,
            consistency=0,
            workers=1,
            no_logprobs=False,
        )
    )
    assert called == ["summary"]  # only the stage with no limit to skip


def test_pipeline_config_is_read_from_the_export_not_guessed():
    """Constants that happen to match production are not evidence — the settings must come from data."""
    from vlm_eval import pipeline_config

    cfg = pipeline_config.load(
        {
            "processing_config": {
                "classification_chunk_size": {"value_int": 25},
                "individual_questions": {"value_json": ["utility_room"]},
                "image_optimization_enabled": {"value_bool": True},
                "image_optimization_max_dimension": {"value_int": 1536},
                "image_optimization_jpeg_quality": {"value_int": 54},
                "image_optimization_target_size_kb": {"value_int": 600},
            }
        }
    )
    assert cfg.chunk_size == 25
    assert cfg.individual_questions == ["utility_room"]
    assert cfg.jpeg_quality == 54  # not the 85 this tool used to assume
    assert cfg.target_size_kb == 600
    assert cfg.faithful and cfg.defaulted == []
    assert cfg.strict() is cfg


def test_pipeline_config_reports_what_it_had_to_guess():
    from vlm_eval import pipeline_config

    cfg = pipeline_config.load({"processing_config": {"classification_chunk_size": {"value_int": 15}}})
    assert cfg.chunk_size == 15
    assert set(cfg.defaulted) == {
        "individual_questions",
        "image_optimization_enabled",
        "image_optimization_max_dimension",
        "image_optimization_jpeg_quality",
        "image_optimization_target_size_kb",
    }
    assert not cfg.faithful
    assert "NOT from the export" in cfg.describe()
    with pytest.raises(SystemExit) as e:
        cfg.strict()
    assert "vlm-eval export" in str(e.value)  # tells you how to fix it


def test_download_does_not_re_encode_already_processed_images(tmp_path, monkeypatch):
    """Production uploads images already resized and compressed; re-encoding adds a second generation
    of loss and hands the models different pixels than the reference model saw."""
    import httpx

    from vlm_eval import dataset

    original = b"\xff\xd8\xff\xe0 pretend this is a production-optimised jpeg"
    monkeypatch.setattr(dataset, "IMAGES", tmp_path)

    def handler(request):
        return httpx.Response(200, content=original)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    item = dataset.Item("img", "http://x/a.jpg", "http://x/a.jpg", "indoor")
    done, failed = dataset.download_all([item])
    assert (done, failed) == (1, [])
    assert (tmp_path / "img.jpg").read_bytes() == original  # byte-for-byte, untouched


def test_grounding_targets_have_no_domain_baked_in(tmp_path, monkeypatch):
    from vlm_eval.tasks import grounding

    assert grounding.TARGETS == {}  # nothing about kitchens or radiators lives in the code
    monkeypatch.setattr("vlm_eval.dataset.DATA", tmp_path)
    with pytest.raises(SystemExit) as e:
        grounding.load_targets()
    assert "grounding_targets.json" in str(e.value)

    (tmp_path / "grounding_targets.json").write_text(json.dumps({"fireplace": "fireplace"}))
    assert grounding.load_targets() == {"fireplace": "fireplace"}
