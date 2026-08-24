"""End-to-end: dataset on disk -> run -> metrics -> review -> reports, with nothing mocked but the model.

Each step consumes what the previous one wrote, so a break anywhere in the chain fails here rather than
three hours into a real run. The backend is a stub that answers like a real one; everything else — file
layout, resume, parsing, metric maths, report rendering — is the production code path.

A second test drives the same chain through the CLI, so the argument wiring is covered too.
"""

import csv
import io
import json
from dataclasses import replace
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
                "api_cost_per_image": 0.0012,
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
    assert "level at **608,333 images/month**" in md
    assert "4 in parallel" in md  # 4000 peak / 1000 per hour
    assert "| always on | $8,760 |" in md
    assert "| autoscaled | $240 |" in md
    assert "$0.20/month" in md  # 10 GB of weights in object storage
    # The verdict is per alternative now, and it reads the same whichever side you start from.
    assert "costs an extra $8,472/year" in md  # an always-on GPU at this volume
    assert "level at **608,333 images/month**" in md  # ...and where that would flip
    assert "growing into it is not an argument" in md  # autoscaled scales like the API does


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
            via="server",
            checkpoint=None,
            device=None,
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
            via="server",
            checkpoint=None,
            device=None,
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

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG", quality=54)
    original = buf.getvalue()
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


def test_sweep_covers_models_without_a_server(monkeypatch):
    """Deleting the shell scripts must not lose Florence/InternVL/PaliGemma: `--via` routes to them,
    and skips the tasks each architecture cannot do."""
    import argparse

    from vlm_eval import cli

    seen = []
    monkeypatch.setattr(cli, "cmd_florence", lambda a: seen.append(("florence", a.task, a.limit)))
    monkeypatch.setattr(cli, "cmd_hf", lambda a: seen.append((a.backend, a.task, a.limit)))
    monkeypatch.setattr(cli, "cmd_metrics", lambda a: None)
    monkeypatch.setattr(cli, "cmd_status", lambda a: None)

    def sweep(via, **over):
        seen.clear()
        args = dict(
            model=via,
            served_name=None,
            base_url=None,
            flavor=None,
            coords=None,
            tagging=50,
            captions=20,
            grounding=10,
            chunk_all=0,
            consistency=0,
            workers=1,
            no_logprobs=True,
            via=via,
            checkpoint=None,
            device=None,
        )
        args.update(over)
        cli.cmd_sweep(argparse.Namespace(**args))
        return [(b, t) for b, t, _ in seen]

    # Florence-2 takes one image at a time -> no property summary
    assert sweep("florence") == [("florence", "captions"), ("florence", "grounding"), ("florence", "tagging")]
    # PaliGemma is single-image and single-turn -> no summary, no grounding sweep stage
    assert sweep("paligemma") == [("paligemma", "captions"), ("paligemma", "tagging")]
    # InternVL is a full chat model -> everything
    assert sweep("internvl") == [
        ("internvl", "summary"),
        ("internvl", "grounding"),
        ("internvl", "captions"),
        ("internvl", "tagging"),
    ]
    # a stage set to 0 is skipped here too
    assert sweep("florence", captions=0) == [("florence", "grounding"), ("florence", "tagging")]


def test_user_presets_override_the_repo_and_carry_the_backend(tmp_path, monkeypatch):
    """Someone else's models must not require editing a tracked file, and a preset can say how the
    model is reached — so `sweep mine` works without repeating --via and --checkpoint."""
    import argparse

    from vlm_eval import cli

    repo = tmp_path / "models.json"
    repo.write_text(json.dumps({"_comment": "ignored", "qwen3": {"run_name": "repo-run", "flavor": "vllm"}}))
    mine = tmp_path / "models.local.json"
    mine.write_text(
        json.dumps(
            {
                "qwen3": {"run_name": "my-run", "flavor": "ollama"},
                "finetune": {"run_name": "ft-v3", "via": "internvl", "checkpoint": "/weights/x", "coords": "abs"},
            }
        )
    )
    monkeypatch.setattr("vlm_eval.config.model_presets", lambda: [repo, mine])

    assert sorted(cli._presets()) == ["finetune", "qwen3"]  # comments dropped, files merged

    a = argparse.Namespace(
        model="qwen3", served_name=None, base_url=None, flavor=None, coords=None, via="server", checkpoint=None
    )
    cli._resolve(a)
    assert (a.model, a.flavor) == ("my-run", "ollama")  # the local file wins

    b = argparse.Namespace(
        model="finetune", served_name=None, base_url=None, flavor=None, coords=None, via="server", checkpoint=None
    )
    cli._resolve(b)
    assert (b.model, b.via, b.checkpoint, b.coords) == ("ft-v3", "internvl", "/weights/x", "abs")

    # An explicit flag still beats the preset.
    c = argparse.Namespace(
        model="finetune",
        served_name=None,
        base_url=None,
        flavor=None,
        coords="norm1000",
        via="florence",
        checkpoint=None,
    )
    cli._resolve(c)
    assert (c.via, c.coords) == ("florence", "norm1000")


def test_unknown_model_name_is_used_as_is(monkeypatch):
    """A one-off model needs no preset at all — pass the flags and the name becomes the run folder."""
    import argparse

    from vlm_eval import cli

    monkeypatch.setattr(cli, "_presets", dict)
    a = argparse.Namespace(
        model="whatever",
        served_name="x:7b",
        base_url="http://h/v1",
        flavor="ollama",
        coords=None,
        via="server",
        checkpoint=None,
    )
    cli._resolve(a)
    assert a.model == "whatever" and a.served_name == "x:7b" and a.coords == "norm1000"


def test_economics_refuses_placeholder_inputs(tmp_path, monkeypatch, capsys):
    """A confident report built on example numbers is worse than no report."""
    from vlm_eval import cli

    data, reports = tmp_path / "data", tmp_path / "reports"
    data.mkdir()
    monkeypatch.setattr("vlm_eval.dataset.DATA", data)
    monkeypatch.setattr(cli, "REPORTS", reports)
    (data / "economics.json").write_text(json.dumps({"api_cost_per_image": 0.001}))

    with pytest.raises(SystemExit) as e:
        cli.main(["economics"])
    message = str(e.value)
    assert "vlm-eval volume" in message and "vlm-eval cost" in message  # says how to fix each one
    assert not (reports / "economics.md").exists()

    # Overridden knowingly: it renders, but the file says so on the first line.
    cli.main(["economics", "--allow-unmeasured"])
    assert (reports / "economics.md").read_text().startswith("> **UNMEASURED INPUTS")


def test_commands_say_what_to_run_instead_of_raising(tmp_path, monkeypatch):
    """Running a step too early must name the command that produces what is missing."""
    from vlm_eval import cli, preconditions

    empty = tmp_path / "data"
    empty.mkdir()
    runs = tmp_path / "runs"
    monkeypatch.setattr("vlm_eval.dataset.DATA", empty)
    monkeypatch.setattr("vlm_eval.runner.RUNS", runs)

    with pytest.raises(SystemExit) as e:
        preconditions.need_dataset(empty)
    assert "vlm-eval export" in str(e.value)

    (empty / "manifest.csv").write_text("image_id\n")
    with pytest.raises(SystemExit) as e:
        preconditions.need_dataset(empty)
    assert "vlm-eval download" in str(e.value)  # manifest exists, images do not

    with pytest.raises(SystemExit) as e:
        preconditions.need_run(runs, "somemodel")
    assert "vlm-eval run somemodel tagging" in str(e.value)

    with pytest.raises(SystemExit) as e:
        preconditions.need_metrics(runs, "somemodel")
    assert "vlm-eval metrics somemodel" in str(e.value)

    with pytest.raises(SystemExit) as e:
        cli.main(["report", "somemodel"])
    assert "vlm-eval metrics" in str(e.value)


def test_stale_results_are_reported_not_silently_used(tmp_path, capsys):
    """Deriving from an out-of-date file looks like success — say so, but do not overrule the user."""
    import os
    import time

    from vlm_eval import preconditions

    source = tmp_path / "run.jsonl"
    target = tmp_path / "metrics.json"
    target.write_text("{}")
    source.write_text("{}")
    os.utime(source, (time.time() + 10, time.time() + 10))  # source is newer

    assert preconditions.stale(target, [source]) == ["run.jsonl"]
    assert preconditions.warn_if_stale(target, [source], "vlm-eval metrics m") is True
    out = capsys.readouterr().out
    assert "older than run.jsonl" in out and "vlm-eval metrics m" in out

    # The other way round: nothing to report.
    os.utime(source, (time.time() - 10, time.time() - 10))
    assert preconditions.warn_if_stale(target, [source], "x") is False


def test_download_rejects_a_url_that_does_not_serve_an_image(tmp_path, monkeypatch, capsys):
    """An expired link or a proxy page answers 200 with HTML. Written as <id>.jpg it looks like a
    successful download and fails hours later, mid-run."""
    import httpx

    from vlm_eval import dataset

    monkeypatch.setattr(dataset, "IMAGES", tmp_path)

    def handler(request):
        return httpx.Response(200, content=b"<html>Access denied</html>", headers={"content-type": "text/html"})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    done, failed = dataset.download_all([dataset.Item("bad", "http://x/a.jpg", "http://x/a.jpg", "indoor")])
    assert (done, failed) == (0, ["bad"])
    assert not (tmp_path / "bad.jpg").exists()  # nothing written
    out = capsys.readouterr().out
    assert "not an image" in out and "text/html" in out


def test_one_unreadable_image_does_not_end_the_run(tmp_path, monkeypatch):
    """A four-hour run must not die on a single bad file: record the failure, keep going."""
    from vlm_eval import dataset, runner

    monkeypatch.setattr(dataset, "IMAGES", tmp_path)
    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "good.jpg")
    items = [dataset.Item("good", "u", "u", "indoor"), dataset.Item("missing", "u", "u", "indoor")]

    cfg = runner.RunConfig(model="m", chunk_size=15, individual=[])
    be = StubBackend({"pool"})
    out = tmp_path / "runs" / "out.jsonl"
    out.parent.mkdir(parents=True)
    n = runner.run_over_items(
        items, lambda it: runner.run_tagging_one(be, it, TAGS, cfg), out, repeats=1, workers=1, log=lambda _: None
    )

    assert n == 2  # both attempted
    rows = {r["image_id"]: r for r in dataset.load_jsonl(out)}
    assert rows["good"]["errors"] == []
    assert "cannot read missing.jpg" in rows["missing"]["errors"][0]  # named, not swallowed


def test_grounding_refuses_to_run_with_no_targets(tmp_path, monkeypatch):
    """Empty targets would write rows with no detections — indistinguishable from a model that found
    nothing. That is the failure this whole tool exists to prevent."""
    from vlm_eval import dataset, runner

    monkeypatch.setattr(dataset, "IMAGES", tmp_path)
    Image.new("RGB", (8, 8), "white").save(tmp_path / "i.jpg")
    cfg = runner.RunConfig(model="m")
    with pytest.raises(ValueError) as e:
        runner.run_grounding_one(StubBackend(set()), dataset.Item("i", "u", "u", "indoor"), {}, cfg)
    assert "grounding_targets.json" in str(e.value)


def test_every_backend_goes_through_the_same_dispatcher(workspace, monkeypatch):
    """A served model and a transformers checkpoint must ask the same questions with the same settings.

    They used not to: the transformers path was a copy that had drifted — it read an emptied constant
    for detection targets, skipped the exported caption header, and ignored production's batch size.
    """
    import argparse

    from vlm_eval import cli, runner

    captured = {}

    def spy(be, *, task, model, cfg, limit=None, workers=1, repeats=1):
        captured[model] = {"task": task, "chunk": cfg.chunk_size, "individual": cfg.individual}

    monkeypatch.setattr(cli, "run_task", spy)
    monkeypatch.setattr(cli, "_backend", lambda a: StubBackend({"pool"}))
    monkeypatch.setattr(cli, "cmd_metrics", lambda a: None)
    monkeypatch.setattr(cli, "_presets", dict)

    class FakeInternVL:
        name = "fake-internvl"

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr("vlm_eval.backends.hf_chat.InternVLBackend", FakeInternVL)

    cli.cmd_run(
        argparse.Namespace(
            model="served",
            task="tagging",
            served_name="x",
            base_url="http://h/v1",
            flavor="ollama",
            coords="abs",
            chunk=None,
            repeats=1,
            workers=1,
            limit=None,
            no_logprobs=True,
        )
    )
    cli.cmd_hf(
        argparse.Namespace(
            backend="internvl",
            task="tagging",
            checkpoint=None,
            model="local",
            device=None,
            chunk=None,
            repeats=1,
            limit=None,
        )
    )

    # The fixture's export says batch by 2 and ask `garden` on its own; both paths must obey it.
    assert captured["served"] == captured["local"] == {"task": "tagging", "chunk": 2, "individual": ["garden"]}
    assert set(captured) == {"served", "local"}
    # And the dispatcher is the real one, not a per-command copy.
    assert runner.tagging_out("served", 2).name == "tagging_chunk2.jsonl"


def test_reference_file_name_is_vendor_neutral_with_a_fallback(tmp_path, monkeypatch):
    """The reference is whatever the pipeline runs today; the file should not name a vendor. Datasets
    exported before the rename must keep working without anyone touching them."""
    from vlm_eval import dataset

    monkeypatch.setattr(dataset, "DATA", tmp_path)

    old = tmp_path / "gemini_tags.jsonl"
    old.write_text(json.dumps({"image_id": "a", "tags": {"pool": 0.8}}) + "\n")
    assert dataset.reference_path("tags").name == "gemini_tags.jsonl"  # legacy dataset still found
    assert dataset.reference_tags_by_image()["a"]["tags"] == {"pool": 0.8}

    new = tmp_path / "reference_tags.jsonl"
    new.write_text(json.dumps({"image_id": "a", "tags": {"garden": 0.9}}) + "\n")
    assert dataset.reference_path("tags").name == "reference_tags.jsonl"  # neutral name wins
    assert dataset.reference_tags_by_image()["a"]["tags"] == {"garden": 0.9}

    # The old entry point keeps working for anything already calling it.
    assert dataset.gemini_tags_by_image is dataset.reference_tags_by_image


def test_backend_capabilities_are_declared_once():
    """`sweep` and the per-backend commands must agree on what each architecture can do. When they were
    two lists, one of them was free to go stale — the same shape of bug as the copied dispatcher."""
    import argparse

    from vlm_eval import cli

    for backend, tasks in cli.BACKEND_TASKS.items():
        if backend == "server":
            continue
        for task in tasks:
            cli._check_task(backend, task)  # every declared task is accepted

    with pytest.raises(SystemExit) as e:
        cli._check_task("paligemma", "summary")  # and anything else is refused, with the reason
    assert "architectural limit" in str(e.value)

    # sweep walks exactly the declared tasks, in the declared order
    seen = []
    import vlm_eval.cli as c

    def fake(a):
        seen.append(a.task)

    original_florence, original_hf, original_metrics = c.cmd_florence, c.cmd_hf, c.cmd_metrics
    c.cmd_florence = fake
    c.cmd_hf = fake
    c.cmd_metrics = lambda a: None
    try:
        c.cmd_sweep(
            argparse.Namespace(
                model="florence",
                served_name=None,
                base_url=None,
                flavor=None,
                coords=None,
                tagging=5,
                captions=5,
                grounding=5,
                chunk_all=0,
                consistency=0,
                workers=1,
                no_logprobs=True,
                via="florence",
                checkpoint=None,
                device=None,
            )
        )
    finally:
        c.cmd_florence, c.cmd_hf, c.cmd_metrics = original_florence, original_hf, original_metrics
    assert seen == cli.BACKEND_TASKS["florence"]


def test_a_card_may_reference_a_priced_option_instead_of_copying_it(tmp_path):
    """Two copies of a price are two chances to disagree; a stale copy is how a cost figure ends up
    eight times off in a document somebody forwards."""
    from vlm_eval import report
    from vlm_eval.economics import Option

    options = [Option(name="L4 spot", kind="per_hour", price=0.58, throughput_per_hour=2000)]
    card = {"projection": {"option": "L4 spot", "note": "verify on real hardware"}}

    resolved = report.resolve_projection(card, options)
    assert resolved["usd_per_hour"] == 0.58 and resolved["images_per_hour"] == 2000
    assert "economics.json" in resolved["source"]  # says where the numbers came from
    assert resolved["note"] == "verify on real hardware"

    md = report.render_model({"name": "M", **card}, {}, options=options)
    assert "**Projected for deployment** — L4 spot" in md
    assert "0.29 USD" in md  # 0.58 / 2000 * 1000, both numbers from the same option

    # A name that does not exist must fail loudly, not silently drop the projection.
    with pytest.raises(SystemExit) as e:
        report.resolve_projection({"projection": {"option": "typo"}}, options)
    assert "not in the economics config" in str(e.value)

    # Inline numbers still work for a card used without any economics config.
    inline = {"projection": {"hardware": "T4", "images_per_hour": 500, "usd_per_hour": 0.5}}
    assert report.resolve_projection(inline, [])["hardware"] == "T4"


def test_a_truncated_answer_is_reported_not_read_as_a_refusal():
    """A reasoning model can spend the whole token budget thinking and return empty content. Parsed
    naively that looks like a model with no opinion; it is a model that never got to speak."""
    from vlm_eval.backends.base import Response
    from vlm_eval.runner import truncation_error

    finished = Response(text='{"pool": true}', latency_s=0.1, finish_reason="stop")
    assert truncation_error(finished, 3000) is None

    thought_itself_out = Response(text="", latency_s=0.1, finish_reason="length", reasoning_chars=4200)
    msg = truncation_error(thought_itself_out, 3000)
    assert "cut off at the 3000-token budget" in msg
    assert "spent it reasoning (4200 characters" in msg
    assert "returned nothing" in msg and "raise the budget" in msg

    cut_mid_answer = Response(text='{"pool": tr', latency_s=0.1, finish_reason="length")
    msg = cut_mid_answer and truncation_error(cut_mid_answer, 500)
    assert "cut off at the 500-token budget" in msg
    assert "returned nothing" not in msg  # it did say something, just not all of it


def test_the_backend_surfaces_finish_reason_and_reasoning(monkeypatch):
    """Both live in the response and were previously dropped, which is why the truncation was invisible."""
    import httpx

    from vlm_eval.backends.openai_compat import OpenAICompatBackend

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", "reasoning": "let me think about this"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 1000},
            },
        )

    be = OpenAICompatBackend("http://x/v1", "m", flavor="ollama", transport=httpx.MockTransport(handler))
    r = be.chat([], "prompt")
    assert r.finish_reason == "length"
    assert r.reasoning_chars == len("let me think about this")
    assert r.text == ""


def test_the_budget_reported_is_the_budget_sent(tmp_path):
    """A message that names a budget the request never carried sends you debugging the wrong machine.

    This exact mismatch happened: the error said "cut off at the 4000-token budget" while the request
    asked for 1000, and the search for the cause went to the inference server instead of here.
    """
    import json as _json

    import httpx
    from PIL import Image as _Image

    from vlm_eval import runner
    from vlm_eval.backends.openai_compat import OpenAICompatBackend

    sent = {}

    def handler(request):
        sent.update(_json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "", "reasoning": "x" * 50}, "finish_reason": "length"}],
                "usage": {"completion_tokens": sent["max_tokens"]},
            },
        )

    be = OpenAICompatBackend("http://x/v1", "m", flavor="ollama", transport=httpx.MockTransport(handler))
    _Image.new("RGB", (8, 8), "white").save(tmp_path / "a.jpg")

    row = runner.run_summary_one(
        be,
        {"property_job_id": "p", "image_ids": ["a"], "property_summary": "ref"},
        "prompt",
        tmp_path,
        3000,
    )
    assert sent["max_tokens"] == 4000  # base 1000 + the model's extra 3000
    assert f"{sent['max_tokens']}-token budget" in row["errors"][0]  # and the message says the same

    # The same must hold for tagging and captions, whose budgets are built the same way.
    import vlm_eval.dataset as _dataset

    original_images = _dataset.IMAGES
    _dataset.IMAGES = tmp_path
    cfg = runner.RunConfig(model="m", chunk_size=15, individual=[], extra_output_tokens=500)
    item = runner.Item("a", "u", "u", "indoor")
    tags = [{"slug": "pool", "question_text": "Pool?", "category": "common", "order": 0}]
    row = runner.run_tagging_one(be, item, tags, cfg)
    assert sent["max_tokens"] == 3500 and "3500-token budget" in row["errors"][0]

    row = runner.run_captions_one(be, item, {"base_caption": "short"}, cfg)
    assert f"{sent['max_tokens']}-token budget" in row["errors"][0]
    _dataset.IMAGES = original_images


def test_resume_refuses_to_mix_results_from_different_settings(tmp_path, capsys):
    """`(image_id, repeat)` says the work was done, not that it is still valid. Raise a token budget and
    every old row still counts as done — two experiments end up in one file and average into one number."""
    from vlm_eval import provenance

    run_file = tmp_path / "tagging_chunk15.jsonl"
    before = provenance.RunFingerprint(
        task="tagging",
        served_name="qwen3-vl:8b",
        chunk_size=15,
        extra_output_tokens=0,
        prompt_digest="aaa",
        model_identity="ollama:901c",
    )
    provenance.check(run_file, before)  # first run: records what produced it
    run_file.write_text('{"image_id": "a"}\n')
    recorded = provenance.load(run_file)
    assert recorded.fingerprint == before and recorded.status == provenance.VERIFIED

    provenance.check(run_file, before)  # same settings: resumes silently
    assert "" == capsys.readouterr().out

    after = provenance.RunFingerprint(
        task="tagging",
        served_name="qwen3-vl:8b",
        chunk_size=15,
        extra_output_tokens=3000,
        prompt_digest="aaa",
        model_identity="ollama:901c",
    )
    with pytest.raises(SystemExit) as e:
        provenance.check(run_file, after)
    message = str(e.value)
    assert "extra_output_tokens: was 0, now 3000" in message  # names what changed
    assert "mix two experiments" in message and "archive it" in message

    # A reworded question counts too — the answers mean something different.
    reworded = provenance.RunFingerprint(
        task="tagging",
        served_name="qwen3-vl:8b",
        chunk_size=15,
        extra_output_tokens=0,
        prompt_digest="bbb",
        model_identity="ollama:901c",
    )
    with pytest.raises(SystemExit) as e:
        provenance.check(run_file, reworded)
    assert "prompt_digest" in str(e.value)


def test_a_file_from_before_fingerprinting_says_so_rather_than_guessing(tmp_path, capsys):
    from vlm_eval import provenance

    run_file = tmp_path / "old.jsonl"
    run_file.write_text('{"image_id": "a"}\n')  # rows, no sidecar
    fp = provenance.RunFingerprint(task="tagging", served_name="m", chunk_size=15, model_identity="ollama:901c")
    provenance.check(run_file, fp)
    out = capsys.readouterr().out
    assert "before runs recorded their settings" in out and "legacy_unknown" in out

    # The current settings are recorded, but the file is NOT promoted to verified: rows written under
    # unknown settings must not become indistinguishable from checked ones.
    recorded = provenance.load(run_file)
    assert recorded.fingerprint == fp
    assert recorded.status == provenance.LEGACY and recorded.unverified_rows == 1

    # ...and the label sticks — a second, matching run does not launder it into a clean measurement.
    provenance.check(run_file, fp)
    assert "legacy_unknown" in capsys.readouterr().out
    assert provenance.load(run_file).status == provenance.LEGACY

    # It travels into the metrics, where a report is built from it.
    described = provenance.describe(run_file)
    assert described["status"] == provenance.LEGACY and "not verified" in described["note"]


def test_an_unfinished_answer_contributes_nothing(tmp_path):
    """Parsing what arrived would let a half-written JSON put real tags into accuracy and recall, and
    the tags it happened to reach are not a sample of anything."""
    from vlm_eval import metrics, runner
    from vlm_eval.backends.base import Response

    class Truncating:
        def chat(self, *a, **k):
            # valid JSON as far as it goes, but the model was cut off
            return Response(text='{"pool": true}', latency_s=0.1, finish_reason="length", reasoning_chars=900)

    from PIL import Image as _Image

    import vlm_eval.dataset as _dataset

    _Image.new("RGB", (8, 8), "white").save(tmp_path / "a.jpg")
    original, _dataset.IMAGES = _dataset.IMAGES, tmp_path
    try:
        cfg = runner.RunConfig(model="m", chunk_size=15, individual=[])
        tags = [{"slug": "pool", "question_text": "Pool?", "category": "common", "order": 0}]
        row = runner.run_tagging_one(Truncating(), runner.Item("a", "u", "u", "indoor"), tags, cfg)
    finally:
        _dataset.IMAGES = original

    assert row["answers"] == {"pool": None}  # not True, even though the text parsed
    assert row["errors"] and "cut off" in row["errors"][0]

    # ...and the loss is reported as its own number, not left inside an error string.
    t = metrics.truncation([row])
    assert t["images_affected"] == 1 and t["pct"] == 100.0


def test_truncation_is_read_from_a_field_not_from_the_wording_of_an_error():
    """The metric used to scan error text for "cut off at the". Reword the message and truncation
    silently reads as zero — the one number whose job is to say the run measured less than it looks."""
    from vlm_eval import metrics

    reworded = {
        "image_id": "a",
        "completion": {"calls": 3, "truncated": 1, "status": "truncated"},
        "errors": ["ран out of room"],
    }
    finished = {"image_id": "b", "completion": {"calls": 3, "truncated": 0, "status": "ok"}, "errors": []}
    t = metrics.truncation([reworded, finished])
    assert t["images_affected"] == 1 and t["pct"] == 50.0
    assert t["rows_without_record"] == 0

    # Rows written before the record existed still count, from the only evidence they carry — and say so.
    legacy = {"image_id": "c", "errors": ["kitchen_island: answer cut off at the 512-token budget"]}
    t = metrics.truncation([legacy])
    assert t["images_affected"] == 1 and t["rows_without_record"] == 1
    assert "predate the completion record" in t["note"]


def test_every_command_that_writes_run_rows_passes_the_provenance_gate():
    """Florence and PaliGemma wrote their rows straight to disk, so the guard was not a project-wide
    guarantee, only a property of one code path. This is the seam that keeps drifting: two things that
    must agree, written in different places."""
    import ast
    import inspect

    from vlm_eval import cli

    tree = ast.parse(inspect.getsource(cli))
    unguarded = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        calls = {
            f"{ast.unparse(n.func)}"
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        if "runner.run_over_items" in calls and "provenance.check" not in calls:
            unguarded.append(fn.name)
    assert not unguarded, f"writes run rows without a provenance check: {unguarded}"


def test_completion_status_distinguishes_failed_from_complete():
    """An exception leaves no answer just like truncation does; a record that says "ok" beside a
    non-empty errors list is a small lie a summary table will repeat."""
    from vlm_eval import runner

    assert runner.completion_record(3, 0)["status"] == "complete"
    assert runner.completion_record(3, 1)["status"] == "truncated"
    assert runner.completion_record(3, 0, failed=2)["status"] == "failed"
    assert runner.completion_record(0, 0)["status"] == "not_called"


def test_replacing_an_image_under_the_same_id_changes_the_fingerprint(tmp_path, monkeypatch):
    """Settings say how images were supposed to be prepared; the digest says what they are. A swapped
    file under an unchanged id is invisible to settings alone."""
    from PIL import Image as _Image

    from vlm_eval import dataset

    (tmp_path / "images").mkdir()
    _Image.new("RGB", (8, 8), "white").save(tmp_path / "images" / "a.jpg")
    monkeypatch.setattr(dataset, "IMAGES", tmp_path / "images")
    items = [dataset.Item("a", "u", "u", "indoor")]

    before = dataset.images_digest(items)
    _Image.new("RGB", (8, 8), "black").save(tmp_path / "images" / "a.jpg")
    assert dataset.images_digest(items) != before

    # A file that vanished is a different dataset too, not a silent shrink.
    (tmp_path / "images" / "a.jpg").unlink()
    assert dataset.images_digest(items) not in (before, dataset.images_digest([]))


def test_each_task_fingerprints_the_images_it_actually_reads(workspace):
    """48 real listing images exist only in properties.jsonl, not the manifest. A manifest-wide digest
    would never notice one being swapped — the exact images a summary is about would be the only ones
    nobody was watching. And the converse: swapping an image a task never touches must not block it."""
    import vlm_eval.cli as cli
    from vlm_eval import dataset
    from vlm_eval.runner import RunConfig

    images = workspace["data"] / "images"
    Image.new("RGB", (32, 32), "blue").save(images / "prop-only.jpg")  # summary-only, not in the manifest
    (workspace["data"] / "properties.jsonl").write_text(
        json.dumps({"property_job_id": "p1", "image_ids": ["img-a", "prop-only"], "property_summary": "x"}) + "\n"
    )

    cfg = RunConfig(model="m", chunk_size=15, individual=[])
    manifest_items = [it for it in dataset.load_manifest() if it.path.exists()]

    def prints():
        tagging_fp = cli._fingerprint("tagging", cfg, {}, "m", {"q": 1}, images=manifest_items)
        summary_fp = cli._fingerprint("summary", cfg, {}, "m", {"p": 1}, images=cli._summary_items())
        return tagging_fp.digest(), summary_fp.digest()

    tag0, sum0 = prints()

    # 1+2. Replacing a manifest image changes tagging; the summary set contains img-a too, so both move.
    Image.new("RGB", (32, 32), "red").save(images / "img-a.jpg")
    tag1, sum1 = prints()
    assert tag1 != tag0 and sum1 != sum0

    # 3. Replacing the property-only image changes summary and leaves tagging alone.
    Image.new("RGB", (32, 32), "green").save(images / "prop-only.jpg")
    tag2, sum2 = prints()
    assert tag2 == tag1 and sum2 != sum1

    # 4. A property image that vanished is a different summary dataset, not a silent shrink.
    (images / "prop-only.jpg").unlink()
    tag3, sum3 = prints()
    assert tag3 == tag2 and sum3 not in (sum1, sum2)


def test_fingerprint_sees_question_selection_not_just_question_texts(workspace):
    """`digest_of` sorts dict keys, so a bare {slug: text} map missed everything structural: which
    category a tag sits in, its position in the chunk order, and which images count as indoor. All
    three change the requests without changing a single question text."""
    import vlm_eval.cli as cli
    from vlm_eval import dataset
    from vlm_eval.runner import RunConfig

    cfg = RunConfig(model="m", chunk_size=15, individual=[])
    items = [it for it in dataset.load_manifest() if it.path.exists()]
    tags = [dict(t) for t in TAGS]

    def print_of(tags_, items_):
        return cli._fingerprint("tagging", cfg, {}, "m", cli._tagging_identity(tags_, items_)).digest()

    base = print_of(tags, items)

    moved = [dict(t) for t in tags]
    moved[0]["category"] = "outdoor" if moved[0]["category"] != "outdoor" else "indoor"
    assert print_of(moved, items) != base  # a recategorised tag is asked of different images

    reordered = [dict(t) for t in tags]
    reordered[0]["order"] = 99
    assert print_of(reordered, items) != base  # chunk composition follows `order`

    retyped = [replace(items[0], image_type="outdoor" if items[0].image_type != "outdoor" else "indoor")] + items[1:]
    assert print_of(tags, retyped) != base  # same pixels, different question set


def test_fingerprint_sees_listing_grouping_and_order(workspace):
    """The byte digest cannot tell a regrouping or a reshuffle that reuses the same files — but the
    model is shown the images per listing, in order, so both are different experiments."""
    import vlm_eval.cli as cli
    from vlm_eval.runner import RunConfig

    cfg = RunConfig(model="m", chunk_size=15, individual=[])
    props_file = workspace["data"] / "properties.jsonl"

    def print_of(props):
        props_file.write_text("".join(json.dumps(p) + "\n" for p in props))
        listings = [[str(p["property_job_id"]), [str(i) for i in p["image_ids"]]] for p in props]
        return cli._fingerprint(
            "summary", cfg, {}, "m", {"rendered": "x", "listings": listings}, images=cli._summary_items()
        ).digest()

    base = print_of([{"property_job_id": "p1", "image_ids": ["img-a", "img-b"], "property_summary": "s"}])
    shuffled = print_of([{"property_job_id": "p1", "image_ids": ["img-b", "img-a"], "property_summary": "s"}])
    regrouped = print_of(
        [
            {"property_job_id": "p1", "image_ids": ["img-a"], "property_summary": "s"},
            {"property_job_id": "p2", "image_ids": ["img-b"], "property_summary": "s"},
        ]
    )
    assert len({base, shuffled, regrouped}) == 3  # same bytes on disk in all three


def test_a_missing_image_among_the_first_twenty_fails_rather_than_sliding(tmp_path):
    """Filtering before capping let image 21 fill a hole among the first 20: the count came out right,
    nothing failed, and the model saw a different listing than production would send."""
    from PIL import Image as _Image

    from vlm_eval import runner
    from vlm_eval.tasks import summary

    ids = [f"i{n:02d}" for n in range(summary.MAX_IMAGES + 1)]  # 21 ids, i05 missing from disk
    for i in ids:
        if i != "i05":
            _Image.new("RGB", (8, 8), "white").save(tmp_path / f"{i}.jpg")

    class Explode:
        def chat(self, *a, **k):
            raise AssertionError("must not reach the model")

    row = runner.run_summary_one(Explode(), {"property_job_id": "p", "image_ids": ids}, "prompt", tmp_path)
    assert row["summary"] is None and row["errors"]  # failed, did not quietly borrow image 21
    assert row["n_images"] == summary.MAX_IMAGES - 1 and row["n_expected"] == summary.MAX_IMAGES
    assert row["image_ids"] == ids[: summary.MAX_IMAGES]


def test_resume_with_identical_settings_appends_nothing_and_keeps_the_sidecar(workspace):
    """The positive half of the guarantee: same code, same route, same settings — the second run is a
    no-op that leaves both the rows and the recorded provenance byte-identical."""
    import vlm_eval.cli as cli
    from vlm_eval import provenance, runner

    be = StubBackend({"pool"})
    be.weights_digest = "stub:deadbeef"
    cfg = runner.RunConfig(model="stub", chunk_size=15, individual=[])
    cli.run_task(be, task="tagging", model="stub", cfg=cfg)

    out = runner.tagging_out("stub", 15)
    rows_before = out.read_text()
    meta_before = provenance.sidecar(out).read_text()
    assert provenance.load(out).status == provenance.VERIFIED

    cli.run_task(be, task="tagging", model="stub", cfg=cfg)  # must resume, not refuse and not redo
    assert out.read_text() == rows_before
    assert provenance.sidecar(out).read_text() == meta_before


def test_resume_refuses_a_different_server_route_or_implementation(tmp_path):
    """Two servers answering to one served name, or an edited parser, are different experiments even
    when every setting the CLI knows about is identical."""
    from vlm_eval import provenance

    run_file = tmp_path / "tagging_chunk15.jsonl"
    base = dict(task="tagging", served_name="qwen3", chunk_size=15, prompt_digest="x", model_identity="ollama:901c")
    provenance.check(run_file, provenance.RunFingerprint(**base, route="ollama@http://127.0.0.1:11434/v1", code="aaa"))
    run_file.write_text('{"image_id": "a"}\n')

    moved = provenance.RunFingerprint(**base, route="vllm@http://10.0.0.5:8000/v1", code="aaa")
    with pytest.raises(SystemExit) as e:
        provenance.check(run_file, moved)
    assert "route" in str(e.value) and "vllm@http://10.0.0.5:8000/v1" in str(e.value)

    edited = provenance.RunFingerprint(**base, route="ollama@http://127.0.0.1:11434/v1", code="bbb")
    with pytest.raises(SystemExit) as e:
        provenance.check(run_file, edited)
    assert "code" in str(e.value)


def test_code_identity_tracks_the_answer_producing_source(tmp_path):
    """Editing the code that builds requests or parses answers changes the identity; the digest is of
    file bytes, so it holds across processes rather than only within one."""
    import types

    from vlm_eval import provenance

    f = tmp_path / "mod.py"
    f.write_text("def parse(x): return x\n")
    mod = types.SimpleNamespace(__name__="mod", __file__=str(f))
    before = provenance.code_identity([mod])
    assert before == provenance.code_identity([mod])  # stable while the file is unchanged

    f.write_text("def parse(x): return x.lower()\n")
    assert provenance.code_identity([mod]) != before


def test_same_name_same_route_different_weights_refuses_resume(tmp_path):
    """`qwen3-vl:8b` is a tag somebody can re-point: pull an update and the same name on the same
    server answers with a different model. The digest of the weights is the identity; the name is not."""
    from vlm_eval import provenance

    run_file = tmp_path / "tagging_chunk15.jsonl"
    base = dict(task="tagging", served_name="qwen3-vl:8b", chunk_size=15, prompt_digest="x", route="ollama@http://o/v1")
    provenance.check(run_file, provenance.RunFingerprint(**base, model_identity="ollama:901cae"))
    run_file.write_text('{"image_id": "a"}\n')

    with pytest.raises(SystemExit) as e:
        provenance.check(run_file, provenance.RunFingerprint(**base, model_identity="ollama:f00d42"))
    assert "model_identity" in str(e.value) and "ollama:f00d42" in str(e.value)


def test_unprovable_weights_refuse_resume_but_allow_a_fresh_run(tmp_path):
    """A backend that cannot prove its weights still gets to run — but never to resume rows it cannot
    vouch for. Resume's one forbidden move is assuming the model stayed the same because its name did."""
    from vlm_eval import provenance

    run_file = tmp_path / "captions.jsonl"
    fp = provenance.RunFingerprint(
        task="captions",
        served_name="m",
        chunk_size=15,
        model_identity="unknown: backend does not report a weights digest",
    )
    provenance.check(run_file, fp)  # fresh file: allowed, recorded honestly

    provenance.check(run_file, fp)  # still empty: nothing to vouch for, still allowed

    run_file.write_text('{"image_id": "a"}\n')
    with pytest.raises(SystemExit) as e:
        provenance.check(run_file, fp)
    msg = str(e.value)
    assert "cannot prove the model weights are unchanged" in msg and "mutable tag" in msg
