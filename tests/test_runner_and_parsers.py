import json

from vlm_eval import runner
from vlm_eval.backends.florence_hf import tag_phrase
from vlm_eval.dataset import Item
from vlm_eval.tasks import captions, grounding, summary


def test_run_over_items_resumes_and_repeats(tmp_path):
    out = tmp_path / "t.jsonl"
    items = [Item(1, "u", "s", "indoor"), Item(2, "u", "s", "outdoor")]
    calls = []

    def fn(it):
        calls.append(it.image_id)
        return {"image_id": it.image_id, "answers": {}}

    n = runner.run_over_items(items, fn, out, repeats=2, workers=2, log=lambda s: None)
    assert n == 4 and sorted(calls) == [1, 1, 2, 2]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {(r["image_id"], r["repeat"]) for r in rows} == {(1, 0), (2, 0), (1, 1), (2, 1)}
    calls.clear()
    assert runner.run_over_items(items, fn, out, repeats=2, workers=1, log=lambda s: None) == 0
    assert calls == []


def test_caption_prompt_and_parse():
    p = {"base_caption": "short", "detailed_caption": "long"}
    txt = captions.prompt_text(p)
    assert txt.startswith("You are a helpful SEO expert in the real estate area.")
    assert txt.endswith("base_caption: short\ndetailed_caption: long")
    assert captions.schema(p)["required"] == ["base_caption", "detailed_caption"]
    parsed = captions.parse('{"base_caption": " a ", "detailed_caption": null}', p)
    assert parsed == {"base_caption": "a", "detailed_caption": None}


def test_summary_parse_and_normalize():
    assert summary.parse('{"property_summary": "Too short."}') is None
    long = "A bright home with **bold** light. The garden is lovely and green. Perfect for families."
    out = summary.parse(json.dumps({"property_summary": long}))
    assert out and "**" not in out
    assert summary.SCHEMA["required"] == ["property_summary"]


def test_grounding_parse_norm1000_and_abs():
    txt = '{"detections": [{"label": "fireplace", "bbox_2d": [100, 200, 500, 900]}]}'
    n = grounding.parse(txt, coords="norm1000", width=1000, height=1000)
    assert n == [{"label": "fireplace", "bbox": [0.1, 0.2, 0.5, 0.9]}]
    a = grounding.parse(txt, coords="abs", width=1000, height=1000)
    assert a == n
    assert grounding.parse("junk", coords="abs", width=10, height=10) == []


def test_tag_phrase_strips_parentheses():
    assert tag_phrase({"slug": "tile_floor", "name": "Tile Floor (Indoor)"}) == "tile floor"
    assert tag_phrase({"slug": "kitchen_island"}) == "kitchen island"
