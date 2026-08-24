import json

from vlm_eval import review

GEM = {1: {"tags": {"pool": 0.8}, "evaluable_slugs": ["pool", "garden"]}}
TAGS = [{"slug": "pool", "question_text": "Pool?"}, {"slug": "garden", "question_text": "Garden?"}]


def test_disagreements_and_sampling():
    rows = [{"image_id": 1, "repeat": 0, "answers": {"pool": False, "garden": True}, "confidence": {"pool": 0.2}}]
    d = review.disagreements(rows, GEM, TAGS)
    assert {(c["slug"], c["gemini"], c["model"]) for c in d} == {("pool", True, False), ("garden", False, True)}
    assert len(review.sample_by_tag(d * 3, per_tag=1)) == 2


def test_html_decisions_roundtrip(tmp_path):
    cases = [{"image_id": 1, "slug": "pool", "question": "Pool?", "gemini": True, "model": False, "model_conf": 0.2}]
    out = review.build_review_html("m", cases, tmp_path / "r.html")
    assert "Download decisions" in out.read_text()
    dec = tmp_path / "d.json"
    dec.write_text(
        json.dumps([{"image_id": 1, "slug": "pool", "truth": True}, {"image_id": 1, "slug": "garden", "truth": None}])
    )
    store = tmp_path / "labels.json"
    assert review.apply_decisions([dec], store) == {"added": 1, "total": 1}
    rows = [{"image_id": 1, "repeat": 0, "answers": {"pool": False}}]
    m = review.manual_agreement(rows, GEM, store)
    assert m == {"n": 1, "model_correct_pct": 0.0, "gemini_correct_pct": 100.0}
