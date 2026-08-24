import json

from vlm_eval.tasks import tagging

TAGS = [
    {"slug": "kitchen", "question_text": "Does this image depict a kitchen?", "category": "indoor", "order": 0},
    {"slug": "garden", "question_text": "Does this image depict a garden?", "category": "outdoor", "order": 0},
    {"slug": "pool", "question_text": "Does this image contain a swimming pool?", "category": "common", "order": 0},
    {"slug": "utility_room", "question_text": "Utility room?", "category": "indoor", "order": 1},
]


def test_questions_for_indoor_is_common_plus_indoor_in_gemini_order():
    q = tagging.questions_for("indoor", TAGS)
    assert list(q) == ["pool", "kitchen", "utility_room"]  # category, order, slug
    assert q["kitchen"] == "Does this image depict a kitchen?"


def test_questions_for_outdoor():
    assert list(tagging.questions_for("outdoor", TAGS)) == ["pool", "garden"]


def test_chunking_matches_gemini_split_and_individual_questions():
    q = tagging.questions_for("indoor", TAGS)
    chunks = tagging.chunk_questions(q, chunk_size=2, individual=["utility_room"])
    # split_dict -> [{pool, kitchen}, {utility_room}] then utility_room is popped out of its chunk
    # and appended as its own single-question call; the now-empty chunk is dropped.
    assert chunks == [{"pool": q["pool"], "kitchen": q["kitchen"]}, {"utility_room": q["utility_room"]}]


def test_chunk_all_in_one():
    q = tagging.questions_for("indoor", TAGS)
    assert tagging.chunk_questions(q, chunk_size=0, individual=[]) == [q]


def test_boolean_schema_has_required_boolean_per_slug():
    schema = tagging.boolean_schema({"a": "q1", "b": "q2"})
    assert schema["type"] == "object"
    assert schema["required"] == ["a", "b"]
    assert schema["properties"]["a"] == {"type": "boolean"}
    assert schema["additionalProperties"] is False


def test_prompt_text_is_the_raw_json_like_gemini():
    q = {"a": "q1"}
    assert tagging.prompt_text(q) == json.dumps(q)


def test_parse_answers_tolerates_strings_and_missing_keys():
    out = tagging.parse_answers('{"a": true, "b": "false", "c": 1}', ["a", "b", "c", "d"])
    assert out == {"a": True, "b": False, "c": True, "d": None}


def test_parse_answers_salvages_fenced_json():
    out = tagging.parse_answers('```json\n{"a": false}\n```', ["a"])
    assert out == {"a": False}


def test_parse_answers_garbage_returns_none_per_slug():
    assert tagging.parse_answers("nope", ["a"]) == {"a": None}


def test_confidence_from_logprobs_reads_prob_at_value_token():
    # token stream for '{"a": true, "b": false}'
    toks = [
        ('{"', -0.01, {}), ("a", -0.01, {}), ('":', -0.01, {}), (" true", -0.105, {" true": -0.105, " false": -2.3}),
        (',"', -0.01, {}), ("b", -0.01, {}), ('":', -0.01, {}), (" false", -0.02, {" false": -0.02, " true": -4.0}),
        ("}", -0.01, {}),
    ]
    conf = tagging.confidence_from_logprobs(toks, ["a", "b"])
    assert round(conf["a"], 2) == 0.90     # P(true) = exp(-0.105)
    assert round(conf["b"], 2) == 0.02     # P(true) = exp(-4.0) from top_logprobs
