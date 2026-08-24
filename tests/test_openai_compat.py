import json

import httpx

from vlm_eval.backends.openai_compat import OpenAICompatBackend


def _server(captured: dict, content: str, logprobs=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["path"] = request.url.path
        choice = {"message": {"role": "assistant", "content": content}}
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
        return httpx.Response(200, json={
            "choices": [choice],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240},
        })
    return httpx.MockTransport(handler)


def test_body_has_images_schema_and_guided_json_for_vllm():
    cap = {}
    be = OpenAICompatBackend("http://x/v1", "qwen", flavor="vllm", transport=_server(cap, '{"a": true}'))
    r = be.chat([b"\xff\xd8img"], '{"a": "q"}', json_schema={"type": "object"}, logprobs=True)
    body = cap["body"]
    assert cap["path"] == "/v1/chat/completions"
    assert body["model"] == "qwen" and body["temperature"] == 0.0
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[-1] == {"type": "text", "text": '{"a": "q"}'}
    assert body["response_format"]["type"] == "json_schema"
    assert body["guided_json"] == {"type": "object"}
    assert body["logprobs"] is True and body["top_logprobs"] == 5
    assert r.text == '{"a": true}'
    assert r.usage["prompt_tokens"] == 1200
    assert r.latency_s >= 0


def test_ollama_flavor_has_no_guided_json_and_parses_logprobs():
    cap = {}
    top = [{"token": " true", "logprob": -0.1}, {"token": " false", "logprob": -2.5}]
    lp = [{"token": " true", "logprob": -0.1, "top_logprobs": top}]
    be = OpenAICompatBackend("http://x/v1", "m", flavor="ollama", transport=_server(cap, "true", lp))
    r = be.chat([], "p", json_schema={"type": "object"})
    assert "guided_json" not in cap["body"]
    assert r.logprobs == [(" true", -0.1, {" true": -0.1, " false": -2.5})]
