"""OpenAI-compatible chat backend: vLLM (`vllm serve`) and Ollama (`/v1`).

Structured output: `response_format={"type":"json_schema",...}` (vLLM + Ollama>=0.5 honour it). For vLLM we
also pass `guided_json` in the body (older servers ignore `response_format` json_schema but accept
`guided_json`). Token logprobs: `logprobs=true, top_logprobs=5` (vLLM yes; Ollama ignores -> None).
"""

import base64
import time
from typing import Any

import httpx

from .base import Response


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        flavor: str = "vllm",
        timeout: float = 600.0,
        api_key: str = "EMPTY",
        transport: httpx.BaseTransport | None = None,
    ):
        self.name = model
        self.model = model
        self.flavor = flavor
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @staticmethod
    def _image_part(data: bytes) -> dict[str, Any]:
        b64 = base64.b64encode(data).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    def build_body(
        self,
        images: list[bytes],
        prompt: str,
        *,
        json_schema: dict | None,
        max_tokens: int,
        temperature: float,
        logprobs: bool,
    ) -> dict[str, Any]:
        content = [self._image_part(img) for img in images] + [{"type": "text", "text": prompt}]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": json_schema, "strict": True},
            }
            if self.flavor == "vllm":
                body["guided_json"] = json_schema
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        return body

    @staticmethod
    def _parse_logprobs(choice: dict) -> list[tuple[str, float, dict[str, float]]] | None:
        lp = (choice.get("logprobs") or {}).get("content")
        if not lp:
            return None
        return [
            (
                t.get("token", ""),
                float(t.get("logprob", 0.0)),
                {x.get("token", ""): float(x.get("logprob", 0.0)) for x in (t.get("top_logprobs") or [])},
            )
            for t in lp
        ]

    def chat(
        self,
        images: list[bytes],
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> Response:
        body = self.build_body(
            images, prompt, json_schema=json_schema, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs
        )
        t0 = time.perf_counter()
        resp = self._client.post("/chat/completions", json=body)
        latency = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        message = choice.get("message") or {}
        # Some servers put chain-of-thought in `reasoning` (Ollama) or `reasoning_content` (others).
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        return Response(
            text=message.get("content") or "",
            finish_reason=choice.get("finish_reason"),
            reasoning_chars=len(reasoning),
            latency_s=latency,
            usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
            logprobs=self._parse_logprobs(choice),
            raw=data,
        )

    def health(self) -> bool:
        try:
            return self._client.get("/models").status_code == 200
        except httpx.HTTPError:
            return False
