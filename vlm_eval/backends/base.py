from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Response:
    text: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens / completion_tokens
    logprobs: list[tuple[str, float, dict[str, float]]] | None = None
    # "stop" when the model finished, "length" when it ran out of budget mid-answer. A truncated
    # answer often arrives as an *empty* string rather than a broken one, which is indistinguishable
    # from a model that had nothing to say unless this is checked.
    finish_reason: str | None = None
    # Reasoning models (Qwen3-VL and friends) emit chain-of-thought into a separate field and can
    # spend the whole token budget there, returning empty content. Recording its length makes that
    # diagnosable instead of mysterious.
    reasoning_chars: int = 0
    raw: Any = None


class Backend(Protocol):
    name: str

    def chat(
        self,
        images: list[bytes],
        prompt: str,
        *,
        json_schema: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        logprobs: bool = False,
    ) -> Response: ...
