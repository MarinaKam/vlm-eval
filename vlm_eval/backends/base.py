from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Response:
    text: str
    latency_s: float
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens / completion_tokens
    logprobs: list[tuple[str, float, dict[str, float]]] | None = None
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
