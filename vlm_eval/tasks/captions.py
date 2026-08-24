"""Caption generation — copy of the production generate_free_text_prompts call.

One call per image, all caption prompts at once, JSON object with one string per prompt key.
Gemini config: temperature 0.2, max_output_tokens 8000.
"""

import json
from typing import Any

from .tagging import _loads_lenient

# The first line is whatever the production prompt opens with — a domain instruction that belongs to
# the pipeline being replayed, not to this tool. It travels in the export as
# `prompt_templates["caption_header"]`; the fallback below is deliberately domain-free.
FALLBACK_HEADER = "You are an assistant that describes images."
INSTRUCTION = (
    "Based on the provided image(s), generate a JSON object with the following keys."
    " Follow the specific instructions for each key:"
)


def header(templates: dict[str, str] | None = None) -> list[str]:
    opening = (templates or {}).get("caption_header") or FALLBACK_HEADER
    return [opening, INSTRUCTION]


TEMPERATURE = 0.2
MAX_TOKENS = 2000  # Gemini uses 8000; captions are short, keep local runs bounded

# Only a starting point for a dataset with no caption prompts exported; the real texts live in
# data/prompts.json.
DEFAULT_PROMPTS = {
    "base_caption": "A short, high-level description of the image.",
    "detailed_caption": "A detailed description including intricate details of the image.",
}


def prompt_text(prompts: dict[str, str], templates: dict[str, str] | None = None) -> str:
    return "\n".join(header(templates) + [f"{key}: {instruction}" for key, instruction in prompts.items()])


def schema(prompts: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in prompts},
        "required": list(prompts),
        "additionalProperties": False,
    }


def parse(text: str, prompts: dict[str, str]) -> dict[str, str | None]:
    obj = _loads_lenient(text or "") or {}
    return {k: (str(obj[k]).strip() if obj.get(k) is not None else None) for k in prompts}


def word_count(text: str | None) -> int:
    return len((text or "").split())


def as_json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)
