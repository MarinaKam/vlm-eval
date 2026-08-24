"""Image-level tagging — a faithful copy of the production Gemini classification call.

Reference implementation (the source app's gemini/classification.py):
  * questions = {slug: question_text} for common + (indoor|outdoor) tags, ordered (category, order, slug)
  * chunks of `classification_chunk_size` (15); `individual_questions` are pulled into single-question calls
  * the *prompt* is literally json.dumps(questions) next to the image — no instruction text
  * response_schema: object with a required boolean per slug; temperature 0
  * confidence: fake (random 0.7-0.9) — here we instead read P(true) from token logprobs when available
"""

import json
import math
import re
from collections.abc import Iterable
from typing import Any

CATEGORY_ORDER = {"common": 0, "indoor": 1, "outdoor": 2}


def questions_for(image_type: str, tags: list[dict]) -> dict[str, str]:
    """Mirror the production question selection: common + one category, ordered."""
    wanted = {"common", "indoor" if image_type == "indoor" else "outdoor"}
    rows = [t for t in tags if t["category"] in wanted]
    rows.sort(key=lambda t: (CATEGORY_ORDER[t["category"]], t.get("order", 0), t["slug"]))
    return {t["slug"]: t["question_text"] for t in rows}


def chunk_questions(questions: dict[str, str], chunk_size: int, individual: Iterable[str]) -> list[dict[str, str]]:
    """Mirror generate_classification_tags chunking. chunk_size<=0 => everything in one call.

    Note: the production code keeps a chunk that became empty after popping an individual question (and would
    issue an empty call); we drop empty chunks.
    """
    keys = list(questions)
    size = len(keys) if chunk_size <= 0 else chunk_size
    chunks = [{k: questions[k] for k in keys[i : i + size]} for i in range(0, len(keys), size)]
    singles = []
    for chunk in chunks:
        singles.extend({slug: chunk.pop(slug)} for slug in individual if slug in chunk)
    return [c for c in chunks if c] + singles


def boolean_schema(questions: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {slug: {"type": "boolean"} for slug in questions},
        "required": list(questions),
        "additionalProperties": False,
    }


def prompt_text(questions: dict[str, str]) -> str:
    return json.dumps(questions)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _loads_lenient(text: str) -> dict | None:
    for candidate in (text, *(m.strip() for m in _FENCE.findall(text))):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


def parse_answers(text: str, slugs: Iterable[str]) -> dict[str, bool | None]:
    obj = _loads_lenient(text or "") or {}
    return {slug: _to_bool(obj.get(slug)) for slug in slugs}


def confidence_from_logprobs(
    tokens: list[tuple[str, float, dict[str, float]]], slugs: Iterable[str]
) -> dict[str, float]:
    """P(true) per slug from a token stream [(token_text, logprob, {top_token: logprob})].

    Walk the emitted text; right after `"<slug>":` the next non-whitespace token is the boolean value.
    P(true) = exp(logprob of the 'true' alternative): the chosen token's logprob if it is 'true', else the
    'true' entry from top_logprobs (0.0 if absent).
    """
    text, positions = "", []
    for tok, lp, top in tokens:
        positions.append((len(text), tok, lp, top))
        text += tok
    out: dict[str, float] = {}
    for slug in slugs:
        m = re.search(r'"' + re.escape(slug) + r'"\s*:\s*', text)
        if not m:
            continue
        for start, tok, lp, top in positions:
            if start + len(tok) <= m.end():
                continue
            if not tok.strip():
                continue
            first = tok.strip().lower()
            if first.startswith("true"):
                out[slug] = math.exp(lp)
            elif first.startswith("false"):
                true_lp = [v for k, v in (top or {}).items() if k.strip().lower().startswith("true")]
                out[slug] = math.exp(max(true_lp)) if true_lp else 0.0
            break
    return out
