"""Property-level summary from multiple images — copy of the production multi-image summary call:
images first, then the prompt; JSON {"property_summary": str};
temperature 0.2; max_output_tokens 1000; MAX_IMAGES_PER_PROPERTY = 20; images resized to 1536px / q85.
"""
import re
from typing import Any

from .tagging import _loads_lenient

MAX_IMAGES = 20
TEMPERATURE = 0.2
MAX_TOKENS = 1000

DEFAULT_PROMPT = """
You are a professional property copywriter for a high-end real estate agency.
Your task is to analyze ALL the provided property images together and write a
**fluid, emotionally engaging, and elegant** property description.

You are viewing multiple images of the same property. Consider them as a complete
visual tour of the home - from exterior views to interior spaces.

The tone should be warm, aspirational, and sophisticated — similar to luxury
real estate listings in publications like *The Times Property*, *Mansion Global*, or *Savills*.

**IMPORTANT: Keep your response between 150-250 words. Do not exceed 250 words.**

Structure the description as a flowing narrative:
- Begin with an opening paragraph capturing the essence of the property and its setting.
- Then describe the interior flow through the home as visible in the images.
- Weave in lifestyle elements, atmosphere, and amenities.
- End with a closing sentence reinforcing why this property is exceptional.

Use graceful, evocative language — show, don't just tell.
Avoid bullet points or mechanical listing of rooms. Your response must be a
**single, well-written prose block**.

Only describe features that are clearly visible in the images.
Do not assume room counts, square footage, or amenities unless they are visually evident.

Analyze all images holistically to create a cohesive property narrative.

Return your response strictly as valid JSON in the following format:

{
  "property_summary": "<string>"
}

Do not include any explanations, markdown, or text outside of this JSON object.
""".strip()

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"property_summary": {"type": "string"}},
    "required": ["property_summary"],
    "additionalProperties": False,
}


def normalize(text: str | None) -> str | None:
    """Copy of prompts._normalize_summary: whitespace, markdown, AI disclaimers, min 50 chars / 2 sentences."""
    if not text:
        return None
    cleaned = " ".join(text.split()).replace("**", "").replace("__", "")
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_([^_]+)_", r"\1", cleaned)
    cleaned = re.sub(r"As an AI.*?\.", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"I cannot.*?\.", "", cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) < 50 or len(re.findall(r"[.!?]+", cleaned)) < 2:
        return None
    return cleaned


def parse(text: str) -> str | None:
    obj = _loads_lenient(text or "")
    if isinstance(obj, dict) and "property_summary" in obj:
        return normalize(str(obj["property_summary"]))
    # Gemini fallback path: raw prose
    if text and len(text) > 100 and not text.lstrip().startswith("{"):
        return normalize(text[:2000])
    return None
