"""Detection / grounding probe: can the VLM localise a property feature (kitchen island, fireplace, radiator)?

Generative VLMs (Qwen2.5-VL / Qwen3-VL / InternVL) are prompted for JSON bounding boxes. Coordinate
conventions differ per model family and are passed in as `coords`:
  * "abs"      — absolute pixels of the input image (Qwen2.5-VL)
  * "norm1000" — 0..1000 normalised (Qwen3-VL, InternVL)
Florence-2 uses its own task tokens (see backends/florence_hf.py) and returns absolute pixels.
Output is normalised to [x1, y1, x2, y2] in 0..1 so boxes are comparable across models.
"""

from typing import Any

from .tagging import _loads_lenient

TARGETS = {
    "kitchen_island": "kitchen island (a freestanding or built-in island counter)",
    "fireplace": "fireplace",
    "radiator": "radiator (wall-mounted heating radiator)",
}


def prompt_text(label: str, description: str) -> str:
    return (
        f"Locate every {description} in the image. Output JSON only: "
        '{"detections": [{"label": "' + label + '", "bbox_2d": [x1, y1, x2, y2]}]}. '
        'If there is none, output {"detections": []}.'
    )


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["label", "bbox_2d"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["detections"],
    "additionalProperties": False,
}


def parse(text: str, *, coords: str, width: int, height: int) -> list[dict[str, Any]]:
    obj = _loads_lenient(text or "") or {}
    dets = obj.get("detections") or []
    out = []
    for d in dets:
        box = d.get("bbox_2d") if isinstance(d, dict) else None
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if coords == "norm1000":
            x1, x2, y1, y2 = x1 / 1000, x2 / 1000, y1 / 1000, y2 / 1000
        elif coords == "abs":
            x1, x2, y1, y2 = x1 / width, x2 / width, y1 / height, y2 / height
        bbox = [round(min(max(v, 0.0), 1.0), 4) for v in (x1, y1, x2, y2)]
        out.append({"label": str(d.get("label", "")), "bbox": bbox})
    return out
