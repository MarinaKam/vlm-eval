"""Florence-2 (microsoft/Florence-2-large, MIT) via transformers — runs on the GPU box (CUDA) or Mac (MPS/CPU).

Florence-2 is a seq2seq vision model with fixed task tokens, not an instruction-following chat VLM:
  <CAPTION> <DETAILED_CAPTION> <MORE_DETAILED_CAPTION>   captions (3 fixed levels of detail)
  <OD>                                                   closed-set object detection (COCO-ish labels)
  <CAPTION_TO_PHRASE_GROUNDING> text                     boxes for phrases in `text`
  <OPEN_VOCABULARY_DETECTION> text                       boxes for an arbitrary class name in `text`
It cannot answer 15 yes/no questions with a JSON schema. "Closest viable implementation" for tagging
(as the ticket asks): OPEN_VOCABULARY_DETECTION per tag name -> tag present iff >=1 box, plus a caption
keyword fallback. No confidence scores are exposed by the model for OVD.
"""

import re
import time
from typing import Any

TASK_CAPTION = "<CAPTION>"
TASK_DETAILED = "<DETAILED_CAPTION>"
TASK_MORE_DETAILED = "<MORE_DETAILED_CAPTION>"
TASK_OD = "<OD>"
TASK_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_OVD = "<OPEN_VOCABULARY_DETECTION>"


def tag_phrase(tag: dict) -> str:
    """Short noun phrase for OVD from the tag's human name (e.g. 'Kitchen Island' -> 'kitchen island')."""
    name = re.sub(r"\(.*?\)", "", tag.get("name") or tag["slug"].replace("_", " ")).strip()
    return name.lower()


class FlorenceBackend:
    name = "florence-2-large"

    def __init__(
        self,
        checkpoint: str = "florence-community/Florence-2-large",
        device: str | None = None,
        dtype: str = "float16",
    ):
        import torch
        from transformers import AutoProcessor

        self.checkpoint = checkpoint
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32

        # The `microsoft/Florence-2-*` repos ship custom modelling code written for older transformers;
        # it raises on 4.50+ (missing `forced_bos_token_id`). The `florence-community/*` ports are the
        # same weights against the class that now lives in transformers, so no trust_remote_code.
        try:
            from transformers import Florence2ForConditionalGeneration as _Model

            kwargs = {}
        except ImportError:  # transformers too old for the native class
            from transformers import AutoModelForCausalLM as _Model

            kwargs = {"trust_remote_code": True}

        self.model = _Model.from_pretrained(checkpoint, torch_dtype=self.torch_dtype, **kwargs).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint, **kwargs)
        self.name = checkpoint.split("/")[-1].lower()

    def run(self, image, task: str, text: str = "", max_new_tokens: int = 1024) -> tuple[Any, float]:
        """Return (parsed_answer, latency_s). `image` is a PIL image."""
        import torch

        prompt = task + (" " + text if text else "")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        out = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(out, task=task, image_size=(image.width, image.height))
        return parsed.get(task), time.perf_counter() - t0

    # ---- task helpers producing the same row shapes as runner.py ------------------------------------

    def captions(self, image) -> dict[str, Any]:
        base, t1 = self.run(image, TASK_CAPTION)
        det, t2 = self.run(image, TASK_MORE_DETAILED)
        return {"captions": {"base_caption": base, "detailed_caption": det}, "latency_s": round(t1 + t2, 3)}

    def ovd(self, image, phrase: str) -> tuple[list[dict], float]:
        res, lat = self.run(image, TASK_OVD, phrase)
        boxes = (res or {}).get("bboxes") or []
        labels = (res or {}).get("bboxes_labels") or []
        w, h = image.width, image.height
        dets = [
            {"label": lbl, "bbox": [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]}
            for b, lbl in zip(boxes, labels)
        ]
        return dets, lat

    def tagging_via_ovd(self, image, questions_tags: list[dict]) -> dict[str, Any]:
        """Tag present iff OVD returns >=1 box for the tag phrase. One forward pass per tag (no batching)."""
        answers, lat_total, raw = {}, 0.0, {}
        for tag in questions_tags:
            dets, lat = self.ovd(image, tag_phrase(tag))
            answers[tag["slug"]] = bool(dets)
            raw[tag["slug"]] = dets
            lat_total += lat
        return {
            "answers": answers,
            "confidence": {},
            "latency_s": round(lat_total, 3),
            "n_calls": len(questions_tags),
            "raw": raw,
        }
