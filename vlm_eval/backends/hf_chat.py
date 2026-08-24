"""Transformers backends for models without an Ollama build, run locally (MPS/CUDA/CPU).

Two flavors:
  * InternVLBackend  — OpenGVLab/InternVL3_5-8B-HF: instruction-tuned chat, same JSON prompts as Qwen
                        (no guided decoding — the lenient parser in tasks/ handles fenced/dirty JSON).
  * PaliGemmaBackend — google/paligemma2-3b-mix-448: single-turn prompt formats only
                        ("answer en <q>", "caption en", "detect <obj>"), gated repo (accept Gemma terms,
                        HF_TOKEN required). Tagging = one "answer en" call per question, yes/no parsed.

Both expose chat(images, prompt, ...) -> Response like the OpenAI backend, so runner.py works unchanged
(json_schema is accepted but ignored — parsing stays lenient).
"""
import io
import time
from typing import Any

from .base import Response


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_pil(images: list[bytes]):
    from PIL import Image as PILImage
    return [PILImage.open(io.BytesIO(b)).convert("RGB") for b in images]


class InternVLBackend:
    def __init__(self, checkpoint: str = "OpenGVLab/InternVL3_5-8B-HF", device: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or _pick_device()
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self.name = checkpoint.split("/")[-1].lower()

    def chat(self, images: list[bytes], prompt: str, *, json_schema: dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, logprobs: bool = False) -> Response:
        import torch

        pils = _to_pil(images)
        content = [{"type": "image", "image": im} for im in pils] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
            )
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return Response(text=text, latency_s=time.perf_counter() - t0)


class PaliGemmaBackend:
    """Single-turn prompt-format model. `chat` treats the prompt as a raw PaliGemma prompt
    (caller must pass 'answer en …' / 'caption en' / 'detect …'); helper methods build rows."""

    def __init__(self, checkpoint: str = "google/paligemma2-3b-mix-448", device: str | None = None):
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.device = device or _pick_device()
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True).to(self.device).eval()
        self.name = checkpoint.split("/")[-1].lower()

    def chat(self, images: list[bytes], prompt: str, *, json_schema: dict | None = None, max_tokens: int = 128,
             temperature: float = 0.0, logprobs: bool = False) -> Response:
        import torch

        pils = _to_pil(images)
        if len(pils) != 1:
            raise ValueError("PaliGemma is single-image")
        inputs = self.processor(text="<image>" + prompt, images=pils[0], return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return Response(text=text.strip(), latency_s=time.perf_counter() - t0)

    def answer_yes_no(self, image: bytes, question: str) -> tuple[bool | None, str, float]:
        r = self.chat([image], f"answer en {question}", max_tokens=8)
        first = r.text.strip().lower().split()[:2]
        joined = " ".join(first)
        if joined.startswith("yes"):
            return True, r.text, r.latency_s
        if joined.startswith("no"):
            return False, r.text, r.latency_s
        return None, r.text, r.latency_s

    def tagging_rows(self, image: bytes, questions: dict[str, str]) -> dict[str, Any]:
        answers, raw, total = {}, {}, 0.0
        for slug, q in questions.items():
            ans, txt, lat = self.answer_yes_no(image, q)
            answers[slug] = ans
            raw[slug] = txt
            total += lat
        return {"answers": answers, "confidence": {}, "latency_s": round(total, 3),
                "n_calls": len(questions), "raw": raw}
