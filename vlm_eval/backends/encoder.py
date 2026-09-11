"""Image-text encoders: one vector per image, one per tag prompt, compared by cosine similarity.

This is deliberately **not** the `chat()` `Backend` protocol next door. An encoder cannot be asked a
question and does not produce an answer; it produces a position in a shared space. Dressing that up as
a chat call would hide the two things this investigation exists to measure — that the comparison is a
dot product rather than a generation, and that a threshold, not the model, decides what counts as a tag.

Everything that varies between checkpoints is read from the checkpoint, never assumed:

  * **the text window** comes from `text_config.max_position_embeddings` — 64 for SigLIP 2, 77 for CLIP.
    A constant here that happened to match today would silently measure the wrong thing tomorrow.
  * **the decision boundary**, for a model that has one. SigLIP is trained with a sigmoid loss and ships
    `logit_scale` and `logit_bias`, so `sigmoid(scale * cosine + bias)` is a calibrated probability and
    the model's own threshold falls out of it. CLIP is trained with a softmax over a batch: its scores
    are meaningful only relative to the other candidates, and it has no such boundary to offer.
  * **the weights' identity** is the commit the cache resolved, so `provenance` can refuse to mix two
    revisions in one run file the way it already does for a served model tag.
"""

import io
import math
from dataclasses import dataclass

import numpy as np

# Prompts are padded to the full window rather than to the longest in the batch: both families were
# trained that way, and batch-dependent padding would make a prompt's vector depend on its neighbours.
PAD = "max_length"


@dataclass(frozen=True)
class EncodedTexts:
    """Vectors for a list of prompts, plus how many of them the encoder could not read in full."""

    vectors: np.ndarray
    token_counts: list[int]
    context_length: int

    @property
    def truncated(self) -> list[int]:
        return [i for i, n in enumerate(self.token_counts) if n > self.context_length]


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _commit_from_cache(repo_id: str) -> str:
    """The revision the cache actually resolved, taken from the snapshot path it lives under.

    A repository name is a moving target exactly like a served model tag: `main` today is not `main`
    next month. The snapshot directory is named after the commit, which is immutable.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id, "config.json")
        if isinstance(path, str) and "/snapshots/" in path:
            return path.split("/snapshots/")[1].split("/")[0]
    except Exception:
        pass
    return f"unknown: no resolved snapshot for {repo_id}"


def default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Encoder:
    """One checkpoint, loaded once. `encode_images` and `encode_texts` return L2-normalised vectors."""

    def __init__(self, name: str, repo_id: str, *, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.name = name
        self.repo_id = repo_id
        # `checkpoint` and `device` are the names the existing fingerprint helpers already look for, so
        # an encoder is identified and routed by the same code that identifies a served model.
        self.checkpoint = repo_id
        self.device = device or default_device()
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(repo_id)
        model = AutoModel.from_pretrained(repo_id, dtype=torch.float32).eval()
        self._model = model.to(self.device)
        self.commit = _commit_from_cache(repo_id)

        text_config = model.config.text_config
        # `tokenizer.model_max_length` is a sentinel on these repositories (10^30), so it cannot be the
        # source for the window; the text tower's position embeddings are the real limit.
        self.context_length = int(text_config.max_position_embeddings)
        self.dim = int(getattr(text_config, "hidden_size", 0))
        self.logit_scale = float(model.logit_scale.detach().exp()) if hasattr(model, "logit_scale") else None
        self.logit_bias = float(model.logit_bias.detach()) if hasattr(model, "logit_bias") else None

    # ------------------------------------------------------------------ identity

    @property
    def calibrated(self) -> bool:
        """Whether the checkpoint ships a trained sigmoid boundary, as SigLIP does and CLIP does not."""
        return self.logit_bias is not None and self.logit_scale is not None

    def probability(self, similarity: float) -> float | None:
        """The model's own P(tag present) for a cosine similarity, or None when it has no opinion."""
        if not self.calibrated:
            return None
        return 1.0 / (1.0 + math.exp(-(self.logit_scale * similarity + self.logit_bias)))

    @property
    def native_threshold(self) -> float | None:
        """The cosine similarity at which the checkpoint's own probability crosses one half."""
        if not self.calibrated:
            return None
        return -self.logit_bias / self.logit_scale

    @property
    def weights_digest(self) -> str:
        """Read by the shared fingerprint helper, which refuses to resume a file it cannot vouch for."""
        if self.commit.startswith("unknown"):
            return self.commit
        return f"hf:{self.repo_id}@{self.commit}"

    # ------------------------------------------------------------------ encoding

    def token_count(self, text: str) -> int:
        """Tokens the prompt really needs, measured without truncating — the evidence that it was cut."""
        return len(self._processor.tokenizer(text, truncation=False)["input_ids"])

    def encode_texts(self, texts: list[str], *, batch_size: int = 64) -> EncodedTexts:
        counts = [self.token_count(t) for t in texts]
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = self._processor.tokenizer(
                texts[start : start + batch_size],
                padding=PAD,
                max_length=self.context_length,
                truncation=True,
                return_tensors="pt",
            )
            chunks.append(self._features(self._model.get_text_features, batch))
        vectors = _l2(np.concatenate(chunks)) if chunks else np.zeros((0, self.dim), dtype=np.float32)
        return EncodedTexts(vectors=vectors, token_counts=counts, context_length=self.context_length)

    def encode_images(self, images: list[bytes]) -> np.ndarray:
        from PIL import Image

        pils = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        batch = self._processor(images=pils, return_tensors="pt")
        return _l2(self._features(self._model.get_image_features, batch))

    def _features(self, fn, batch) -> np.ndarray:
        """Both families return a pooled-output object here, not a bare tensor."""
        moved = {k: v.to(self.device) for k, v in batch.items()}
        with self._torch.no_grad():
            out = fn(**moved)
        tensor = getattr(out, "pooler_output", out)
        return tensor.detach().to("cpu").float().numpy()
