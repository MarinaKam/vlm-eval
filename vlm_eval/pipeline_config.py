"""The production settings the harness must reproduce, read from the export — never guessed.

The whole claim of this tool is "we ask your pipeline's questions, in your pipeline's batches, on your
pipeline's pixels". That only holds if the numbers come from the exported configuration. Constants in
code that happen to match production today are not evidence: change the value in production and a
hardcoded harness keeps measuring the old thing, silently.

So: every setting here is read from `data/prompts.json` (`processing_config`, as the export wrote it).
A fallback exists for datasets exported before a setting was captured, and using one is reported, not
assumed to be fine — `strict()` turns any fallback into an error.
"""

from dataclasses import dataclass, field
from typing import Any

# Used only when the export does not carry the setting. Matching production by coincidence is exactly
# the failure this module exists to prevent, so every use is recorded in `defaulted`.
FALLBACK = {
    "classification_chunk_size": 15,
    "individual_questions": [],
    "image_optimization_enabled": True,
    "image_optimization_max_dimension": 1536,
    "image_optimization_jpeg_quality": 85,
    "image_optimization_target_size_kb": 500,
}


def _unwrap(raw: Any) -> Any:
    """The export stores `{"value_int": 15}` / `{"value_json": [...]}`; take whatever is inside."""
    if isinstance(raw, dict):
        for key in ("value_int", "value_float", "value_bool", "value_json", "value_text"):
            if key in raw and raw[key] is not None:
                return raw[key]
        return None
    return raw


@dataclass(frozen=True)
class PipelineConfig:
    chunk_size: int
    individual_questions: list[str]
    optimize_enabled: bool
    max_dimension: int
    jpeg_quality: int
    target_size_kb: int
    defaulted: list[str] = field(default_factory=list)

    @property
    def faithful(self) -> bool:
        """True when every setting came from the export rather than a fallback."""
        return not self.defaulted

    def describe(self) -> str:
        lines = [
            f"  questions per call        {self.chunk_size}",
            f"  asked on their own        {', '.join(self.individual_questions) or '(none)'}",
            f"  image max dimension       {self.max_dimension}px",
            f"  image JPEG quality        {self.jpeg_quality}",
            f"  image target size         {self.target_size_kb} KB",
            f"  re-encode images          {'yes' if self.optimize_enabled else 'no'}",
        ]
        if self.defaulted:
            lines.append(f"  NOT from the export       {', '.join(self.defaulted)}  <- guessed, may not match")
        return "\n".join(lines)

    def strict(self) -> "PipelineConfig":
        """Refuse to run on guessed settings."""
        if self.defaulted:
            raise SystemExit(
                "These pipeline settings are missing from the export and would be guessed: "
                + ", ".join(self.defaulted)
                + "\nRe-run `vlm-eval export` so the harness reproduces production, or pass --allow-defaults "
                "to accept the fallbacks knowingly."
            )
        return self


def load(prompts: dict | None = None) -> PipelineConfig:
    """Build the config from an already-loaded `prompts.json` (or read it)."""
    if prompts is None:
        from .dataset import load_prompts

        prompts = load_prompts()
    raw = prompts.get("processing_config") or {}

    defaulted: list[str] = []

    def get(key: str) -> Any:
        if key in raw:
            value = _unwrap(raw[key])
            if value is not None:
                return value
        defaulted.append(key)
        return FALLBACK[key]

    return PipelineConfig(
        chunk_size=int(get("classification_chunk_size")),
        individual_questions=list(get("individual_questions") or []),
        optimize_enabled=bool(get("image_optimization_enabled")),
        max_dimension=int(get("image_optimization_max_dimension")),
        jpeg_quality=int(get("image_optimization_jpeg_quality")),
        target_size_kb=int(get("image_optimization_target_size_kb")),
        defaulted=defaulted,
    )
