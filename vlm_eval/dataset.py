"""Dataset access: manifest, reference-model (Gemini) outputs, tag questions, prompts, images on disk."""

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image as PILImage

from .config import DATA, ROOT  # noqa: F401  (ROOT re-exported for callers)

IMAGES = DATA / "images"

# same resize/compress defaults as the production pipeline (max_dimension 1536, JPEG q85, target 500 KB)
MAX_DIM = 1536
JPEG_Q = 85
TARGET_KB = 500


@dataclass(frozen=True)
class Item:
    image_id: str  # source-DB PKs (UUID strings)
    url: str
    s3_url: str
    image_type: str

    @property
    def path(self) -> Path:
        return IMAGES / f"{self.image_id}.jpg"


def load_manifest(path: Path = DATA / "manifest.csv") -> list[Item]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Create it first: scripts/export_staging_dataset.py against the source DB, "
            "or scripts/smoke_manifest.py <dir> <indoor|outdoor> for a local smoke test."
        )
    with path.open() as fh:
        return [Item(str(r["image_id"]), r["url"], r["s3_url"], r["image_type"]) for r in csv.DictReader(fh)]


def load_tags(path: Path | None = None, *, active_only: bool = True) -> list[dict]:
    path = path or (DATA / "tags.json" if (DATA / "tags.json").exists() else DATA / "tags_from_migrations.json")
    tags = json.loads(path.read_text())
    return [t for t in tags if t.get("is_active", True)] if active_only else tags


def load_prompts(path: Path = DATA / "prompts.json") -> dict:
    if not path.exists():
        return {"caption_prompts": {}, "prompt_templates": {}, "processing_config": {}}
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gemini_tags_by_image(path: Path = DATA / "gemini_tags.jsonl") -> dict[str, dict]:
    return {str(row["image_id"]): row for row in load_jsonl(path)}


def property_items(path: Path = DATA / "properties.jsonl") -> list[Item]:
    """Every image belonging to an exported listing, as downloadable Items (deduplicated)."""
    out: dict[str, Item] = {}
    for row in load_jsonl(path):
        for image_id, url in zip(row["image_ids"], row.get("s3_urls") or []):
            if url and image_id not in out:
                out[image_id] = Item(str(image_id), url, url, "property")
    return list(out.values())


def optimize(image_data: bytes, max_dim: int = MAX_DIM, quality: int = JPEG_Q, target_kb: int = TARGET_KB) -> bytes:
    """Same resize/compress as production so every model sees the same pixels the reference model saw."""
    img = PILImage.open(io.BytesIO(image_data))
    if img.mode in ("RGBA", "P", "LA"):
        bg = PILImage.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((int(w * r), int(h * r)), PILImage.Resampling.LANCZOS)
    q = quality
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        out = buf.getvalue()
        if len(out) <= target_kb * 1024 or q <= 20:
            return out
        q -= 10


def image_size(data: bytes) -> tuple[int, int]:
    return PILImage.open(io.BytesIO(data)).size


def download_all(items: list[Item], *, force: bool = False, timeout: float = 30.0) -> tuple[int, list[int]]:
    """Fetch s3_url -> data/images/<id>.jpg (optimized). Returns (downloaded, failed_ids)."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    done, failed = 0, []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for it in items:
            if it.path.exists() and not force:
                continue
            try:
                r = client.get(it.s3_url or it.url)
                r.raise_for_status()
                it.path.write_bytes(optimize(r.content))
                done += 1
                if done % 50 == 0:
                    print(f"  downloaded {done}...", flush=True)
            except (httpx.HTTPError, OSError) as exc:
                failed.append(it.image_id)
                print(f"download failed image_id={it.image_id}: {exc}")
    return done, failed
