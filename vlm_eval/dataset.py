"""Dataset access: manifest, reference-model (Gemini) outputs, tag questions, prompts, images on disk."""

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image as PILImage

from .config import DATA, ROOT  # noqa: F401  (ROOT re-exported for callers)

IMAGES = DATA / "images"

# Resize/compress settings are read from the export (see pipeline_config); these are only the
# fallbacks for a dataset exported before they were captured.
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


def load_manifest(path: Path | None = None) -> list[Item]:
    # Resolved on call, not at import: a default bound at import time would ignore VLM_EVAL_DATA_DIR
    # and quietly read the wrong directory.
    path = path or DATA / "manifest.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Create it first: scripts/export_staging_dataset.py against the source DB, "
            "or scripts/smoke_manifest.py <dir> <indoor|outdoor> for a local smoke test."
        )
    with path.open() as fh:
        return [Item(str(r["image_id"]), r["url"], r["s3_url"], r["image_type"]) for r in csv.DictReader(fh)]


def images_digest(items: list[Item] | None = None) -> str:
    """One digest over the actual bytes of every manifest image on disk.

    Encoding *settings* in a fingerprint say how the images were supposed to be prepared; this says
    what they actually are. Re-download at a different quality, or replace one file under the same
    id, and the digest changes — settings alone would not notice the second case. Reading ~170 MB
    takes well under a second, which is cheaper than ever wondering.

    Missing files are folded in by name: a run over 900 of 1000 images is a different dataset than a
    run over all of them.
    """
    items = items if items is not None else load_manifest()
    agg = hashlib.sha256()
    for it in sorted(items, key=lambda x: x.image_id):
        agg.update(it.image_id.encode())
        if it.path.exists():
            agg.update(hashlib.sha256(it.path.read_bytes()).digest())
        else:
            agg.update(b"missing")
    return agg.hexdigest()[:16]


def load_tags(path: Path | None = None, *, active_only: bool = True) -> list[dict]:
    path = path or (DATA / "tags.json" if (DATA / "tags.json").exists() else DATA / "tags_from_migrations.json")
    tags = json.loads(path.read_text())
    return [t for t in tags if t.get("is_active", True)] if active_only else tags


def load_prompts(path: Path | None = None) -> dict:
    path = path or DATA / "prompts.json"
    if not path.exists():
        return {"caption_prompts": {}, "prompt_templates": {}, "processing_config": {}}
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def reference_path(kind: str, data: Path | None = None) -> Path:
    """Where the reference answers live. `kind` is "tags" or "captions".

    The tool has no opinion about which system is the reference: it is whatever your pipeline runs
    today — a paid API, a model you already host, last quarter's checkpoint. Early datasets named these
    files after one vendor; both names are accepted, and the neutral one wins when both exist.
    """
    data = data or DATA
    neutral = data / f"reference_{kind}.jsonl"
    return neutral if neutral.exists() else data / f"gemini_{kind}.jsonl"


def reference_tags_by_image(path: Path | None = None) -> dict[str, dict]:
    return {str(row["image_id"]): row for row in load_jsonl(path or reference_path("tags"))}


# Older name, kept so existing scripts and notebooks keep working.
gemini_tags_by_image = reference_tags_by_image


def property_items(path: Path | None = None) -> list[Item]:
    """Every image belonging to an exported listing, as downloadable Items (deduplicated)."""
    path = path or DATA / "properties.jsonl"
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


def is_image(data: bytes) -> bool:
    """Do these bytes decode as an image at all?

    A URL that has expired, a signed link that went stale, or a proxy interstitial all return HTTP 200
    with an HTML body. Written to disk as `<id>.jpg` that looks like a successful download and fails
    hours later, mid-run, one image at a time.
    """
    try:
        PILImage.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def download_all(
    items: list[Item],
    *,
    force: bool = False,
    timeout: float = 30.0,
    reencode: bool = False,
) -> tuple[int, list[str]]:
    """Fetch each image to data/images/<id>.jpg. Returns (downloaded, failed_ids).

    By default the bytes are stored exactly as served. Production uploads its images *already* resized
    and compressed, so re-encoding them here would add a second generation of JPEG loss and hand the
    models slightly different pixels than the reference model saw — the opposite of what this tool
    claims. Pass `reencode=True` only for a source that serves originals.
    """
    IMAGES.mkdir(parents=True, exist_ok=True)
    done, failed = 0, []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for it in items:
            if it.path.exists() and not force:
                continue
            try:
                r = client.get(it.s3_url or it.url)
                r.raise_for_status()
                if not is_image(r.content):
                    raise ValueError(
                        f"served {len(r.content)} bytes that are not an image "
                        f"(content-type {r.headers.get('content-type', 'unknown')})"
                    )
                it.path.write_bytes(optimize(r.content) if reencode else r.content)
                done += 1
                if done % 50 == 0:
                    print(f"  downloaded {done}...", flush=True)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                failed.append(it.image_id)
                print(f"download failed image_id={it.image_id}: {exc}")
    if failed:
        print(
            f"\n{len(failed)} image(s) could not be fetched and are absent from the dataset. Runs will "
            "skip them; the counts in every report are over the images that exist."
        )
    return done, failed
