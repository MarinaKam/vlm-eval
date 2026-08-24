"""Build a tiny data/manifest.csv from local JPGs to smoke-test the harness before the staging export.

    python scripts/smoke_manifest.py <dir_with_jpgs> <indoor|outdoor> [limit]

Copies (optimized) images into data/images/<n>.jpg with ids 900001+, no Gemini ground truth.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_eval.dataset import DATA, IMAGES, optimize  # noqa: E402

src, kind = Path(sys.argv[1]), sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 5
IMAGES.mkdir(parents=True, exist_ok=True)
files = sorted([*src.glob("*.jpg"), *src.glob("*.jpeg"), *src.glob("*.png")])[:limit]
with (DATA / "manifest.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image_id", "url", "s3_url", "image_type", "user_id", "job_created_at"])
    for n, f in enumerate(files, start=900001):
        (IMAGES / f"{n}.jpg").write_bytes(optimize(f.read_bytes()))
        w.writerow([n, str(f), "", kind, "", ""])
print(f"smoke manifest: {len(files)} images from {src} as {kind}")
