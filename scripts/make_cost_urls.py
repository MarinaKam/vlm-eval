"""Build a URL list for the source app's token-cost measurement command.

The reference API bills per call, and the production pipeline re-sends the image with every chunk of
questions. This produces the input file needed to measure what that costs, and what asking every
question in a single call would cost instead.

    python scripts/make_cost_urls.py --type indoor --limit 20

Writes <data-dir>/cost_urls_<type>.txt with "url,image_id" per line — the format the measurement
command expects. The file lives in the data dir, which is gitignored: image URLs are private.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vlm_eval.config import DATA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["indoor", "outdoor"], default="indoor")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--manifest", type=Path, default=DATA / "manifest.csv")
    a = ap.parse_args()

    if not a.manifest.exists():
        raise SystemExit(f"{a.manifest} not found — run the dataset export first")

    with a.manifest.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["image_type"] == a.type and r["s3_url"]]
    picked = rows[: a.limit]
    if not picked:
        raise SystemExit(f"no {a.type} rows with a stored URL in {a.manifest}")

    out = DATA / f"cost_urls_{a.type}.txt"
    out.write_text(
        f"# {len(picked)} {a.type} images from {a.manifest.name}\n"
        + "".join(f"{r['s3_url']},{r['image_id']}\n" for r in picked)
    )
    print(f"wrote {len(picked)} urls -> {out}")


if __name__ == "__main__":
    main()
