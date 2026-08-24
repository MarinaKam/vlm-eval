"""Manual review of Gemini-vs-model disagreements (the ticket's "manually labelled subset", minimal form).

`build_review_html` writes a static page listing (image, tag, question, Gemini verdict, model verdict) for a
sample of disagreement cases; the reviewer picks the truth per row and clicks "Download decisions" which
saves a JSON file. `apply_decisions` merges such files into data/manual_labels.json, and
`manual_agreement` scores any model run against those human labels.
"""

import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import DATA, IMAGES


def disagreements(rows: list[dict], gemini: dict[int, dict], tags: list[dict]) -> list[dict]:
    q = {t["slug"]: t["question_text"] for t in tags}
    out = []
    for r in rows:
        if r.get("repeat", 0) != 0:
            continue
        g = gemini.get(r["image_id"])
        if not g:
            continue
        g_pos = set(g.get("tags", {}))
        for slug in set(g.get("evaluable_slugs") or g_pos) & set(r["answers"]):
            ans = r["answers"].get(slug)
            if ans is None or ans == (slug in g_pos):
                continue
            out.append(
                {
                    "image_id": r["image_id"],
                    "slug": slug,
                    "question": q.get(slug, slug),
                    "gemini": slug in g_pos,
                    "model": bool(ans),
                    "model_conf": (r.get("confidence") or {}).get(slug),
                }
            )
    return out


def sample_by_tag(cases: list[dict], per_tag: int = 5, seed: int = 7104) -> list[dict]:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_tag[c["slug"]].append(c)
    rng = random.Random(seed)
    picked = []
    for slug, lst in sorted(by_tag.items()):
        rng.shuffle(lst)
        picked.extend(lst[:per_tag])
    return picked


def build_review_html(model: str, cases: list[dict], out: Path) -> Path:
    rows = []
    for i, c in enumerate(cases):
        img = (IMAGES / f"{c['image_id']}.jpg").resolve()
        rows.append(f"""
<tr data-i="{i}">
 <td><img src="file://{img}" loading="lazy"></td>
 <td><b>{html.escape(c["slug"])}</b><br><small>{html.escape(c["question"])}</small><br>
     Gemini: <b>{c["gemini"]}</b> · {html.escape(model)}: <b>{c["model"]}</b>
     {"" if c.get("model_conf") is None else f"(p={c['model_conf']:.2f})"}</td>
 <td><label><input type=radio name=r{i} value=true> present</label><br>
     <label><input type=radio name=r{i} value=false> absent</label><br>
     <label><input type=radio name=r{i} value=unsure> unsure</label></td>
</tr>""")
    data = json.dumps([{"image_id": c["image_id"], "slug": c["slug"]} for c in cases])
    page = f"""<!doctype html><meta charset=utf-8><title>Tag review — {html.escape(model)}</title>
<style>body{{font-family:system-ui;margin:16px}} img{{max-width:420px;max-height:320px}}
td{{vertical-align:top;padding:6px;border-bottom:1px solid #ddd}}</style>
<h2>Review disagreements: Gemini vs {html.escape(model)} ({len(cases)} cases)</h2>
<p>Pick the truth for each row, then <button onclick="dl()">Download decisions</button></p>
<table>{"".join(rows)}</table>
<script>
const CASES={data};
function dl(){{const out=[];CASES.forEach((c,i)=>{{const v=document.querySelector('input[name=r'+i+']:checked');
 if(v) out.push({{...c, truth: v.value==='unsure'?null:(v.value==='true'), reviewer_model:{json.dumps(model)}}});}});
 const b=new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download='decisions_{model}.json';a.click();}}
</script>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


def apply_decisions(files: list[Path], store: Path | None = None) -> dict[str, Any]:
    store = store or DATA / "manual_labels.json"
    labels: dict[str, dict] = json.loads(store.read_text()) if store.exists() else {}
    added = 0
    for f in files:
        for d in json.loads(Path(f).read_text()):
            if d.get("truth") is None:
                continue
            key = f"{d['image_id']}:{d['slug']}"
            labels[key] = {"image_id": d["image_id"], "slug": d["slug"], "truth": bool(d["truth"])}
            added += 1
    store.write_text(json.dumps(labels, indent=1))
    return {"added": added, "total": len(labels)}


def manual_agreement(rows: list[dict], gemini: dict[str, dict], store: Path | None = None) -> dict:
    """On human-labelled (image, tag) pairs: accuracy of the model and of Gemini. Answers the question
    'when they disagree, who is right?'."""
    store = store or DATA / "manual_labels.json"
    if not store.exists():
        return {"n": 0}
    labels = json.loads(store.read_text())
    by_img = {r["image_id"]: r for r in rows if r.get("repeat", 0) == 0}
    n = model_ok = gemini_ok = 0
    for lab in labels.values():
        r = by_img.get(lab["image_id"])
        if not r or lab["slug"] not in r["answers"] or r["answers"][lab["slug"]] is None:
            continue
        n += 1
        model_ok += int(bool(r["answers"][lab["slug"]]) == lab["truth"])
        g_pos = lab["slug"] in gemini.get(lab["image_id"], {}).get("tags", {})
        gemini_ok += int(g_pos == lab["truth"])
    return {
        "n": n,
        "model_correct_pct": round(100 * model_ok / n, 1) if n else None,
        "gemini_correct_pct": round(100 * gemini_ok / n, 1) if n else None,
    }
