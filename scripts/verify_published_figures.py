"""Recompute every published figure from the raw run files, with an independent implementation.

Deliberately imports nothing from `vlm_eval`: a bug in metrics.py must not be able to confirm itself.
Only the standard library, reading `data/` and `runs/` directly.

    python scripts/verify_published_figures.py

The expected values below are the ones written into `reports/`. A MISMATCH means either a report is
wrong or the data has moved on since it was written — both worth knowing before anyone forwards the
document. Edit the expectations when you re-generate the reports; that edit is the point at which
somebody consciously accepts the new number.

This is a project-specific script: the checks name our models and our figures. It is kept because a
report nobody re-derives is a report nobody can trust.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, RUNS = ROOT / "data", ROOT / "runs"
Q25, Q3 = "qwen2.5vl-7b-ollama", "qwen3-vl-8b-instruct-ollama"

ok = fail = 0


def check(label, claimed, actual, tol=0.05):
    global ok, fail
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        good = abs(claimed - actual) <= tol
    else:
        good = claimed == actual
    mark = "OK  " if good else "MISMATCH"
    if good:
        ok += 1
    else:
        fail += 1
    print(f"  [{mark}] {label:52} published {claimed}   actual {actual}")


skipped = 0


def missing(what: str, path) -> bool:
    """A section whose data is not on disk is skipped out loud, never silently passed.

    Crashing here used to take the other nine sections with it — a verification script that dies on
    one absent file verifies nothing at all.
    """
    global skipped
    if Path(path).exists():
        return False
    skipped += 1
    print(f"  [SKIP] {what}: no data at {Path(path).name}")
    return True


def jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


_rt = DATA / "reference_tags.jsonl"
ref = {r["image_id"]: r for r in jsonl(_rt if _rt.exists() else DATA / "gemini_tags.jsonl")}


def score(rows):
    """TP/FP/FN/TN over the tags the reference actually judged."""
    tp = fp = fn = tn = null = 0
    for r in rows:
        g = ref.get(r["image_id"])
        if not g:
            continue
        pos = set(g["tags"])
        for slug in set(g.get("evaluable_slugs") or pos) & set(r["answers"]):
            a = r["answers"][slug]
            if a is None:
                null += 1
            elif a and slug in pos:
                tp += 1
            elif a:
                fp += 1
            elif slug in pos:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn, null


print("=== 1. Full Qwen2.5-VL tagging run ===")
rows = [r for r in jsonl(RUNS / Q25 / "tagging_chunk15.jsonl") if r.get("repeat", 0) == 0]
tp, fp, fn, tn, null = score(rows)
judged = tp + fp + fn + tn
check("images", 1000, len(rows), 0)
check("tag decisions", 30912, judged + null, 0)
check("accuracy %", 96.8, round(100 * (tp + tn) / judged, 1))
check("precision %", 89.4, round(100 * tp / (tp + fp), 1))
check("recall %", 73.5, round(100 * tp / (tp + fn), 1))
check("false positive rate %", 0.9, round(100 * fp / (fp + tn), 1))
check("unparsed answers", 0, null, 0)
check("rows with an error", 0, sum(1 for r in rows if r.get("errors")), 0)

print("\n=== 2. Stability across repeats ===")
by_img = defaultdict(list)
for r in jsonl(RUNS / Q25 / "tagging_chunk15.jsonl"):
    by_img[r["image_id"]].append(r)
multi = {k: v for k, v in by_img.items() if len(v) >= 2}
flips = sum(1 for v in multi.values() for slug in v[0]["answers"] if len({x["answers"].get(slug) for x in v}) > 1)
decisions = sum(len(v[0]["answers"]) for v in multi.values())
check("images run more than once", 100, len(multi), 0)
check("decisions compared", 5018, decisions, 0)
check("answers that changed", 0, flips, 0)

print("\n=== 3. Batches of 15 against all questions in one call ===")
allq = jsonl(RUNS / Q25 / "tagging_chunkall.jsonl")
ids = {r["image_id"] for r in allq}
base = {r["image_id"]: r for r in rows if r["image_id"] in ids}
same = diff = 0
for r in allq:
    b = base.get(r["image_id"])
    if not b:
        continue
    for slug, v in b["answers"].items():
        w = r["answers"].get(slug)
        if w is None:
            continue
        same += v == w
        diff += v != w
check("images", 300, len(allq), 0)
check("answers that agree %", 98.9, round(100 * same / (same + diff), 1))
check("calls per image (batched)", 4.9, round(statistics.mean(r["n_calls"] for r in base.values()), 1))
check("calls per image (one call)", 2.0, round(statistics.mean(r["n_calls"] for r in allq), 1))

print("\n=== 4. Captions ===")
caps = jsonl(RUNS / Q25 / "captions.jsonl")
gem_caps = {
    r["image_id"]: r["captions"]
    for r in jsonl(
        DATA / ("reference_captions.jsonl" if (DATA / "reference_captions.jsonl").exists() else "gemini_captions.jsonl")
    )
}
check("images", 1000, len(caps), 0)
for key, mine, theirs in (("base_caption", 10.3, 11), ("detailed_caption", 29.6, 54)):
    m = [len(c.split()) for r in caps if (c := r["captions"].get(key))]
    g = [len(c.split()) for r in caps if (c := gem_caps.get(r["image_id"], {}).get(key))]
    check(f"{key}: words from the model", mine, round(statistics.mean(m), 1), 0.15)
    check(f"{key}: words from the reference", theirs, round(statistics.mean(g)), 1)

print("\n=== 5. Grounding ===")
gr = jsonl(RUNS / Q25 / "grounding.jsonl")
check("images", 300, len(gr), 0)
stats = defaultdict(lambda: [0, 0, 0, 0])  # pos_img, pos_found, neg_img, neg_found
for r in gr:
    g = ref.get(r["image_id"], {})
    ev = set(g.get("evaluable_slugs") or [])
    for label, dets in (r.get("detections") or {}).items():
        if dets is None or label not in ev:
            continue
        s = stats[label]
        if label in g["tags"]:
            s[0] += 1
            s[1] += bool(dets)
        else:
            s[2] += 1
            s[3] += bool(dets)
for label, (pi, pf, ni, nf) in sorted(stats.items()):
    print(f"       {label:16} tag present {pf}/{pi}, tag absent {nf}/{ni}")
check("kitchen_island found where tagged", 6, stats["kitchen_island"][1], 0)
check("radiator found where tagged", 10, stats["radiator"][1], 0)
check("false boxes in total", 0, sum(s[3] for s in stats.values()), 0)

print("\n=== 6. Head to head on the shared images ===")
_h2h = RUNS / Q3 / "tagging_chunk15.jsonl"
if not missing("head to head", _h2h):
    q3rows = jsonl(_h2h)
    common = {r["image_id"] for r in q3rows} & {r["image_id"] for r in rows}
    a = score([r for r in q3rows if r["image_id"] in common])
    b = score([r for r in rows if r["image_id"] in common])
    check("shared images", 280, len(common), 0)
    check("recall Qwen3 %", 82.8, round(100 * a[0] / (a[0] + a[2]), 1))
    check("recall Qwen2.5 %", 74.8, round(100 * b[0] / (b[0] + b[2]), 1))

print("\n=== 7. Adjudicated disagreements ===")
if missing("adjudication", DATA / "manual_labels.json"):
    labels = {}
else:
    labels = json.loads((DATA / "manual_labels.json").read_text())
tagged = {r["image_id"]: r for r in rows}
n = mo = ge = 0
for lab in labels.values():
    r = tagged.get(lab["image_id"])
    if not r or r["answers"].get(lab["slug"]) is None:
        continue
    n += 1
    mo += bool(r["answers"][lab["slug"]]) == lab["truth"]
    ge += (lab["slug"] in ref.get(lab["image_id"], {}).get("tags", {})) == lab["truth"]
check("verdicts given", 140, n, 0)
check("model was right %", 51, round(100 * mo / n), 1)
check("reference was right %", 49, round(100 * ge / n), 1)

print("\n=== 8. The reference is not deterministic ===")
old = DATA / "archive_20img" / "cost_chunk15.csv"
new = DATA / "cost_chunk15.csv"


def tags_csv(p):
    with open(p) as fh:
        return {
            r["image_id"]: {t.strip() for t in (r["classification_tags"] or "").split(";") if t.strip()}
            for r in csv.DictReader(fh)
        }


if old.exists():
    x, y = tags_csv(old), tags_csv(new)
    shared = set(x) & set(y)
    identical = sum(1 for k in shared if x[k] == y[k])
    check("images compared", 20, len(shared), 0)
    check("identical tag sets %", 85, round(100 * identical / len(shared)), 1)

print("\n=== 9. Cost by batch size ===")
for chunk, claimed in ((15, 0.001216), (47, 0.000865)):
    f = DATA / f"cost_chunk{chunk}.csv"
    with open(f) as fh:
        costs = [float(r["cost_2_5"]) for r in csv.DictReader(fh)]
    check(f"chunk {chunk}: $/image", claimed, round(statistics.mean(costs), 6), 1e-6)
with open(DATA / "cost_chunk15.csv") as fh:
    a15 = statistics.mean(float(r["cost_2_5"]) for r in csv.DictReader(fh))
with open(DATA / "cost_chunk47.csv") as fh:
    a47 = statistics.mean(float(r["cost_2_5"]) for r in csv.DictReader(fh))
check("saving at 47 questions %", 29, round(100 * (1 - a47 / a15)), 1)

print("\n=== 10. Tags lost at 47 questions ===")
t15, t47 = tags_csv(DATA / "cost_chunk15.csv"), tags_csv(DATA / "cost_chunk47.csv")
shared = set(t15) & set(t47)
check("tags at batches of 15", 226, sum(len(t15[k]) for k in shared), 0)
check("tags at 47 in one call", 204, sum(len(t47[k]) for k in shared), 0)

print(f"\n{'=' * 70}\n{ok} matched, {fail} mismatched, {skipped} skipped for missing data")
