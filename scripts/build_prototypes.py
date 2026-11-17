#!/usr/bin/env python
"""Draft `data/prototypes.json` from `data/tags.json` — caption-shaped prompts for an encoder.

Why a draft rather than a finished file: the prompts an encoder needs are captions ("a bicycle leaning
against a wall"), while a tag definition is an instruction written for a model that reads instructions.
Deriving the caption from the tag's own name gets most of them right and some of them wrong — a tag
named "Watermark" or "Low Quality" is not a thing a photograph can contain in those words. Those
need a human sentence, so every entry is written with `"reviewed_by": "generator"` and the harness
prints how many are still unreviewed.

Nothing about any particular domain lives in this file: the names, the questions and the exclusion
phrasing all come from `data/tags.json`, and the edited result stays in `data/`, which is gitignored.

    python scripts/build_prototypes.py            # writes data/prototypes.json, refuses to clobber
    python scripts/build_prototypes.py --force    # overwrite, losing hand-written edits
"""

import argparse
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

FRAMES = (
    "a photo of {subject}",
    "{subject}",
    "a photo showing {subject}",
    "a photograph of {subject}",
)

# Where a definition says what *not* to count, the words after the marker are the look-alikes worth
# having as negative prototypes. The split is crude on purpose; that is what the review flag is for.
EXCLUSION = re.compile(
    r"(?:do not count|don't count|do not tag|ignore|answer false for|not include|excluding)\b(.*?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
SPLIT = re.compile(r",|\band\b|\bor\b|;")
# Trimmed only from the front of a candidate phrase. Stripping them everywhere turned "bicycles in
# paintings" into "bicycles paintings" — prepositions carry the meaning of an exclusion.
LEADING_FILLER = {
    "items",
    "item",
    "that",
    "those",
    "these",
    "are",
    "is",
    "as",
    "any",
    "even",
    "when",
    "they",
    "look",
    "similar",
    "such",
    "it",
    "if",
    "you",
    "see",
    "them",
    "which",
    "the",
    "a",
    "an",
    "in",
    "on",
}
# A mass or plural noun takes no article. Everything else gets one, which is what a caption looks like.
NO_ARTICLE_SUFFIX = ("s", "ing", "ware", "work")
NO_ARTICLE_EXACT = {"access", "glass", "grass"}


def subject_of(tag: dict) -> str:
    """A noun phrase for the tag, from its name, with any clarifying parenthetical dropped.

    Wrong often enough to need review — a tag named "Watermark" is not something a photo *contains* —
    and right often enough to be a better starting point than an empty file.
    """
    name = re.sub(r"\(.*?\)", " ", tag["name"])
    return re.sub(r"\s+", " ", name).strip().lower()


def with_article(subject: str) -> str:
    """`a bicycle`, but `stairs` and `air conditioning` unchanged."""
    head = subject.split()[-1] if subject.split() else subject
    if head in NO_ARTICLE_EXACT or head.endswith(NO_ARTICLE_SUFFIX):
        return subject
    return f"{'an' if subject[:1] in 'aeiou' else 'a'} {subject}"


def negatives_of(tag: dict, subject: str, *, limit: int = 4) -> list[str]:
    """Look-alikes named in the definition's exclusion clause, as caption-shaped negatives.

    A clause like "answer false for items that are NOT bicycles, even when they look similar: scooters,
    motorcycles" names the *feature* before it names the look-alikes, so a colon means everything useful
    sits behind it. Without that rule the tag's own subject becomes a negative
    and subtracts the thing being looked for.

    A phrase that mentions the subject is deliberately kept: "bicycles in paintings" can be the single
    most useful negative a `bicycle` definition has, and a rule that drops anything echoing the subject
    throws away exactly the exclusions worth encoding. Only an exact restatement of the subject goes.

    The text is unwrapped first. Definitions may be hard-wrapped where they are stored, and splitting on
    the line breaks cuts a two-word look-alike such as "traffic cone" into "traffic" — a negative that
    would penalise far more images than the exclusion meant.
    """
    found: list[str] = []
    unwrapped = re.sub(r"\s+", " ", tag["question_text"])
    for clause in EXCLUSION.findall(unwrapped):
        if ":" in clause:
            clause = clause.split(":", 1)[1]
        for piece in SPLIT.split(clause):
            words = piece.strip().strip(".").split()
            while words and words[0].lower() in LEADING_FILLER:
                words.pop(0)
            phrase = " ".join(words).strip().lower()
            if not (3 <= len(phrase) <= 40) or phrase == subject or phrase in found:
                continue
            found.append(phrase)
    return [f"a photo of {with_article(p)}" for p in found[:limit]]


def draft(tags: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tag in tags:
        subject = subject_of(tag)
        phrase = with_article(subject)
        out[tag["slug"]] = {
            "subject": subject,
            "positive": [frame.format(subject=phrase) for frame in FRAMES],
            "negative": negatives_of(tag, subject),
            "reviewed_by": "generator",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", type=Path, default=DATA / "tags.json")
    ap.add_argument("--out", type=Path, default=DATA / "prototypes.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file and lose its edits")
    args = ap.parse_args()

    if not args.tags.exists():
        print(f"no {args.tags} — run `vlm-eval export` first", file=sys.stderr)
        return 1
    if args.out.exists() and not args.force:
        print(f"{args.out} exists; hand-written prompts live there. Re-run with --force to replace it.")
        return 1

    tags = json.loads(args.tags.read_text())
    entries = draft(tags)
    payload = {
        "note": (
            "Caption-shaped prompts per tag. `positive` is what the tag looks like; `negative` is what "
            "it is mistaken for. `reviewed_by` is 'generator' for prompts derived mechanically from the "
            "tag name and 'human' only after a person has read them; any other value is reported "
            "separately and does not count as reviewed."
        ),
        "tags": entries,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with_negatives = sum(1 for v in entries.values() if v["negative"])
    print(f"wrote {args.out}: {len(entries)} tags, {with_negatives} with negatives extracted from the definition")
    print("every entry is flagged for review — edit the ones whose `subject` is not something a photo can show")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
