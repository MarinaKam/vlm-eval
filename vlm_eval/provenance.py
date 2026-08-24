"""What a run file was produced by, so results from different settings cannot silently mix.

Resume used to key on `(image_id, repeat)` alone. That is enough to avoid repeating work and not nearly
enough to say the work is still valid: raise a token budget, change a prompt, point at another
checkpoint, and every existing row is still "done". The rows stay, the new ones are appended beside
them, and one file ends up holding two experiments that get averaged into a single number.

So each run file carries a sidecar naming the configuration that produced it. Resuming under a
different configuration is refused, not merged — the fix is to archive the file or start a new run
name, and either way somebody decides rather than discovers it later in a report.

A file that already existed before any of this was recorded is a third case, and the tempting mistake
is to quietly stamp it with today's settings. Its rows would then look exactly as verified as rows that
really were checked. Instead its status is written down as `legacy_unknown` and stays that way for the
life of the file, travels into `metrics.json`, and prints on every run that touches it.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VERIFIED = "verified"
LEGACY = "legacy_unknown"
UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class RunFingerprint:
    """Everything that changes what an answer means. Anything omitted here can drift unnoticed."""

    task: str
    served_name: str | None
    chunk_size: int
    individual: tuple[str, ...] = ()
    extra_output_tokens: int = 0
    # A digest of what was actually sent: the rendered prompt and the JSON schema, not just the
    # question texts. Reword the wrapper around the questions and the request changes even though every
    # question is identical.
    prompt_digest: str = ""
    coords: str | None = None
    backend: str = ""
    checkpoint: str | None = None
    logprobs: bool = False
    image_prep: str = ""  # how the images on disk were encoded — resize, quality, target size
    images_digest: str = ""  # digest of the image bytes actually on disk — the pixels, not the plan
    # Where the requests actually went: flavor@base_url for a server, in-process/<device> for a local
    # checkpoint. Two servers answering to the same served name are two different experiments.
    route: str = ""
    # Digest of the answer-producing source: the runner, the task module, the backend. Deliberately NOT
    # a git SHA — editing a report or a docstring elsewhere must not refuse a resume, but editing the
    # code that builds requests or parses answers must.
    code: str = ""
    # The weights themselves. A served name like `qwen3-vl:8b` is a mutable tag: pull an update and the
    # same name answers with a different model. Ollama's manifest digest / an HF revision are the
    # immutable identities; a backend that cannot prove one records `unknown: <why>`, and an unknown
    # identity refuses to resume a non-empty file — the one thing resume must never do is assume the
    # model stayed the same because its name did.
    model_identity: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def differences(self, other: "RunFingerprint") -> list[str]:
        mine, theirs = asdict(self), asdict(other)
        return [f"{k}: was {theirs[k]!r}, now {mine[k]!r}" for k in mine if mine[k] != theirs[k]]


@dataclass(frozen=True)
class Provenance:
    fingerprint: RunFingerprint
    status: str = VERIFIED
    # Rows that were already in the file when the fingerprint was first written — produced by settings
    # nobody recorded, and not made trustworthy by anything that happens afterwards.
    unverified_rows: int = 0

    @property
    def trustworthy(self) -> bool:
        return self.status == VERIFIED


def _unproven(identity: str) -> bool:
    return not identity or identity.startswith("unknown")


def code_identity(modules: list) -> str:
    """Digest of the source files behind the given modules, as found on disk right now."""
    agg = hashlib.sha256()
    for m in modules:
        f = getattr(m, "__file__", None)
        agg.update((m.__name__ if hasattr(m, "__name__") else str(m)).encode())
        if f and Path(f).exists():
            agg.update(Path(f).read_bytes())
    return agg.hexdigest()[:16]


def digest_of(payload: Any) -> str:
    """Stable digest of prompt content, so a reworded question counts as a different run."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[
        :16
    ]


def sidecar(run_file: Path) -> Path:
    return run_file.with_suffix(run_file.suffix + ".meta.json")


def _count_rows(run_file: Path) -> int:
    try:
        return sum(1 for line in run_file.read_text().splitlines() if line.strip())
    except OSError:
        return 0


def load(run_file: Path) -> Provenance | None:
    path = sidecar(run_file)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    fp = dict(raw["fingerprint"])
    fp["individual"] = tuple(fp.get("individual") or ())
    return Provenance(
        fingerprint=RunFingerprint(**fp),
        status=raw.get("status", VERIFIED),
        unverified_rows=int(raw.get("unverified_rows", 0)),
    )


def save(run_file: Path, prov: Provenance) -> None:
    payload = {
        "status": prov.status,
        "unverified_rows": prov.unverified_rows,
        "fingerprint": asdict(prov.fingerprint),
    }
    sidecar(run_file).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list))


def describe(run_file: Path) -> dict[str, Any]:
    """The one-line provenance of a run file, for `metrics.json` and the reports built from it."""
    prov = load(run_file)
    if prov is None:
        return {"status": UNRECORDED, "note": "no sidecar — nothing is known about how this file was produced"}
    out: dict[str, Any] = {"status": prov.status, "digest": prov.fingerprint.digest()}
    if prov.status == LEGACY:
        out["unverified_rows"] = prov.unverified_rows
        out["note"] = (
            f"{prov.unverified_rows} row(s) predate provenance recording; their settings are asserted, "
            "not verified — do not publish this file as a clean measurement"
        )
    return out


def check(run_file: Path, fp: RunFingerprint, *, log=print) -> None:
    """Refuse to append to a file produced under different settings.

    A file with no sidecar predates this check. Its rows are marked `legacy_unknown` permanently rather
    than adopted as verified: appending to it is allowed, publishing it as a clean measurement is a
    decision somebody has to make with the label in front of them.
    """
    if not run_file.exists():
        save(run_file, Provenance(fingerprint=fp, status=VERIFIED))
        return

    previous = load(run_file)
    if previous is None:
        rows = _count_rows(run_file)
        save(run_file, Provenance(fingerprint=fp, status=LEGACY, unverified_rows=rows))
        log(
            f"NOTE: {run_file.name} already held {rows} row(s) before runs recorded their settings.\n"
            f"      Marked '{LEGACY}' for the life of the file — the current settings are recorded, but\n"
            "      nothing proves those rows were produced under them. For a clean measurement, archive\n"
            f"      the file: mv {run_file} {run_file}.legacy"
        )
        return

    if previous.fingerprint.digest() == fp.digest() and _unproven(fp.model_identity) and _count_rows(run_file):
        raise SystemExit(
            f"{run_file.name} has rows, and the backend cannot prove the model weights are unchanged "
            f"({fp.model_identity}).\nA served name is a mutable tag — the same name may now answer "
            "with different weights, and resuming would mix two models in one file. Either\n"
            f"  archive it:  mv {run_file} {run_file}.old  (and {sidecar(run_file).name})\n"
            "  or use a backend that reports an immutable model digest."
        )

    if previous.fingerprint.digest() != fp.digest():
        raise SystemExit(
            f"{run_file.name} already holds results produced under different settings:\n  - "
            + "\n  - ".join(fp.differences(previous.fingerprint))
            + "\n\nResuming would mix two experiments in one file and average them into one number. Either\n"
            f"  archive it:  mv {run_file} {run_file}.old  (and {sidecar(run_file).name})\n"
            "  or run under a different model name so the results land separately."
        )

    if previous.status == LEGACY:
        log(
            f"NOTE: {run_file.name} is marked '{LEGACY}' — {previous.unverified_rows} row(s) in it were\n"
            "      written before provenance was recorded. Settings match for everything since."
        )
