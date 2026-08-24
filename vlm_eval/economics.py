"""Self-host vs pay-per-call arithmetic.

The quality metrics answer "can we switch". This answers "should we", which is usually decided by three
numbers nobody measured: how many images per month actually go through, what one image costs on the API
today, and how bursty the traffic is. All three come from measurement, not estimation:

  * volume  — `scripts/count_volume.py` against the production database
  * api_cost_per_image — the token-cost command at the chunk size production uses
  * peak    — the busiest hour, from the same volume count

A GPU is rented by the hour, so below a certain volume it idles most of the time and costs more than the
API. And a burst that the vendor absorbs elastically needs real hardware on your side.
"""

from dataclasses import dataclass, field
from typing import Any

HOURS_PER_MONTH = 730


@dataclass(frozen=True)
class Hosting:
    """One way to run the model. `always_on` bills for the whole month regardless of volume — a
    dedicated node, or a pod that must stay warm. Otherwise you pay for the hours the work takes,
    which only works if the pipeline tolerates a cold start."""

    name: str
    usd_per_hour: float
    always_on: bool = False
    cold_start_min: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Inputs:
    api_cost_per_image: float  # measured, current settings
    api_cost_per_image_optimized: float | None = None  # measured, cheaper settings (if tested)
    gpu_usd_per_hour: float = 0.5832  # spot L4 by default
    gpu_images_per_hour: float = 2000.0  # measure this; published benchmarks are a starting point
    scenarios: list[tuple[str, int]] = field(default_factory=list)  # (label, images per month)
    peak_hour_images: int | None = None
    busy_hours_pct: float | None = None
    gpu_name: str = "L4 (spot)"
    hosting: list[Hosting] = field(default_factory=list)
    weights_gb: float | None = None  # model weights parked in object storage
    storage_usd_per_gb_month: float = 0.02  # standard regional object storage


def monthly_gpu_cost(images_per_month: float, inp: Inputs) -> float:
    """Autoscaled: pay for the hours the work actually takes, capped at running the GPU non-stop."""
    hours = images_per_month / inp.gpu_images_per_hour
    return min(hours, HOURS_PER_MONTH) * inp.gpu_usd_per_hour


def break_even_images(inp: Inputs, *, on_demand_hourly: float | None = None) -> float:
    """Volume at which a GPU running non-stop costs the same as the API."""
    hourly = on_demand_hourly if on_demand_hourly is not None else inp.gpu_usd_per_hour
    return hourly * HOURS_PER_MONTH / inp.api_cost_per_image


def peak_analysis(inp: Inputs) -> dict[str, Any] | None:
    """What the busiest hour demands of self-hosted capacity."""
    if not inp.peak_hour_images:
        return None
    hours_for_one_gpu = inp.peak_hour_images / inp.gpu_images_per_hour
    return {
        "peak_hour_images": inp.peak_hour_images,
        "hours_for_one_gpu": round(hours_for_one_gpu, 1),
        "gpus_to_absorb_in_one_hour": max(1, round(hours_for_one_gpu)),
        "busy_hours_pct": inp.busy_hours_pct,
    }


def hosting_cost(images_per_month: float, h: Hosting, inp: Inputs) -> float:
    """Monthly cost of one hosting option at a given volume."""
    if h.always_on:
        return h.usd_per_hour * HOURS_PER_MONTH
    hours = images_per_month / inp.gpu_images_per_hour
    return min(hours, HOURS_PER_MONTH) * h.usd_per_hour


def storage_cost(inp: Inputs) -> float | None:
    """Keeping the weights in object storage — small, but it is not zero."""
    if inp.weights_gb is None:
        return None
    return inp.weights_gb * inp.storage_usd_per_gb_month


def hosting_table(inp: Inputs) -> str:
    """Compare hosting options across the volume scenarios."""
    if not inp.hosting or not inp.scenarios:
        return ""
    head = [
        "| hosting option | $/hour | " + " | ".join(f"{lbl} ({v:,}/mo)" for lbl, v in inp.scenarios) + " |",
        "|---|---|" + "---|" * len(inp.scenarios),
    ]
    body = []
    for h in inp.hosting:
        cells = [f"${hosting_cost(v, h, inp) * 12:,.0f}/yr" for _, v in inp.scenarios]
        body.append(f"| {h.name} | ${h.usd_per_hour:.4f} | " + " | ".join(cells) + " |")
    api = [
        "| **the API (no hosting)** | — | "
        + " | ".join(f"${v * inp.api_cost_per_image * 12:,.0f}/yr" for _, v in inp.scenarios)
        + " |"
    ]
    notes = [f"- **{h.name}** — {h.note}" for h in inp.hosting if h.note]
    cold = [
        f"- **{h.name}** needs ~{h.cold_start_min:g} min to come up from cold." for h in inp.hosting if h.cold_start_min
    ]
    out = ["", "## Where to run it", "", *head, *body, *api]
    if notes or cold:
        out += ["", *notes, *cold]
    stor = storage_cost(inp)
    if stor is not None:
        out += [
            "",
            f"Keeping {inp.weights_gb:g} GB of weights in object storage adds ${stor:.2f}/month "
            f"(${stor * 12:.2f}/year) — negligible next to compute, but it is where the weights live "
            "and it is what makes a cold start reproducible.",
        ]
    return "\n".join(out)


def rows(inp: Inputs) -> list[dict[str, Any]]:
    out = []
    for label, v in inp.scenarios:
        api = v * inp.api_cost_per_image * 12
        opt = v * inp.api_cost_per_image_optimized * 12 if inp.api_cost_per_image_optimized else None
        own = monthly_gpu_cost(v, inp) * 12
        out.append(
            {
                "label": label,
                "per_month": v,
                "api_year": api,
                "optimized_year": opt,
                "own_year": own,
                "saving_year": api - own,
            }
        )
    return out


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def render(inp: Inputs, *, currency_note: str = "") -> str:
    data = rows(inp)
    be = break_even_images(inp)
    peak = peak_analysis(inp)
    biggest = max(data, key=lambda r: r["per_month"]) if data else None

    lines = [
        "# Self-host vs pay-per-call",
        "",
        "The quality numbers say whether a switch is *possible*. These say whether it is *worth it*.",
        "The API charges per image; a GPU charges per hour. Below a certain volume the GPU idles most of",
        "the time and costs more.",
        "",
        "## Measured inputs",
        "",
        "| input | value |",
        "|---|---|",
        f"| Cost per image on the API (current settings) | ${inp.api_cost_per_image:.6f} |",
    ]
    if inp.api_cost_per_image_optimized:
        saving = 100 * (1 - inp.api_cost_per_image_optimized / inp.api_cost_per_image)
        lines.append(f"| Cost per image, cheaper settings | ${inp.api_cost_per_image_optimized:.6f} (−{saving:.0f}%) |")
    lines += [
        f"| GPU | {inp.gpu_name}, ${inp.gpu_usd_per_hour:.4f}/hour |",
        f"| GPU throughput | {inp.gpu_images_per_hour:,.0f} images/hour |",
    ]
    if peak:
        lines.append(f"| Busiest hour observed | {peak['peak_hour_images']:,} images |")
    if inp.busy_hours_pct is not None:
        lines.append(f"| Hours with any work | {inp.busy_hours_pct}% |")

    lines += [
        "",
        "## What it costs",
        "",
        "| scenario | images/month | API per year | optimized | self-hosted per year | saving/year |",
        "|---|---|---|---|---|---|",
    ]
    for r in data:
        lines.append(
            f"| {r['label']} | {r['per_month']:,} | {_money(r['api_year'])} | {_money(r['optimized_year'])} "
            f"| {_money(r['own_year'])} | {_money(r['saving_year'])} |"
        )

    lines += ["", f"**Break-even for a GPU running non-stop: {be:,.0f} images/month.**"]
    if biggest and biggest["per_month"]:
        lines.append(f"That is {be / biggest['per_month']:.1f}× the highest scenario above ({biggest['label']}).")

    if peak:
        lines += [
            "",
            "## The real obstacle is the shape of the traffic, not the price",
            "",
            f"The busiest hour was **{peak['peak_hour_images']:,} images in sixty minutes**. One GPU at "
            f"{inp.gpu_images_per_hour:,.0f} images/hour needs **{peak['hours_for_one_gpu']} hours** for "
            f"that. Absorbing it within the hour takes **{peak['gpus_to_absorb_in_one_hour']} GPUs at once**.",
            "",
            "A hosted API absorbs that invisibly — elasticity sits with the vendor and is priced in. "
            "Self-hosting answers a burst either by paying for idle capacity or by delaying the queue.",
        ]
        if inp.busy_hours_pct is not None:
            lines.append(
                f"With work in only {inp.busy_hours_pct}% of hours, capacity sized for the peak "
                f"would sit idle over {100 - inp.busy_hours_pct:.0f}% of the time."
            )

    table = hosting_table(inp)
    if table:
        lines.append(table)

    if biggest:
        s = biggest["saving_year"]
        lines += ["", "## Verdict", ""]
        if biggest["per_month"] >= be:
            lines.append(
                f"At {biggest['per_month']:,} images/month self-hosting saves **{_money(s)}/year** "
                "and is past break-even even for a GPU that never sleeps. Worth building."
            )
        elif s > 5000:
            lines.append(
                f"At {biggest['per_month']:,} images/month self-hosting saves **{_money(s)}/year**. "
                "That justifies the engineering — provided the burst above has an answer."
            )
        else:
            lines.append(
                f"At {biggest['per_month']:,} images/month self-hosting saves **{_money(s)}/year** — less "
                "than a week of engineering time, before counting maintenance and the burst problem. "
                f"**Not yet.** Revisit when volume approaches {be * 0.5:,.0f}/month."
            )
        if inp.api_cost_per_image_optimized and biggest["optimized_year"]:
            delta = biggest["api_year"] - biggest["optimized_year"]
            lines.append("")
            lines.append(
                f"Cheaper API settings save {_money(delta)}/year at the same volume — "
                f"{'more' if delta > s else 'less'} than switching models, and without new "
                "infrastructure. Verify quality first: fewer, larger calls can change the answers."
            )
    if currency_note:
        lines += ["", f"_{currency_note}_"]
    return "\n".join(lines) + "\n"
