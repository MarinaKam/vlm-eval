"""Which way of running the pipeline costs less, and from what volume.

There is no "the API" and "self-hosting" here — only **options**, each billed one of two ways:

* **per image** — a hosted API, or anything charged per unit of work. Cost scales with volume and never
  idles.
* **per hour** — a GPU you rent, a pod in a cluster, a machine under a desk. Cost scales with *time*,
  so below some volume it is paying to idle, and above it the per-image option is paying a margin.

That framing is symmetric on purpose. Moving off a paid API and moving onto one are the same
calculation read in opposite directions, and a tool that hardcodes the direction quietly assumes the
answer. Mark whichever option you run today as `current`; everything else is compared against it.

The arithmetic is trivial. The measurements are the whole point: volume from `vlm-eval volume`, price
per image from `vlm-eval cost`, throughput from a real run on the hardware you would actually use.
"""

from dataclasses import dataclass, field
from typing import Any

HOURS_PER_MONTH = 730


@dataclass(frozen=True)
class Option:
    """One way of running the pipeline.

    `kind="per_image"` needs `price` only. `kind="per_hour"` needs `price` and `throughput_per_hour`,
    plus `always_on=True` if it cannot be scaled down between bursts (an interactive endpoint, or an
    autoscaler your queue cannot afford to wait for).
    """

    name: str
    kind: str  # "per_image" | "per_hour"
    price: float
    throughput_per_hour: float | None = None
    always_on: bool = False
    cold_start_min: float | None = None
    # Charged every month whatever the volume: a cluster control-plane fee, a reserved disk, a
    # persistent endpoint. Easy to forget because it does not appear on the per-hour price, and on a
    # small workload it can be most of the bill.
    fixed_monthly: float = 0.0
    fixed_note: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("per_image", "per_hour"):
            raise ValueError(f"{self.name}: kind must be per_image or per_hour, not {self.kind!r}")
        if self.kind == "per_hour" and not self.throughput_per_hour:
            raise ValueError(
                f"{self.name}: a per-hour option needs throughput_per_hour — without it there is no way "
                "to turn a monthly volume into hours, and any cost figure would be invented"
            )

    def monthly_cost(self, images_per_month: float) -> float:
        return self.fixed_monthly + self.variable_monthly(images_per_month)

    def variable_monthly(self, images_per_month: float) -> float:
        if self.kind == "per_image":
            return images_per_month * self.price
        if self.always_on:
            return HOURS_PER_MONTH * self.price
        hours = images_per_month / float(self.throughput_per_hour)
        return min(hours, HOURS_PER_MONTH) * self.price


@dataclass(frozen=True)
class Inputs:
    options: list[Option]
    current: str  # the option in use today
    scenarios: list[tuple[str, int]] = field(default_factory=list)
    peak_hour_images: int | None = None
    busy_hours_pct: float | None = None
    weights_gb: float | None = None
    storage_usd_per_gb_month: float = 0.02

    def __post_init__(self) -> None:
        names = [o.name for o in self.options]
        if self.current not in names:
            raise ValueError(f"current={self.current!r} is not one of the options: {names}")

    @property
    def current_option(self) -> Option:
        return next(o for o in self.options if o.name == self.current)

    @property
    def alternatives(self) -> list[Option]:
        return [o for o in self.options if o.name != self.current]


def crossover(current: Option, other: Option) -> float | None:
    """Monthly volume at which the two cost the same, if they ever do.

    Two options billed the same way never cross — one is simply cheaper per image at every volume. A
    per-image option and an *always-on* per-hour one cross where the fixed monthly cost is covered. An
    autoscaled per-hour option scales with volume just like a per-image one, so it does not cross
    either: worth knowing, because it means "wait until we grow" is not an argument for it.
    """
    per_image = [o for o in (current, other) if o.kind == "per_image"]
    hourly = [o for o in (current, other) if o.kind == "per_hour"]
    if len(per_image) != 1 or len(hourly) != 1:
        return None
    a, b = per_image[0], hourly[0]
    # cost_a(v) = fixed_a + price_a*v ; cost_b(v) = fixed_b + (price_b/thr)*v when autoscaled,
    # or fixed_b + 730*price_b when always on. They cross where the slopes differ.
    slope_a = a.price
    slope_b = 0.0 if b.always_on else b.price / float(b.throughput_per_hour)
    const_a = a.fixed_monthly
    const_b = b.fixed_monthly + (HOURS_PER_MONTH * b.price if b.always_on else 0.0)
    if slope_a == slope_b:
        return None
    v = (const_b - const_a) / (slope_a - slope_b)
    return v if v > 0 else None


def check_measured(inp: Inputs) -> list[str]:
    """Inputs that are still placeholders rather than measurements."""
    problems = []
    if not inp.scenarios:
        problems.append("no volume scenarios — run `vlm-eval volume` and put real numbers in the config")
    if inp.current_option.kind == "per_image" and inp.current_option.price == 0.001:
        problems.append("the current option's price is the example value — run `vlm-eval cost --chunks <N>`")
    if inp.peak_hour_images is None:
        problems.append("peak_hour_images is missing — `vlm-eval volume` prints the busiest hour")
    return problems


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:,.0f}"


def _billing(o: Option) -> str:
    if o.kind == "per_image":
        base = f"${o.price:.6f}/image"
    else:
        base = f"${o.price:.4f}/hour" + (
            " (always on)" if o.always_on else f", {o.throughput_per_hour:,.0f} images/hour"
        )
    if o.fixed_monthly:
        base += f" + ${o.fixed_monthly:,.2f}/month fixed"
        if o.fixed_note:
            base += f" ({o.fixed_note})"
    return base


def render(inp: Inputs, *, currency_note: str = "") -> str:
    current = inp.current_option
    lines = [
        "# What it costs to run this, and which way is cheaper",
        "",
        f"Today: **{current.name}** — {_billing(current)}.",
        "",
        "Options billed per image scale with volume and never idle. Options billed per hour scale with",
        "time, so below some volume they pay to idle — and above it, the per-image option pays a margin.",
        "",
        "## The options",
        "",
        "| option | billed | notes |",
        "|---|---|---|",
    ]
    for o in inp.options:
        mark = " **(current)**" if o.name == inp.current else ""
        extra = o.note
        if o.cold_start_min:
            extra = (extra + "; " if extra else "") + f"~{o.cold_start_min:g} min to come up from cold"
        lines.append(f"| {o.name}{mark} | {_billing(o)} | {extra} |")

    if inp.scenarios:
        lines += [
            "",
            "## Cost per year",
            "",
            "| option | " + " | ".join(f"{lbl} ({v:,}/mo)" for lbl, v in inp.scenarios) + " |",
            "|---|" + "---|" * len(inp.scenarios),
        ]
        for o in inp.options:
            mark = " **(current)**" if o.name == inp.current else ""
            cells = [_money(o.monthly_cost(v) * 12) for _, v in inp.scenarios]
            lines.append(f"| {o.name}{mark} | " + " | ".join(cells) + " |")

        lines += ["", "## Against what you run today", ""]
        biggest_label, biggest = max(inp.scenarios, key=lambda s: s[1])
        for other in inp.alternatives:
            delta = (current.monthly_cost(biggest) - other.monthly_cost(biggest)) * 12
            verb = "saves" if delta > 0 else "costs an extra"
            point = crossover(current, other)
            line = f"- **{other.name}** {verb} {_money(abs(delta))}/year at {biggest:,} images/month ({biggest_label})."
            if point is not None:
                side = "above" if other.kind == "per_hour" else "below"
                line += f" The two are level at **{point:,.0f} images/month**; it wins {side} that."
            elif other.kind == current.kind:
                line += " Both are billed the same way, so the ratio holds at any volume."
            else:
                line += " Both scale with volume, so this ratio holds at any size — growing into it is not an argument."
            lines.append(line)

    if inp.peak_hour_images:
        lines += ["", "## The shape of the traffic, not just the total", ""]
        lines.append(f"The busiest hour carried **{inp.peak_hour_images:,} images**.")
        for o in inp.options:
            if o.kind != "per_hour":
                continue
            hours = inp.peak_hour_images / float(o.throughput_per_hour)
            lines.append(
                f"- **{o.name}** needs {hours:.1f} hours for that, or {max(1, round(hours))} in parallel "
                "to absorb it within the hour."
            )
        lines.append(
            "\nA per-image option absorbs a burst invisibly — the elasticity is the vendor's and it is "
            "priced in. Per-hour capacity answers a burst by paying for idle time or by making the queue "
            "wait."
        )
        if inp.busy_hours_pct is not None:
            lines.append(
                f"With work in only {inp.busy_hours_pct}% of hours, capacity sized for the peak sits idle "
                f"more than {100 - inp.busy_hours_pct:.0f}% of the time."
            )

    if inp.weights_gb is not None:
        cost = inp.weights_gb * inp.storage_usd_per_gb_month
        lines += [
            "",
            f"Keeping {inp.weights_gb:g} GB of weights in object storage adds ${cost:.2f}/month "
            f"(${cost * 12:.2f}/year) — negligible beside compute, but it is what makes a cold start "
            "reproducible.",
        ]

    if currency_note:
        lines += ["", f"_{currency_note}_"]
    return "\n".join(lines) + "\n"


def from_config(cfg: dict[str, Any]) -> tuple[Inputs, str]:
    """Build Inputs from the JSON config, accepting the older one-directional shape.

    The old shape named one side `api_cost_per_image` and the rest `hosting`. It is read as "a
    per-image option, currently in use, plus per-hour alternatives" — which is what it always meant.
    """
    cfg = dict(cfg)
    note = cfg.pop("note", "")
    scenarios = [tuple(x) for x in cfg.pop("scenarios", [])]

    if "options" in cfg:
        options = [Option(**o) for o in cfg.pop("options")]
        current = cfg.pop("current", options[0].name)
    else:
        throughput = cfg.pop("gpu_images_per_hour", 2000)
        options = [Option(name="paid API (per image)", kind="per_image", price=cfg.pop("api_cost_per_image"))]
        current = options[0].name
        cheaper = cfg.pop("api_cost_per_image_optimized", None)
        if cheaper:
            options.append(Option(name="paid API, cheaper settings", kind="per_image", price=cheaper))
        for h in cfg.pop("hosting", []) or []:
            options.append(
                Option(
                    name=h["name"],
                    kind="per_hour",
                    price=h["usd_per_hour"],
                    throughput_per_hour=throughput,
                    always_on=h.get("always_on", False),
                    cold_start_min=h.get("cold_start_min"),
                    note=h.get("note", ""),
                )
            )
        if len(options) == 1:
            options.append(
                Option(
                    name=cfg.pop("gpu_name", "self-hosted GPU"),
                    kind="per_hour",
                    price=cfg.pop("gpu_usd_per_hour", 0.5832),
                    throughput_per_hour=throughput,
                )
            )
        cfg.pop("gpu_name", None)
        cfg.pop("gpu_usd_per_hour", None)

    return Inputs(options=options, current=current, scenarios=scenarios, **cfg), note
