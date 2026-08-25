"""Count processed images per month (READ-ONLY) — the number that decides self-host vs pay-per-call.

Like the export script, this one is written against a particular Django schema: adapt the model names
and field names to your own, or replace it with the equivalent query for your backend.

Run through the wrapper so the source app's environment is loaded:

    python scripts/run_source_manage.py shell < scripts/count_volume.py

Prints, per calendar month: completed image jobs, failed jobs, distinct listings, and the
indoor/outdoor split where it can be inferred (outdoor images ask fewer questions and cost less).
Nothing is written anywhere.
"""

from collections import Counter, defaultdict
from datetime import timedelta

from computer_vision.models import ImageProcessingJob, PropertyProcessingJob
from django.db import connection
from django.db.models import Count
from django.db.models.functions import TruncMonth
from django.utils import timezone

# Postgres-level guard: every transaction in this session becomes read-only, so an accidental write
# fails with an error instead of touching the database. Safe to point at production.
with connection.cursor() as _cur:
    _cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
print("session set to read only")

MONTHS_BACK = 12
since = timezone.now() - timedelta(days=30 * MONTHS_BACK)

rows = (
    ImageProcessingJob.objects.filter(created_at__gte=since)
    .annotate(month=TruncMonth("created_at"))
    .values("month", "status")
    .annotate(n=Count("id"))
    .order_by("month")
)

per_month = defaultdict(Counter)
for r in rows:
    per_month[r["month"].date().replace(day=1)][r["status"]] += r["n"]

props = (
    PropertyProcessingJob.objects.filter(created_at__gte=since)
    .annotate(month=TruncMonth("created_at"))
    .values("month")
    .annotate(n=Count("id"))
)
props_by_month = {p["month"].date().replace(day=1): p["n"] for p in props}

print(f"{'month':<10} {'completed':>10} {'failed':>8} {'other':>7} {'listings':>9}")
print("-" * 48)
totals = Counter()
for month in sorted(per_month):
    c = per_month[month]
    completed = c.get("completed", 0)
    failed = c.get("failed", 0)
    other = sum(v for k, v in c.items() if k not in ("completed", "failed"))
    totals["completed"] += completed
    totals["failed"] += failed
    print(f"{month.isoformat():<10} {completed:>10,} {failed:>8,} {other:>7,} {props_by_month.get(month, 0):>9,}")

months = len(per_month)
if months:
    print("-" * 48)
    print(f"{'TOTAL':<10} {totals['completed']:>10,} {totals['failed']:>8,}")
    print(f"\naverage completed images per month: {totals['completed'] / months:,.0f}  (over {months} months)")
    recent = sorted(per_month)[-3:]
    if recent:
        r_sum = sum(per_month[m].get("completed", 0) for m in recent)
        print(f"average over last {len(recent)} months:        {r_sum / len(recent):,.0f}")

# --- how uneven the load is -------------------------------------------------------------
# For a scale-up-on-demand GPU the average volume is the wrong number: what decides the machine
# count is the busiest hour, and what decides the bill is the share of hours with no work at all.
from django.db.models.functions import TruncDay, TruncHour  # noqa: E402

days = (
    ImageProcessingJob.objects.filter(created_at__gte=since, status="completed")
    .annotate(d=TruncDay("created_at"))
    .values("d")
    .annotate(n=Count("id"))
    .order_by("-n")[:5]
)
hours = (
    ImageProcessingJob.objects.filter(created_at__gte=since, status="completed")
    .annotate(h=TruncHour("created_at"))
    .values("h")
    .annotate(n=Count("id"))
    .order_by("-n")[:5]
)
hour_rows = (
    ImageProcessingJob.objects.filter(created_at__gte=since, status="completed")
    .annotate(h=TruncHour("created_at"))
    .values("h")
    .annotate(n=Count("id"))
)
busy_hours = len(hour_rows)
span_hours = MONTHS_BACK * 30 * 24

print("\nbusiest days:")
for r in days:
    print(f"  {r['d'].date()}  {r['n']:,} images")
print("\nbusiest hours:")
for r in hours:
    print(f"  {r['h']:%Y-%m-%d %H:00}  {r['n']:,} images")
print(f"\nhours with at least one job: {busy_hours:,} of ~{span_hours:,} ({100 * busy_hours / span_hours:.1f}%)")
print("(the lower that share, the more a scale-to-zero GPU beats one that is always on)")

# --- the shape of one upload ----------------------------------------------------------------------
# Monthly and hourly totals say how much work arrives; they do not say how it arrives. A client who
# uploads one listing of 40 photographs and a client who uploads 40 listings of one are the same point
# on the hourly chart and completely different problems for a queue. What a person waiting on a result
# experiences is the size of their own upload, not the monthly average — so measure that distribution
# and let the report answer "what if someone sends 500 at once" with a number instead of a shrug.
print("\n=== the shape of one upload ===")

sizes = sorted(
    PropertyProcessingJob.objects.filter(created_at__gte=since)
    .annotate(n=Count("images"))
    .filter(n__gt=0)
    .values_list("n", flat=True)
)
if sizes:

    def pct(p: float) -> int:
        return sizes[min(len(sizes) - 1, int(p * len(sizes)))]

    print(f"listings measured: {len(sizes):,}   images in them: {sum(sizes):,}")
    print(f"  images per listing   median {pct(0.5)}   p90 {pct(0.9)}   p99 {pct(0.99)}   max {max(sizes)}")
    buckets = Counter()
    for n in sizes:
        buckets["1" if n == 1 else "2-9" if n < 10 else "10-24" if n < 25 else "25-49" if n < 50 else "50+"] += 1
    print("  distribution:")
    for label in ("1", "2-9", "10-24", "25-49", "50+"):
        n = buckets.get(label, 0)
        print(f"    {label:>6} images  {n:6,} listings ({100 * n / len(sizes):5.1f}%)")
else:
    print("  no listings in the window — nothing to measure")

# How many uploads land in the same hour: one 500-image listing needs a fast worker, five hundred
# 1-image listings arriving together need a deep queue. They size differently.
per_hour = (
    PropertyProcessingJob.objects.filter(created_at__gte=since)
    .annotate(h=TruncHour("created_at"))
    .values("h")
    .annotate(n=Count("id"))
    .order_by("-n")[:5]
)
print("\n  busiest hours by number of uploads (not images):")
for r in per_hour:
    print(f"    {r['h']:%Y-%m-%d %H:00}  {r['n']:,} listings")
