"""Count processed images per month (READ-ONLY) — the number that decides self-host vs pay-per-call.

Run through the wrapper so the source app's environment is loaded:

    python scripts/run_source_manage.py shell < scripts/count_volume.py

Prints, per calendar month: completed image jobs, failed jobs, distinct listings, and the
indoor/outdoor split where it can be inferred (outdoor images ask fewer questions and cost less).
Nothing is written anywhere.
"""

from collections import Counter, defaultdict
from datetime import timedelta

from computer_vision.models import ImageProcessingJob, PropertyProcessingJob
from django.db.models import Count
from django.db.models.functions import TruncMonth
from django.utils import timezone

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

# --- насколько нагрузка неравномерна -------------------------------------------------------------
# Для схемы «поднимаем GPU под нагрузку» важен не средний объём, а пики: сколько изображений
# приходит в самый загруженный час и какая доля часов вообще пустая.
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

print("\nсамые загруженные дни:")
for r in days:
    print(f"  {r['d'].date()}  {r['n']:,} изображений")
print("\nсамые загруженные часы:")
for r in hours:
    print(f"  {r['h']:%Y-%m-%d %H:00}  {r['n']:,} изображений")
print(f"\nчасов с хотя бы одной задачей: {busy_hours:,} из ~{span_hours:,} ({100 * busy_hours / span_hours:.1f}%)")
print("(чем ниже эта доля, тем выгоднее поднимать GPU только под нагрузку)")
