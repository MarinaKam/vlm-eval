# Economics

Quality decides whether you can switch. Whether you should is arithmetic: a paid API charges per
image, a GPU charges per hour, and below some volume the GPU is mostly idle and costs more. The
arithmetic is simple; the inputs are what matter, and each one is measured by a command.

```
volume ─┐
cost ───┴─► data/economics.json ─► economics ─► reports/economics.md
```

`volume` and `cost` run inside a source application and follow the same template caveat as the dataset
export ([datasets.md](datasets.md#the-export-templates)). If they do not fit your stack, obtain the
same numbers any other way and write `data/economics.json` by hand.

## How much do you process?

```bash
vlm-eval volume
vlm-eval volume --db-from /path/to/other-deployment.env
```

Prints completed and failed image jobs per calendar month over the last twelve months, the busiest
days and hours, the share of hours that had any work, and the size distribution of one listing (a
group of images submitted together). The share of busy hours decides whether per-hour capacity can be
scaled to zero between bursts; the busiest hour decides how much capacity a burst needs.

The script first switches its database session to read-only at the Postgres level, so an accidental
write fails with a database error. `--db-from` takes only `DATABASE_URL` from the named env file and
keeps the rest of the local environment, which is the way to read another deployment's database
without loading its other settings.

## Is the batch size justified?

A pipeline that splits its questions into batches re-sends the image with every batch, and that is
usually the dominant cost. `cost` measures what a batch size costs and what a larger one changes:

```bash
vlm-eval cost --chunks 15 47 --images 60
```

`--chunks` is questions per API call, one or more values to compare; `--images` is how many images to
measure on (default 60); `--type` selects `indoor` or `outdoor` images (default `indoor`); `--refresh`
re-measures instead of reusing an existing result file.

For each batch size the command prints API calls per image, input tokens and price per image, with the
change relative to the first size. With more than one size it also prints how the answers differ:
the share of images with identical tag sets, tag agreement, and which tags were lost or gained. A
cheaper batch size is not free if it drops tags. Part of any difference is the API answering
differently on a re-run, so measure that baseline by running the same size twice
(`vlm-eval cost --chunks 15 --refresh`) before attributing a difference to the batch size.

Do not assume the result carries over between models. How much the answers change when all questions
go into one call depends on the model, and it has to be measured on the model you use.

### The cost command contract

`cost` does not call the API itself. It runs a management command in your application, named by
`VLM_EVAL_COST_COMMAND`, once per batch size:

```
manage.py <command> --urls-file <file> --out <csv> --no-gpu --assume-type <indoor|outdoor> --chunk-size <N>
```

The URL file has one `url,image_id` line per image and is built from the manifest. The command must
write a CSV with these columns:

| Column | Content |
|---|---|
| `image_id` (or `url`) | which image the row is about |
| `classification_tags` | the tags the API returned, separated by `;` |
| `cost_2_5` | price of the image in US dollars |
| `n_calls` | API calls made for the image |
| `prompt_tokens` | input tokens for the image |

The column names are fixed by the reader in `vlm_eval/cli.py`. Results are kept as
`data/cost_chunk<N>.csv` and listed by `vlm-eval status`.

## Is switching worth it?

```bash
vlm-eval economics                  # reads data/economics.json, writes reports/economics.md
vlm-eval economics --config path/to/other.json
```

Run without a config file, the command prints an example. The config lists **options**, each billed
per image or per hour, and names the one you run today as `current`:

```json
{
  "current": "paid API",
  "options": [
    {"name": "paid API", "kind": "per_image", "price": 0.002},
    {"name": "GPU, autoscaled", "kind": "per_hour", "price": 0.60,
     "throughput_per_hour": 2000, "cold_start_min": 3},
    {"name": "GPU, always on", "kind": "per_hour", "price": 0.60,
     "throughput_per_hour": 2000, "always_on": true, "fixed_monthly": 70,
     "fixed_note": "cluster management fee"}
  ],
  "scenarios": [["last 3 months", 20000], ["projected", 60000]],
  "peak_hour_images": 5000,
  "busy_hours_pct": 10
}
```

All numbers above are placeholders. Replace every one with a measurement.

| Key | Meaning |
|---|---|
| `current` | name of the option in use today; defaults to the first option |
| `options[].kind` | `per_image` or `per_hour` |
| `options[].price` | US dollars per image, or per hour |
| `options[].throughput_per_hour` | images per hour; required for `per_hour` |
| `options[].always_on` | `true` if the capacity cannot be scaled down between bursts |
| `options[].cold_start_min` | minutes to come up from cold, shown in the report |
| `options[].fixed_monthly`, `fixed_note` | a cost charged every month whatever the volume, and what it is |
| `options[].note` | free text shown in the options table |
| `scenarios` | `[label, images per month]` pairs |
| `peak_hour_images` | images in the busiest hour, from `volume` |
| `busy_hours_pct` | share of hours with any work, from `volume` |
| `weights_gb`, `storage_usd_per_gb_month` | optional: cost of keeping model weights in object storage |
| `note` | optional text appended to the report |

The report contains the options and how each is billed, the cost per year of each option in each
scenario, what each alternative saves or costs against the current option at the largest scenario, the
monthly volume at which a per-image option and a per-hour option cost the same where such a volume
exists, and what
the busiest hour demands of per-hour capacity.

Points that are easy to get wrong:

- **Per-hour options need a throughput.** There is no other way to turn a monthly volume into hours.
  Use a figure measured on the hardware you would deploy on (`vlm-eval perf`), not one from a laptop.
- **Fixed monthly costs.** A cluster management fee, a reserved disk or a persistent endpoint does not
  appear in the hourly price, and on a small workload it can be most of the bill. It is worth listing
  "capacity in a cluster already paid for" and "a cluster stood up for this" as two options.
- **An autoscaled per-hour option with no fixed cost does not cross a per-image one.** Both scale
  with volume, so the ratio between them holds at any size and growth is not an argument for
  switching. A break-even volume exists when one side carries a constant monthly cost: an always-on
  machine, or a fixed fee.
- **The peak matters more than the average.** A vendor absorbs a burst invisibly. Your own hardware
  answers it by paying for idle capacity or by making the queue wait.

### Refusing placeholder inputs

`economics` refuses to render when its inputs were never measured: when there are no volume scenarios,
when `peak_hour_images` is missing, or when the current per-image price is still exactly `0.001`. A
confident report built on placeholders is worse than no report. `--allow-unmeasured` renders anyway and
puts a warning at the top of the file.

The check recognises only those three cases. It cannot tell whether the other numbers were measured.

## Symmetry

The calculation has no built-in direction. Moving off a paid API and moving onto one are the same
arithmetic with a different option marked `current`. A config in the older one-directional shape
(`api_cost_per_image`, `hosting`, `gpu_usd_per_hour`, `gpu_images_per_hour`) is still read.
