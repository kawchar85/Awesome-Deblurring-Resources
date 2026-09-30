# Structured research data

This directory is the machine-readable layer for the resource list. The public README is still unchanged during the migration phase.

## Files

- `taxonomy.yaml` — controlled vocabulary for tasks, input signals/sensors, and method families.
- `papers.yaml` — initial structured paper catalog (currently 2025–2026).
- `papers-YYYY.yaml` — year-sharded paper catalogs used as historical years are migrated.
- `migration.yaml` — authoritative migration state for complete vs pending years.
- `datasets.yaml` — structured deblurring dataset/benchmark records.

The validator aggregates `papers.yaml` and every `papers-*.yaml` file into one logical paper catalog. Year sharding is only a storage/maintenance choice; task/signal/method views remain taxonomy-driven and should not duplicate paper records.

## Paper schema

Each paper record has:

```yaml
- id: 2025-example-paper
  year: 2025
  venue: CVPR
  title: "Example Paper"
  paper_url: "https://..."
  code_url: "https://..."      # optional
  project_url: "https://..."   # optional
  tasks: [motion-deblurring]
  signals: [rgb, events]
  methods: [transformer]
  datasets: [ExampleDataset]
```

### Why three classification dimensions?

`tasks`, `signals`, and `methods` answer different questions:

- **tasks** — what restoration/reconstruction problem is solved?
- **signals** — what inputs or sensors are used?
- **methods** — what technical family is central to the solution?

For example, an event-guided diffusion model for motion deblurring is represented as:

```yaml
tasks: [motion-deblurring]
signals: [rgb, events]
methods: [diffusion]
```

This is more stable than putting `event-deblurring`, `diffusion-deblurring`, and `motion-deblurring` into one flat tag namespace.

## Dataset schema

```yaml
- id: QPDD
  name: QPDD
  url: "https://..."
  description: "..."
  tasks: [defocus-deblurring]
  signals: [quad-pixel]
  capture: real
```

`capture` currently uses a lightweight descriptive vocabulary such as `real`, `synthetic`, `mixed`, `high-fps-video`, or `synthetic-from-high-fps`. It is intentionally not part of the controlled research taxonomy yet.

## Migration policy

The migration is incremental. `migration.yaml` is authoritative:

```yaml
complete_years: [2024, 2025, 2026]
pending_years: [2019, 2020, 2021, 2022, 2023]
```

A year in `complete_years` must have structured paper records. Years in `pending_years` continue to use the public README as their authoritative listing until migrated and reviewed.

Do not generate the public README entirely from structured data until all historical years have moved from `pending_years` to `complete_years`.

## Validation

Run:

```bash
python -m pip install pyyaml
python scripts/validate_data.py
```

The validator checks required fields, unique paper/dataset IDs across all paper files, duplicate normalized paper titles, taxonomy values, dataset references, URL shape, and migration-year consistency.
