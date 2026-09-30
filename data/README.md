# Structured research data

This directory is the machine-readable layer for the resource list. The public README is still unchanged during the migration phase.

## Files

- `taxonomy.yaml` — controlled vocabulary for tasks, input signals/sensors, and method families.
- `papers.yaml` — structured paper records. The `migration` block states which years are complete.
- `datasets.yaml` — structured deblurring dataset/benchmark records.

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

The migration is incremental. A year listed in `papers.yaml -> migration.complete_years` is considered fully represented in structured data. Years in `pending_years` continue to use the README as their authoritative public listing until migrated and reviewed.

Do not generate the public README entirely from `papers.yaml` until all historical years have moved from `pending_years` to `complete_years`.

## Validation

Run:

```bash
python -m pip install pyyaml
python scripts/validate_data.py
```

The validator checks required fields, unique paper/dataset IDs, duplicate normalized paper titles, taxonomy values, dataset references, and URL shape.
