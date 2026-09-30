# Structured research data

This directory is the machine-readable source for the deblurring resource catalog. The historical 2019–2026 migration is complete; the public README can now be generated from this structured layer in the next phase.

## Files

- `taxonomy.yaml` — controlled vocabulary for tasks, input signals/sensors, and method families.
- `papers.yaml` — the initial structured catalog containing the 2025–2026 records.
- `papers-YYYY.yaml` — year-sharded historical paper catalogs for 2019–2024.
- `migration.yaml` — authoritative migration state.
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

## Migration status

`migration.yaml` is authoritative. The historical migration covered by the current repository is complete:

```yaml
complete_years: [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
pending_years: []
```

Every complete year must have structured paper records and pass global validation. New years should be added directly to the structured catalog rather than maintained only in the public README.

The next phase is to generate the public README from this data so the structured catalog becomes the single source of truth for both content and presentation.

## Validation

Run:

```bash
python -m pip install pyyaml
python scripts/validate_data.py
```

The validator checks required fields, unique paper/dataset IDs across all paper files, duplicate normalized paper titles, taxonomy values, dataset references, URL shape, and migration-year consistency.
