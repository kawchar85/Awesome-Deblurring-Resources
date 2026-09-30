#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_papers() -> list[dict]:
    paths = [DATA / "papers.yaml", *sorted(DATA.glob("papers-*.yaml"))]
    papers: list[dict] = []
    for path in paths:
        doc = load_yaml(path) or {}
        papers.extend(doc.get("papers", []))
    return papers


def label(value: str) -> str:
    special = {
        "3d-reconstruction": "3D Reconstruction",
        "rgb": "RGB",
        "rgb-d": "RGB-D",
        "gan": "GAN",
        "cnn": "CNN",
        "raw": "RAW",
    }
    return special.get(value, value.replace("-", " ").title())


def md_link(text: str, url: str | None) -> str:
    return f"[{text}]({url})" if url else text


def resource_link(paper: dict) -> str:
    if paper.get("code_url"):
        return md_link("Code", paper["code_url"])
    if paper.get("project_url"):
        return md_link("Project", paper["project_url"])
    return "—"


def tags(values: list[str]) -> str:
    return " · ".join(f"`{value}`" for value in values) if values else "—"


def write_readme_preview(out: Path, papers: list[dict], datasets: list[dict], migration: dict) -> None:
    years = sorted({p["year"] for p in papers}, reverse=True)
    year_counts = Counter(p["year"] for p in papers)
    task_counts = Counter(task for p in papers for task in p.get("tasks", []))
    signal_counts = Counter(signal for p in papers for signal in p.get("signals", []))
    method_counts = Counter(method for p in papers for method in p.get("methods", []))

    lines: list[str] = [
        "# Awesome Image & Video Deblurring",
        "",
        "A curated, structured collection of research papers, implementations, datasets, and benchmarks for image and video deblurring.",
        "",
        f"**Coverage:** {min(years)}–{max(years)} · **Papers:** {len(papers)} · **Datasets:** {len(datasets)}",
        "",
        "> This file is generated from the structured research data. Edit the YAML catalog, not generated tables.",
        "",
        "## Browse",
        "",
        "- [By task](docs/by-task.md) — motion, defocus, blind, video, blur synthesis, 3D reconstruction, and related problems.",
        "- [By signal / sensor](docs/by-signal.md) — events, gyro, dual/quad pixel, spike, stereo, RAW, and more.",
        "- [By method](docs/by-method.md) — diffusion, transformers, state-space models, kernel estimation, Gaussian splatting, and more.",
        "- [Datasets & benchmarks](#datasets--benchmarks)",
        "",
        "### By year",
        "",
        " | ".join(f"[{year}](#{year}-papers) ({year_counts[year]})" for year in years),
        "",
        "### Research map",
        "",
        "| Dimension | Most represented categories | Full index |",
        "|---|---|---|",
        f"| Tasks | {', '.join(f'{label(k)} ({v})' for k, v in task_counts.most_common(6))} | [Browse tasks](docs/by-task.md) |",
        f"| Signals | {', '.join(f'{label(k)} ({v})' for k, v in signal_counts.most_common(6))} | [Browse signals](docs/by-signal.md) |",
        f"| Methods | {', '.join(f'{label(k)} ({v})' for k, v in method_counts.most_common(6))} | [Browse methods](docs/by-method.md) |",
        "",
    ]

    for year in years:
        lines += [
            f"## {year} Papers",
            "",
            "| Venue | Paper | Task | Resource |",
            "|---|---|---|---|",
        ]
        for paper in [p for p in papers if p["year"] == year]:
            lines.append(
                f"| {paper['venue']} | {md_link(paper['title'], paper['paper_url'])} | "
                f"{tags(paper.get('tasks', []))} | {resource_link(paper)} |"
            )
        lines.append("")

    lines += [
        "## Datasets & Benchmarks",
        "",
        "| Dataset | Focus | Signals | Capture | Link |",
        "|---|---|---|---|---|",
    ]
    for dataset in datasets:
        lines.append(
            f"| {dataset['name']} | {tags(dataset.get('tasks', []))} | {tags(dataset.get('signals', []))} | "
            f"{dataset.get('capture', '—')} | {md_link('Resource', dataset['url'])} |"
        )

    lines += [
        "",
        "## Data & Maintenance",
        "",
        f"Structured migration: **{', '.join(map(str, migration.get('complete_years', [])))}** complete; "
        f"**{len(migration.get('pending_years', []))}** pending years.",
        "",
        "- Research data: [`data/`](data/)",
        "- Taxonomy: [`data/taxonomy.yaml`](data/taxonomy.yaml)",
        "- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)",
        "- Validation: `python scripts/validate_data.py`",
        "- Generation: `python scripts/generate_catalog.py --output-dir .`",
        "",
        "## Contributing",
        "",
        "Contributions are welcome for missing papers, official code/project links, datasets, and corrections. Please follow the contribution policy and prefer primary sources.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")


def write_index(out: Path, title: str, dimension: str, papers: list[dict], taxonomy: dict) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for paper in papers:
        for value in paper.get(dimension, []):
            groups[value].append(paper)

    vocabulary = taxonomy.get(dimension, {})
    lines = [
        f"# {title}",
        "",
        "Generated from the structured research catalog.",
        "",
        "## Index",
        "",
    ]
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        lines.append(f"- [{label(key)}](#{key}) — {len(groups[key])} papers")

    for key in sorted(groups):
        lines += ["", f"## {label(key)}", ""]
        description = vocabulary.get(key)
        if description:
            lines += [description, ""]
        lines += ["| Year | Venue | Paper | Resource |", "|---:|---|---|---|"]
        for paper in sorted(groups[key], key=lambda p: (-p["year"], p["venue"], p["title"].lower())):
            lines.append(
                f"| {paper['year']} | {paper['venue']} | {md_link(paper['title'], paper['paper_url'])} | {resource_link(paper)} |"
            )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate navigable Markdown views from structured deblurring research data.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "generated-preview")
    args = parser.parse_args()

    papers = load_papers()
    datasets = (load_yaml(DATA / "datasets.yaml") or {}).get("datasets", [])
    taxonomy = load_yaml(DATA / "taxonomy.yaml") or {}
    migration = load_yaml(DATA / "migration.yaml") or {}

    output = args.output_dir
    docs = output / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    write_readme_preview(output / "README.md", papers, datasets, migration)
    write_index(docs / "by-task.md", "Papers by Task", "tasks", papers, taxonomy)
    write_index(docs / "by-signal.md", "Papers by Signal / Sensor", "signals", papers, taxonomy)
    write_index(docs / "by-method.md", "Papers by Method", "methods", papers, taxonomy)

    print(
        f"Generated catalog views for {len(papers)} papers and {len(datasets)} datasets "
        f"covering {min(p['year'] for p in papers)}–{max(p['year'] for p in papers)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
