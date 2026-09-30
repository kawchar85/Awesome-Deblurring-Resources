#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def check_vocab(values, allowed, field, owner, errors):
    if not isinstance(values, list):
        errors.append(f"{owner}: {field} must be a list")
        return
    for value in values:
        if value not in allowed:
            errors.append(f"{owner}: unknown {field} value {value!r}")


def main() -> int:
    errors: list[str] = []

    taxonomy = load_yaml(DATA / "taxonomy.yaml")
    papers_doc = load_yaml(DATA / "papers.yaml")
    datasets_doc = load_yaml(DATA / "datasets.yaml")

    allowed_tasks = set(taxonomy["tasks"])
    allowed_signals = set(taxonomy["signals"])
    allowed_methods = set(taxonomy["methods"])

    datasets = datasets_doc.get("datasets", [])
    dataset_ids: set[str] = set()

    for dataset in datasets:
        owner = f"dataset {dataset.get('id', '<missing-id>')}"
        for field in ("id", "name", "url", "description", "tasks", "signals", "capture"):
            if field not in dataset:
                errors.append(f"{owner}: missing required field {field!r}")

        dataset_id = dataset.get("id")
        if dataset_id:
            if dataset_id in dataset_ids:
                errors.append(f"duplicate dataset id: {dataset_id}")
            dataset_ids.add(dataset_id)

        if dataset.get("url") and not valid_url(dataset["url"]):
            errors.append(f"{owner}: invalid url {dataset['url']!r}")

        check_vocab(dataset.get("tasks"), allowed_tasks, "tasks", owner, errors)
        check_vocab(dataset.get("signals"), allowed_signals, "signals", owner, errors)

    papers = papers_doc.get("papers", [])
    paper_ids: set[str] = set()
    normalized_titles: dict[str, str] = {}

    complete_years = set(papers_doc.get("migration", {}).get("complete_years", []))
    pending_years = set(papers_doc.get("migration", {}).get("pending_years", []))
    overlap = complete_years & pending_years
    if overlap:
        errors.append(f"migration years appear in both complete and pending: {sorted(overlap)}")

    for paper in papers:
        owner = f"paper {paper.get('id', '<missing-id>')}"
        for field in ("id", "year", "venue", "title", "paper_url", "tasks", "signals", "methods", "datasets"):
            if field not in paper:
                errors.append(f"{owner}: missing required field {field!r}")

        paper_id = paper.get("id")
        if paper_id:
            if paper_id in paper_ids:
                errors.append(f"duplicate paper id: {paper_id}")
            paper_ids.add(paper_id)

        title = paper.get("title")
        if title:
            normalized = normalize_title(title)
            if normalized in normalized_titles:
                errors.append(
                    f"duplicate normalized paper title: {title!r} and {normalized_titles[normalized]!r}"
                )
            else:
                normalized_titles[normalized] = title

        year = paper.get("year")
        if year is not None and year not in complete_years:
            errors.append(f"{owner}: year {year} is not listed in migration.complete_years")

        for url_field in ("paper_url", "code_url", "project_url"):
            value = paper.get(url_field)
            if value and not valid_url(value):
                errors.append(f"{owner}: invalid {url_field} {value!r}")

        check_vocab(paper.get("tasks"), allowed_tasks, "tasks", owner, errors)
        check_vocab(paper.get("signals"), allowed_signals, "signals", owner, errors)
        check_vocab(paper.get("methods"), allowed_methods, "methods", owner, errors)

        refs = paper.get("datasets")
        if not isinstance(refs, list):
            errors.append(f"{owner}: datasets must be a list")
        else:
            for ref in refs:
                if ref not in dataset_ids:
                    errors.append(f"{owner}: references unknown dataset {ref!r}")

    if errors:
        print("Structured-data validation failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(papers)} papers across {sorted(complete_years)}; "
        f"{len(datasets)} datasets; taxonomy references are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
