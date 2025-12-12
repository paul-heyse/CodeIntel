"""Dataset conformance validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import jsonschema

from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.validation.contract import (
    _schema_path,
    collect_contract_issues,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from duckdb import DuckDBPyConnection

    from codeintel.storage.datasets.registry import DatasetRegistry

__all__ = [
    "ConformanceIssue",
    "ConformanceReport",
    "run_conformance",
]


@dataclass(frozen=True)
class ConformanceIssue:
    """Single contract violation discovered during conformance checks."""

    dataset: str | None
    message: str


@dataclass(frozen=True)
class ConformanceReport:
    """Collection of contract issues discovered in a run."""

    issues: list[ConformanceIssue]

    @property
    def ok(self) -> bool:
        """True when no issues were found."""
        return not self.issues


def _validate_schema_rows(
    con: DuckDBPyConnection,
    registry: DatasetRegistry,
    *,
    schema_base_dir: Path,
    sample_size: int = 50,
) -> Iterable[ConformanceIssue]:
    """Validate sampled rows against JSON Schemas when available.

    Yields
    ------
    ConformanceIssue
        Issues discovered while validating sampled rows.
    """
    for name, ds in registry.by_name.items():
        if ds.json_schema_id is None:
            continue
        schema_file = _schema_path(ds.json_schema_id, base_dir=schema_base_dir)
        if not schema_file.exists():
            continue
        if ds.schema is None:
            continue
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        try:
            rows = con.table(ds.table_key).limit(sample_size).fetchall()
        except duckdb.Error as exc:
            yield ConformanceIssue(dataset=name, message=f"Failed to sample rows: {exc}")
            continue
        columns = [col.name for col in ds.schema.columns if col.name is not None]
        for idx, row in enumerate(rows):
            record = dict(zip(columns, row, strict=True))
            errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
            if errors:
                yield ConformanceIssue(
                    dataset=name,
                    message=f"Row {idx} failed JSON Schema validation: {errors[0].message}",
                )


def run_conformance(
    con: DuckDBPyConnection,
    *,
    schema_base_dir: Path,
    sample_rows: bool = False,
    sample_size: int = 50,
) -> ConformanceReport:
    """Run contract conformance checks and optionally sample row validation.

    Returns
    -------
    ConformanceReport
        Aggregated contract issues discovered during validation.
    """
    registry = load_dataset_registry(con)
    issues: list[ConformanceIssue] = [
        ConformanceIssue(dataset=None, message=msg)
        for msg in collect_contract_issues(con, schema_base_dir=schema_base_dir)
    ]
    if sample_rows:
        issues.extend(
            _validate_schema_rows(
                con,
                registry,
                schema_base_dir=schema_base_dir,
                sample_size=sample_size,
            )
        )
    return ConformanceReport(issues=issues)
