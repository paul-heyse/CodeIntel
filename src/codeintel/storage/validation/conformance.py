"""Dataset conformance validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import jsonschema

from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.validation.contract import collect_contract_issues

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        """True when no issues were found.

        Returns
        -------
        bool
            Whether no issues were found.
        """
        return not self.issues


def _get_generated_schema(dataset_name: str) -> dict[str, object] | None:
    """Get a generated JSON Schema for the dataset name.

    Parameters
    ----------
    dataset_name
        Dataset name.

    Returns
    -------
    dict[str, object] | None
        Generated JSON Schema, or None if not available.
    """
    try:
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            get_json_schema_for_dataset_name,
        )

        return get_json_schema_for_dataset_name(dataset_name)
    except Exception:  # noqa: BLE001
        return None


def _validate_schema_rows(
    con: DuckDBPyConnection,
    registry: DatasetRegistry,
    *,
    sample_size: int = 50,
) -> Iterable[ConformanceIssue]:
    """Validate sampled rows against JSON Schemas when available.

    Parameters
    ----------
    con
        DuckDB connection.
    registry
        Dataset registry.
    sample_size
        Number of rows to sample per table.

    Yields
    ------
    ConformanceIssue
        Issues discovered while validating sampled rows.
    """
    for name, ds in registry.by_name.items():
        if ds.json_schema_id is None:
            continue
        schema = _get_generated_schema(name)
        if schema is None:
            continue
        if ds.schema is None:
            continue
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
    sample_rows: bool = False,
    sample_size: int = 50,
) -> ConformanceReport:
    """Run contract conformance checks and optionally sample row validation.

    Parameters
    ----------
    con
        DuckDB connection.
    sample_rows
        Whether to validate sampled rows.
    sample_size
        Number of rows to sample per table.

    Returns
    -------
    ConformanceReport
        Aggregated contract issues discovered during validation.
    """
    registry = load_dataset_registry(con)
    issues: list[ConformanceIssue] = [
        ConformanceIssue(dataset=None, message=msg) for msg in collect_contract_issues(con)
    ]
    if sample_rows:
        issues.extend(
            _validate_schema_rows(
                con,
                registry,
                sample_size=sample_size,
            )
        )
    return ConformanceReport(issues=issues)
