"""Dataset conformance validation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import duckdb
import jsonschema

from codeintel.core.errors.schema import SchemaError
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.contracts.json_schema import get_json_schema_for_table_key
from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.validation.contract import collect_contract_issues

if TYPE_CHECKING:
    from collections.abc import Iterable

    from duckdb import DuckDBPyConnection

    from codeintel.storage.datasets.registry import DatasetRegistry

log = logging.getLogger(__name__)

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


def _get_generated_schema(table_key: str) -> dict[str, object] | None:
    """Get a generated JSON Schema for the table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    dict[str, object] | None
        Generated JSON Schema, or None if not available.
    """
    try:
        return get_json_schema_for_table_key(table_key)
    except SchemaError as e:
        log.debug("Schema lookup failed for %s: %s", table_key, e)
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
        schema = _get_generated_schema(ds.table_key)
        if schema is None:
            continue
        if ds.schema is None:
            continue
        validator = jsonschema.Draft202012Validator(schema)
        try:
            reader = (
                con.table(ds.table_key)
                .limit(sample_size)
                .fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
            )
        except duckdb.Error as exc:
            yield ConformanceIssue(dataset=name, message=f"Failed to sample rows: {exc}")
            continue
        row_index = 0
        for batch in reader:
            for record in batch.to_pylist():
                errors = sorted(validator.iter_errors(record), key=lambda e: e.path)
                if errors:
                    yield ConformanceIssue(
                        dataset=name,
                        message=(
                            f"Row {row_index} failed JSON Schema validation: {errors[0].message}"
                        ),
                    )
                row_index += 1


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
