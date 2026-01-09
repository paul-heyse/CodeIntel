"""Dataset conformance validation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.columnar.conversion import tabular_to_arrow_reader
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.validation.columnar import (
    ColumnarValidationContext,
    TableValidationError,
    validate_record_batch_reader,
)
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


def _validate_schema_rows(
    con: DuckDBPyConnection,
    registry: DatasetRegistry,
    *,
    sample_size: int = 50,
) -> Iterable[ConformanceIssue]:
    """Validate sampled rows against TableSchema contracts.

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
        if ds.schema is None:
            continue
        try:
            reader = tabular_to_arrow_reader(
                con.table(ds.table_key).limit(sample_size),
                batch_size=DEFAULT_ARROW_BATCH_SIZE,
            )
        except DuckDBError as exc:
            yield ConformanceIssue(dataset=name, message=f"Failed to sample rows: {exc}")
            continue
        context = ColumnarValidationContext(
            table_schema=ds.schema,
            schema_observation=None,
            validation_profile="strict",
        )
        try:
            validated = validate_record_batch_reader(
                ds.table_key,
                reader,
                context=context,
                mode="strict",
            )
            for _ in validated:
                pass
        except TableValidationError as exc:
            yield ConformanceIssue(dataset=name, message=str(exc))


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
        ConformanceIssue(dataset=None, message=msg)
        for msg in collect_contract_issues(con, missing_ok=True)
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
