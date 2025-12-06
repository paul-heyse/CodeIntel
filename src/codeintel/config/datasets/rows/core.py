"""Core entity row models and serializers.

This module provides row types for core DuckDB tables:
- IngestRunRow and IngestRunLike for core.ingest_runs (dataclass - not duplicated)
- DocstringRow for core.docstrings (TypedDict - not duplicated in compute)
- ConfigValueRow for analytics.config_values (TypedDict - not duplicated in compute)
- GoidRow and GoidCrosswalkRow are re-exported from codeintel.graphs.data_models.rows

The canonical dataclass definitions for GOID rows live in
codeintel.graphs.data_models.rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, TypedDict

# Re-export canonical row types from data_models (single source of truth)
from codeintel.core.data_models.rows import GoidCrosswalkRow, GoidRow

# Export to_tuple as method references for backward compatibility
goid_to_tuple = GoidRow.to_tuple
goid_crosswalk_to_tuple = GoidCrosswalkRow.to_tuple


class IngestRunStatus(StrEnum):
    """Outcome for an ingestion step run."""

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class IngestRunMode(StrEnum):
    """High-level mode for a dataset step."""

    FULL = "full"
    INCREMENTAL = "incremental"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IngestRunRow:
    """Row shape for control-plane ingest runs persisted to DuckDB.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA being processed.
    step
        Pipeline step name.
    run_id
        Unique run identifier.
    mode
        Ingest run mode (string representation).
    started_at
        Run start timestamp.
    finished_at
        Run completion timestamp (None if still running).
    duration_s
        Run duration in seconds.
    rows_inserted
        Number of rows inserted.
    rows_deleted
        Number of rows deleted.
    status
        Run status (string representation).
    error_kind
        Error category if failed.
    error_message
        Error message if failed.
    datasets
        JSON string of datasets processed.
    modules_total
        Total modules processed.
    modules_changed
        Number of changed modules.
    modules_deleted
        Number of deleted modules.
    modules_changed_ratio
        Ratio of changed modules.
    modules_deleted_ratio
        Ratio of deleted modules.
    use_full_rebuild
        Whether full rebuild was used.
    """

    repo: str
    commit: str
    step: str
    run_id: str
    mode: str
    started_at: datetime
    finished_at: datetime | None
    duration_s: float | None
    rows_inserted: int
    rows_deleted: int
    status: str
    error_kind: str | None
    error_message: str | None
    datasets: str
    modules_total: int | None
    modules_changed: int | None
    modules_deleted: int | None
    modules_changed_ratio: float | None
    modules_deleted_ratio: float | None
    use_full_rebuild: bool | None


class IngestRunLike(Protocol):
    """Structural contract for ingest run serialization.

    This protocol defines the shape expected by ingest_run_to_tuple().
    """

    repo: str
    commit: str
    step: str
    run_id: str
    mode: IngestRunMode
    started_at: datetime
    finished_at: datetime | None
    duration_s: float | None
    rows_inserted: int
    rows_deleted: int
    status: IngestRunStatus
    error_kind: str | None
    error_message: str | None
    datasets: tuple[str, ...]
    modules_total: int | None
    modules_changed: int | None
    modules_deleted: int | None
    modules_changed_ratio: float | None
    modules_deleted_ratio: float | None
    use_full_rebuild: bool | None


def ingest_run_to_tuple(run: IngestRunLike) -> tuple[object, ...]:
    """Serialize an IngestRun into the INSERT column order for core.ingest_runs.

    Parameters
    ----------
    run
        The ingest run to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by ingest_runs INSERTs.
    """
    return (
        run.repo,
        run.commit,
        run.step,
        run.run_id,
        run.mode.value,
        run.started_at,
        run.finished_at,
        run.duration_s,
        run.rows_inserted,
        run.rows_deleted,
        run.status.value,
        run.error_kind,
        run.error_message,
        json.dumps(list(run.datasets)),
        run.modules_total,
        run.modules_changed,
        run.modules_deleted,
        run.modules_changed_ratio,
        run.modules_deleted_ratio,
        run.use_full_rebuild,
    )


class DocstringRow(TypedDict):
    """Row shape for core.docstrings inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    rel_path
        Relative file path.
    module
        Module name.
    qualname
        Fully qualified name.
    kind
        Entity kind.
    lineno
        Starting line number.
    end_lineno
        Ending line number.
    raw_docstring
        Raw docstring text.
    style
        Docstring style (numpy, google, etc.).
    short_desc
        Short description.
    long_desc
        Long description.
    params
        Parameter documentation (JSON).
    returns
        Return value documentation (JSON).
    raises
        Exception documentation (JSON).
    examples
        Example documentation (JSON).
    created_at
        Creation timestamp.
    """

    repo: str
    commit: str
    rel_path: str
    module: str
    qualname: str
    kind: str
    lineno: int | None
    end_lineno: int | None
    raw_docstring: str | None
    style: str | None
    short_desc: str | None
    long_desc: str | None
    params: object
    returns: object
    raises: object
    examples: object
    created_at: datetime


def docstring_row_to_tuple(row: DocstringRow) -> tuple[object, ...]:
    """Serialize a DocstringRow into the INSERT column order.

    Parameters
    ----------
    row
        The docstring row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by docstrings INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["rel_path"],
        row["module"],
        row["qualname"],
        row["kind"],
        row["lineno"],
        row["end_lineno"],
        row["raw_docstring"],
        row["style"],
        row["short_desc"],
        row["long_desc"],
        row["params"],
        row["returns"],
        row["raises"],
        row["examples"],
        row["created_at"],
    )


class ConfigValueRow(TypedDict):
    """Row shape for analytics.config_values inserts.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    config_path
        Path to the configuration file.
    format
        Configuration format (yaml, json, etc.).
    key
        Configuration key.
    reference_paths
        List of paths referencing this config.
    reference_modules
        List of modules referencing this config.
    reference_count
        Number of references.
    """

    repo: str
    commit: str
    config_path: str
    format: str
    key: str
    reference_paths: list[str]
    reference_modules: list[str]
    reference_count: int


def config_value_to_tuple(row: ConfigValueRow) -> tuple[object, ...]:
    """Serialize a ConfigValueRow into the INSERT column order.

    Parameters
    ----------
    row
        The config value row to serialize.

    Returns
    -------
    tuple[object, ...]
        Values in the order expected by config_values INSERTs.
    """
    return (
        row["repo"],
        row["commit"],
        row["config_path"],
        row["format"],
        row["key"],
        row["reference_paths"],
        row["reference_modules"],
        row["reference_count"],
    )


__all__ = [
    "ConfigValueRow",
    "DocstringRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "IngestRunLike",
    "IngestRunMode",
    "IngestRunRow",
    "IngestRunStatus",
    "config_value_to_tuple",
    "docstring_row_to_tuple",
    "goid_crosswalk_to_tuple",
    "goid_to_tuple",
    "ingest_run_to_tuple",
]
