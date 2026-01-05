"""Finding types, persistence, and severity handling for graph validation.

This module provides graph-specific validation types and utilities,
extending the core validation infrastructure with graph-specific features.

The helper functions are re-exported from core to maintain a consistent API.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import msgspec
import pyarrow as pa

from codeintel.build.graphs.runtime import GraphRuntime
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.hashing import schema_digest, schema_hash
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.validation import (
    BaseValidationOptions,
    GraphValidationReporter,
    ValidationSeverity,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
)
from codeintel.core.validation.reporters import GRAPH_VALIDATION_TABLE_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.graphs.runtime import GraphRuntimeOptions


SAMPLE_LIMIT = 5
SYMBOL_COMMUNITY_MIN = 2
CONFIG_KEY_MIN_THRESHOLD = 2
HUB_MIN_DEGREE_FLOOR = 10
HUB_DEGREE_RATIO = 0.1
CALL_SCC_MIN = 5

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphValidationOptions(BaseValidationOptions):
    """Options for controlling graph validation behavior.

    Extend ``BaseValidationOptions`` with graph-specific options.
    Currently no additional fields, but this allows for future extension.

    Attributes
    ----------
    severity_overrides
        Mapping of rule names to severity levels. Use "*" for all.
    hard_fail
        Whether to raise an exception on error-level findings.
    max_findings_per_rule
        Maximum findings to collect per rule.
    """


def resolve_validation_options(
    runtime: GraphRuntime | GraphRuntimeOptions,
    options: GraphValidationOptions | None,
) -> GraphValidationOptions:
    """Determine effective validation options from runtime feature flags.

    Parameters
    ----------
    runtime
        Runtime or options containing feature flags.
    options
        Explicit options to use if provided.

    Returns
    -------
    GraphValidationOptions
        Options merged with any feature flag overrides.
    """
    if options is not None:
        return options
    features = runtime.options.features if isinstance(runtime, GraphRuntime) else runtime.features
    strict = features.validation_strict if features is not None else None
    if strict:
        return GraphValidationOptions(severity_overrides={"*": "error"}, hard_fail=True)
    return GraphValidationOptions()


def hub_threshold(node_count: int) -> int:
    """Compute a hub threshold that scales with graph size.

    Parameters
    ----------
    node_count
        Number of nodes in the graph.

    Returns
    -------
    int
        Degree threshold used to flag hubs.
    """
    return max(HUB_MIN_DEGREE_FLOOR, int(node_count * HUB_DEGREE_RATIO))


def persist_findings(
    dataset_root_dir: Path | None,
    findings: list[dict[str, object]],
    repo: str,
    commit: str,
) -> None:
    """Persist validation findings to the analytics Parquet dataset.

    Parameters
    ----------
    dataset_root_dir
        Root directory for Parquet dataset snapshots.
    findings
        List of findings to persist.
    repo
        Repository identifier.
    commit
        Commit identifier.
    """
    if not findings:
        return
    reporter = GraphValidationReporter(repo=repo, commit=commit)
    for finding in findings:
        graph_name = str(finding.get("check_name") or "graph_validation")
        entity_ref = finding.get("path") or finding.get("entity_id") or finding.get("graph_name")
        entity_id = str(entity_ref) if entity_ref is not None else graph_name
        issue = str(finding.get("issue") or finding.get("severity") or graph_name)
        severity = str(finding.get("severity") or "info")
        rel_path = finding.get("path")
        detail = str(finding.get("detail") or "")
        metadata = finding.get("context")
        extras = {
            "severity": severity,
            "rel_path": str(rel_path) if rel_path is not None else None,
            "metadata": metadata,
        }
        reporter.record(
            graph_name=graph_name,
            entity_id=entity_id,
            issue=issue,
            detail=detail,
            extras=extras,
        )
    if not reporter.rows:
        return
    _persist_findings_parquet(dataset_root_dir, reporter.rows, repo=repo, commit=commit)


def _persist_findings_parquet(
    dataset_root_dir: Path | None,
    rows: Sequence[Mapping[str, object] | msgspec.Struct],
    *,
    repo: str,
    commit: str,
) -> bool:
    dataset_root = dataset_root_dir
    if dataset_root is None:
        log.info("Graph validation persistence skipped; dataset_root_dir is not configured.")
        return False
    snapshot_id = commit.strip()
    if not snapshot_id:
        log.warning("Graph validation persistence skipped; snapshot_id missing.")
        return False
    normalized_rows = _normalize_rows(rows)
    table = _rows_to_arrow_table(GRAPH_VALIDATION_TABLE_KEY, normalized_rows)
    table_schema = table_schema_from_arrow_schema(
        arrow_schema=table.schema,
        table_key=GRAPH_VALIDATION_TABLE_KEY,
    )
    schema_hash_value = schema_hash(table_schema)
    schema_digest_value = schema_digest(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    schema_metadata = _graph_validation_metadata(
        GraphValidationMetadataInput(
            table_schema=table_schema,
            schema_hash_value=schema_hash_value,
            schema_digest_value=schema_digest_value,
            partition_columns=partition_columns,
            repo=repo,
            commit=commit,
        )
    )
    options = ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        existing_data_behavior="delete_matching",
        persist_manifest=True,
        schema_hash=schema_hash_value,
        manifest_extras={"table_schema": table_schema.to_json_obj()},
        schema_metadata=schema_metadata,
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=GRAPH_VALIDATION_TABLE_KEY,
        snapshot_id=snapshot_id,
        data=table,
        options=options,
    )
    return True


def _normalize_rows(
    rows: Sequence[Mapping[str, object] | msgspec.Struct],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        row_mapping: Mapping[str, object]
        if isinstance(row, Mapping):
            row_mapping = row
        else:
            builtins = msgspec.to_builtins(row)
            if not isinstance(builtins, Mapping):
                continue
            row_mapping = cast("Mapping[str, object]", builtins)
        metadata = row_mapping.get("metadata")
        if metadata is None:
            normalized.append(dict(row_mapping))
            continue
        updated = dict(row_mapping)
        updated["metadata"] = _normalize_metadata(metadata)
        normalized.append(updated)
    return normalized


def _normalize_metadata(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _rows_to_arrow_table(table_key: str, rows: Sequence[Mapping[str, object]]) -> pa.Table:
    columns = columns_for_table_key(table_key)
    if columns is None:
        return pa.Table.from_pylist(list(rows))
    ordered_rows: list[dict[str, object]] = []
    for row in rows:
        ordered = {name: row.get(name) for name in columns}
        ordered_rows.append(ordered)
    return pa.Table.from_pylist(ordered_rows)


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    column_names = table_schema.column_names()
    if "repo" in column_names and "commit" in column_names:
        return ("repo", "commit")
    return ()


@dataclass(frozen=True)
class GraphValidationMetadataInput:
    table_schema: TableSchema
    schema_hash_value: str
    schema_digest_value: str
    partition_columns: tuple[str, ...]
    repo: str
    commit: str


def _graph_validation_metadata(
    inputs: GraphValidationMetadataInput,
) -> dict[str, object]:
    columns_json = {col.name: col.type for col in inputs.table_schema.columns}
    nullability_json = {col.name: col.nullable for col in inputs.table_schema.columns}
    return {
        "codeintel.table_key": inputs.table_schema.table_key,
        "codeintel.domain": inputs.table_schema.schema,
        "codeintel.target": "graph_validation",
        "codeintel.schema_hash": inputs.schema_hash_value,
        "codeintel.schema_digest": inputs.schema_digest_value,
        "codeintel.columns_json": columns_json,
        "codeintel.nullability_json": nullability_json,
        "codeintel.primary_keys_json": list(inputs.table_schema.primary_key),
        "codeintel.partition_columns_json": list(inputs.partition_columns),
        "codeintel.build_id": inputs.commit,
        "codeintel.repo": inputs.repo,
        "codeintel.commit": inputs.commit,
        "codeintel.snapshot_id": inputs.commit,
        "codeintel.generated_at": datetime.now(tz=UTC).isoformat(),
        "codeintel.hamilton.node": "graph_validation_runner",
        "codeintel.hamilton.graph_version": "manual",
        "codeintel.inputs_json": [],
    }


__all__ = [
    "CALL_SCC_MIN",
    "CONFIG_KEY_MIN_THRESHOLD",
    "HUB_DEGREE_RATIO",
    "HUB_MIN_DEGREE_FLOOR",
    "SAMPLE_LIMIT",
    "SYMBOL_COMMUNITY_MIN",
    "GraphValidationOptions",
    "ValidationSeverity",
    "apply_severity_overrides",
    "cap_findings",
    "has_error_findings",
    "hub_threshold",
    "persist_findings",
    "resolve_validation_options",
]
