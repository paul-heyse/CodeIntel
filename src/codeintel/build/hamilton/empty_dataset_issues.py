"""Persist empty dataset guardrail diagnostics as build datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.datasets.manifests import load_dataset_manifest
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema, resolve_stable_sort_keys
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from codeintel.core.manifests import ArrowDatasetManifest

log = logging.getLogger(__name__)

EMPTY_DATASET_ISSUES_TABLE_KEY = "build.empty_dataset_issues"

_REQUIRED_TABLE_MIN_ROWS: Final[Mapping[str, int]] = {
    "graph.cdg_edges": 1,
    "graph.cpg_edges_calls": 1,
    "graph.cpg_edges_ret_to_call": 1,
    "analytics.config_data_flow": 1,
    "analytics.config_graph_metrics_keys": 1,
    "analytics.config_graph_metrics_modules": 1,
}


@dataclass(frozen=True, slots=True)
class _IssueContext:
    dataset_root: Path
    snapshot_id: str
    run_id: str
    repo: str
    commit: str
    dependency_map: Mapping[str, Sequence[str]]


def persist_empty_dataset_issues(
    *,
    env: BuildEnv,
    run_id: str,
    catalog: DagCatalog,
) -> bool:
    """Persist empty dataset diagnostics as a dataset.

    Returns
    -------
    bool
        True when rows were written, otherwise False.
    """
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        log.info("Empty dataset diagnostics skipped; dataset_root_dir unavailable.")
        return False
    snapshot_id = env.commit.strip()
    if not snapshot_id:
        log.warning("Empty dataset diagnostics skipped; snapshot_id missing.")
        return False
    if not run_id:
        log.warning("Empty dataset diagnostics skipped; run_id missing.")
        return False

    dependency_map = _dependency_chain_map(catalog, _REQUIRED_TABLE_MIN_ROWS)
    context = _IssueContext(
        dataset_root=dataset_root,
        snapshot_id=snapshot_id,
        run_id=run_id,
        repo=env.repo,
        commit=env.commit,
        dependency_map=dependency_map,
    )
    rows = _issue_rows(
        required=_REQUIRED_TABLE_MIN_ROWS,
        context=context,
    )
    if rows:
        table, _ = table_for_rows(EMPTY_DATASET_ISSUES_TABLE_KEY, rows)
    else:
        table = empty_table_for_table(EMPTY_DATASET_ISSUES_TABLE_KEY)

    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(EMPTY_DATASET_ISSUES_TABLE_KEY)
    schema_hash_value = schema_hash(table_schema)
    partition_columns = _partition_columns_for_schema(table_schema)
    options = ArrowDatasetWriteOptions(
        partition_columns=partition_columns,
        schema_hash=schema_hash_value,
        manifest_extras={"table_schema": table_schema.to_json_obj()},
        stable_sort_keys=resolve_stable_sort_keys(table_schema),
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=EMPTY_DATASET_ISSUES_TABLE_KEY,
        snapshot_id=snapshot_id,
        data=table,
        options=options,
    )
    return bool(rows)


def _issue_rows(
    *,
    required: Mapping[str, int],
    context: _IssueContext,
) -> list[dict[str, object]]:
    recorded_at = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for table_key, min_rows in required.items():
        manifest = load_dataset_manifest(
            dataset_root=context.dataset_root,
            table_key=table_key,
            snapshot_id=context.snapshot_id,
        )
        row_count, status = _manifest_status(manifest, min_rows=min_rows)
        rows.append(
            {
                "run_id": context.run_id,
                "repo": context.repo,
                "commit": context.commit,
                "table_key": table_key,
                "required_min_rows": min_rows,
                "row_count": row_count,
                "status": status,
                "dependency_chain": list(context.dependency_map.get(table_key, ())),
                "recorded_at": recorded_at,
            }
        )
    return rows


def _manifest_status(
    manifest: ArrowDatasetManifest | None,
    *,
    min_rows: int,
) -> tuple[int | None, str]:
    if manifest is None:
        return None, "missing_manifest"
    row_count = manifest.row_count
    if row_count is None:
        return None, "missing_row_count"
    status = "ok" if row_count >= min_rows else "empty"
    return row_count, status


def _dependency_chain_map(
    catalog: DagCatalog,
    table_keys: Mapping[str, int] | Iterable[str],
) -> dict[str, list[str]]:
    targets = table_keys.keys() if isinstance(table_keys, Mapping) else table_keys
    return {table_key: _dependency_chain_for_table(catalog, table_key) for table_key in targets}


def _dependency_chain_for_table(catalog: DagCatalog, table_key: str) -> list[str]:
    output = catalog.table_outputs.get(table_key)
    if output is None:
        return []
    target_name = output.producer_target
    deps = catalog.target_dependencies.get(target_name, ())
    chain = [target_name]
    chain.extend(dep for dep in sorted(deps) if dep != target_name)
    return chain


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    names = set(table_schema.column_names())
    if "repo" in names and "commit" in names:
        return ("repo", "commit")
    return ()


__all__ = [
    "EMPTY_DATASET_ISSUES_TABLE_KEY",
    "persist_empty_dataset_issues",
]
