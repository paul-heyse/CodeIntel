"""Post-run quality output emitters for non-DAG analytics."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence

import pyarrow as pa

from codeintel.build.analytics.py_cpg_quality_report import (
    PY_CPG_QUALITY_REPORT_TABLE_KEY,
    PyCpgQualityInputs,
    build_py_cpg_quality_report_rows,
)
from codeintel.build.analytics.scip_diagnostics_rollups import (
    SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY,
    SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY,
    SCIP_DIAGNOSTICS_TABLE_KEY,
    SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY,
    build_scip_diagnostics_rollups,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.validation.runner import (
    GraphValidationRunRequest,
    run_graph_validations_with_runner,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.cpg.constants import (
    CPG_EDGES_TABLE_KEY,
    CPG_NODES_TABLE_KEY,
    PY_BC_BLOCKS_TABLE_KEY,
    PY_BC_CFG_EDGES_TABLE_KEY,
    PY_BC_DEFUSE_EVENTS_TABLE_KEY,
    PY_BC_INSTRUCTIONS_TABLE_KEY,
    PY_INSPECT_OBJECTS_TABLE_KEY,
    PY_SYM_SCOPES_TABLE_KEY,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    scan_parquet_dataset_with_telemetry,
)
from codeintel.core.execution.ids import RUN_PREFIX_ANALYTICS, new_run_id
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema, resolve_stable_sort_keys
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.reporters import (
    GRAPH_VALIDATION_TABLE_KEY,
    GraphValidationReporter,
)

log = logging.getLogger(__name__)

_SCIP_DIAGNOSTICS_COLUMNS = (
    "repo",
    "commit",
    "rel_path",
    "severity",
    "source",
    "code",
    "message",
)

_PY_CPG_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    PY_BC_INSTRUCTIONS_TABLE_KEY: ("repo", "commit", "span_start_byte"),
    PY_SYM_SCOPES_TABLE_KEY: ("repo", "commit", "anchor_ast_node_id"),
    PY_BC_BLOCKS_TABLE_KEY: (
        "repo",
        "commit",
        "code_unit_id",
        "block_id",
        "start_offset",
        "first_instr_index",
    ),
    PY_INSPECT_OBJECTS_TABLE_KEY: ("repo", "commit"),
    PY_BC_CFG_EDGES_TABLE_KEY: ("repo", "commit", "code_unit_id", "src_block_id", "dst_block_id"),
    PY_BC_DEFUSE_EVENTS_TABLE_KEY: ("repo", "commit", "event_kind", "space"),
    CPG_NODES_TABLE_KEY: ("repo", "commit", "cpg_node_id", "node_kind"),
    CPG_EDGES_TABLE_KEY: (
        "repo",
        "commit",
        "edge_kind",
        "edge_layer",
        "extras_kv",
        "src_cpg_node_id",
        "dst_cpg_node_id",
    ),
}


def persist_post_run_quality_outputs(*, env: BuildEnv, run_id: str) -> None:
    """Persist post-run quality outputs for non-DAG analytics."""
    persist_graph_validation(env=env)
    persist_scip_diagnostics_rollups(env=env)
    persist_py_cpg_quality_report(env=env, run_id=run_id)


def persist_graph_validation(*, env: BuildEnv) -> bool:
    """Persist graph validation findings outside the DAG.

    Returns
    -------
    bool
        True when the dataset is written, otherwise False.
    """
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        log.info("Graph validation skipped; dataset_root_dir unavailable.")
        return False
    if not env.commit.strip():
        log.warning("Graph validation skipped; snapshot_id missing.")
        return False

    request = GraphValidationRunRequest(
        snapshot=env.snapshot,
        runtime=GraphRuntimeOptions(
            snapshot=env.snapshot,
            dataset_root_dir=dataset_root,
        ),
        dataset_root_dir=dataset_root,
    )
    try:
        report = run_graph_validations_with_runner(request=request)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("Graph validation failed; error=%s", exc)
        return False

    reporter = GraphValidationReporter(repo=env.repo, commit=env.commit)
    _report_findings(reporter, report.findings)
    rows = reporter.to_rows()
    return _write_dataset_table(env=env, table_key=GRAPH_VALIDATION_TABLE_KEY, rows=rows)


def persist_scip_diagnostics_rollups(*, env: BuildEnv) -> bool:
    """Persist SCIP diagnostics rollups outside the DAG.

    Returns
    -------
    bool
        True when all rollup datasets are written, otherwise False.
    """
    reader = _scan_snapshot_reader(
        env=env,
        table_key=SCIP_DIAGNOSTICS_TABLE_KEY,
        columns=_SCIP_DIAGNOSTICS_COLUMNS,
    )
    if reader is None:
        log.info("SCIP diagnostics rollups skipped; source dataset unavailable.")
        return False

    rows: list[dict[str, object]] = []
    for batch in reader:
        rows.extend(iter_rows(batch))
    rollups = build_scip_diagnostics_rollups(repo=env.repo, commit=env.commit, rows=rows)

    summary_written = _write_dataset_table(
        env=env,
        table_key=SCIP_DIAGNOSTICS_SUMMARY_TABLE_KEY,
        rows=rollups.summary_rows,
    )
    by_file_written = _write_dataset_table(
        env=env,
        table_key=SCIP_DIAGNOSTICS_BY_FILE_TABLE_KEY,
        rows=rollups.by_file_rows,
    )
    top_written = _write_dataset_table(
        env=env,
        table_key=SCIP_DIAGNOSTICS_TOP_MESSAGES_TABLE_KEY,
        rows=rollups.top_message_rows,
    )
    return summary_written and by_file_written and top_written


def persist_py_cpg_quality_report(*, env: BuildEnv, run_id: str) -> bool:
    """Persist the Python CPG quality report outside the DAG.

    Returns
    -------
    bool
        True when the dataset is written, otherwise False.
    """
    tables: dict[str, pa.Table] = {}
    for table_key, columns in _PY_CPG_COLUMNS_BY_TABLE.items():
        table = _scan_snapshot_table(
            env=env,
            table_key=table_key,
            columns=columns,
        )
        if table is None:
            log.info("Python CPG quality report skipped; dataset missing: %s", table_key)
            return False
        tables[table_key] = table

    rows = build_py_cpg_quality_report_rows(
        repo=env.repo,
        commit=env.commit,
        run_id=_resolve_run_id(env=env, run_id=run_id),
        inputs=PyCpgQualityInputs(
            instructions=tables[PY_BC_INSTRUCTIONS_TABLE_KEY],
            scopes=tables[PY_SYM_SCOPES_TABLE_KEY],
            blocks=tables[PY_BC_BLOCKS_TABLE_KEY],
            inspect_objects=tables[PY_INSPECT_OBJECTS_TABLE_KEY],
            cfg_edges=tables[PY_BC_CFG_EDGES_TABLE_KEY],
            defuse_events=tables[PY_BC_DEFUSE_EVENTS_TABLE_KEY],
            cpg_nodes=tables[CPG_NODES_TABLE_KEY],
            cpg_edges=tables[CPG_EDGES_TABLE_KEY],
        ),
    )
    return _write_dataset_table(
        env=env,
        table_key=PY_CPG_QUALITY_REPORT_TABLE_KEY,
        rows=rows,
    )


def _scan_snapshot_table(
    *,
    env: BuildEnv,
    table_key: str,
    columns: Sequence[str] | None,
) -> pa.Table | None:
    reader = _scan_snapshot_reader(
        env=env,
        table_key=table_key,
        columns=columns,
    )
    if reader is None:
        return None
    table = reader_to_table(reader)
    finalized = finalize_table(
        table,
        spec=FinalizeSpec(table_key=table_key, mode="tolerant"),
    )
    return finalized.good


def _scan_snapshot_reader(
    *,
    env: BuildEnv,
    table_key: str,
    columns: Sequence[str] | None,
) -> pa.RecordBatchReader | None:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        log.info("Post-run scan skipped; dataset_root_dir unavailable.")
        return None
    snapshot_id = env.commit.strip()
    if not snapshot_id:
        log.warning("Post-run scan skipped; snapshot_id missing.")
        return None

    reader, telemetry = scan_parquet_dataset_with_telemetry(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        options=ParquetScanOptions(
            columns=tuple(columns) if columns is not None else None,
            repo=env.repo,
            commit=env.commit,
            implicit_ordering=True,
            require_sequenced_output=True,
            metrics_enabled=True,
        ),
    )
    if telemetry is not None:
        log.debug("Post-run scan telemetry: %s", telemetry.to_mapping())
    if reader is None:
        return None
    missing = [name for name in ("repo", "commit") if name not in reader.schema.names]
    if missing:
        log.warning("Post-run scan missing scope columns: %s table_key=%s", missing, table_key)
        return None
    return reader


def _write_dataset_table(
    *,
    env: BuildEnv,
    table_key: str,
    rows: Iterable[Mapping[str, object]] | Iterable[Sequence[object]],
) -> bool:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        log.info("Post-run dataset write skipped; dataset_root_dir unavailable.")
        return False
    snapshot_id = env.commit.strip()
    if not snapshot_id:
        log.warning("Post-run dataset write skipped; snapshot_id missing.")
        return False

    table, _ = table_for_rows(table_key, rows)
    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(table_key)
    options = ArrowDatasetWriteOptions(
        partition_columns=_partition_columns_for_schema(table_schema),
        schema_hash=schema_hash(table_schema),
        manifest_extras={"table_schema": table_schema.to_json_obj()},
        stable_sort_keys=resolve_stable_sort_keys(table_schema),
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
        options=options,
    )
    return True


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    names = set(table_schema.column_names())
    if "repo" in names and "commit" in names:
        return ("repo", "commit")
    return ()


def _resolve_run_id(*, env: BuildEnv, run_id: str) -> str:
    if run_id:
        return run_id
    run_context = env.run_context
    if run_context is None:
        return new_run_id(RUN_PREFIX_ANALYTICS)
    return run_context.run_id


def _report_findings(
    reporter: GraphValidationReporter,
    findings: Iterable[Mapping[str, object]],
) -> None:
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


__all__ = [
    "persist_graph_validation",
    "persist_post_run_quality_outputs",
    "persist_py_cpg_quality_report",
    "persist_scip_diagnostics_rollups",
]
