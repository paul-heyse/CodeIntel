"""Post-run quality output emitters for non-DAG analytics."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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
from codeintel.build.analytics.utilities.finalize import (
    finalize_analytics_reader,
    finalize_analytics_result,
    finalize_artifact_counts,
    finalize_artifact_table_key,
)
from codeintel.build.analytics.utilities.pipeline import (
    AnalyticsPipelineRunRequest,
    run_analytics_pipeline,
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
from codeintel.build.tabular.finalize_ops import FinalizeResult
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
)
from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.ordering import OrderingSpec, SortDirection, SortKey
from codeintel.core.columnar.plan_ops import QueryPlanOptions
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    columnar_batch_collector_for_table_key,
    table_for_rows,
)
from codeintel.core.columnar.run_manifest import (
    RunManifestOptions,
    run_manifest_options_for_context,
    write_run_manifest,
)
from codeintel.core.columnar.streaming import ScanTelemetry, scan_telemetry_for_queryspec
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetWriteOptions,
    scan_dataset,
    write_dataset,
)
from codeintel.core.execution.ids import RUN_PREFIX_ANALYTICS, new_run_id
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema, resolve_canonical_sort_keys
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

ORDER_ASC: SortDirection = "ascending"

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
        output_dir=env.paths.build_dir / "quality-results" / "graph_validation",
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
    table = _scan_snapshot_table(
        env=env,
        table_key=SCIP_DIAGNOSTICS_TABLE_KEY,
        columns=_SCIP_DIAGNOSTICS_COLUMNS,
    )
    if table is None:
        log.info("SCIP diagnostics rollups skipped; source dataset unavailable.")
        return False

    rollups = build_scip_diagnostics_rollups(
        repo=env.repo,
        commit=env.commit,
        rows=table,
        ctx=env.execution_context,
    )

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
    result = _scan_snapshot_result(
        env=env,
        table_key=table_key,
        columns=columns,
    )
    if result is None:
        return None
    return result.good


def _scan_snapshot_result(
    *,
    env: BuildEnv,
    table_key: str,
    columns: Sequence[str] | None,
) -> FinalizeResult | None:
    result: FinalizeResult | None = None
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        log.info("Post-run scan skipped; dataset_root_dir unavailable.")
    else:
        snapshot_id = env.commit.strip()
        if not snapshot_id:
            log.warning("Post-run scan skipped; snapshot_id missing.")
        else:
            dataset = _scan_snapshot_dataset(
                dataset_root=dataset_root,
                table_key=table_key,
                snapshot_id=snapshot_id,
            )
            if dataset is None:
                return None
            resolved_columns = _resolve_snapshot_columns(dataset, columns)
            if resolved_columns is None and columns is not None:
                return None
            predicate = _snapshot_predicate(dataset.schema, repo=env.repo, commit=env.commit)
            query_spec = _query_spec_for_snapshot(
                dataset,
                columns=resolved_columns,
                predicate=predicate,
            )
            telemetry = scan_telemetry_for_queryspec(dataset, spec=query_spec)
            log.debug(
                "Post-run scan telemetry: fragments=%s rows=%s",
                telemetry.fragment_count,
                telemetry.estimated_rows,
            )
            _emit_run_manifest(
                env=env,
                table_key=table_key,
                telemetry=telemetry,
                implicit_ordering=True,
            )
            options = QueryPlanOptions(
                implicit_ordering=True,
                require_sequenced_output=True,
            )
            ctx = _scan_execution_context(env)
            try:
                result = run_analytics_pipeline(
                    AnalyticsPipelineRunRequest(
                        source=dataset,
                        spec=query_spec,
                        table_key=table_key,
                        ctx=ctx,
                        options=options,
                    )
                )
            except (
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
                TypeError,
                ValueError,
            ) as exc:
                log.warning("Post-run scan failed; table_key=%s error=%s", table_key, exc)
                result = None
            if result is not None:
                missing = [
                    name for name in ("repo", "commit") if name not in result.good.schema.names
                ]
                if missing:
                    log.warning(
                        "Post-run scan missing scope columns: %s table_key=%s",
                        missing,
                        table_key,
                    )
                    result = None
    return result


def _scan_execution_context(env: BuildEnv) -> ExecutionContext:
    runtime_ctx = env.execution_context
    resolved = resolve_columnar_context(runtime_ctx)
    if resolved is None:
        return ExecutionContext(use_threads=True)
    return ExecutionContext(use_threads=True, runtime_profile=resolved.runtime_profile)


def _scan_snapshot_dataset(
    *,
    dataset_root: Path,
    table_key: str,
    snapshot_id: str,
) -> ds.Dataset | None:
    try:
        return scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except FileNotFoundError:
        log.warning("Post-run scan missing dataset: %s", table_key)
        return None
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        log.warning("Post-run scan failed; table_key=%s error=%s", table_key, exc)
        return None


def _snapshot_predicate(
    schema: pa.Schema,
    *,
    repo: str,
    commit: str,
) -> Expression | None:
    names = set(schema.names)
    expressions: list[Expression] = []
    if "repo" in names:
        expressions.append(E.field("repo") == E.scalar(repo))
    if "commit" in names:
        expressions.append(E.field("commit") == E.scalar(commit))
    if not expressions:
        return None
    predicate = expressions[0]
    for expr in expressions[1:]:
        predicate &= expr
    return predicate


def _resolve_snapshot_columns(
    dataset: ds.Dataset,
    columns: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if columns is None:
        return None
    available = set(dataset.schema.names)
    missing = [name for name in columns if name not in available]
    if missing:
        log.warning(
            "Post-run scan missing columns: %s table_key=%s",
            ", ".join(missing),
            dataset.schema,
        )
        return None
    return tuple(columns)


def _query_spec_for_snapshot(
    dataset: ds.Dataset,
    *,
    columns: tuple[str, ...] | None,
    predicate: Expression | None,
) -> QuerySpec:
    projection = _projection_spec_for_columns(dataset, columns)
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _projection_spec_for_columns(
    dataset: ds.Dataset,
    columns: tuple[str, ...] | None,
) -> ProjectionSpec:
    if columns is None:
        return ProjectionSpec(base_cols=tuple(dataset.schema.names))
    return ProjectionSpec(base_cols=columns)


def _emit_run_manifest(
    *,
    env: BuildEnv,
    table_key: str,
    telemetry: ScanTelemetry,
    implicit_ordering: bool,
) -> None:
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    sort_keys = resolve_canonical_sort_keys(table_schema)
    ordering = _scan_ordering(
        sort_keys=sort_keys,
        implicit_ordering=implicit_ordering,
    )
    filename = f"run_manifest_{table_key.replace('.', '_')}.json"
    resolved_ctx = resolve_columnar_context(env.execution_context)
    options = run_manifest_options_for_context(
        ctx=resolved_ctx,
        ordering=ordering,
        scan_telemetry=telemetry,
        options=RunManifestOptions(
            extras={"table_key": table_key, "snapshot_id": env.commit},
            filename=filename,
        ),
    )
    write_run_manifest(_post_run_output_dir(env), options=options)


def _scan_ordering(
    *,
    sort_keys: tuple[str, ...] | None,
    implicit_ordering: bool,
) -> OrderingSpec | None:
    if sort_keys:
        keys: list[SortKey] = [(key, ORDER_ASC) for key in sort_keys]
        return OrderingSpec.explicit(keys=keys, reason="canonical sort keys")
    if implicit_ordering:
        return OrderingSpec.implicit(reason="implicit scan ordering")
    return OrderingSpec.unordered(reason="scan unordered")


def _emit_finalize_artifacts(
    *,
    env: BuildEnv,
    table_key: str,
    result: FinalizeResult,
) -> None:
    output_dir = _post_run_output_dir(env)
    _write_artifact_table(
        output_dir,
        table_key=table_key,
        artifact="errors",
        table=result.errors,
    )
    _write_artifact_table(
        output_dir,
        table_key=table_key,
        artifact="alignment",
        table=result.alignment,
    )
    _write_artifact_table(
        output_dir,
        table_key=table_key,
        artifact="stats",
        table=result.stats,
    )


def _write_artifact_table(
    output_dir: Path,
    *,
    table_key: str,
    artifact: str,
    table: pa.Table,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{table_key.replace('.', '_')}_{artifact}.parquet"
    try:
        pq.write_table(table, output_dir / filename)
    except (OSError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        log.warning(
            "Post-run artifact write failed; table_key=%s artifact=%s error=%s",
            table_key,
            artifact,
            exc,
        )


def _post_run_output_dir(env: BuildEnv) -> Path:
    return env.paths.build_dir / "quality-results" / "post_run_scans"


def _write_dataset_table(
    *,
    env: BuildEnv,
    table_key: str,
    rows: ColumnarRowBuffer | Iterable[Mapping[str, object]] | Iterable[Sequence[object]],
) -> bool:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        log.info("Post-run dataset write skipped; dataset_root_dir unavailable.")
        return False
    snapshot_id = env.commit.strip()
    if not snapshot_id:
        log.warning("Post-run dataset write skipped; snapshot_id missing.")
        return False

    if isinstance(rows, ColumnarRowBuffer):
        collector = columnar_batch_collector_for_table_key(table_key)
        collector.extend(rows)
        result = finalize_analytics_reader(table_key, collector.to_reader())
    else:
        table, _ = table_for_rows(table_key, rows)
        result = finalize_analytics_result(table_key, table)
    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(table_key)
    finalize_counts = finalize_artifact_counts(result)
    _emit_finalize_artifacts(env=env, table_key=table_key, result=result)
    _write_finalize_artifact_datasets(
        dataset_root=dataset_root,
        snapshot_id=snapshot_id,
        base_table_key=table_key,
        result=result,
    )
    table = result.good
    options = ArrowDatasetWriteOptions(
        partition_columns=_partition_columns_for_schema(table_schema),
        schema_hash=schema_hash(table_schema),
        manifest_extras=_manifest_extras(table_schema, finalize_counts=finalize_counts),
        stable_sort_keys=_resolve_manifest_sort_keys(table_schema),
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


def _resolve_manifest_sort_keys(table_schema: TableSchema) -> tuple[str, ...] | None:
    return resolve_canonical_sort_keys(table_schema)


def _manifest_extras(
    table_schema: TableSchema,
    *,
    finalize_counts: Mapping[str, int] | None = None,
    artifact_for: str | None = None,
    artifact_type: str | None = None,
) -> dict[str, object]:
    extras: dict[str, object] = {"table_schema": table_schema.to_json_obj()}
    if finalize_counts is not None:
        extras["finalize"] = dict(finalize_counts)
    if artifact_for is not None:
        extras["artifact_for"] = artifact_for
    if artifact_type is not None:
        extras["artifact_type"] = artifact_type
    return extras


def _write_finalize_artifact_datasets(
    *,
    dataset_root: Path,
    snapshot_id: str,
    base_table_key: str,
    result: FinalizeResult,
) -> None:
    _write_finalize_artifact_dataset(
        dataset_root=dataset_root,
        snapshot_id=snapshot_id,
        base_table_key=base_table_key,
        artifact="errors",
        table=result.errors,
    )
    _write_finalize_artifact_dataset(
        dataset_root=dataset_root,
        snapshot_id=snapshot_id,
        base_table_key=base_table_key,
        artifact="alignment",
        table=result.alignment,
    )
    _write_finalize_artifact_dataset(
        dataset_root=dataset_root,
        snapshot_id=snapshot_id,
        base_table_key=base_table_key,
        artifact="stats",
        table=result.stats,
    )


def _write_finalize_artifact_dataset(
    *,
    dataset_root: Path,
    snapshot_id: str,
    base_table_key: str,
    artifact: str,
    table: pa.Table,
) -> None:
    artifact_table_key = finalize_artifact_table_key(base_table_key, artifact)
    try:
        table_schema = table_schema_from_arrow_schema(
            arrow_schema=table.schema,
            table_key=artifact_table_key,
        )
        options = ArrowDatasetWriteOptions(
            partition_columns=_partition_columns_for_schema(table_schema),
            schema_hash=schema_hash(table_schema),
            manifest_extras=_manifest_extras(
                table_schema,
                artifact_for=base_table_key,
                artifact_type=artifact,
            ),
            stable_sort_keys=_resolve_manifest_sort_keys(table_schema),
        )
        write_dataset(
            dataset_root=dataset_root,
            table_key=artifact_table_key,
            snapshot_id=snapshot_id,
            data=table,
            options=options,
        )
    except (OSError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        log.warning(
            "Post-run finalize artifact write failed; table_key=%s artifact=%s error=%s",
            base_table_key,
            artifact,
            exc,
        )


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
