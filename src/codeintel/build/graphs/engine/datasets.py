"""Parquet dataset helpers for graph engines and validation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codeintel.build.graphs.assembly import iter_normalized_tuples
from codeintel.build.scopes.snapshot import SnapshotScanContext
from codeintel.core.columnar.arrowdsl import (
    ExecutionPlan,
    PipelineRunOptions,
    run_pipeline,
)
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    runtime_profile_from_settings,
)
from codeintel.core.columnar.finalize_ops import (
    FinalizeResult,
    finalize_spec_for_table,
)
from codeintel.core.columnar.ordering import SortDirection, SortKey
from codeintel.core.columnar.plan_ops import (
    Plan,
    QueryPlanOptions,
    build_query_plan_for_context,
)
from codeintel.core.columnar.queryspec import QuerySpec, projection_spec_from_columns
from codeintel.core.columnar.streaming import scan_telemetry_for_queryspec
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetWriteOptions,
    scan_dataset,
    write_dataset,
)
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.core.schemas.arrow_polars import table_schema_from_arrow_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.primitives import TableSchema, resolve_canonical_sort_keys
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.columnar.profiles import RuntimeProfile

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotScanRequest:
    """Scan request for dataset snapshots."""

    dataset_root: Path
    table_key: str
    snapshot_id: str
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None = None
    provenance: bool = False
    repo: str | None = None
    commit: str | None = None
    batch_size: int | None = None
    batch_readahead: int | None = None
    fragment_readahead: int | None = None
    use_threads: bool | None = None
    cache_metadata: bool | None = None
    parquet_pre_buffer: bool | None = None
    parquet_use_buffered_stream: bool | None = None
    parquet_buffer_size: int | None = None
    unify_schemas: bool = True
    scan_context: SnapshotScanContext | None = None
    apply_filter: bool = True
    implicit_ordering: bool | None = True
    require_sequenced_output: bool | None = True
    metrics_enabled: bool = True
    execution_ctx: ExecutionContext | None = None


@dataclass(frozen=True, slots=True)
class GraphViewScanOptions:
    """Overrides for snapshot scan behavior in graph views."""

    apply_filter: bool = True
    implicit_ordering: bool | None = True
    require_sequenced_output: bool | None = True
    metrics_enabled: bool = True
    provenance: bool = False
    execution_ctx: ExecutionContext | None = None


@dataclass(frozen=True, slots=True)
class GraphViewFactory:
    """Factory for graph views backed by snapshot datasets."""

    dataset_root: Path
    snapshot_id: str
    scan_context: SnapshotScanContext

    @classmethod
    def for_snapshot(
        cls,
        dataset_root: Path,
        *,
        repo: str | None,
        commit: str,
    ) -> GraphViewFactory:
        """Build a graph view factory aligned to a snapshot.

        Parameters
        ----------
        dataset_root
            Root directory for Parquet dataset snapshots.
        repo
            Repository identifier anchoring the view.
        commit
            Commit hash anchoring the snapshot.

        Returns
        -------
        GraphViewFactory
            Factory configured for the snapshot.
        """
        scan_context = SnapshotScanContext(
            repo=repo,
            commit=commit,
            settings=load_runtime_settings().build.arrow_scan,
        )
        return cls(dataset_root=dataset_root, snapshot_id=commit, scan_context=scan_context)

    def load_reader(
        self,
        *,
        table_key: str,
        columns: Sequence[str] | Mapping[str, pc.Expression] | None = None,
        scan_options: GraphViewScanOptions | None = None,
    ) -> pa.RecordBatchReader | None:
        """Return a record batch reader for a snapshot table.

        Parameters
        ----------
        table_key
            Dataset table key.
        columns
            Optional column selection for the scan.
        scan_options
            Optional scan overrides (filter, ordering, metrics).

        Returns
        -------
        pyarrow.RecordBatchReader | None
            Reader for the dataset snapshot or None when missing.
        """
        resolved_scan_options = scan_options or GraphViewScanOptions()
        resolved_columns: tuple[str, ...] | Mapping[str, pc.Expression] | None
        if isinstance(columns, Mapping):
            resolved_columns = columns
        elif columns is None:
            resolved_columns = None
        else:
            resolved_columns = tuple(columns)
        request = SnapshotScanRequest(
            dataset_root=self.dataset_root,
            table_key=table_key,
            snapshot_id=self.snapshot_id,
            columns=resolved_columns,
            provenance=resolved_scan_options.provenance,
            repo=self.scan_context.repo,
            commit=self.scan_context.commit,
            scan_context=self.scan_context,
            apply_filter=resolved_scan_options.apply_filter,
            implicit_ordering=resolved_scan_options.implicit_ordering,
            require_sequenced_output=resolved_scan_options.require_sequenced_output,
            metrics_enabled=resolved_scan_options.metrics_enabled,
            execution_ctx=resolved_scan_options.execution_ctx,
        )
        return scan_snapshot_reader(request)

    def load_plan(
        self,
        *,
        table_key: str,
        columns: Sequence[str] | Mapping[str, pc.Expression] | None = None,
        scan_options: GraphViewScanOptions | None = None,
    ) -> Plan | None:
        """Return a scan plan for a snapshot table.

        Parameters
        ----------
        table_key
            Dataset table key.
        columns
            Optional column selection for the scan.
        scan_options
            Optional scan overrides (filter, ordering, metrics).

        Returns
        -------
        Plan | None
            Plan for the dataset snapshot or None when missing.
        """
        resolved_scan_options = scan_options or GraphViewScanOptions()
        resolved_columns: tuple[str, ...] | Mapping[str, pc.Expression] | None
        if isinstance(columns, Mapping):
            resolved_columns = columns
        elif columns is None:
            resolved_columns = None
        else:
            resolved_columns = tuple(columns)
        request = SnapshotScanRequest(
            dataset_root=self.dataset_root,
            table_key=table_key,
            snapshot_id=self.snapshot_id,
            columns=resolved_columns,
            provenance=resolved_scan_options.provenance,
            repo=self.scan_context.repo,
            commit=self.scan_context.commit,
            scan_context=self.scan_context,
            apply_filter=resolved_scan_options.apply_filter,
            implicit_ordering=resolved_scan_options.implicit_ordering,
            require_sequenced_output=resolved_scan_options.require_sequenced_output,
            metrics_enabled=resolved_scan_options.metrics_enabled,
            execution_ctx=resolved_scan_options.execution_ctx,
        )
        return scan_snapshot_plan(request)

    @staticmethod
    def iter_tuples(
        reader: pa.RecordBatchReader,
        *,
        columns: Sequence[str] | None = None,
    ) -> Iterable[tuple[object, ...]]:
        """Yield normalized row tuples from a record batch reader.

        Parameters
        ----------
        reader
            Reader supplying record batches.
        columns
            Optional column selection for tuple materialization.

        Yields
        ------
        tuple[object, ...]
            Row tuples in column order after normalization.
        """
        yield from iter_normalized_tuples(reader, columns=columns)


@dataclass(frozen=True, slots=True)
class GraphRunMetadata:
    """Run metadata captured for graph inputs and outputs."""

    determinism_tier: DedupeTier
    runtime_profile: str | None
    scan_profile: str | None

    def manifest_extras(self) -> dict[str, object]:
        """Return a manifest extras payload for this run metadata.

        Returns
        -------
        dict[str, object]
            Manifest extras payload containing run metadata fields.
        """
        extras: dict[str, object] = {"determinism_tier": self.determinism_tier}
        if self.runtime_profile is not None:
            extras["runtime_profile"] = self.runtime_profile
        if self.scan_profile is not None:
            extras["scan_profile"] = self.scan_profile
        return extras


@dataclass(frozen=True, slots=True)
class _FinalizeArtifactContext:
    dataset_root: Path
    snapshot_id: str
    base_table_key: str
    run_metadata: GraphRunMetadata | None


def resolve_dataset_root(
    _snapshot: SnapshotRef,
    dataset_root_dir: Path | None,
) -> Path | None:
    """Resolve the dataset root directory for a snapshot.

    Parameters
    ----------
    _snapshot
        Snapshot reference for repository context.
    dataset_root_dir
        Optional explicit dataset root directory.

    Returns
    -------
    pathlib.Path | None
        Resolved dataset root directory or None when not found.
    """
    return dataset_root_dir


def dataset_snapshot_exists(
    dataset_root: Path | None,
    table_key: str,
    snapshot_id: str,
) -> bool:
    """Return True when a dataset snapshot directory exists.

    Parameters
    ----------
    dataset_root
        Root directory for datasets.
    table_key
        Dataset table key.
    snapshot_id
        Snapshot identifier for the dataset.

    Returns
    -------
    bool
        True when the snapshot directory exists, otherwise False.
    """
    if dataset_root is None:
        return False
    try:
        snapshot_dir = dataset_snapshot_dir(
            dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except SnapshotIdError as exc:
        LOG.warning("Invalid snapshot_id for %s: %s", table_key, exc)
        return False
    return snapshot_dir.is_dir()


def scan_snapshot_plan(request: SnapshotScanRequest) -> Plan | None:
    """Return a scan plan for a dataset snapshot or None when missing.

    Parameters
    ----------
    request
        Snapshot scan request describing the dataset and filters.

    Returns
    -------
    Plan | None
        Plan for the dataset snapshot or None when missing.
    """
    dataset = _scan_dataset(request.dataset_root, request.table_key, request.snapshot_id)
    if dataset is None:
        return None
    scan_ctx = request.scan_context or SnapshotScanContext(
        repo=request.repo,
        commit=request.commit,
        settings=load_runtime_settings().build.arrow_scan,
    )
    filter_expression = scan_ctx.filter_expr(dataset.schema) if request.apply_filter else None
    resolved_columns = _resolve_columns(dataset, request.columns)
    if resolved_columns is None and request.columns is not None:
        return None
    query_spec = _query_spec_for_request(
        dataset,
        columns=resolved_columns,
        predicate=filter_expression,
    )
    if request.metrics_enabled:
        _log_scan_telemetry(
            dataset,
            table_key=request.table_key,
            snapshot_id=request.snapshot_id,
            query_spec=query_spec,
        )
    options = QueryPlanOptions(
        provenance=request.provenance,
        implicit_ordering=request.implicit_ordering,
        require_sequenced_output=request.require_sequenced_output,
    )
    options = _apply_canonical_order_by(request, options=options)
    return build_query_plan_for_context(
        dataset,
        spec=query_spec,
        ctx=request.execution_ctx,
        options=options,
    )


def scan_snapshot_reader(
    request: SnapshotScanRequest,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a dataset snapshot or None when missing.

    Parameters
    ----------
    request
        Snapshot scan request describing the dataset and filters.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader for the dataset snapshot or None when missing.
    """
    plan = scan_snapshot_plan(request)
    if plan is None:
        return None
    use_threads = _resolve_use_threads(request)
    execution_ctx = _resolve_execution_context(request, use_threads=use_threads)
    return ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)


def scan_snapshot_reader_with_columns(
    request: SnapshotScanRequest,
    *,
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
) -> pa.RecordBatchReader | None:
    """Return a RecordBatchReader for a dataset snapshot with selected columns.

    Returns
    -------
    pyarrow.RecordBatchReader | None
        Reader for the dataset snapshot or None when missing.
    """
    updated = replace(request, columns=columns)
    return scan_snapshot_reader(updated)


def scan_snapshot_table(
    request: SnapshotScanRequest,
) -> pa.Table | None:
    """Return a materialized Arrow Table for a dataset snapshot.

    Returns
    -------
    pyarrow.Table | None
        Arrow table for the dataset snapshot or None when missing.
    """
    plan = scan_snapshot_plan(request)
    if plan is None:
        return None
    use_threads = _resolve_use_threads(request)
    execution_ctx = _resolve_execution_context(request, use_threads=use_threads)
    result = run_pipeline(
        plan=ExecutionPlan.from_plan(plan),
        finalize=finalize_spec_for_table(request.table_key, mode="tolerant"),
        options=PipelineRunOptions(ctx=execution_ctx),
    )
    return result.good


def graph_execution_context(
    *,
    determinism: DedupeTier,
    provenance: bool,
) -> ExecutionContext:
    """Return an ExecutionContext configured for graph scans.

    Returns
    -------
    ExecutionContext
        Execution context configured for graph scan behavior.
    """
    runtime_profile = _resolve_graph_runtime_profile(default_name="graph_views")
    return ExecutionContext(
        determinism=determinism,
        provenance=provenance,
        runtime_profile=runtime_profile,
    )


def graph_run_metadata(
    *,
    determinism: DedupeTier,
    execution_ctx: ExecutionContext | None,
) -> GraphRunMetadata:
    """Build graph run metadata from determinism and execution context.

    Returns
    -------
    GraphRunMetadata
        Run metadata for graph outputs and artifacts.
    """
    profile = execution_ctx.runtime_profile if execution_ctx is not None else None
    runtime_name = profile.name if profile is not None else None
    scan_profile = profile.scan_profile if profile is not None else None
    return GraphRunMetadata(
        determinism_tier=determinism,
        runtime_profile=runtime_name,
        scan_profile=scan_profile,
    )


def _scan_dataset(dataset_root: Path, table_key: str, snapshot_id: str) -> ds.Dataset | None:
    try:
        return scan_dataset(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
    except FileNotFoundError:
        LOG.warning("Dataset snapshot missing for %s@%s", table_key, snapshot_id)
    except (OSError, ValueError, pa.ArrowInvalid) as exc:
        LOG.warning("Dataset scan failed for %s@%s: %s", table_key, snapshot_id, exc)
    return None


def _resolve_columns(
    dataset: ds.Dataset,
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
) -> tuple[str, ...] | Mapping[str, pc.Expression] | None:
    if columns is None:
        return None
    if isinstance(columns, Mapping):
        return columns
    available = set(dataset.schema.names)
    missing = [name for name in columns if name not in available]
    if missing:
        LOG.warning(
            "Dataset columns missing: %s (table=%s)",
            ", ".join(missing),
            dataset.schema,
        )
        return None
    return columns


def _apply_canonical_order_by(
    request: SnapshotScanRequest,
    *,
    options: QueryPlanOptions,
) -> QueryPlanOptions:
    if options.order_by is not None:
        return options
    ctx = request.execution_ctx
    if ctx is None or ctx.resolve_determinism() != "canonical":
        return options
    schema = get_schema_service().get_table_schema(request.table_key)
    canonical_keys = resolve_canonical_sort_keys(schema)
    if not canonical_keys:
        return options
    direction: SortDirection = "ascending"
    order_by: tuple[SortKey, ...] = tuple((key, direction) for key in canonical_keys)
    return replace(options, order_by=order_by)


def _query_spec_for_request(
    dataset: ds.Dataset,
    *,
    columns: tuple[str, ...] | Mapping[str, pc.Expression] | None,
    predicate: pc.Expression | None,
) -> QuerySpec:
    projection = projection_spec_from_columns(
        columns,
        default_columns=tuple(dataset.schema.names),
    )
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _log_scan_telemetry(
    dataset: ds.Dataset,
    *,
    table_key: str,
    snapshot_id: str,
    query_spec: QuerySpec,
) -> None:
    telemetry = scan_telemetry_for_queryspec(dataset, spec=query_spec)
    LOG.debug(
        "Dataset scan telemetry table=%s snapshot=%s fragments=%s rows=%s filter=%s",
        table_key,
        snapshot_id,
        telemetry.fragment_count,
        telemetry.estimated_rows,
        query_spec.scan_filter_expression(),
    )


def _resolve_use_threads(request: SnapshotScanRequest) -> bool:
    if request.use_threads is not None:
        return request.use_threads
    ctx = request.execution_ctx
    if ctx is None:
        return True
    return ctx.resolve_use_threads()


def _resolve_execution_context(
    request: SnapshotScanRequest,
    *,
    use_threads: bool,
) -> ExecutionContext:
    execution_ctx = request.execution_ctx
    if execution_ctx is None:
        return ExecutionContext(use_threads=use_threads)
    if request.use_threads is None:
        return execution_ctx
    return replace(execution_ctx, use_threads=use_threads)


def persist_finalize_artifacts(
    *,
    dataset_root: Path,
    snapshot_id: str,
    base_table_key: str,
    result: FinalizeResult,
    run_metadata: GraphRunMetadata | None = None,
) -> None:
    """Persist finalize artifacts for a graph input table."""
    context = _FinalizeArtifactContext(
        dataset_root=dataset_root,
        snapshot_id=snapshot_id,
        base_table_key=base_table_key,
        run_metadata=run_metadata,
    )
    _write_finalize_artifact_dataset(
        context=context,
        artifact="errors",
        table=result.errors,
    )
    _write_finalize_artifact_dataset(
        context=context,
        artifact="alignment",
        table=result.alignment,
    )
    _write_finalize_artifact_dataset(
        context=context,
        artifact="stats",
        table=result.stats,
    )


def _resolve_graph_runtime_profile(
    *,
    default_name: str,
) -> RuntimeProfile | None:
    settings = load_runtime_settings()
    profile = runtime_profile_from_settings(settings.columnar)
    if profile is None:
        return None
    scan_settings = settings.build.arrow_scan
    scan_profile = profile.scan_profile or scan_settings.profile
    use_threads = (
        profile.use_threads if profile.use_threads is not None else scan_settings.use_threads
    )
    implicit_ordering = (
        True if profile.implicit_ordering is None else profile.implicit_ordering
    )
    require_sequenced_output = (
        True
        if profile.require_sequenced_output is None
        else profile.require_sequenced_output
    )
    return replace(
        profile,
        name=profile.name or default_name,
        scan_profile=scan_profile,
        implicit_ordering=implicit_ordering,
        require_sequenced_output=require_sequenced_output,
        use_threads=use_threads,
    )


def _write_finalize_artifact_dataset(
    *,
    context: _FinalizeArtifactContext,
    artifact: str,
    table: pa.Table,
) -> None:
    artifact_table_key = _finalize_artifact_table_key(context.base_table_key, artifact)
    try:
        table_schema = table_schema_from_arrow_schema(
            arrow_schema=table.schema,
            table_key=artifact_table_key,
        )
        manifest_extras = _artifact_manifest_extras(
            table_schema,
            artifact_for=context.base_table_key,
            artifact_type=artifact,
            run_metadata=context.run_metadata,
        )
        options = ArrowDatasetWriteOptions(
            partition_columns=_partition_columns_for_schema(table_schema),
            schema_hash=schema_hash(table_schema),
            manifest_extras=manifest_extras,
            stable_sort_keys=_resolve_manifest_sort_keys(table_schema),
        )
        write_dataset(
            dataset_root=context.dataset_root,
            table_key=artifact_table_key,
            snapshot_id=context.snapshot_id,
            data=table,
            options=options,
        )
    except (OSError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        LOG.warning(
            "Finalize artifact write failed; table_key=%s artifact=%s error=%s",
            context.base_table_key,
            artifact,
            exc,
        )


def _finalize_artifact_table_key(base_table_key: str, artifact: str) -> str:
    return f"{base_table_key}__{artifact}"


def _partition_columns_for_schema(table_schema: TableSchema) -> tuple[str, ...]:
    column_names = set(table_schema.column_names())
    if "repo" in column_names and "commit" in column_names:
        return ("repo", "commit")
    return ()


def _resolve_manifest_sort_keys(table_schema: TableSchema) -> tuple[str, ...] | None:
    return resolve_canonical_sort_keys(table_schema)


def _artifact_manifest_extras(
    table_schema: TableSchema,
    *,
    artifact_for: str,
    artifact_type: str,
    run_metadata: GraphRunMetadata | None,
) -> dict[str, object]:
    extras: dict[str, object] = {
        "table_schema": table_schema.to_json_obj(),
        "artifact_for": artifact_for,
        "artifact_type": artifact_type,
        "written_at": datetime.now(tz=UTC).isoformat(),
    }
    if run_metadata is not None:
        extras["graph_run"] = run_metadata.manifest_extras()
    return extras


__all__ = [
    "GraphRunMetadata",
    "GraphViewFactory",
    "SnapshotScanRequest",
    "dataset_snapshot_exists",
    "graph_execution_context",
    "graph_run_metadata",
    "persist_finalize_artifacts",
    "resolve_dataset_root",
    "scan_snapshot_plan",
    "scan_snapshot_reader",
    "scan_snapshot_reader_with_columns",
    "scan_snapshot_table",
]
