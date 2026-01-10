"""Main orchestration for running graph validations.

This module provides the high-level functions for executing the
full validation suite and coordinating individual checks.

Architecture Notes
------------------
This module imports from graphs.runtime for GraphRuntime access.

All validations use CheckProtocol-based validation via core.validation.ValidationRunner.
Validation is expected to run post-materialization against Parquet-backed base tables.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.utilities.catalogs import (
    CatalogProviderRequest,
    catalog_provider_from_frames,
)
from codeintel.build.graphs.engine.datasets import (
    SnapshotScanRequest,
    dataset_snapshot_exists,
    resolve_dataset_root,
    scan_snapshot_reader,
)
from codeintel.build.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.graphs.validation.checks.anomaly import (
    ALL_ANOMALY_CHECKS,
    SubsystemDisagreementCheck,
    SymbolCommunityCheck,
)
from codeintel.build.graphs.validation.checks.database import (
    ALL_DATABASE_CHECKS,
    CallsiteSpanMismatchCheck,
    MissingFunctionGoidsCheck,
    OrphanModulesCheck,
)
from codeintel.build.graphs.validation.checks.structure import (
    ALL_STRUCTURE_CHECKS,
    CallGraphStructureCheck,
    ConfigKeyCheck,
    ImportGraphStructureCheck,
    SymbolGraphCheck,
)
from codeintel.build.graphs.validation.context import GraphValidationContext
from codeintel.build.graphs.validation.findings import (
    persist_findings,
    resolve_validation_options,
)
from codeintel.build.scopes.snapshot import SnapshotScanContext
from codeintel.build.settings import get_arrow_scan_settings, get_columnar_runtime_settings
from codeintel.core.columnar.execution_context import (
    ExecutionContext as ColumnarExecutionContext,
)
from codeintel.core.columnar.execution_context import (
    RuntimeProfile,
    resolve_arrow_scan_settings,
    runtime_profile_from_settings,
)
from codeintel.core.columnar.iter import iter_rows_limit
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.queryspec import (
    PROVENANCE_FIELDS,
    QuerySpec,
    projection_spec_from_schema_defaults,
)
from codeintel.core.columnar.run_manifest import (
    RunManifestOptions,
    run_manifest_options_for_context,
    write_run_manifest,
)
from codeintel.core.columnar.runtime import apply_runtime_profile
from codeintel.core.columnar.streaming import ScanTelemetry, scan_telemetry_for_queryspec
from codeintel.core.constants import DEFAULT_ARROW_USE_THREADS
from codeintel.core.datasets.arrow_store import scan_dataset
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.validation.runner import ValidationRunner

if TYPE_CHECKING:
    from codeintel.build.graphs.engine import NxGraphEngine
    from codeintel.build.graphs.validation.findings import (
        GraphValidationOptions,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.columnar.ordering import SortKey
    from codeintel.core.validation.runner import CheckProtocol, ValidationReport
    from codeintel.storage.catalog import FunctionCatalogProvider


# =============================================================================
# Check Registration
# =============================================================================


_ROW_FILTER_LIMIT = 1_000_000
_MANIFEST_TABLE_CANDIDATES: tuple[str, ...] = (
    "graph.call_graph_nodes",
    "graph.call_graph_edges",
    "core.modules",
    "core.goids",
)


def create_validation_runner(
    options: GraphValidationOptions | None = None,
) -> ValidationRunner[GraphValidationContext]:
    """Create a ValidationRunner with all registered graph checks.

    Parameters
    ----------
    options
        Optional validation options for severity overrides and capping.

    Returns
    -------
    ValidationRunner
        Configured runner with all graph validation checks.
    """
    runner: ValidationRunner[GraphValidationContext] = ValidationRunner(
        options=options,
    )

    # Register all check classes
    all_checks: list[GraphCheckBase] = [
        # Database integrity checks
        *(check() for check in ALL_DATABASE_CHECKS),
        # Structure checks
        CallGraphStructureCheck(),
        ImportGraphStructureCheck(),
        SymbolGraphCheck(),
        ConfigKeyCheck(),
        # Anomaly checks
        SymbolCommunityCheck(),
        SubsystemDisagreementCheck(),
    ]

    for check in all_checks:
        runner.register(check)

    return runner


# =============================================================================
# Primary Validation Functions
# =============================================================================


@dataclass(frozen=True)
class GraphValidationRunRequest:
    """Inputs required to run graph validations."""

    snapshot: SnapshotRef
    runtime: GraphRuntime | GraphRuntimeOptions
    catalog_provider: FunctionCatalogProvider | None = None
    options: GraphValidationOptions | None = None
    dataset_root_dir: Path | None = None
    execution_ctx: ColumnarExecutionContext | None = None
    output_dir: Path | None = None


def run_graph_validations_with_runner(
    *,
    request: GraphValidationRunRequest,
) -> ValidationReport:
    """Run graph validations using core ValidationRunner.

    This function uses the CheckProtocol-based validation approach,
    enabling unified validation infrastructure across the codebase.

    Parameters
    ----------
    request : GraphValidationRunRequest
        Run parameters including snapshot, runtime, and optional overrides.

    Returns
    -------
    ValidationReport
        Validation report with all findings and statistics.

    Raises
    ------
    RuntimeError
        When hard_fail is enabled and error-level findings are present.
    """
    snapshot = request.snapshot
    validation_opts = resolve_validation_options(
        runtime=request.runtime,
        options=request.options,
    )
    active_log = logging.getLogger(__name__)

    dataset_root_dir = resolve_dataset_root(snapshot, request.dataset_root_dir)
    execution_ctx = _resolve_columnar_ctx(request)
    log_db_snapshot(
        dataset_root_dir,
        snapshot.repo,
        snapshot.commit,
        active_log,
        execution_ctx=execution_ctx,
    )

    catalog_provider = request.catalog_provider or _catalog_provider_from_dataset(
        dataset_root_dir=dataset_root_dir,
        snapshot=snapshot,
        execution_ctx=execution_ctx,
    )
    catalog = catalog_provider.catalog() if catalog_provider is not None else None

    resolved_runtime = resolve_validation_runtime(
        snapshot=snapshot,
        runtime=request.runtime,
        dataset_root_dir=dataset_root_dir,
    )

    # Build context for validation checks
    ctx = GraphValidationContext(
        dataset_root_dir=dataset_root_dir,
        repo=snapshot.repo,
        commit=snapshot.commit,
        engine=resolved_runtime.engine,
        catalog=catalog,
        runtime=resolved_runtime,
        logger=active_log,
    )

    missing_by_check = _parquet_validation_skips(dataset_root_dir, snapshot.commit, active_log)
    check_filter = _parquet_check_filter(missing_by_check, active_log) if missing_by_check else None

    # Create and run the validation runner
    runner = create_validation_runner(options=validation_opts)
    report = runner.run(ctx, check_filter=check_filter)

    # Persist findings
    persist_findings(dataset_root_dir, report.findings, snapshot.repo, snapshot.commit)
    _emit_validation_run_manifest(
        request=request,
        dataset_root_dir=dataset_root_dir,
        execution_ctx=execution_ctx,
    )

    active_log.info(
        "Graph validation completed for %s@%s: %d finding(s), %d checks run, %d skipped, %d failed",
        snapshot.repo,
        snapshot.commit,
        len(report.findings),
        report.checks_run,
        report.checks_skipped,
        report.checks_failed,
    )

    if validation_opts.hard_fail and report.has_errors:
        message = "Graph validation failed with error-level findings"
        raise RuntimeError(message)

    return report


def warn_graph_structure(
    engine: NxGraphEngine,
    repo: str,
    commit: str,
    log: logging.Logger | None = None,
) -> list[dict[str, object]]:
    """Run graph validations and return the resulting findings.

    Parameters
    ----------
    engine
        Graph engine bound to a snapshot.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Optional logger used for validation output.

    Returns
    -------
    list[dict[str, object]]
        Validation findings emitted by registered checks.
    """
    active_log = log or logging.getLogger(__name__)
    snapshot = engine.snapshot
    runtime = resolve_validation_runtime(
        snapshot=snapshot,
        runtime=GraphRuntimeOptions(
            snapshot=snapshot,
            engine=engine,
            dataset_root_dir=engine.dataset_root_dir,
        ),
        dataset_root_dir=engine.dataset_root_dir,
    )
    validation_opts = resolve_validation_options(runtime=runtime, options=None)
    runner = create_validation_runner(options=validation_opts)
    catalog_provider = _catalog_provider_from_dataset(
        dataset_root_dir=engine.dataset_root_dir,
        snapshot=snapshot,
    )
    catalog = catalog_provider.catalog() if catalog_provider is not None else None
    ctx = GraphValidationContext(
        dataset_root_dir=engine.dataset_root_dir,
        repo=repo,
        commit=commit,
        engine=engine,
        catalog=catalog,
        runtime=runtime,
        logger=active_log,
    )
    missing_by_check = _parquet_validation_skips(engine.dataset_root_dir, commit, active_log)
    check_filter = _parquet_check_filter(missing_by_check, active_log) if missing_by_check else None
    report = runner.run(ctx, check_filter=check_filter)
    return report.findings


# =============================================================================
# Helper Functions
# =============================================================================


def _scan_snapshot_reader(request: SnapshotScanRequest) -> pa.RecordBatchReader | None:
    return scan_snapshot_reader(request)


def _resolve_columnar_ctx(
    request: GraphValidationRunRequest,
) -> ColumnarExecutionContext:
    if request.execution_ctx is not None:
        return request.execution_ctx
    scan_settings = get_arrow_scan_settings()
    resolved_use_threads = (
        DEFAULT_ARROW_USE_THREADS
        if scan_settings.use_threads is None
        else scan_settings.use_threads
    )
    runtime_profile = runtime_profile_from_settings(get_columnar_runtime_settings())
    if runtime_profile is None:
        runtime_profile = RuntimeProfile(name="graph_validation")
    use_threads = (
        runtime_profile.use_threads
        if runtime_profile.use_threads is not None
        else scan_settings.use_threads
    )
    implicit_ordering = (
        True if runtime_profile.implicit_ordering is None else runtime_profile.implicit_ordering
    )
    require_sequenced_output = (
        True
        if runtime_profile.require_sequenced_output is None
        else runtime_profile.require_sequenced_output
    )
    runtime_profile = replace(
        runtime_profile,
        name=runtime_profile.name or "graph_validation",
        implicit_ordering=implicit_ordering,
        require_sequenced_output=require_sequenced_output,
        use_threads=use_threads,
    )
    return ColumnarExecutionContext(
        use_threads=resolved_use_threads,
        runtime_profile=runtime_profile,
        scan_settings=scan_settings,
    )


def _emit_validation_run_manifest(
    *,
    request: GraphValidationRunRequest,
    dataset_root_dir: Path | None,
    execution_ctx: ColumnarExecutionContext,
) -> None:
    output_dir = request.output_dir
    snapshot_id = request.snapshot.commit
    if output_dir is None or dataset_root_dir is None or not snapshot_id:
        return
    table_key = _manifest_table_key(dataset_root_dir, snapshot_id)
    if table_key is None:
        return
    telemetry = _scan_telemetry_for_manifest(
        dataset_root_dir=dataset_root_dir,
        table_key=table_key,
        snapshot=request.snapshot,
        execution_ctx=execution_ctx,
    )
    ordering = _manifest_ordering(table_key, execution_ctx=execution_ctx)
    threading_snapshot = apply_runtime_profile(
        execution_ctx.runtime_profile,
        settings=resolve_arrow_scan_settings(execution_ctx),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        options = run_manifest_options_for_context(
            ctx=execution_ctx,
            ordering=ordering,
            scan_telemetry=telemetry,
            options=RunManifestOptions(
                extras={
                    "table_key": table_key,
                    "snapshot_id": snapshot_id,
                    "repo": request.snapshot.repo,
                    "arrow_threading": threading_snapshot.to_mapping(),
                },
                filename="run_manifest_graph_validation.json",
            ),
        )
        write_run_manifest(output_dir, options=options)
    except (OSError, TypeError, ValueError) as exc:
        logging.getLogger(__name__).warning("Graph validation manifest emission failed: %s", exc)


def _manifest_table_key(dataset_root_dir: Path, snapshot_id: str) -> str | None:
    for table_key in _MANIFEST_TABLE_CANDIDATES:
        if dataset_snapshot_exists(dataset_root_dir, table_key, snapshot_id):
            return table_key
    return None


def _scan_telemetry_for_manifest(
    *,
    dataset_root_dir: Path,
    table_key: str,
    snapshot: SnapshotRef,
    execution_ctx: ColumnarExecutionContext,
) -> ScanTelemetry:
    try:
        dataset = scan_dataset(
            dataset_root=dataset_root_dir,
            table_key=table_key,
            snapshot_id=snapshot.commit,
        )
    except (FileNotFoundError, OSError, ValueError, pa.ArrowInvalid):
        return _empty_scan_telemetry()
    scan_ctx = SnapshotScanContext.from_snapshot(snapshot)
    predicate = scan_ctx.filter_expr(dataset.schema)
    table_schema = get_schema_service().get_table_schema(table_key)
    available_columns = tuple(dataset.schema.names)
    provenance_columns = _provenance_columns_for_spec(
        execution_ctx=execution_ctx,
        available_columns=available_columns,
    )
    projection = projection_spec_from_schema_defaults(
        None,
        table_schema=table_schema,
        available_columns=available_columns,
        provenance_columns=provenance_columns,
    )
    spec = QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )
    return scan_telemetry_for_queryspec(dataset, spec=spec)


def _provenance_columns_for_spec(
    *,
    execution_ctx: ColumnarExecutionContext,
    available_columns: Sequence[str],
) -> tuple[str, ...]:
    provenance = execution_ctx.provenance
    profile = execution_ctx.runtime_profile
    if profile is not None:
        provenance = profile.resolve_provenance(default=provenance)
    if execution_ctx.resolve_determinism() == "canonical":
        provenance = True
    if not provenance:
        return ()
    available = set(available_columns)
    return tuple(
        output_name
        for output_name, _source_name in PROVENANCE_FIELDS
        if output_name in available
    )


def _empty_scan_telemetry() -> ScanTelemetry:
    return ScanTelemetry(fragment_count=None, estimated_rows=None)


def _manifest_ordering(
    table_key: str,
    *,
    execution_ctx: ColumnarExecutionContext,
) -> OrderingSpec | None:
    implicit_ordering = True
    profile = execution_ctx.runtime_profile
    if profile is not None:
        implicit_ordering = profile.resolve_implicit_ordering(default=implicit_ordering)
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        schema = None
    sort_keys = resolve_canonical_sort_keys(schema)
    if sort_keys:
        keys: list[SortKey] = [(key, "ascending") for key in sort_keys]
        return OrderingSpec.explicit(keys=keys, reason="canonical sort keys")
    if implicit_ordering:
        return OrderingSpec.implicit(reason="implicit scan ordering")
    return OrderingSpec.unordered(reason="scan unordered")


def _profile_name(execution_ctx: ColumnarExecutionContext) -> str | None:
    profile = execution_ctx.runtime_profile
    return None if profile is None else profile.name


def _scan_profile_name(execution_ctx: ColumnarExecutionContext) -> str | None:
    profile = execution_ctx.runtime_profile
    return None if profile is None else profile.scan_profile


def _catalog_provider_from_dataset(
    *,
    dataset_root_dir: Path | None,
    snapshot: SnapshotRef,
    execution_ctx: ColumnarExecutionContext | None = None,
) -> FunctionCatalogProvider | None:
    if dataset_root_dir is None:
        return None
    goids_reader = _scan_snapshot_reader(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.goids",
            snapshot_id=snapshot.commit,
            columns=(
                "goid_h128",
                "urn",
                "rel_path",
                "kind",
                "qualname",
                "start_line",
                "end_line",
                "repo",
                "commit",
            ),
            repo=snapshot.repo,
            commit=snapshot.commit,
            execution_ctx=execution_ctx,
        )
    )
    modules_reader = _scan_snapshot_reader(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.modules",
            snapshot_id=snapshot.commit,
            columns=("path", "module", "repo", "commit"),
            repo=snapshot.repo,
            commit=snapshot.commit,
            execution_ctx=execution_ctx,
        )
    )
    if goids_reader is None or modules_reader is None:
        return None
    return catalog_provider_from_frames(
        CatalogProviderRequest(
            goids_frame=goids_reader,
            modules_frame=modules_reader,
        )
    )


def resolve_validation_runtime(
    *,
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions,
    dataset_root_dir: Path | None,
) -> GraphRuntime:
    """Resolve the runtime for validation, ensuring snapshot consistency.

    Parameters
    ----------
    snapshot : SnapshotRef
        Repository snapshot reference.
    runtime : GraphRuntime | GraphRuntimeOptions
        Runtime or options to resolve.
    dataset_root_dir : Path | None
        Dataset root used for Parquet-backed graph loading.

    Returns
    -------
    GraphRuntime
        Resolved runtime with consistent snapshot.

    Raises
    ------
    ValueError
        When runtime snapshot doesn't match the provided snapshot.
    """
    runtime_snapshot = (
        runtime.options.snapshot if isinstance(runtime, GraphRuntime) else runtime.snapshot
    )
    if runtime_snapshot is not None and (
        runtime_snapshot.repo != snapshot.repo or runtime_snapshot.commit != snapshot.commit
    ):
        message = "GraphRuntime snapshot mismatch for validation run"
        raise ValueError(message)

    if isinstance(runtime, GraphRuntime):
        return runtime

    options = runtime if runtime.snapshot is not None else replace(runtime, snapshot=snapshot)
    if options.dataset_root_dir is None and dataset_root_dir is not None:
        options = replace(options, dataset_root_dir=dataset_root_dir)
    return resolve_graph_runtime(snapshot, options)


def log_db_snapshot(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
    *,
    execution_ctx: ColumnarExecutionContext | None = None,
) -> None:
    """Record table counts to aid debugging validation state."""

    def _count(
        table_key: str,
        *,
        row_filter: Callable[[dict[str, object]], bool] | None = None,
    ) -> int:
        if dataset_root_dir is None:
            return -1
        reader = _scan_snapshot_reader(
            SnapshotScanRequest(
                dataset_root=dataset_root_dir,
                table_key=table_key,
                snapshot_id=commit,
                columns=None,
                repo=repo,
                commit=commit,
                execution_ctx=execution_ctx,
            )
        )
        if reader is None:
            return -1
        if row_filter is None:
            return sum(batch.num_rows for batch in reader)
        count = 0
        for row in iter_rows_limit(reader, limit=_ROW_FILTER_LIMIT):
            if row_filter(row):
                count += 1
        return count

    def _kind_matches(values: set[str]) -> Callable[[dict[str, object]], bool]:
        def _predicate(row: dict[str, object]) -> bool:
            kind = row.get("kind")
            return kind is not None and str(kind) in values

        return _predicate

    counts = {
        "modules": _count("core.modules"),
        "goids": _count("core.goids"),
        "module_goids": _count(
            "core.goids",
            row_filter=_kind_matches({"module"}),
        ),
        "class_goids": _count(
            "core.goids",
            row_filter=_kind_matches({"class"}),
        ),
        "function_goids": _count(
            "core.goids",
            row_filter=_kind_matches({"function", "method"}),
        ),
        "call_nodes": _count("graph.call_graph_nodes"),
        "call_edges": _count("graph.call_graph_edges"),
    }
    snapshot = (
        f"[graph_validation] repo={repo} commit={commit} "
        f"modules={counts['modules']} goids={counts['goids']} "
        f"module_goids={counts['module_goids']} class_goids={counts['class_goids']} "
        f"function_goids={counts['function_goids']} "
        f"call_nodes={counts['call_nodes']} call_edges={counts['call_edges']}"
    )
    log.info(snapshot)
    _append_log(snapshot)


def _append_log(message: str) -> None:
    """Append a timestamped line to build/logs/pipeline.log for offline inspection."""
    log_path = Path("build/logs/pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


def _parquet_check_filter(
    missing_by_check: dict[type[GraphCheckBase], tuple[str, ...]],
    log: logging.Logger,
) -> Callable[[CheckProtocol[GraphValidationContext]], bool]:
    def _filter(check: CheckProtocol[GraphValidationContext]) -> bool:
        if not isinstance(check, GraphCheckBase):
            return True
        missing = missing_by_check.get(type(check))
        if not missing:
            return True
        log.warning(
            "Skipping graph validation check %s; missing Parquet tables: %s",
            check.name,
            ", ".join(missing),
        )
        return False

    return _filter


def _parquet_validation_skips(
    dataset_root_dir: Path | None,
    snapshot_id: str,
    _log: logging.Logger,
) -> dict[type[GraphCheckBase], tuple[str, ...]]:
    checks = {
        MissingFunctionGoidsCheck: ("core.ast_nodes", "core.goids"),
        CallsiteSpanMismatchCheck: ("graph.call_graph_edges",),
        OrphanModulesCheck: ("core.modules", "core.goids"),
        SymbolCommunityCheck: ("analytics.symbol_graph_metrics_modules",),
        SubsystemDisagreementCheck: ("analytics.subsystem_agreement",),
    }
    missing_by_check: dict[type[GraphCheckBase], tuple[str, ...]] = {}
    for check_cls, table_keys in checks.items():
        missing = _missing_parquet_tables(dataset_root_dir, snapshot_id, table_keys)
        if missing:
            missing_by_check[check_cls] = tuple(missing)
    return missing_by_check


def _missing_parquet_tables(
    dataset_root_dir: Path | None,
    snapshot_id: str,
    table_keys: tuple[str, ...],
) -> list[str]:
    if dataset_root_dir is None:
        return list(table_keys)
    return [
        table_key
        for table_key in table_keys
        if not dataset_snapshot_exists(dataset_root_dir, table_key, snapshot_id)
    ]


# =============================================================================
# Exports
# =============================================================================

# All check class tuples for external registration
ALL_GRAPH_CHECKS: tuple[type[GraphCheckBase], ...] = (
    *ALL_DATABASE_CHECKS,
    *ALL_STRUCTURE_CHECKS,
    *ALL_ANOMALY_CHECKS,
)

__all__ = [
    # Check class tuples
    "ALL_GRAPH_CHECKS",
    # Functions
    "GraphValidationRunRequest",
    "create_validation_runner",
    "log_db_snapshot",
    "resolve_validation_runtime",
    "run_graph_validations_with_runner",
]
