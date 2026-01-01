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
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

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
from codeintel.core.catalog import load_function_catalog
from codeintel.core.validation.runner import ValidationRunner
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression, Expression
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from codeintel.build.graphs.engine import NxGraphEngine
    from codeintel.build.graphs.validation.findings import (
        GraphValidationOptions,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.core.validation.runner import CheckProtocol, ValidationReport
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Check Registration
# =============================================================================


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
        MissingFunctionGoidsCheck(),
        CallsiteSpanMismatchCheck(),
        OrphanModulesCheck(),
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


def _ensure_dataset_root(
    gateway: StorageGateway,
    dataset_root_dir: Path | None,
) -> None:
    if dataset_root_dir is None:
        return
    if gateway.datasets.dataset_root_dir == dataset_root_dir:
        return
    gateway.datasets = gateway.datasets.with_dataset_root(dataset_root_dir)


def run_graph_validations_with_runner(
    gateway: StorageGateway,
    *,
    request: GraphValidationRunRequest,
) -> ValidationReport:
    """Run graph validations using core ValidationRunner.

    This function uses the CheckProtocol-based validation approach,
    enabling unified validation infrastructure across the codebase.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway for database access.
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

    _ensure_dataset_root(gateway, request.dataset_root_dir)
    log_db_snapshot(gateway, snapshot.repo, snapshot.commit, active_log)

    catalog = (
        request.catalog_provider.catalog()
        if request.catalog_provider is not None
        else load_function_catalog(gateway, repo=snapshot.repo, commit=snapshot.commit)
    )

    resolved_runtime = resolve_validation_runtime(
        gateway,
        snapshot=snapshot,
        runtime=request.runtime,
    )

    # Build context for validation checks
    ctx = GraphValidationContext(
        gateway=gateway,
        repo=snapshot.repo,
        commit=snapshot.commit,
        engine=resolved_runtime.engine,
        catalog=catalog,
        runtime=resolved_runtime,
        logger=active_log,
    )

    missing_by_check = _parquet_validation_skips(gateway, active_log)
    check_filter = (
        _parquet_check_filter(missing_by_check, active_log) if missing_by_check else None
    )

    # Create and run the validation runner
    runner = create_validation_runner(options=validation_opts)
    report = runner.run(ctx, check_filter=check_filter)

    # Persist findings
    persist_findings(gateway, report.findings, snapshot.repo, snapshot.commit)

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
        engine.gateway,
        snapshot=snapshot,
        runtime=GraphRuntimeOptions(snapshot=snapshot, engine=engine),
    )
    validation_opts = resolve_validation_options(runtime=runtime, options=None)
    runner = create_validation_runner(options=validation_opts)
    catalog = load_function_catalog(engine.gateway, repo=repo, commit=commit)
    ctx = GraphValidationContext(
        gateway=engine.gateway,
        repo=repo,
        commit=commit,
        engine=engine,
        catalog=catalog,
        runtime=runtime,
        logger=active_log,
    )
    missing_by_check = _parquet_validation_skips(engine.gateway, active_log)
    check_filter = (
        _parquet_check_filter(missing_by_check, active_log) if missing_by_check else None
    )
    report = runner.run(ctx, check_filter=check_filter)
    return report.findings


# =============================================================================
# Helper Functions
# =============================================================================


def resolve_validation_runtime(
    gateway: StorageGateway,
    *,
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions,
) -> GraphRuntime:
    """Resolve the runtime for validation, ensuring snapshot consistency.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway for database access.
    snapshot : SnapshotRef
        Repository snapshot reference.
    runtime : GraphRuntime | GraphRuntimeOptions
        Runtime or options to resolve.

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
    return resolve_graph_runtime(gateway, snapshot, options)


def log_db_snapshot(gateway: StorageGateway, repo: str, commit: str, log: logging.Logger) -> None:
    """Record table counts to aid debugging validation state.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway for database access.
    repo : str
        Repository identifier.
    commit : str
        Commit identifier.
    log : logging.Logger
        Logger for output.
    """

    def _count(
        table_key: str,
        *,
        predicate: Expression | None = None,
    ) -> int:
        try:
            if not _require_parquet_table(gateway, table_key, log):
                return -1
            relation = gateway.relation_from_table_key(table_key)
            if predicate is not None:
                relation = relation.filter(predicate)
            result = relation.aggregate("count(*) as cnt").fetchone()
            if result is None:
                return 0
            return int(cast("SupportsInt", result[0]))
        except DuckDBError as exc:
            log.warning("Validation snapshot count failed for %s: %s", table_key, exc)
            return -1

    snapshot_predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
        ColumnExpression("commit") == ConstantExpression(commit)
    )
    counts = {
        "modules": _count("core.modules", predicate=snapshot_predicate),
        "goids": _count("core.goids", predicate=snapshot_predicate),
        "module_goids": _count(
            "core.goids",
            predicate=snapshot_predicate
            & (ColumnExpression("kind") == ConstantExpression("module")),
        ),
        "class_goids": _count(
            "core.goids",
            predicate=snapshot_predicate
            & (ColumnExpression("kind") == ConstantExpression("class")),
        ),
        "function_goids": _count(
            "core.goids",
            predicate=snapshot_predicate
            & ColumnExpression("kind").isin(
                ConstantExpression("function"),
                ConstantExpression("method"),
            ),
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
    gateway: StorageGateway,
    log: logging.Logger,
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
        missing = _missing_parquet_tables(gateway, table_keys, log)
        if missing:
            missing_by_check[check_cls] = tuple(missing)
    return missing_by_check


def _missing_parquet_tables(
    gateway: StorageGateway,
    table_keys: tuple[str, ...],
    log: logging.Logger,
) -> list[str]:
    return [
        table_key
        for table_key in table_keys
        if not _require_parquet_table(gateway, table_key, log)
    ]


def _require_parquet_table(gateway: StorageGateway, table_key: str, log: logging.Logger) -> bool:
    schema, table = split_table_key(table_key)
    try:
        row = gateway.execute(
            """
            SELECT table_type
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            LIMIT 1
            """,
            [schema, table],
        ).fetchone()
    except DuckDBError as exc:
        log.warning("Validation table lookup failed for %s: %s", table_key, exc)
        return False
    if row is None:
        log.warning("Validation table missing: %s", table_key)
        return False
    table_type = str(row[0] or "").upper()
    if table_type not in {"BASE TABLE", "TABLE"}:
        log.warning("Validation expects Parquet base table for %s, found %s", table_key, table_type)
        return False
    return True


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
