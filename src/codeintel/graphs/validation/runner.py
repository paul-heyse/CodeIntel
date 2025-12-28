"""Main orchestration for running graph validations.

This module provides the high-level functions for executing the
full validation suite and coordinating individual checks.

Architecture Notes
------------------
This module imports from graphs.runtime for GraphRuntime access.

All validations use CheckProtocol-based validation via core.validation.ValidationRunner.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

from codeintel.core.catalog import load_function_catalog
from codeintel.core.validation.runner import ValidationRunner
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.graphs.validation.checks.anomaly import (
    ALL_ANOMALY_CHECKS,
    SubsystemDisagreementCheck,
    SymbolCommunityCheck,
)
from codeintel.graphs.validation.checks.database import (
    ALL_DATABASE_CHECKS,
    CallsiteSpanMismatchCheck,
    MissingFunctionGoidsCheck,
    OrphanModulesCheck,
)
from codeintel.graphs.validation.checks.structure import (
    ALL_STRUCTURE_CHECKS,
    CallGraphStructureCheck,
    ConfigKeyCheck,
    ImportGraphStructureCheck,
    SymbolGraphCheck,
)
from codeintel.graphs.validation.context import GraphValidationContext
from codeintel.graphs.validation.findings import (
    persist_findings,
    resolve_validation_options,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.helpers.sql_params import render_sql

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.core.validation.runner import ValidationReport
    from codeintel.graphs.engine import GraphEngine, NxGraphEngine
    from codeintel.graphs.validation.base import GraphCheckBase
    from codeintel.graphs.validation.findings import (
        GraphValidationOptions,
    )
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


def run_graph_validations_with_runner(
    gateway: StorageGateway,
    *,
    snapshot: SnapshotRef,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions,
    options: GraphValidationOptions | None = None,
) -> ValidationReport:
    """Run graph validations using core ValidationRunner.

    This function uses the CheckProtocol-based validation approach,
    enabling unified validation infrastructure across the codebase.

    Parameters
    ----------
    gateway : StorageGateway
        Storage gateway for database access.
    snapshot : SnapshotRef
        Repository snapshot reference.
    catalog_provider : FunctionCatalogProvider | None
        Optional catalog provider for function metadata.
    runtime : GraphRuntime | GraphRuntimeOptions
        Runtime or options for graph access.
    options : GraphValidationOptions | None
        Optional validation options.

    Returns
    -------
    ValidationReport
        Validation report with all findings and statistics.

    Raises
    ------
    RuntimeError
        When hard_fail is enabled and error-level findings are present.
    """
    validation_opts = resolve_validation_options(runtime=runtime, options=options)
    active_log = logging.getLogger(__name__)
    repo = snapshot.repo
    commit = snapshot.commit

    log_db_snapshot(gateway, repo, commit, active_log)

    catalog = (
        catalog_provider.catalog()
        if catalog_provider is not None
        else load_function_catalog(gateway, repo=snapshot.repo, commit=snapshot.commit)
    )

    resolved_runtime = resolve_validation_runtime(
        gateway,
        snapshot=snapshot,
        runtime=runtime,
    )
    engine: GraphEngine = resolved_runtime.engine

    # Build context for validation checks
    ctx = GraphValidationContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        engine=engine,
        catalog=catalog,
        runtime=resolved_runtime,
        logger=active_log,
    )

    # Create and run the validation runner
    runner = create_validation_runner(options=validation_opts)
    report = runner.run(ctx)

    # Persist findings
    persist_findings(gateway, report.findings, repo, commit)

    active_log.info(
        "Graph validation completed for %s@%s: %d finding(s), %d checks run, %d skipped, %d failed",
        repo,
        commit,
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
    report = runner.run(ctx)
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
        where: str | None = None,
        params: dict[str, object] | None = None,
    ) -> int:
        sql = f"SELECT COUNT(*) FROM {table_key}"
        if where:
            sql += f" WHERE {where}"
        try:
            result = gateway.con.sql(render_sql(sql, params)).fetchone()
            if result is None:
                return 0
            return int(cast("SupportsInt", result[0]))
        except DuckDBError as exc:
            log.warning("Validation snapshot count failed for %s: %s", table_key, exc)
            return -1

    counts = {
        "modules": _count(
            "core.modules",
            where="repo = $repo AND commit = $commit",
            params={"repo": repo, "commit": commit},
        ),
        "goids": _count(
            "core.goids",
            where="repo = $repo AND commit = $commit",
            params={"repo": repo, "commit": commit},
        ),
        "module_goids": _count(
            "core.goids",
            where="repo = $repo AND commit = $commit AND kind = 'module'",
            params={"repo": repo, "commit": commit},
        ),
        "class_goids": _count(
            "core.goids",
            where="repo = $repo AND commit = $commit AND kind = 'class'",
            params={"repo": repo, "commit": commit},
        ),
        "function_goids": _count(
            "core.goids",
            where="repo = $repo AND commit = $commit AND kind IN ('function', 'method')",
            params={"repo": repo, "commit": commit},
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
    "create_validation_runner",
    "log_db_snapshot",
    "resolve_validation_runtime",
    "run_graph_validations_with_runner",
]
