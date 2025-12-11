"""Main orchestration for running graph validations.

This module provides the high-level functions for executing the
full validation suite and coordinating individual checks.

Architecture Notes
------------------
This module imports from analytics.graph_runtime for GraphRuntime access.
This is an intentional delegation - the graphs package orchestrates validation
but delegates runtime resolution to analytics (Option B architecture).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ibis.expr.types import Table

from codeintel.analytics.runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.catalog import load_function_catalog
from codeintel.graphs.engine import GraphEngine
from codeintel.graphs.validation.checks import (
    warn_callsite_span_mismatches,
    warn_graph_structure,
    warn_missing_function_goids,
    warn_orphan_modules,
)
from codeintel.graphs.validation.findings import (
    GraphValidationOptions,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
    persist_findings,
    resolve_validation_options,
)
from codeintel.storage.gateway import DuckDBError, StorageGateway

if TYPE_CHECKING:
    from codeintel.graphs.catalog import FunctionCatalogProvider


def run_graph_validations(
    gateway: StorageGateway,
    *,
    snapshot: SnapshotRef,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions,
    options: GraphValidationOptions | None = None,
) -> None:
    """
    Emit warnings for common graph integrity issues.

    Checks include:
    - Files with functions in AST that are missing GOIDs.
    - Call graph edges whose callsites lie outside caller spans.
    - Modules with no GOIDs (orphans).

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
    findings = []
    findings.extend(warn_missing_function_goids(gateway, repo, commit, active_log))
    findings.extend(warn_callsite_span_mismatches(gateway, catalog, repo, commit, active_log))
    findings.extend(warn_orphan_modules(gateway, repo, commit, active_log, catalog))
    findings.extend(warn_graph_structure(engine, repo, commit, active_log))
    normalized_findings = apply_severity_overrides(findings, validation_opts.severity_overrides)
    capped_findings = cap_findings(normalized_findings, validation_opts.max_findings_per_rule)
    persist_findings(gateway, capped_findings, repo, commit)
    active_log.info(
        "Graph validation completed for %s@%s: %d finding(s)",
        repo,
        commit,
        len(capped_findings),
    )
    if validation_opts.hard_fail and has_error_findings(capped_findings):
        message = "Graph validation failed with error-level findings"
        raise RuntimeError(message)


def resolve_validation_runtime(
    gateway: StorageGateway,
    *,
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions,
) -> GraphRuntime:
    """
    Resolve the runtime for validation, ensuring snapshot consistency.

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
    """
    Record table counts to aid debugging validation state.

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
    def _count(table_expr: Table, *, filters: list | None = None) -> int:
        try:
            expr = table_expr if not filters else table_expr.filter(*filters)
            return int(expr.count().execute())
        except DuckDBError as exc:  # pragma: no cover - defensive logging
            log.warning("Validation snapshot count failed for %s: %s", table_expr, exc)
            return -1

    modules_tbl = gateway.ibis.table("core.modules")
    goids_tbl = gateway.ibis.table("core.goids")
    call_nodes_tbl = gateway.ibis.table("graph.call_graph_nodes")
    call_edges_tbl = gateway.ibis.table("graph.call_graph_edges")

    counts = {
        "modules": _count(
            modules_tbl,
            filters=[(modules_tbl.repo == repo) & (modules_tbl.commit == commit)],
        ),
        "goids": _count(
            goids_tbl,
            filters=[(goids_tbl.repo == repo) & (goids_tbl.commit == commit)],
        ),
        "module_goids": _count(
            goids_tbl,
            filters=[
                (goids_tbl.repo == repo)
                & (goids_tbl.commit == commit)
                & (goids_tbl.kind == "module")
            ],
        ),
        "class_goids": _count(
            goids_tbl,
            filters=[
                (goids_tbl.repo == repo)
                & (goids_tbl.commit == commit)
                & (goids_tbl.kind == "class")
            ],
        ),
        "function_goids": _count(
            goids_tbl,
            filters=[
                (goids_tbl.repo == repo)
                & (goids_tbl.commit == commit)
                & goids_tbl.kind.isin(["function", "method"])
            ],
        ),
        "call_nodes": _count(call_nodes_tbl),
        "call_edges": _count(call_edges_tbl),
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


__all__ = [
    "log_db_snapshot",
    "resolve_validation_runtime",
    "run_graph_validations",
]
