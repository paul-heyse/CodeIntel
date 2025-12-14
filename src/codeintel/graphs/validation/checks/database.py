"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation; legacy
function wrappers are provided for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import ibis

from codeintel.graphs.validation.base import GraphCheckBase
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import filter_by, ibis_bool, isin_values

if TYPE_CHECKING:
    import logging

    from codeintel.core.catalog import FunctionCatalog
    from codeintel.core.validation import ValidationSeverity
    from codeintel.graphs.validation.context import GraphValidationContext
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Check Classes (CheckProtocol-compliant)
# =============================================================================


class MissingFunctionGoidsCheck(GraphCheckBase):
    """Check for files with functions in AST that are missing GOIDs."""

    check_name: ClassVar[str] = "missing_function_goids"
    check_description: ClassVar[str] = "Detect files with functions missing GOIDs"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute missing function GOIDs check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for files with missing function GOIDs.
        """
        _ = self  # Instance method required for CheckProtocol
        if ctx.gateway is None:
            return []
        return _warn_missing_function_goids_impl(ctx.gateway, ctx.repo, ctx.commit, ctx.logger)


class CallsiteSpanMismatchCheck(GraphCheckBase):
    """Check for call graph edges whose callsites lie outside caller spans."""

    check_name: ClassVar[str] = "callsite_span_mismatch"
    check_description: ClassVar[str] = "Detect callsites outside caller spans"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute callsite span mismatch check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway and catalog.

        Returns
        -------
        list[dict[str, object]]
            Findings for callsite span mismatches.
        """
        _ = self  # Instance method required for CheckProtocol
        if ctx.gateway is None or ctx.catalog is None:
            return []
        return _warn_callsite_span_mismatches_impl(
            ctx.gateway, ctx.catalog, ctx.repo, ctx.commit, ctx.logger
        )


class OrphanModulesCheck(GraphCheckBase):
    """Check for modules with no GOIDs (orphans)."""

    check_name: ClassVar[str] = "orphan_modules"
    check_description: ClassVar[str] = "Detect modules with no GOIDs"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute orphan modules check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway and catalog.

        Returns
        -------
        list[dict[str, object]]
            Findings for orphan modules.
        """
        _ = self  # Instance method required for CheckProtocol
        if ctx.gateway is None or ctx.catalog is None:
            return []
        return _warn_orphan_modules_impl(ctx.gateway, ctx.repo, ctx.commit, ctx.logger, ctx.catalog)


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _warn_missing_function_goids_impl(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for files with functions in AST that are missing GOIDs (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for files with missing function GOIDs.
    """
    try:
        ast_nodes = cast("Any", gateway.ibis.table("core.ast_nodes"))
        goids = cast("Any", gateway.ibis.table("core.goids"))

        funcs = (
            filter_by(
                ast_nodes,
                ibis_bool(ast_nodes.repo == repo),
                ibis_bool(ast_nodes.commit == commit),
                isin_values(ast_nodes.node_type, ["FunctionDef", "AsyncFunctionDef"]),
            )
            .group_by(ast_nodes.path)
            .aggregate(function_count=ast_nodes.path.count())
            .rename({"path": "rel_path"})
        )

        goid_counts = (
            filter_by(
                goids,
                ibis_bool(goids.repo == repo),
                ibis_bool(goids.commit == commit),
                isin_values(goids.kind, ["function", "method"]),
            )
            .group_by(goids.rel_path)
            .aggregate(goid_count=goids.rel_path.count())
        )

        joined = funcs.left_join(
            goid_counts,
            predicates=[(funcs.rel_path, goid_counts.rel_path)],
        )
        findings = (
            joined.select(
                funcs.rel_path,
                funcs.function_count,
                ibis.coalesce(goid_counts.goid_count, ibis.literal(0)).name("goid_count"),
            )
            .filter(ibis.coalesce(goid_counts.goid_count, ibis.literal(0)) < funcs.function_count)
            .order_by(funcs.rel_path)
        )

        rows = findings.execute()
    except (DuckDBError, AttributeError):
        log.info("Skipping missing_function_goids check due to incomplete AST data")
        return []

    if getattr(rows, "empty", True):
        return []
    sample_rows = list(rows.itertuples(index=False, name=None))[:5]
    sample = ", ".join(str(path) for path, _, _ in sample_rows)
    log.warning(
        "Validation: %d file(s) have functions without GOIDs (sample: %s)",
        len(rows),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "missing_function_goids",
            "severity": "warning",
            "path": path,
            "detail": f"{function_count} functions, {goid_count} GOIDs",
            "context": {"function_count": function_count, "goid_count": goid_count},
        }
        for path, function_count, goid_count in rows.itertuples(index=False, name=None)
    ]


def _warn_callsite_span_mismatches_impl(
    gateway: StorageGateway,
    catalog: FunctionCatalog,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for call graph edges outside caller spans (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for callsite span mismatches.
    """
    spans_by_goid = {span.goid: span for span in catalog.function_spans}
    try:
        edges = cast("Any", gateway.ibis.table("graph.call_graph_edges"))
        rows = (
            filter_by(
                edges,
                ibis_bool(edges.repo == repo),
                ibis_bool(edges.commit == commit),
                ibis_bool(edges.callsite_line.notnull()),
            )
            .select(edges.caller_goid_h128, edges.callsite_path, edges.callsite_line)
            .execute()
        )
    except DuckDBError:
        return []

    mismatches = []
    for goid, path, line in rows.itertuples(index=False, name=None):
        span = spans_by_goid.get(int(goid)) if goid is not None else None
        if span is None:
            continue
        if line < span.start_line or line > span.end_line:
            mismatches.append((path, line, span.start_line, span.end_line))

    if not mismatches:
        return []
    sample = ", ".join(f"{path}:{line}" for path, line, _, _ in mismatches[:5])
    log.warning(
        "Validation: %d call graph edges fall outside caller spans (sample: %s)",
        len(mismatches),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "callsite_span_mismatch",
            "severity": "warning",
            "path": path,
            "detail": f"callsite {line} outside span {start}-{end}",
            "context": {"callsite_line": line, "start_line": start, "end_line": end},
        }
        for path, line, start, end in mismatches
    ]


def _warn_orphan_modules_impl(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    log: logging.Logger,
    catalog: FunctionCatalog,
) -> list[dict[str, object]]:
    """Check for modules with no GOIDs (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for orphan modules.
    """
    query_failed = False
    try:
        modules = cast("Any", gateway.ibis.table("core.modules"))
        goids = cast("Any", gateway.ibis.table("core.goids"))
        module_rows = filter_by(modules, modules.repo == repo, modules.commit == commit)

        module_goids = (
            filter_by(
                goids,
                goids.repo == repo,
                goids.commit == commit,
                goids.kind == "module",
            )
            .group_by(goids.rel_path)
            .aggregate(cnt=goids.rel_path.count())
        )

        joined = module_rows.left_join(
            module_goids, predicates=[(module_rows.path, module_goids.rel_path)]
        )
        rows_df = (
            joined.filter(ibis_bool(module_goids.cnt.isnull())).select(module_rows.path).execute()
        )
        rows = [(path,) for (path,) in rows_df.itertuples(index=False, name=None)]

        if rows:
            stats_df = (
                joined.select(
                    module_rows.path,
                    ibis.coalesce(module_goids.cnt, ibis.literal(0)).name("module_goids"),
                )
                .order_by("module_goids", module_rows.path)
                .limit(5)
                .execute()
            )
            sample_detail = ", ".join(
                f"{path} (module_goids={cnt})"
                for path, cnt in stats_df.itertuples(index=False, name=None)
            )
            log.info(
                "Orphan module debug: repo=%s commit=%s sample=%s",
                repo,
                commit,
                sample_detail,
            )
    except DuckDBError:
        query_failed = True
        rows = []

    if query_failed and catalog.module_by_path:
        rows = [(path,) for path in catalog.module_by_path]

    if not rows:
        return []
    sample = ", ".join(str(path) for (path,) in rows[:5])
    log.warning("Validation: %d module(s) have no GOIDs (sample: %s)", len(rows), sample)
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "orphan_module",
            "severity": "warning",
            "path": path,
            "detail": "module has no GOIDs",
            "context": {},
        }
        for (path,) in rows
    ]


# =============================================================================
# All Check Classes (for runner registration)
# =============================================================================

ALL_DATABASE_CHECKS: tuple[type[GraphCheckBase], ...] = (
    MissingFunctionGoidsCheck,
    CallsiteSpanMismatchCheck,
    OrphanModulesCheck,
)

__all__ = [
    # Check classes
    "ALL_DATABASE_CHECKS",
    "CallsiteSpanMismatchCheck",
    "MissingFunctionGoidsCheck",
    "OrphanModulesCheck",
]
