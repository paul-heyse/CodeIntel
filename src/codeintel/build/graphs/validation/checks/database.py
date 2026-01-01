"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    import logging

    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.catalog import FunctionCatalog
    from codeintel.core.validation import ValidationSeverity
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
        predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
            ColumnExpression("commit") == ConstantExpression(commit)
        )
        node_types = [ConstantExpression("FunctionDef"), ConstantExpression("AsyncFunctionDef")]
        funcs = (
            gateway.relation_from_table_key("core.ast_nodes")
            .filter(predicate)
            .filter(ColumnExpression("node_type").isin(*node_types))
            .aggregate("count(*) as function_count", "path")
            .set_alias("funcs")
        )
        kind_literals = [ConstantExpression("function"), ConstantExpression("method")]
        goid_counts = (
            gateway.relation_from_table_key("core.goids")
            .filter(predicate)
            .filter(ColumnExpression("kind").isin(*kind_literals))
            .aggregate("count(*) as goid_count", "rel_path")
            .set_alias("goid_counts")
        )
        relation = (
            funcs.join(goid_counts, "funcs.path = goid_counts.rel_path", how="left")
            .select(
                "funcs.path as rel_path",
                "funcs.function_count",
                "coalesce(goid_counts.goid_count, 0) as goid_count",
            )
            .filter("coalesce(goid_counts.goid_count, 0) < funcs.function_count")
            .order("funcs.path")
        )
        rows = relation.fetchall()
    except DuckDBError:
        log.info("Skipping missing_function_goids check due to incomplete AST data")
        return []

    if not rows:
        return []
    sample_rows = rows[:5]
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
        for path, function_count, goid_count in rows
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
        predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
            ColumnExpression("commit") == ConstantExpression(commit)
        )
        relation = (
            gateway.relation_from_table_key("graph.call_graph_edges")
            .filter(predicate)
            .filter(~ColumnExpression("callsite_line").isnull())
            .select("caller_goid_h128", "callsite_path", "callsite_line")
        )
        rows = relation.fetchall()
    except DuckDBError:
        return []

    mismatches = []
    for goid, path, line in rows:
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
        predicate = (ColumnExpression("repo") == ConstantExpression(repo)) & (
            ColumnExpression("commit") == ConstantExpression(commit)
        )
        module_goids = (
            gateway.relation_from_table_key("core.goids")
            .filter(predicate)
            .filter(ColumnExpression("kind") == ConstantExpression("module"))
            .aggregate("count(*) as cnt", "rel_path")
            .set_alias("module_goids")
        )
        modules = (
            gateway.relation_from_table_key("core.modules").filter(predicate).set_alias("modules")
        )
        relation = (
            modules.join(module_goids, "modules.path = module_goids.rel_path", how="left")
            .filter(ColumnExpression("module_goids.cnt").isnull())
            .select("modules.path")
        )
        rows = [(path,) for (path,) in relation.fetchall()]

        count_row = modules.aggregate("count(*) as cnt").fetchone()
        module_count = 0 if count_row is None else int(count_row[0])
        if rows:
            stats_rel = (
                modules.join(module_goids, "modules.path = module_goids.rel_path", how="left")
                .select(
                    "modules.path",
                    "coalesce(module_goids.cnt, 0) as module_goids",
                )
                .order("module_goids, modules.path")
                .limit(5)
            )
            sample_detail = ", ".join(
                f"{path} (module_goids={cnt})" for path, cnt in stats_rel.fetchall()
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
        module_count = 0

    if query_failed and catalog.module_by_path:
        rows = [(path,) for path in catalog.module_by_path]

    if not rows and module_count == 0 and catalog.module_by_path:
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
