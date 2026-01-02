"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from codeintel.build.graphs.engine.datasets import SnapshotScanRequest, scan_snapshot_lazyframe
from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_int, coerce_str

if TYPE_CHECKING:
    import logging

    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.catalog import FunctionCatalog
    from codeintel.core.validation import ValidationSeverity


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
        return _warn_missing_function_goids_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


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
        if ctx.catalog is None:
            return []
        return _warn_callsite_span_mismatches_impl(
            ctx.dataset_root_dir,
            ctx.catalog,
            ctx.repo,
            ctx.commit,
            ctx.logger,
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
        if ctx.catalog is None:
            return []
        return _warn_orphan_modules_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
            ctx.catalog,
        )


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _warn_missing_function_goids_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for files with functions in AST that are missing GOIDs (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for files with missing function GOIDs.
    """
    if dataset_root_dir is None:
        return []
    ast_frame = scan_snapshot_lazyframe(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.ast_nodes",
            snapshot_id=commit,
            columns=("path", "node_type"),
            repo=None,
            commit=None,
        )
    )
    if ast_frame is None:
        return []
    goids_frame = scan_snapshot_lazyframe(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.goids",
            snapshot_id=commit,
            columns=("rel_path", "kind", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if goids_frame is None:
        return []
    funcs = (
        ast_frame.filter(pl.col("node_type").is_in(["FunctionDef", "AsyncFunctionDef"]))
        .group_by("path")
        .agg(pl.len().alias("function_count"))
    )
    goid_counts = (
        goids_frame.filter(pl.col("kind").is_in(["function", "method"]))
        .group_by("rel_path")
        .agg(pl.len().alias("goid_count"))
        .rename({"rel_path": "path"})
    )
    joined = funcs.join(goid_counts, on="path", how="left").with_columns(
        pl.col("goid_count").fill_null(0)
    )
    rows = [
        (
            coerce_str(row.get("path"), ctx="missing_function_goids.rel_path"),
            coerce_int(row.get("function_count"), ctx="missing_function_goids.function_count"),
            coerce_int(row.get("goid_count"), ctx="missing_function_goids.goid_count"),
        )
        for row in joined.filter(pl.col("goid_count") < pl.col("function_count"))
        .sort("path")
        .collect()
        .to_dicts()
    ]

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
    dataset_root_dir: Path | None,
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
    if dataset_root_dir is None:
        return []
    frame = scan_snapshot_lazyframe(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="graph.call_graph_edges",
            snapshot_id=commit,
            columns=("caller_goid_h128", "callsite_path", "callsite_line", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if frame is None:
        return []
    rows = frame.filter(pl.col("callsite_line").is_not_null()).collect().to_dicts()

    mismatches = []
    for row in rows:
        goid_int = normalize_decimal_id(row.get("caller_goid_h128"))
        if goid_int is None:
            continue
        span = spans_by_goid.get(goid_int)
        if span is None:
            continue
        line_value = coerce_int(row.get("callsite_line"), ctx="callsite_line")
        if line_value < span.start_line or line_value > span.end_line:
            mismatches.append(
                (
                    coerce_str(row.get("callsite_path"), ctx="callsite_path"),
                    line_value,
                    span.start_line,
                    span.end_line,
                )
            )

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
    dataset_root_dir: Path | None,
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
    if dataset_root_dir is None:
        return []
    modules_frame = scan_snapshot_lazyframe(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.modules",
            snapshot_id=commit,
            columns=("path", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    goids_frame = scan_snapshot_lazyframe(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.goids",
            snapshot_id=commit,
            columns=("rel_path", "kind", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if modules_frame is None or goids_frame is None:
        if catalog.module_by_path:
            rows = [(path,) for path in catalog.module_by_path]
            module_count = 0
        else:
            return []
    else:
        module_goids = (
            goids_frame.filter(pl.col("kind") == "module")
            .group_by("rel_path")
            .agg(pl.len().alias("cnt"))
            .rename({"rel_path": "path"})
        )
        modules = modules_frame.select("path")
        joined = modules.join(module_goids, on="path", how="left")
        rows = [
            (coerce_str(row.get("path"), ctx="orphan_modules.path"),)
            for row in joined.filter(pl.col("cnt").is_null()).collect().to_dicts()
        ]
        module_count = int(modules.select(pl.len()).collect().to_series()[0])
        if rows:
            sample = (
                joined.with_columns(pl.col("cnt").fill_null(0).alias("module_goids"))
                .select("path", "module_goids")
                .sort(["module_goids", "path"])
                .limit(5)
                .collect()
                .to_dicts()
            )
            sample_detail = ", ".join(
                f"{row['path']} (module_goids={coerce_int(row['module_goids'], ctx='module_goids')})"
                for row in sample
            )
            log.info(
                "Orphan module debug: repo=%s commit=%s sample=%s",
                repo,
                commit,
                sample_detail,
            )

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
