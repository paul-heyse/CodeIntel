"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from codeintel.build.graphs.engine.datasets import SnapshotScanRequest, scan_snapshot_table
from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.hamilton.native.graphs.cpg.bytecode import instruction_cpg_id
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.query_results import coerce_int, coerce_str
from codeintel.core.serialization.payload import decode_payload

if TYPE_CHECKING:
    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.catalog.function_span import FunctionSpan
    from codeintel.core.validation import ValidationSeverity
    from codeintel.storage.catalog import FunctionCatalog


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


class SymtableResolutionEdgesCheck(GraphCheckBase):
    """Check for symtable bindings missing resolution edges."""

    check_name: ClassVar[str] = "symtable_resolution_edges"
    check_description: ClassVar[str] = (
        "Detect global/nonlocal/free bindings missing resolution edges"
    )
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute symtable resolution edge check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for missing resolution edges.
        """
        _ = self
        return _warn_missing_symtable_resolution_edges_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


class SymtableFreevarsCheck(GraphCheckBase):
    """Check for symtable freevars mismatching bytecode freevars."""

    check_name: ClassVar[str] = "symtable_freevars_mismatch"
    check_description: ClassVar[str] = (
        "Detect mismatches between symtable frees and bytecode freevars"
    )
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute symtable freevar mismatch check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for symtable/bytecode mismatches.
        """
        _ = self
        return _warn_symtable_freevar_mismatch_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


class BytecodeCfgEdgeIntegrityCheck(GraphCheckBase):
    """Check for bytecode CFG edges referencing missing blocks."""

    check_name: ClassVar[str] = "bytecode_cfg_edge_integrity"
    check_description: ClassVar[str] = "Detect bytecode CFG edges with missing blocks"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute bytecode CFG edge integrity check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for missing CFG blocks.
        """
        _ = self
        return _warn_missing_bytecode_blocks_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


class BytecodeDefuseBindingSpaceCheck(GraphCheckBase):
    """Check for def/use binding edges with mismatched binding kinds."""

    check_name: ClassVar[str] = "bytecode_defuse_binding_space"
    check_description: ClassVar[str] = "Detect def/use bindings that disagree with bytecode space"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute def/use binding space check.

        Parameters
        ----------
        ctx
            Graph validation context with gateway.

        Returns
        -------
        list[dict[str, object]]
            Findings for mismatched def/use binding kinds.
        """
        _ = self
        return _warn_defuse_binding_space_mismatch_impl(
            ctx.dataset_root_dir,
            ctx.repo,
            ctx.commit,
            ctx.logger,
        )


class BytecodeLoadFastBindingCheck(GraphCheckBase):
    """Check LOAD_FAST binding edges resolve to locals or params."""

    check_name: ClassVar[str] = "bytecode_load_fast_binding"
    check_description: ClassVar[str] = "Detect LOAD_FAST uses without local/param binding edges"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute LOAD_FAST binding checks.

        Returns
        -------
        list[dict[str, object]]
            Findings for missing LOAD_FAST binding edges.
        """
        _ = self
        return _warn_missing_defuse_binding_edges_impl(
            _DefuseBindingCheckRequest(
                dataset_root_dir=ctx.dataset_root_dir,
                repo=ctx.repo,
                commit=ctx.commit,
                log=ctx.logger,
                space="local",
                allowed_binding_kinds={"local", "param"},
                check_name=self.check_name,
                detail="LOAD_FAST use missing local/param binding edge",
            )
        )


class BytecodeLoadDerefBindingCheck(GraphCheckBase):
    """Check LOAD_DEREF binding edges resolve to free/nonlocal bindings."""

    check_name: ClassVar[str] = "bytecode_load_deref_binding"
    check_description: ClassVar[str] = "Detect LOAD_DEREF uses without free/nonlocal binding edges"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute LOAD_DEREF binding checks.

        Returns
        -------
        list[dict[str, object]]
            Findings for missing LOAD_DEREF binding edges.
        """
        _ = self
        return _warn_missing_defuse_binding_edges_impl(
            _DefuseBindingCheckRequest(
                dataset_root_dir=ctx.dataset_root_dir,
                repo=ctx.repo,
                commit=ctx.commit,
                log=ctx.logger,
                space="free",
                allowed_binding_kinds={"free_ref", "nonlocal_ref"},
                check_name=self.check_name,
                detail="LOAD_DEREF use missing free/nonlocal binding edge",
            )
        )


class BytecodeLoadGlobalBindingCheck(GraphCheckBase):
    """Check LOAD_GLOBAL binding edges resolve to module globals."""

    check_name: ClassVar[str] = "bytecode_load_global_binding"
    check_description: ClassVar[str] = "Detect LOAD_GLOBAL uses without global binding edges"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute LOAD_GLOBAL binding checks.

        Returns
        -------
        list[dict[str, object]]
            Findings for missing LOAD_GLOBAL binding edges.
        """
        _ = self
        return _warn_missing_defuse_binding_edges_impl(
            _DefuseBindingCheckRequest(
                dataset_root_dir=ctx.dataset_root_dir,
                repo=ctx.repo,
                commit=ctx.commit,
                log=ctx.logger,
                space="global",
                allowed_binding_kinds={"global_ref"},
                check_name=self.check_name,
                detail="LOAD_GLOBAL use missing global binding edge",
            )
        )


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _scan_snapshot_table(request: SnapshotScanRequest) -> pa.Table | None:
    return scan_snapshot_table(request)


def _function_span_resolver(spans: Sequence[FunctionSpan]) -> SpanResolver[int]:
    resolver = SpanResolver.for_lines(path_normalizer=lambda value: value)
    for span in spans:
        resolver.add_span(span.rel_path, span.start_line, span.end_line, span.goid)
    return resolver


def _function_counts_by_path(ast_table: pa.Table) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in iter_rows(ast_table):
        if row.get("node_type") not in {"FunctionDef", "AsyncFunctionDef"}:
            continue
        path = row.get("path")
        if not isinstance(path, str):
            continue
        counts[path] = counts.get(path, 0) + 1
    return counts


def _goid_counts_by_path(goids_table: pa.Table) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in iter_rows(goids_table):
        if row.get("kind") not in {"function", "method"}:
            continue
        path = row.get("rel_path")
        if not isinstance(path, str):
            continue
        counts[path] = counts.get(path, 0) + 1
    return counts


def _missing_function_goid_rows(
    function_counts: dict[str, int],
    goid_counts: dict[str, int],
) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for path, function_count in sorted(function_counts.items()):
        goid_count = goid_counts.get(path, 0)
        if goid_count < function_count:
            rows.append(
                (
                    coerce_str(path, ctx="missing_function_goids.rel_path"),
                    function_count,
                    goid_count,
                )
            )
    return rows


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
    ast_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.ast_nodes",
            snapshot_id=commit,
            columns=("path", "node_type"),
            repo=None,
            commit=None,
        )
    )
    if ast_table is None:
        return []
    goids_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.goids",
            snapshot_id=commit,
            columns=("rel_path", "kind", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if goids_table is None:
        return []
    function_counts = _function_counts_by_path(ast_table)
    goid_counts = _goid_counts_by_path(goids_table)
    rows = _missing_function_goid_rows(function_counts, goid_counts)

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
    span_resolver = _function_span_resolver(catalog.function_spans)
    if dataset_root_dir is None:
        return []
    table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="graph.call_graph_edges",
            snapshot_id=commit,
            columns=("caller_goid_h128", "callsite_path", "callsite_line", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if table is None:
        return []
    rows = [row for row in iter_rows(table) if row.get("callsite_line") is not None]

    mismatches = []
    for row in rows:
        goid_int = normalize_decimal_id(row.get("caller_goid_h128"))
        if goid_int is None:
            continue
        span = spans_by_goid.get(goid_int)
        if span is None:
            continue
        line_value = coerce_int(row.get("callsite_line"), ctx="callsite_line")
        callsite_path = row.get("callsite_path")
        match_kind = "NONE"
        candidate_count = 0
        if isinstance(callsite_path, str):
            match = span_resolver.resolve(callsite_path, line_value, line_value)
            match_kind = match.match_kind
            candidate_count = match.candidate_count
        if line_value < span.start_line or line_value > span.end_line:
            mismatches.append(
                (
                    coerce_str(callsite_path, ctx="callsite_path"),
                    line_value,
                    span.start_line,
                    span.end_line,
                    match_kind,
                    candidate_count,
                )
            )

    if not mismatches:
        return []
    sample = ", ".join(f"{path}:{line}" for path, line, *_ in mismatches[:5])
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
            "context": {
                "callsite_line": line,
                "start_line": start,
                "end_line": end,
                "match_kind": match_kind,
                "candidate_count": candidate_count,
            },
        }
        for path, line, start, end, match_kind, candidate_count in mismatches
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
    modules_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.modules",
            snapshot_id=commit,
            columns=("path", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    goids_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.goids",
            snapshot_id=commit,
            columns=("rel_path", "kind", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    if modules_table is None or goids_table is None:
        if catalog.module_by_path:
            rows = [(path,) for path in catalog.module_by_path]
            module_count = 0
        else:
            return []
    else:
        module_goid_counts: dict[str, int] = {}
        for row in iter_rows(goids_table):
            if row.get("kind") != "module":
                continue
            rel_path = row.get("rel_path")
            if not isinstance(rel_path, str):
                continue
            module_goid_counts[rel_path] = module_goid_counts.get(rel_path, 0) + 1

        module_paths = [
            coerce_str(row.get("path"), ctx="orphan_modules.path")
            for row in iter_rows(modules_table)
            if row.get("path") is not None
        ]
        module_count = len(module_paths)
        rows = [(path,) for path in sorted(module_paths) if path not in module_goid_counts]
        if rows:
            sample = sorted(
                (
                    (path, module_goid_counts.get(path, 0))
                    for path in module_paths
                    if path is not None
                ),
                key=lambda item: (item[1], item[0]),
            )[:5]
            sample_detail_parts = [
                f"{path} (module_goids={count})" for path, count in sample if path is not None
            ]
            sample_detail = ", ".join(sample_detail_parts)
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


def _warn_missing_symtable_resolution_edges_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    if dataset_root_dir is None:
        return []
    bindings_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_bindings",
            snapshot_id=commit,
            columns=("rel_path", "binding_id", "binding_kind", "name", "scope_id"),
            repo=repo,
            commit=commit,
        )
    )
    edges_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_resolution_edges",
            snapshot_id=commit,
            columns=("rel_path", "src_binding_id"),
            repo=repo,
            commit=commit,
        )
    )
    if bindings_table is None or edges_table is None:
        return []
    edge_keys = _symtable_resolution_edge_keys(edges_table)
    missing = _missing_symtable_resolution_rows(bindings_table, edge_keys)
    if not missing:
        return []
    log.warning(
        "Validation: %d symtable bindings missing resolution edges",
        len(missing),
    )
    return _symtable_resolution_findings(missing, repo=repo, commit=commit)


def _symtable_resolution_edge_keys(edges_table: pa.Table) -> set[tuple[str, str]]:
    edge_keys: set[tuple[str, str]] = set()
    for row in iter_rows(edges_table):
        rel_path = row.get("rel_path")
        src_binding_id = row.get("src_binding_id")
        if isinstance(rel_path, str) and isinstance(src_binding_id, str):
            edge_keys.add((rel_path, src_binding_id))
    return edge_keys


def _missing_symtable_resolution_rows(
    bindings_table: pa.Table,
    edge_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    ref_kinds = {"global_ref", "nonlocal_ref", "free_ref"}
    missing: list[dict[str, object]] = []
    for row in iter_rows(bindings_table):
        if row.get("binding_kind") not in ref_kinds:
            continue
        rel_path = row.get("rel_path")
        binding_id = row.get("binding_id")
        if not isinstance(rel_path, str) or not isinstance(binding_id, str):
            continue
        if (rel_path, binding_id) in edge_keys:
            continue
        missing.append(row)
    return missing


def _symtable_resolution_findings(
    rows: Sequence[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for row in rows:
        rel_path = coerce_str(row.get("rel_path"), ctx="symtable_resolution_edges.rel_path")
        binding_id = coerce_str(row.get("binding_id"), ctx="symtable_resolution_edges.binding_id")
        binding_kind = coerce_str(
            row.get("binding_kind"),
            ctx="symtable_resolution_edges.binding_kind",
        )
        name = coerce_str(row.get("name"), ctx="symtable_resolution_edges.name")
        scope_id = coerce_str(row.get("scope_id"), ctx="symtable_resolution_edges.scope_id")
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "symtable_resolution_edges",
                "severity": "warning",
                "path": rel_path,
                "detail": f"missing resolution for {name} ({binding_kind})",
                "context": {
                    "binding_id": binding_id,
                    "binding_kind": binding_kind,
                    "scope_id": scope_id,
                    "name": name,
                },
            }
        )
    return findings


def _warn_symtable_freevar_mismatch_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    if dataset_root_dir is None:
        return []
    scopes_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_scopes",
            snapshot_id=commit,
            columns=("rel_path", "scope_id", "qualpath"),
            repo=repo,
            commit=commit,
        )
    )
    partitions_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_function_partitions",
            snapshot_id=commit,
            columns=("rel_path", "scope_id", "frees"),
            repo=repo,
            commit=commit,
        )
    )
    code_units_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_bc_code_units",
            snapshot_id=commit,
            columns=("rel_path", "qualpath", "freevars"),
            repo=repo,
            commit=commit,
        )
    )
    if scopes_table is None or partitions_table is None or code_units_table is None:
        return []
    scope_by_key = _scope_qualpath_map(scopes_table)
    freevars_by_unit = _freevars_by_unit(code_units_table)
    mismatches = _freevar_mismatches(
        partitions_table,
        scope_by_key=scope_by_key,
        freevars_by_unit=freevars_by_unit,
        repo=repo,
        commit=commit,
    )
    if mismatches:
        log.warning("Validation: %d symtable/bytecode freevar mismatches", len(mismatches))
    return mismatches


def _scope_qualpath_map(scopes_table: pa.Table) -> dict[tuple[str, str], str]:
    scope_by_key: dict[tuple[str, str], str] = {}
    for row in iter_rows(scopes_table):
        rel_path = row.get("rel_path")
        scope_id = row.get("scope_id")
        qualpath = row.get("qualpath")
        if isinstance(rel_path, str) and isinstance(scope_id, str) and isinstance(qualpath, str):
            scope_by_key[rel_path, scope_id] = qualpath
    return scope_by_key


def _freevars_by_unit(code_units_table: pa.Table) -> dict[tuple[str, str], list[str]]:
    freevars_by_unit: dict[tuple[str, str], list[str]] = {}
    for row in iter_rows(code_units_table):
        rel_path = row.get("rel_path")
        qualpath = row.get("qualpath")
        freevars = row.get("freevars")
        if not isinstance(rel_path, str) or not isinstance(qualpath, str):
            continue
        if not isinstance(freevars, list):
            continue
        freevars_by_unit[rel_path, qualpath] = [item for item in freevars if isinstance(item, str)]
    return freevars_by_unit


def _freevar_mismatches(
    partitions_table: pa.Table,
    *,
    scope_by_key: dict[tuple[str, str], str],
    freevars_by_unit: dict[tuple[str, str], list[str]],
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for row in iter_rows(partitions_table):
        rel_path = row.get("rel_path")
        scope_id = row.get("scope_id")
        frees = row.get("frees")
        if not isinstance(rel_path, str) or not isinstance(scope_id, str):
            continue
        if not isinstance(frees, list):
            continue
        qualpath = scope_by_key.get((rel_path, scope_id))
        if qualpath is None:
            continue
        freevars = freevars_by_unit.get((rel_path, qualpath))
        if freevars is None:
            continue
        frees_set = {item for item in frees if isinstance(item, str)}
        freevars_set = set(freevars)
        if frees_set == freevars_set:
            continue
        mismatches.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "symtable_freevars_mismatch",
                "severity": "warning",
                "path": coerce_str(rel_path, ctx="symtable_freevars.rel_path"),
                "detail": f"freevars mismatch for {qualpath}",
                "context": {
                    "qualpath": coerce_str(qualpath, ctx="symtable_freevars.qualpath"),
                    "symtable_frees": sorted(frees_set),
                    "bytecode_freevars": sorted(freevars_set),
                },
            }
        )
    return mismatches


def _warn_missing_bytecode_blocks_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    if dataset_root_dir is None:
        return []
    edges_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_bc_cfg_edges",
            snapshot_id=commit,
            columns=("rel_path", "code_unit_id", "src_block_id", "dst_block_id"),
            repo=repo,
            commit=commit,
        )
    )
    blocks_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_bc_blocks",
            snapshot_id=commit,
            columns=("rel_path", "code_unit_id", "block_id"),
            repo=repo,
            commit=commit,
        )
    )
    if edges_table is None or blocks_table is None:
        return []
    block_keys = {
        (
            coerce_str(row.get("rel_path"), ctx="bytecode_cfg_blocks.rel_path"),
            coerce_str(row.get("code_unit_id"), ctx="bytecode_cfg_blocks.code_unit_id"),
            coerce_str(row.get("block_id"), ctx="bytecode_cfg_blocks.block_id"),
        )
        for row in iter_rows(blocks_table)
    }
    missing: list[dict[str, object]] = []
    for row in iter_rows(edges_table):
        rel_path = coerce_str(row.get("rel_path"), ctx="bytecode_cfg_edges.rel_path")
        code_unit_id = coerce_str(row.get("code_unit_id"), ctx="bytecode_cfg_edges.code_unit_id")
        src_block = coerce_str(row.get("src_block_id"), ctx="bytecode_cfg_edges.src_block_id")
        dst_block = coerce_str(row.get("dst_block_id"), ctx="bytecode_cfg_edges.dst_block_id")
        if rel_path is None or code_unit_id is None or src_block is None or dst_block is None:
            continue
        if (rel_path, code_unit_id, src_block) not in block_keys or (
            rel_path,
            code_unit_id,
            dst_block,
        ) not in block_keys:
            missing.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "check_name": "bytecode_cfg_edge_integrity",
                    "severity": "warning",
                    "path": rel_path,
                    "detail": "CFG edge references missing block",
                    "context": {
                        "code_unit_id": code_unit_id,
                        "src_block_id": src_block,
                        "dst_block_id": dst_block,
                    },
                }
            )
    if missing:
        log.warning("Validation: %d bytecode CFG edges missing blocks", len(missing))
    return missing


def _warn_defuse_binding_space_mismatch_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    if dataset_root_dir is None:
        return []
    edges_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="graph.cpg_edges",
            snapshot_id=commit,
            columns=("rel_path", "edge_kind", "extras_json"),
            repo=repo,
            commit=commit,
        )
    )
    if edges_table is None:
        return []
    expected = {
        "local": {"local", "param"},
        "global": {"global_ref"},
        "free": {"free_ref", "nonlocal_ref"},
    }
    mismatches: list[dict[str, object]] = []
    for row in iter_rows(edges_table):
        edge_kind = row.get("edge_kind")
        if edge_kind not in {"DEFINES_BINDING", "USES_BINDING"}:
            continue
        rel_path = coerce_str(row.get("rel_path"), ctx="defuse_binding_space.rel_path")
        extras = decode_payload(row.get("extras_json"))
        if not isinstance(extras, dict):
            continue
        space = extras.get("space")
        binding_kind = extras.get("binding_kind")
        if not isinstance(space, str) or not isinstance(binding_kind, str):
            continue
        allowed = expected.get(space)
        if allowed is None or binding_kind in allowed:
            continue
        mismatches.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "bytecode_defuse_binding_space",
                "severity": "warning",
                "path": rel_path,
                "detail": "def/use binding kind mismatches bytecode space",
                "context": {
                    "space": space,
                    "binding_kind": binding_kind,
                    "edge_kind": row.get("edge_kind"),
                },
            }
        )
    if mismatches:
        log.warning(
            "Validation: %d bytecode def/use bindings mismatch expected space",
            len(mismatches),
        )
    return mismatches


@dataclass(frozen=True, slots=True)
class _DefuseBindingCheckRequest:
    dataset_root_dir: Path | None
    repo: str
    commit: str
    log: logging.Logger
    space: str
    allowed_binding_kinds: set[str]
    check_name: str
    detail: str


@dataclass(frozen=True, slots=True)
class _DefuseBindingEventContext:
    repo: str
    commit: str
    space: str
    allowed_binding_kinds: set[str]
    check_name: str
    detail: str


def _warn_missing_defuse_binding_edges_impl(
    request: _DefuseBindingCheckRequest,
) -> list[dict[str, object]]:
    if request.dataset_root_dir is None:
        return []
    events_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=request.dataset_root_dir,
            table_key="core.py_bc_defuse_events",
            snapshot_id=request.commit,
            columns=(
                "repo",
                "commit",
                "rel_path",
                "code_unit_id",
                "instr_id",
                "event_kind",
                "space",
            ),
            repo=request.repo,
            commit=request.commit,
        )
    )
    edges_table = _scan_snapshot_table(
        SnapshotScanRequest(
            dataset_root=request.dataset_root_dir,
            table_key="graph.cpg_edges",
            snapshot_id=request.commit,
            columns=("src_cpg_node_id", "edge_kind", "extras_json", "rel_path"),
            repo=request.repo,
            commit=request.commit,
        )
    )
    if events_table is None or edges_table is None:
        return []
    edges_by_src = _defuse_edges_by_source(list(iter_rows(edges_table)))
    missing = _missing_defuse_binding_edges(
        list(iter_rows(events_table)),
        edges_by_src=edges_by_src,
        context=_DefuseBindingEventContext(
            repo=request.repo,
            commit=request.commit,
            space=request.space,
            allowed_binding_kinds=request.allowed_binding_kinds,
            check_name=request.check_name,
            detail=request.detail,
        ),
    )
    if missing:
        request.log.warning("Validation: %d %s issues detected", len(missing), request.check_name)
    return missing


def _defuse_edges_by_source(
    edges: list[dict[str, object]],
) -> dict[tuple[int, str], set[str]]:
    edge_rows = [row for row in edges if row.get("edge_kind") == "USES_BINDING"]
    edges_by_src: dict[tuple[int, str], set[str]] = {}
    for row in edge_rows:
        src_id = normalize_decimal_id(row.get("src_cpg_node_id"))
        if src_id is None:
            continue
        extras = decode_payload(row.get("extras_json"))
        if not isinstance(extras, dict):
            continue
        edge_space = extras.get("space")
        binding_kind = extras.get("binding_kind")
        if not isinstance(edge_space, str) or not isinstance(binding_kind, str):
            continue
        edges_by_src.setdefault((int(src_id), edge_space), set()).add(binding_kind)
    return edges_by_src


def _missing_defuse_binding_edges(
    events: list[dict[str, object]],
    *,
    edges_by_src: dict[tuple[int, str], set[str]],
    context: _DefuseBindingEventContext,
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for row in events:
        if row.get("event_kind") != "USE" or row.get("space") != context.space:
            continue
        rel_path = coerce_str(row.get("rel_path"), ctx=f"{context.check_name}.rel_path")
        code_unit_id = coerce_str(
            row.get("code_unit_id"),
            ctx=f"{context.check_name}.code_unit_id",
        )
        instr_id = coerce_str(row.get("instr_id"), ctx=f"{context.check_name}.instr_id")
        if None in {rel_path, code_unit_id, instr_id}:
            continue
        src_id = instruction_cpg_id(
            repo=context.repo,
            commit=context.commit,
            rel_path=rel_path,
            code_unit_id=code_unit_id,
            instr_id=instr_id,
        )
        binding_kinds = edges_by_src.get((src_id, context.space))
        if binding_kinds and binding_kinds.intersection(context.allowed_binding_kinds):
            continue
        missing.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "check_name": context.check_name,
                "severity": "warning",
                "path": rel_path,
                "detail": context.detail,
                "context": {
                    "space": context.space,
                    "allowed_binding_kinds": sorted(context.allowed_binding_kinds),
                    "code_unit_id": code_unit_id,
                    "instr_id": instr_id,
                },
            }
        )
    return missing


# =============================================================================
# All Check Classes (for runner registration)
# =============================================================================

ALL_DATABASE_CHECKS: tuple[type[GraphCheckBase], ...] = (
    MissingFunctionGoidsCheck,
    CallsiteSpanMismatchCheck,
    OrphanModulesCheck,
    SymtableResolutionEdgesCheck,
    SymtableFreevarsCheck,
    BytecodeCfgEdgeIntegrityCheck,
    BytecodeDefuseBindingSpaceCheck,
    BytecodeLoadFastBindingCheck,
    BytecodeLoadDerefBindingCheck,
    BytecodeLoadGlobalBindingCheck,
)

__all__ = [
    # Check classes
    "ALL_DATABASE_CHECKS",
    "BytecodeCfgEdgeIntegrityCheck",
    "BytecodeDefuseBindingSpaceCheck",
    "BytecodeLoadDerefBindingCheck",
    "BytecodeLoadFastBindingCheck",
    "BytecodeLoadGlobalBindingCheck",
    "CallsiteSpanMismatchCheck",
    "MissingFunctionGoidsCheck",
    "OrphanModulesCheck",
    "SymtableFreevarsCheck",
    "SymtableResolutionEdgesCheck",
]
