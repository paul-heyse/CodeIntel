"""Database integrity validation checks.

This module contains validation checks that verify data integrity
by querying the database for inconsistencies.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from codeintel.build.graphs.engine.datasets import SnapshotScanRequest, scan_snapshot_reader
from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.hamilton.native.graphs.cpg import instruction_cpg_id
from codeintel.build.tabular.conversion import arrow_reader_to_lazyframe
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.query_results import coerce_int, coerce_str
from codeintel.core.serialization.payload import decode_payload

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _scan_snapshot_frame(request: SnapshotScanRequest) -> pl.LazyFrame | None:
    reader = scan_snapshot_reader(request)
    if reader is None:
        return None
    return arrow_reader_to_lazyframe(reader)


def _function_span_resolver(spans: Sequence[FunctionSpan]) -> SpanResolver[int]:
    resolver = SpanResolver.for_lines(path_normalizer=lambda value: value)
    for span in spans:
        resolver.add_span(span.rel_path, span.start_line, span.end_line, span.goid)
    return resolver


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
    ast_frame = _scan_snapshot_frame(
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
    goids_frame = _scan_snapshot_frame(
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
    span_resolver = _function_span_resolver(catalog.function_spans)
    if dataset_root_dir is None:
        return []
    frame = _scan_snapshot_frame(
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
    modules_frame = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.modules",
            snapshot_id=commit,
            columns=("path", "repo", "commit"),
            repo=repo,
            commit=commit,
        )
    )
    goids_frame = _scan_snapshot_frame(
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
            sample_detail_parts: list[str] = []
            for row in sample:
                module_goids = coerce_int(row.get("module_goids"), ctx="module_goids")
                sample_detail_parts.append(f"{row['path']} (module_goids={module_goids})")
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
    bindings = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_bindings",
            snapshot_id=commit,
            columns=("rel_path", "binding_id", "binding_kind", "name", "scope_id"),
            repo=repo,
            commit=commit,
        )
    )
    edges = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_resolution_edges",
            snapshot_id=commit,
            columns=("rel_path", "src_binding_id"),
            repo=repo,
            commit=commit,
        )
    )
    if bindings is None or edges is None:
        return []
    ref_kinds = ["global_ref", "nonlocal_ref", "free_ref"]
    refs = bindings.filter(pl.col("binding_kind").is_in(ref_kinds))
    joined = refs.join(
        edges,
        left_on=["rel_path", "binding_id"],
        right_on=["rel_path", "src_binding_id"],
        how="left",
    )
    missing = joined.filter(pl.col("src_binding_id").is_null()).collect().to_dicts()
    if not missing:
        return []
    log.warning(
        "Validation: %d symtable bindings missing resolution edges",
        len(missing),
    )
    findings: list[dict[str, object]] = []
    for row in missing:
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
    scopes = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_scopes",
            snapshot_id=commit,
            columns=("rel_path", "scope_id", "qualpath"),
            repo=repo,
            commit=commit,
        )
    )
    partitions = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_sym_function_partitions",
            snapshot_id=commit,
            columns=("rel_path", "scope_id", "frees"),
            repo=repo,
            commit=commit,
        )
    )
    code_units = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_bc_code_units",
            snapshot_id=commit,
            columns=("rel_path", "qualpath", "freevars"),
            repo=repo,
            commit=commit,
        )
    )
    if scopes is None or partitions is None or code_units is None:
        return []
    with_qualpath = partitions.join(
        scopes,
        on=["rel_path", "scope_id"],
        how="left",
    )
    joined = with_qualpath.join(code_units, on=["rel_path", "qualpath"], how="left")
    mismatches: list[dict[str, object]] = []
    for row in joined.collect().to_dicts():
        rel_path = coerce_str(row.get("rel_path"), ctx="symtable_freevars.rel_path")
        qualpath = coerce_str(row.get("qualpath"), ctx="symtable_freevars.qualpath")
        frees = row.get("frees")
        freevars = row.get("freevars")
        if not isinstance(frees, list) or not isinstance(freevars, list):
            continue
        frees_set = {item for item in frees if isinstance(item, str)}
        freevars_set = {item for item in freevars if isinstance(item, str)}
        if frees_set == freevars_set:
            continue
        mismatches.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "symtable_freevars_mismatch",
                "severity": "warning",
                "path": rel_path,
                "detail": f"freevars mismatch for {qualpath}",
                "context": {
                    "qualpath": qualpath,
                    "symtable_frees": sorted(frees_set),
                    "bytecode_freevars": sorted(freevars_set),
                },
            }
        )
    if mismatches:
        log.warning("Validation: %d symtable/bytecode freevar mismatches", len(mismatches))
    return mismatches


def _warn_missing_bytecode_blocks_impl(
    dataset_root_dir: Path | None,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    if dataset_root_dir is None:
        return []
    edges = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_bc_cfg_edges",
            snapshot_id=commit,
            columns=("rel_path", "code_unit_id", "src_block_id", "dst_block_id"),
            repo=repo,
            commit=commit,
        )
    )
    blocks = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="core.py_bc_blocks",
            snapshot_id=commit,
            columns=("rel_path", "code_unit_id", "block_id"),
            repo=repo,
            commit=commit,
        )
    )
    if edges is None or blocks is None:
        return []
    block_keys = {
        (
            coerce_str(row.get("rel_path"), ctx="bytecode_cfg_blocks.rel_path"),
            coerce_str(row.get("code_unit_id"), ctx="bytecode_cfg_blocks.code_unit_id"),
            coerce_str(row.get("block_id"), ctx="bytecode_cfg_blocks.block_id"),
        )
        for row in blocks.collect().to_dicts()
    }
    missing: list[dict[str, object]] = []
    for row in edges.collect().to_dicts():
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
    edges = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=dataset_root_dir,
            table_key="graph.cpg_edges",
            snapshot_id=commit,
            columns=("rel_path", "edge_kind", "extras_json"),
            repo=repo,
            commit=commit,
        )
    )
    if edges is None:
        return []
    expected = {
        "local": {"local", "param"},
        "global": {"global_ref"},
        "free": {"free_ref", "nonlocal_ref"},
    }
    filtered = edges.filter(
        pl.col("edge_kind").is_in(["DEFINES_BINDING", "USES_BINDING"])
    ).collect()
    mismatches: list[dict[str, object]] = []
    for row in filtered.to_dicts():
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
    events = _scan_snapshot_frame(
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
    edges = _scan_snapshot_frame(
        SnapshotScanRequest(
            dataset_root=request.dataset_root_dir,
            table_key="graph.cpg_edges",
            snapshot_id=request.commit,
            columns=("src_cpg_node_id", "edge_kind", "extras_json", "rel_path"),
            repo=request.repo,
            commit=request.commit,
        )
    )
    if events is None or edges is None:
        return []
    edges_by_src = _defuse_edges_by_source(edges)
    missing = _missing_defuse_binding_edges(
        events,
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


def _defuse_edges_by_source(edges: pl.LazyFrame) -> dict[tuple[int, str], set[str]]:
    edge_rows = edges.filter(pl.col("edge_kind") == "USES_BINDING").collect().to_dicts()
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
    events: pl.LazyFrame,
    *,
    edges_by_src: dict[tuple[int, str], set[str]],
    context: _DefuseBindingEventContext,
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for row in (
        events.filter((pl.col("event_kind") == "USE") & (pl.col("space") == context.space))
        .collect()
        .to_dicts()
    ):
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
