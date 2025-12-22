"""Consolidated native Hamilton graph targets.

This module consolidates multiple graph-domain targets that share similar
execution patterns and are frequently evolved together:

Support targets (from support_targets.py):
- ``goids``: Extract GOIDs and crosswalks from repository source.
- ``symbol_uses``: Build symbol-use edges from SCIP occurrences.
- ``call_graph_views``: Materialize derived views over call graph edges.

Metrics targets (from metrics_targets.py):
- ``graph_metrics``: Computes graph-derived analytics tables.
- ``graph_validation``: Runs integrity checks on graph tables.

Phase 3 consolidation uses executor_materialize template for Pattern D targets.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import (
    check_output_custom,
    pipe_input,
    schema,
    source,
    step,
    value,
)

from codeintel.analytics.graphs import (
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
)
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult, to_execution_result
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.helpers import (
    filter_paths,
    get_source_root,
    is_test_path,
)
from codeintel.build.hamilton.materialize_options import materialize_options
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.options.graphs import GoidBuilderOptions, SymbolUsesOptions
from codeintel.build.hamilton.native.target_override_tables import (
    CALL_GRAPH_VIEWS_OVERRIDE_TABLES,
    GOIDS_OVERRIDE_TABLES,
    GRAPH_METRICS_OVERRIDE_TABLES,
    GRAPH_VALIDATION_OVERRIDE_TABLES,
    SYMBOL_USES_OVERRIDE_TABLES,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_materialize, tag_tool
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph
from codeintel.config.primitives import GraphBackendConfig
from codeintel.core.ibis_typing import (
    and_predicates,
    col_nunique,
    fillna,
    filter_by,
    is_null,
    not_null,
)
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import goid as goid_compute
from codeintel.graphs.compute import symbols as symbols_compute
from codeintel.graphs.runtime import (
    GraphMetricsOptions,
    build_graph_runtime,
)
from codeintel.storage.gateway import DuckDBError, ibis_facade

if TYPE_CHECKING:
    from codeintel.graphs.compute.goid import GoidCrosswalkRow, GoidRow
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
LOG = log

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)

GOIDS_TARGET_NAME = "goids"
SYMBOL_USES_TARGET_NAME = "symbol_uses"
CALL_GRAPH_VIEWS_TARGET_NAME = "call_graph_views"
GRAPH_METRICS_TARGET_NAME = "graph_metrics"
GRAPH_VALIDATION_TARGET_NAME = "graph_validation"

GOIDS_GOIDS_TABLE_KEY = "core.goids"
GOIDS_CROSSWALK_TABLE_KEY = "core.goid_crosswalk"
GOIDS_TABLE_KEYS = (
    GOIDS_GOIDS_TABLE_KEY,
    GOIDS_CROSSWALK_TABLE_KEY,
)

SYMBOL_USE_EDGES_TABLE_KEY = "graph.symbol_use_edges"
SYMBOL_USES_TABLE_KEYS = (SYMBOL_USE_EDGES_TABLE_KEY,)

CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS = "graph.v_function_call_counts"
CALL_GRAPH_VIEWS_CALL_DEPTH_STATS = "graph.v_call_depth_stats"
CALL_GRAPH_VIEWS_TABLE_KEYS = (
    CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS,
    CALL_GRAPH_VIEWS_CALL_DEPTH_STATS,
)

GRAPH_METRICS_TABLE_KEYS = (
    "analytics.graph_metrics_functions",
    "analytics.graph_metrics_functions_ext",
    "analytics.graph_metrics_modules",
    "analytics.graph_metrics_modules_ext",
    "analytics.graph_stats",
)

GRAPH_VALIDATION_TABLE_KEY = "analytics.graph_validation"
GRAPH_VALIDATION_TABLE_KEYS = (GRAPH_VALIDATION_TABLE_KEY,)

register_output_targets(
    make_output_target(
        name=GOIDS_TARGET_NAME,
        module="graphs",
        description="GOID resolution and crosswalk construction.",
        options=TargetSpecOptions(
            table_keys=GOIDS_TABLE_KEYS,
            override_tables=GOIDS_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=SYMBOL_USES_TARGET_NAME,
        module="graphs",
        description="Symbol definition-to-use edge extraction.",
        options=TargetSpecOptions(
            table_keys=SYMBOL_USES_TABLE_KEYS,
            override_tables=SYMBOL_USES_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=CALL_GRAPH_VIEWS_TARGET_NAME,
        module="graphs",
        description="Derived views over call graph for analytics.",
        options=TargetSpecOptions(
            table_keys=CALL_GRAPH_VIEWS_TABLE_KEYS,
            override_tables=CALL_GRAPH_VIEWS_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=GRAPH_METRICS_TARGET_NAME,
        module="graphs",
        description="Graph topology metrics for functions and modules.",
        options=TargetSpecOptions(
            table_keys=GRAPH_METRICS_TABLE_KEYS,
            override_tables=GRAPH_METRICS_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=GRAPH_VALIDATION_TARGET_NAME,
        module="graphs",
        description="Graph integrity validation checks.",
        options=TargetSpecOptions(
            table_keys=GRAPH_VALIDATION_TABLE_KEYS,
            override_tables=GRAPH_VALIDATION_OVERRIDE_TABLES,
        ),
    ),
)

_GRAPH_METRICS_OUTPUT_TABLES = GRAPH_METRICS_TABLE_KEYS

# ---------------------------------------------------------------------------
# Result dataclasses for goids target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoidExtractionInputs:
    """Inputs for GOID extraction.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    now
        Timestamp for row creation.
    options
        GOID builder options.
    module_name
        Module name for the current file.
    normalized_path
        Normalized relative path for the current file.
    """

    repo: str
    commit: str
    now: datetime
    options: GoidBuilderOptions
    module_name: str
    normalized_path: str


@dataclass(frozen=True)
class GoidExtractResult:
    """Result from GOID extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    goid_count
        Number of GOIDs extracted.
    crosswalk_count
        Number of crosswalk entries extracted.
    table_counts
        Row counts per produced table.
    skipped
        Whether extraction was skipped.
    skip_reason
        Optional reason for skipping extraction.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    goid_count: int = 0
    crosswalk_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Result dataclasses for symbol_uses target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolUsesExtractResult:
    """Result from symbol uses extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    edge_count
        Number of symbol use edges extracted.
    table_counts
        Row counts per produced table.
    skipped
        Whether extraction was skipped.
    skip_reason
        Optional reason for skipping extraction.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    edge_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


@tag_helper()
def goids__execution_result(t__goids__extract: GoidExtractResult) -> ExecutionResult:
    """Convert goids extract result to the executor boundary type.

    Returns
    -------
    ExecutionResult
        Canonical execution result.
    """
    return to_execution_result(t__goids__extract, default_error="GOID extraction failed")


@tag_helper()
def symbol_uses__execution_result(
    t__symbol_uses__extract: SymbolUsesExtractResult,
) -> ExecutionResult:
    """Convert symbol_uses extract result to the executor boundary type.

    Returns
    -------
    ExecutionResult
        Canonical execution result.
    """
    return to_execution_result(
        t__symbol_uses__extract, default_error="Symbol uses extraction failed"
    )


# ---------------------------------------------------------------------------
# Result dataclasses for graph_validation target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphValidationResult:
    """Result from graph validation.

    Attributes
    ----------
    success
        Whether validation passed (no errors).
    error_count
        Number of validation errors found.
    errors
        List of validation error messages.
    table_counts
        Row counts per output (validation errors).
    error
        Fatal error message if validation failed.
    """

    success: bool
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Helper functions for goids target
# ---------------------------------------------------------------------------


@tag_helper()
def _get_tracked_files(gateway: StorageGateway, repo: str, commit: str) -> list[str]:
    """Get list of tracked Python files from core.modules.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    list[str]
        List of relative paths to Python files.
    """
    try:
        modules = ibis_facade.table(gateway, "core.modules")
        expr = (
            filter_by(modules, modules.repo == repo, modules.commit == commit)
            .select(modules.path)
            .distinct()
            .order_by(modules.path)
        )
        df = expr.execute()
        return [str(path) for (path,) in df.itertuples(index=False, name=None)]
    except DuckDBError:
        return []


@tag_helper()
def _path_to_module_name(rel_path: str) -> str:
    """Convert relative path to module name.

    Parameters
    ----------
    rel_path
        Relative file path.

    Returns
    -------
    str
        Module name.
    """
    path = Path(rel_path)
    parts = list(path.parts)
    if path.suffix == ".py":
        parts[-1] = path.stem
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@tag_helper()
def _process_ast_node(
    node: ast.AST,
    parent_qualname: str | None,
    *,
    context: GoidExtractionInputs,
    goid_rows: list[GoidRow],
    crosswalk_rows: list[GoidCrosswalkRow],
) -> None:
    """Process an AST node recursively.

    Parameters
    ----------
    node
        The AST node to process.
    parent_qualname
        Qualified name of the parent node.
    context
        GOID extraction context.
    goid_rows
        List to append GOID rows to.
    crosswalk_rows
        List to append crosswalk rows to.
    """
    options = context.options
    name: str | None = None
    start_line: int = 0
    end_line: int | None = None

    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        name = node.name
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

    if name is not None:
        module_name = context.module_name
        if not options.include_private and name.startswith("_") and name != "__init__":
            for child in ast.iter_child_nodes(node):
                _process_ast_node(
                    child,
                    parent_qualname,
                    context=context,
                    goid_rows=goid_rows,
                    crosswalk_rows=crosswalk_rows,
                )
            return
        qualname = f"{parent_qualname}.{name}" if parent_qualname else f"{module_name}.{name}"
        kind = goid_compute.determine_kind(
            type(node).__name__, parent_qualname, context.normalized_path, module_name
        )

        descriptor = goid_compute.GoidDescriptor(
            repo=context.repo,
            commit=context.commit,
            language="python",
            rel_path=context.normalized_path,
            kind=kind,
            qualname=qualname,
            start_line=start_line,
            end_line=end_line,
        )
        result = goid_compute.compute_goid_result(descriptor)
        goid_rows.append(
            goid_compute.build_goid_row(descriptor, result.goid_h128, result.urn, context.now)
        )
        crosswalk_rows.append(
            goid_compute.build_crosswalk_row(descriptor, result.urn, module_name, context.now)
        )

        for child in ast.iter_child_nodes(node):
            _process_ast_node(
                child,
                qualname,
                context=context,
                goid_rows=goid_rows,
                crosswalk_rows=crosswalk_rows,
            )
    else:
        for child in ast.iter_child_nodes(node):
            _process_ast_node(
                child,
                parent_qualname,
                context=context,
                goid_rows=goid_rows,
                crosswalk_rows=crosswalk_rows,
            )


@tag_helper()
def _extract_entities_from_file(
    file_path: Path,
    context: GoidExtractionInputs,
) -> tuple[list[GoidRow], list[GoidCrosswalkRow]]:
    """Extract entities from a Python file and compute GOIDs.

    Parameters
    ----------
    file_path
        Absolute path to the file.
    context
        Extraction context with repo, commit, module metadata, and options.

    Returns
    -------
    tuple[list[GoidRow], list[GoidCrosswalkRow]]
        GOID rows and crosswalk rows.
    """
    if not file_path.exists():
        return [], []

    try:
        source = file_path.read_text(encoding="utf8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return [], []

    goid_rows: list[GoidRow] = []
    crosswalk_rows: list[GoidCrosswalkRow] = []

    module_name = context.module_name
    normalized_path = context.normalized_path

    module_descriptor = goid_compute.GoidDescriptor(
        repo=context.repo,
        commit=context.commit,
        language="python",
        rel_path=normalized_path,
        kind="module",
        qualname=module_name,
        start_line=1,
        end_line=len(source.splitlines()) if source else 1,
    )
    module_result = goid_compute.compute_goid_result(module_descriptor)
    goid_rows.append(
        goid_compute.build_goid_row(
            module_descriptor, module_result.goid_h128, module_result.urn, context.now
        )
    )
    crosswalk_rows.append(
        goid_compute.build_crosswalk_row(
            module_descriptor, module_result.urn, module_name, context.now
        )
    )

    for child in ast.iter_child_nodes(tree):
        _process_ast_node(
            child,
            module_name,
            context=context,
            goid_rows=goid_rows,
            crosswalk_rows=crosswalk_rows,
        )

    return goid_rows, crosswalk_rows


# ---------------------------------------------------------------------------
# Helper functions for symbol_uses target
# ---------------------------------------------------------------------------


@tag_helper()
def _load_symbol_occurrences(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[symbols_compute.SymbolOccurrence]:
    """Load SCIP symbol occurrences from database.

    Returns
    -------
    list[symbols_compute.SymbolOccurrence]
        Parsed symbol occurrences for the snapshot.
    """
    try:
        scip_tbl = ibis_facade.table(gateway, "core.scip_occurrences")
        expr = filter_by(scip_tbl, scip_tbl.repo == repo, scip_tbl.commit == commit).select(
            scip_tbl.symbol, scip_tbl.rel_path, scip_tbl.line, scip_tbl.roles
        )
        rows = expr.execute()

        return [
            symbols_compute.SymbolOccurrence(
                symbol=str(symbol),
                rel_path=normalize_path(str(rel_path)),
                line=int(line or 0),
                roles=symbols_compute.parse_symbol_roles(roles),
            )
            for symbol, rel_path, line, roles in rows.itertuples(index=False, name=None)
        ]
    except DuckDBError:
        return []


@tag_helper()
def _load_module_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module name by path mapping.

    Returns
    -------
    dict[str, str]
        Mapping of normalized relative paths to module names.
    """
    try:
        modules_tbl = ibis_facade.table(gateway, "core.modules")
        expr = filter_by(
            modules_tbl, modules_tbl.repo == repo, modules_tbl.commit == commit
        ).select(modules_tbl.path, modules_tbl.module)
        rows = expr.execute()
        return {
            normalize_path(str(path)): str(module)
            for path, module in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper()
def _load_path_to_goid_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Load GOID by path mapping.

    Returns
    -------
    dict[str, int]
        Mapping of normalized relative paths to module GOIDs.
    """
    try:
        goids_tbl = ibis_facade.table(gateway, GOIDS_GOIDS_TABLE_KEY)
        expr = filter_by(
            goids_tbl,
            goids_tbl.repo == repo,
            goids_tbl.commit == commit,
            goids_tbl.kind == "module",
        ).select(goids_tbl.rel_path, goids_tbl.goid_h128)
        rows = expr.execute()
        return {
            normalize_path(str(rel_path)): int(goid)
            for rel_path, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper()
def _enrich_edges_with_goids(
    edges: list[symbols_compute.SymbolUseEdge],
    path_to_goid: dict[str, int],
) -> list[symbols_compute.SymbolUseEdge]:
    """Enrich symbol use edges with GOIDs.

    Returns
    -------
    list[symbols_compute.SymbolUseEdge]
        New edges with GOID fields populated when available.
    """
    enriched: list[symbols_compute.SymbolUseEdge] = []
    for edge in edges:
        def_goid = path_to_goid.get(edge.def_path)
        use_goid = path_to_goid.get(edge.use_path)
        enriched.append(
            symbols_compute.SymbolUseEdge(
                symbol=edge.symbol,
                def_path=edge.def_path,
                use_path=edge.use_path,
                same_file=edge.same_file,
                same_module=edge.same_module,
                def_goid=def_goid,
                use_goid=use_goid,
            )
        )
    return enriched


@tag_helper()
def _filter_symbol_occurrences(
    occurrences: list[symbols_compute.SymbolOccurrence],
    *,
    options: SymbolUsesOptions,
) -> list[symbols_compute.SymbolOccurrence]:
    """Filter symbol occurrences by scope and test inclusion.

    Returns
    -------
    list[symbols_compute.SymbolOccurrence]
        Filtered occurrences.
    """
    filtered = list(occurrences)

    if options.scope_paths:
        prefixes = tuple(options.scope_paths)
        filtered = [occ for occ in filtered if occ.rel_path.startswith(prefixes)]

    if not options.include_tests:
        filtered = [occ for occ in filtered if not is_test_path(occ.rel_path)]

    return filtered


# ---------------------------------------------------------------------------
# Helper functions for graph_metrics target
# ---------------------------------------------------------------------------


@tag_helper()
def _count_rows(
    gateway: StorageGateway,
    table: str,
    repo: str,
    commit: str,
) -> int:
    """Count rows in a table for the given snapshot.

    Parameters
    ----------
    gateway
        Storage gateway.
    table
        Table name.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Row count.
    """
    try:
        tbl = ibis_facade.table(gateway, table)
        filtered = filter_by(tbl, tbl.repo == repo, tbl.commit == commit)
        result_df = filtered.aggregate(row_count=tbl.repo.count()).execute()
        return int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0
    except (RuntimeError, ValueError, OSError, KeyError):
        return 0


# ---------------------------------------------------------------------------
# Helper functions for graph_validation target
# ---------------------------------------------------------------------------


@tag_helper()
def _validate_call_graph_integrity(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[str]:
    """Validate call graph edge integrity.

    Returns
    -------
    list[str]
        Validation error messages.
    """
    errors: list[str] = []

    try:
        edges = ibis_facade.table(gateway, "graph.call_graph_edges")
        nodes = ibis_facade.table(gateway, "graph.call_graph_nodes")

        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        caller_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.caller_goid_h128, nodes.goid_h128)]
        )
        orphan_callers_expr = caller_join.filter(is_null(nodes.goid_h128)).count()
        orphan_callers = int(cast("SupportsInt", orphan_callers_expr.execute()))
        if orphan_callers > 0:
            errors.append(f"Found {orphan_callers} call graph edges with orphan caller GOIDs")

        callee_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.callee_goid_h128, nodes.goid_h128)]
        )
        orphan_callees_expr = callee_join.filter(
            and_predicates(not_null(scoped_edges.callee_goid_h128), is_null(nodes.goid_h128))
        ).count()
        orphan_callees = int(cast("SupportsInt", orphan_callees_expr.execute()))
        if orphan_callees > 0:
            log.debug(
                "validation: %d call graph edges have unresolved callee GOIDs",
                orphan_callees,
            )
    except DuckDBError as exc:
        log.debug("validation: Could not validate call graph: %s", exc)

    return errors


@tag_helper()
def _validate_import_graph_integrity(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[str]:
    """Validate import graph integrity.

    Returns
    -------
    list[str]
        Validation error messages.
    """
    errors: list[str] = []

    try:
        edges = ibis_facade.table(gateway, "graph.import_graph_edges")
        modules = ibis_facade.table(gateway, "graph.import_modules")
        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        joined = scoped_edges.left_join(
            modules,
            predicates=[
                (scoped_edges.src_module, modules.module),
                (scoped_edges.repo, modules.repo),
                (scoped_edges.commit, modules.commit),
            ],
        )
        orphan_src_expr = joined.filter(is_null(modules.module)).count()
        orphan_src = int(cast("SupportsInt", orphan_src_expr.execute()))
        if orphan_src > 0:
            errors.append(f"Found {orphan_src} import edges with missing source modules")

    except DuckDBError as exc:
        log.debug("validation: Could not validate import graph: %s", exc)

    return errors


@tag_helper()
def _validate_cfg_integrity(
    gateway: StorageGateway,
    _repo: str,
    _commit: str,
) -> list[str]:
    """Validate CFG integrity.

    Returns
    -------
    list[str]
        Validation error messages.
    """
    errors: list[str] = []

    try:
        edges = ibis_facade.table(gateway, "graph.cfg_edges")
        blocks = ibis_facade.table(gateway, "graph.cfg_blocks")

        joined = edges.left_join(
            blocks,
            predicates=[
                (edges.src_block_id, blocks.block_id),
                (edges.function_goid_h128, blocks.function_goid_h128),
            ],
        )
        orphan_edges_expr = joined.filter(is_null(blocks.block_id)).count()
        orphan_edges = int(cast("SupportsInt", orphan_edges_expr.execute()))
        if orphan_edges > 0:
            errors.append(f"Found {orphan_edges} CFG edges with missing source blocks")

    except DuckDBError as exc:
        log.debug("validation: Could not validate CFG: %s", exc)

    return errors


# ---------------------------------------------------------------------------
# goids target - compute and materialize
# ---------------------------------------------------------------------------


@tag_tool(domain="graphs", target=GOIDS_TARGET_NAME)
def t__goids__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> GoidExtractResult:
    """Execute GOID extraction on repository modules.

    This is the compute node for the goids target. It parses Python source
    files, extracts modules, classes, and functions, and computes stable
    GOIDs for each entity.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    GoidExtractResult
        Result containing GOID and crosswalk row counts.

    Notes
    -----
    Produces:
    - core.goids: GOID records for all entities
    - core.goid_crosswalk: GOID crosswalk records
    """
    if t__modules.status != "succeeded":
        return GoidExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = load_target_options(
            env,
            target_name=GOIDS_TARGET_NAME,
            options_type=GoidBuilderOptions,
        )

        source_root = env.snapshot.repo_root or get_source_root(env.gateway, repo, commit)

        tracked_files = filter_paths(
            _get_tracked_files(env.gateway, repo, commit),
            scope_paths=opts.scope_paths,
            include_tests=opts.include_tests,
        )

        if not tracked_files:
            log.info("goids: No tracked files found, skipping")
            return GoidExtractResult(
                success=True,
                goid_count=0,
                crosswalk_count=0,
                table_counts={
                    GOIDS_GOIDS_TABLE_KEY: 0,
                    GOIDS_CROSSWALK_TABLE_KEY: 0,
                },
            )

        now = datetime.now(UTC)
        all_goid_rows: list[GoidRow] = []
        all_crosswalk_rows: list[GoidCrosswalkRow] = []

        for rel_path in tracked_files:
            rows = _extract_entities_from_file(
                source_root / rel_path,
                GoidExtractionInputs(
                    repo=repo,
                    commit=commit,
                    now=now,
                    options=opts,
                    module_name=_path_to_module_name(rel_path),
                    normalized_path=normalize_path(rel_path),
                ),
            )
            all_goid_rows.extend(rows[0])
            all_crosswalk_rows.extend(rows[1])

        log.info(
            "goids: Extracted %d GOIDs and %d crosswalk entries from %d files",
            len(all_goid_rows),
            len(all_crosswalk_rows),
            len(tracked_files),
        )

        options = materialize_options(env, owner_target=GOIDS_TARGET_NAME, mode="replace")
        goid_result = env.warehouse.materialize_rows(
            GOIDS_GOIDS_TABLE_KEY,
            [row.to_tuple() for row in all_goid_rows],
            columns=None,
            options=options,
        )
        crosswalk_result = env.warehouse.materialize_rows(
            GOIDS_CROSSWALK_TABLE_KEY,
            [row.to_tuple() for row in all_crosswalk_rows],
            columns=None,
            options=options,
        )
        goid_count = int(goid_result.rows_written or 0)
        crosswalk_count = int(crosswalk_result.rows_written or 0)

        log.info(
            "goids: Persisted %d GOIDs and %d crosswalk entries",
            goid_count,
            crosswalk_count,
        )

        return GoidExtractResult(
            success=True,
            goid_count=goid_count,
            crosswalk_count=crosswalk_count,
            table_counts={
                GOIDS_GOIDS_TABLE_KEY: goid_count,
                GOIDS_CROSSWALK_TABLE_KEY: crosswalk_count,
            },
        )

    except Exception as exc:
        log.exception("GOID extraction failed")
        return GoidExtractResult(
            success=False,
            error=str(exc),
        )


@tag_materialize(domain="graphs", target=GOIDS_TARGET_NAME)
def t__goids(
    env: BuildEnv,
    graph: TargetGraph,
    goids__execution_result: ExecutionResult,
) -> TargetRunRecord:
    """Materialize GOIDs target with validation.

    This is the entry point for the goids target. It orchestrates
    GOID extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    goids__execution_result
        Execution result derived from upstream extract node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, GOIDS_TARGET_NAME, goids__execution_result)


# ---------------------------------------------------------------------------
# symbol_uses target - compute and materialize
# ---------------------------------------------------------------------------


@tag_tool(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def t__symbol_uses__extract(
    env: BuildEnv,
    t__scip: TargetRunRecord,
    t__modules: TargetRunRecord,
    t__goids: TargetRunRecord,
) -> SymbolUsesExtractResult:
    """Execute symbol use extraction from SCIP data.

    Returns
    -------
    SymbolUsesExtractResult
        Status and per-table row counts for extracted edges.
    """
    for name, record in [("scip", t__scip), ("modules", t__modules), ("goids", t__goids)]:
        if record.status != "succeeded":
            return SymbolUsesExtractResult(
                success=False,
                error=f"Upstream {name} target failed: {record.error}",
            )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = load_target_options(
            env,
            target_name=SYMBOL_USES_TARGET_NAME,
            options_type=SymbolUsesOptions,
        )

        occurrences = _load_symbol_occurrences(gateway, repo, commit)

        if not occurrences:
            log.info("symbol_uses: No SCIP occurrences found, skipping")
            return SymbolUsesExtractResult(
                success=True,
                edge_count=0,
                table_counts={SYMBOL_USE_EDGES_TABLE_KEY: 0},
            )

        occurrences = _filter_symbol_occurrences(occurrences, options=opts)

        module_map = _load_module_map(gateway, repo, commit)
        path_to_goid = _load_path_to_goid_map(gateway, repo, commit)

        def_map = symbols_compute.build_def_map(occurrences)
        edges = symbols_compute.build_use_edges(
            occurrences,
            def_map=def_map,
            module_by_path=module_map,
        )

        enriched_edges = _enrich_edges_with_goids(edges, path_to_goid)
        rows = symbols_compute.edges_to_rows(enriched_edges)
        row_result = env.warehouse.materialize_rows(
            SYMBOL_USE_EDGES_TABLE_KEY,
            [row.to_tuple() for row in rows],
            columns=None,
            options=materialize_options(
                env,
                owner_target=SYMBOL_USES_TARGET_NAME,
                mode="replace",
            ),
        )
        row_count = int(row_result.rows_written or 0)
        return SymbolUsesExtractResult(
            success=True,
            edge_count=row_count,
            table_counts={SYMBOL_USE_EDGES_TABLE_KEY: row_count},
        )
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        log.exception("symbol_uses: extraction failed")
        return SymbolUsesExtractResult(success=False, error=str(exc))


@tag_materialize(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def t__symbol_uses(
    env: BuildEnv,
    graph: TargetGraph,
    symbol_uses__execution_result: ExecutionResult,
) -> TargetRunRecord:
    """Materialize symbol_uses target.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return executor_materialize(env, graph, SYMBOL_USES_TARGET_NAME, symbol_uses__execution_result)


# ---------------------------------------------------------------------------
# call_graph_views target - compute and materialize (uses Ibis materializations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallGraphCallCountStats:
    """Intermediate aggregations for function call count view.

    Attributes
    ----------
    callee_stats
        Aggregation keyed by caller function, including callee counts.
    caller_stats
        Aggregation keyed by callee function, including caller counts.
    """

    callee_stats: ir.Table
    caller_stats: ir.Table


@dataclass(frozen=True)
class CallGraphDepthTables:
    """Intermediate tables for call depth view.

    Attributes
    ----------
    all_funcs
        Distinct set of all functions participating in the call graph.
    caller_funcs
        Distinct set of caller functions (used to mark leaf nodes).
    """

    all_funcs: ir.Table
    caller_funcs: ir.Table


def _call_graph_views_filter_edges(edges: ir.Table, env: BuildEnv) -> ir.Table:
    """Filter call graph edges to the current snapshot.

    Parameters
    ----------
    edges
        Ibis table expression for graph.call_graph_edges.
    env
        Build environment providing the snapshot filter.

    Returns
    -------
    ir.Table
        Filtered call graph edges scoped to the current snapshot.
    """
    return filter_by(edges, edges.repo == env.snapshot.repo, edges.commit == env.snapshot.commit)


def _call_graph_views_build_call_count_stats(edges: ir.Table) -> CallGraphCallCountStats:
    """Build per-function call count aggregations from filtered call graph edges.

    Parameters
    ----------
    edges
        Filtered call graph edges scoped to the current snapshot.

    Returns
    -------
    CallGraphCallCountStats
        Aggregations used to build the call count view.
    """
    callee_stats: ir.Table = edges.group_by(
        function_goid_h128=edges.caller_goid_h128,
    ).aggregate(
        num_callees=ibis._.count(),
        num_unique_callees=col_nunique(edges.callee_goid_h128),
    )

    caller_stats: ir.Table = (
        filter_by(edges, not_null(edges.callee_goid_h128))
        .group_by(
            function_goid_h128=edges.callee_goid_h128,
        )
        .aggregate(num_callers=ibis._.count())
    )

    return CallGraphCallCountStats(callee_stats=callee_stats, caller_stats=caller_stats)


def _call_graph_views_finalize_call_counts(
    stats: CallGraphCallCountStats, env: BuildEnv
) -> ir.Table:
    """Join call count aggregations and emit final view expression.

    Parameters
    ----------
    stats
        Intermediate call count aggregations.
    env
        Build environment providing the snapshot to embed in output rows.

    Returns
    -------
    ir.Table
        Final call count view expression.
    """
    result = stats.callee_stats.join(
        stats.caller_stats,
        predicates=[
            stats.callee_stats.function_goid_h128 == stats.caller_stats.function_goid_h128,
        ],
        how="outer",
    )
    return result.select(
        repo=ibis.literal(env.snapshot.repo),
        commit=ibis.literal(env.snapshot.commit),
        function_goid_h128=result.function_goid_h128,
        num_callees=fillna(result.num_callees, ibis.literal(0)),
        num_unique_callees=fillna(result.num_unique_callees, ibis.literal(0)),
        num_callers=fillna(result.num_callers, ibis.literal(0)),
    )


def _call_graph_views_prepare_depth_tables(edges: ir.Table) -> CallGraphDepthTables:
    """Prepare intermediate tables for depth stats computation.

    Parameters
    ----------
    edges
        Filtered call graph edges scoped to the current snapshot.

    Returns
    -------
    CallGraphDepthTables
        Intermediate tables required to compute depth stats.
    """
    caller_funcs: ir.Table = edges.select(
        caller_function_goid_h128=edges.caller_goid_h128,
    ).distinct()
    callee_funcs: ir.Table = (
        filter_by(edges, not_null(edges.callee_goid_h128))
        .select(
            function_goid_h128=edges.callee_goid_h128,
        )
        .distinct()
    )
    all_funcs: ir.Table = (
        caller_funcs.select(function_goid_h128=caller_funcs.caller_function_goid_h128)
        .union(callee_funcs)
        .distinct()
    )
    return CallGraphDepthTables(all_funcs=all_funcs, caller_funcs=caller_funcs)


def _call_graph_views_finalize_depth_stats(tables: CallGraphDepthTables, env: BuildEnv) -> ir.Table:
    """Compute final call depth stats view expression.

    Parameters
    ----------
    tables
        Intermediate tables prepared from filtered call graph edges.
    env
        Build environment providing the snapshot to embed in output rows.

    Returns
    -------
    ir.Table
        Final call depth stats view expression.
    """
    joined = tables.all_funcs.left_join(
        tables.caller_funcs,
        predicates=[
            tables.all_funcs.function_goid_h128 == tables.caller_funcs.caller_function_goid_h128,
        ],
    )
    is_leaf = is_null(joined.caller_function_goid_h128)
    return joined.select(
        repo=ibis.literal(env.snapshot.repo),
        commit=ibis.literal(env.snapshot.commit),
        function_goid_h128=joined.function_goid_h128,
        max_call_depth=ibis.ifelse(is_leaf, ibis.literal(0), ibis.literal(1)),
        is_leaf=is_leaf,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node(CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS),
    env=source("env"),
    graph=source("graph"),
    target_name=value(CALL_GRAPH_VIEWS_TARGET_NAME),
    table_key=value(CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS),
)
@pipe_input(
    step(_call_graph_views_filter_edges, env=source("env")),
    step(_call_graph_views_build_call_count_stats),
    step(_call_graph_views_finalize_call_counts, env=source("env")),
    namespace="call_graph_function_call_counts",
    on_input="q__graph__call_graph_edges",
)
@tag_compute(
    domain="graphs",
    target=CALL_GRAPH_VIEWS_TARGET_NAME,
    target_="call_graph_function_call_counts",
    extra_tags={"output_kind": "view"},
)
@check_output_custom(
    *build_table_contract(
        required_columns=[
            "repo",
            "commit",
            "function_goid_h128",
            "num_callees",
            "num_unique_callees",
            "num_callers",
        ],
        no_nulls=["repo", "commit"],
    )
)
@schema.output(
    ("repo", "string"),
    ("commit", "string"),
    ("function_goid_h128", "int64"),
    ("num_callees", "int64"),
    ("num_unique_callees", "int64"),
    ("num_callers", "int64"),
    target_="call_graph_function_call_counts",
)
def call_graph_function_call_counts(
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Compute per-function call count statistics from call graph edges.

    Returns
    -------
    ir.Table
        Ibis expression producing call count view rows.
    """
    return q__graph__call_graph_edges


@SaveToObjectMetadataDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node(CALL_GRAPH_VIEWS_CALL_DEPTH_STATS),
    env=source("env"),
    graph=source("graph"),
    target_name=value(CALL_GRAPH_VIEWS_TARGET_NAME),
    table_key=value(CALL_GRAPH_VIEWS_CALL_DEPTH_STATS),
)
@pipe_input(
    step(_call_graph_views_filter_edges, env=source("env")),
    step(_call_graph_views_prepare_depth_tables),
    step(_call_graph_views_finalize_depth_stats, env=source("env")),
    namespace="call_graph_call_depth_stats",
    on_input="q__graph__call_graph_edges",
)
@tag_compute(
    domain="graphs",
    target=CALL_GRAPH_VIEWS_TARGET_NAME,
    target_="call_graph_depth_stats",
    extra_tags={"output_kind": "view"},
)
@check_output_custom(
    *build_table_contract(
        required_columns=[
            "repo",
            "commit",
            "function_goid_h128",
            "max_call_depth",
            "is_leaf",
        ],
        no_nulls=["repo", "commit"],
    )
)
@schema.output(
    ("repo", "string"),
    ("commit", "string"),
    ("function_goid_h128", "int64"),
    ("max_call_depth", "int64"),
    ("is_leaf", "boolean"),
    target_="call_graph_depth_stats",
)
def call_graph_depth_stats(
    q__graph__call_graph_edges: ir.Table,
) -> ir.Table:
    """Compute call depth statistics (simplified version).

    Returns
    -------
    ir.Table
        Ibis expression producing call depth view rows.
    """
    return q__graph__call_graph_edges


@tag_materialize(domain="graphs", target=CALL_GRAPH_VIEWS_TARGET_NAME)
def t__call_graph_views(
    env: BuildEnv,
    graph: TargetGraph,
    m__graph__v_function_call_counts: MaterializationMetadata,
    m__graph__v_call_depth_stats: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize call graph view expressions to DuckDB.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    LOG.info("Materializing call_graph_views to DuckDB")

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=CALL_GRAPH_VIEWS_TARGET_NAME,
        materializations={
            CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS: m__graph__v_function_call_counts,
            CALL_GRAPH_VIEWS_CALL_DEPTH_STATS: m__graph__v_call_depth_stats,
        },
    )


# ---------------------------------------------------------------------------
# graph_metrics target - compute and materialize
# ---------------------------------------------------------------------------


@tag_tool(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def t__graph_metrics__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
) -> ExecutionResult:
    """Compute graph metrics from call graph data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__call_graph
        Upstream call_graph target result (for dependency).

    Returns
    -------
    ExecutionResult
        Result containing table row counts.
    """
    if t__call_graph.status != "succeeded":
        return ExecutionResult.failed(f"Upstream call_graph target failed: {t__call_graph.error}")

    try:
        gateway = env.gateway
        snapshot = env.snapshot
        repo, commit = snapshot.repo, snapshot.commit

        log.info(
            "graph_metrics: Computing metrics for repo=%s commit=%s",
            repo,
            commit,
        )

        backend_config = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
        base_runtime_options = load_graph_runtime_options(
            env, target_name=GRAPH_METRICS_TARGET_NAME
        )
        runtime_options = replace(
            base_runtime_options,
            snapshot=snapshot,
            backend=backend_config,
        )
        runtime = build_graph_runtime(gateway, runtime_options)

        options = load_target_options(
            env,
            target_name=GRAPH_METRICS_TARGET_NAME,
            options_type=GraphMetricsOptions,
        )
        deps = GraphMetricsDeps(
            catalog_provider=None,
            runtime=runtime,
        )
        compute_graph_metrics(gateway, snapshot, options=options, deps=deps)

        compute_graph_metrics_functions_ext(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        compute_graph_metrics_modules_ext(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        compute_graph_stats(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        row_counts: dict[str, int] = {}
        for table in _GRAPH_METRICS_OUTPUT_TABLES:
            row_counts[table] = _count_rows(gateway, table, repo, commit)

        log.info("graph_metrics: Computed metrics row_counts=%s", row_counts)

        return ExecutionResult.ok(table_counts=row_counts)

    except (RuntimeError, ValueError, OSError) as exc:
        log.exception("Graph metrics computation failed")
        return ExecutionResult.failed(str(exc))


@tag_materialize(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def t__graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__graph_metrics__compute: ExecutionResult,
) -> TargetRunRecord:
    """Materialize graph metrics target with validation.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return executor_materialize(env, graph, GRAPH_METRICS_TARGET_NAME, t__graph_metrics__compute)


# ---------------------------------------------------------------------------
# graph_validation target - compute and materialize
# ---------------------------------------------------------------------------


@tag_tool(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def t__graph_validation__check(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__import_graph: TargetRunRecord,
    t__cfg: TargetRunRecord,
) -> GraphValidationResult:
    """Run validation checks on all graph data.

    Returns
    -------
    GraphValidationResult
        Validation status and any discovered issues.
    """
    deps = [("call_graph", t__call_graph), ("import_graph", t__import_graph), ("cfg", t__cfg)]
    for name, record in deps:
        if record.status != "succeeded":
            return GraphValidationResult(
                success=False,
                error=f"Upstream {name} target failed: {record.error}",
            )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit

        all_errors: list[str] = []

        call_graph_errors = _validate_call_graph_integrity(gateway, repo, commit)
        all_errors.extend(call_graph_errors)

        import_graph_errors = _validate_import_graph_integrity(gateway, repo, commit)
        all_errors.extend(import_graph_errors)

        cfg_errors = _validate_cfg_integrity(gateway, repo, commit)
        all_errors.extend(cfg_errors)

        for error in all_errors:
            log.warning("graph_validation: %s", error)

        log.info(
            "graph_validation: Completed with %d issues found for repo=%s commit=%s",
            len(all_errors),
            repo,
            commit,
        )

        return GraphValidationResult(
            success=len(all_errors) == 0,
            error_count=len(all_errors),
            errors=all_errors,
            table_counts={GRAPH_VALIDATION_TABLE_KEY: len(all_errors)},
        )

    except Exception as exc:
        log.exception("Graph validation failed")
        return GraphValidationResult(
            success=False,
            error=str(exc),
        )


@tag_materialize(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def t__graph_validation(
    env: BuildEnv,
    graph: TargetGraph,
    t__graph_validation__check: GraphValidationResult,
) -> TargetRunRecord:
    """Materialize graph validation target.

    This target requires custom error handling to join validation errors
    into a readable message, so it does not use executor_materialize.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, GRAPH_VALIDATION_TARGET_NAME)

    if executor.should_skip():
        return executor.skip()

    if t__graph_validation__check.error:
        return executor.fail(RuntimeError(t__graph_validation__check.error))

    if not t__graph_validation__check.success:
        errors_msg = "\n".join(t__graph_validation__check.errors)
        return executor.fail(RuntimeError(f"Graph validation failed:\n{errors_msg}"))

    def compute() -> dict[str, int]:
        return dict(t__graph_validation__check.table_counts)

    return executor.execute(compute)


__all__ = [
    "GoidExtractResult",
    "GoidExtractionInputs",
    "GraphValidationResult",
    "SymbolUsesExtractResult",
    "call_graph_depth_stats",
    "call_graph_function_call_counts",
    "t__call_graph_views",
    "t__goids",
    "t__goids__extract",
    "t__graph_metrics",
    "t__graph_metrics__compute",
    "t__graph_validation",
    "t__graph_validation__check",
    "t__symbol_uses",
    "t__symbol_uses__extract",
]
