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

Graph metrics target materializes DAG-visible rows via DuckDBRowsSaver.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, replace
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
)

from codeintel.analytics.graphs.graph_metrics import (
    GraphMetricsDeps,
    build_graph_metric_filters,
    build_graph_metrics_rows,
)
from codeintel.analytics.graphs.graph_metrics_ext import (
    build_graph_metrics_functions_ext_rows,
)
from codeintel.analytics.graphs.graph_stats import build_graph_stats_rows
from codeintel.analytics.graphs.module_graph_metrics_ext import (
    build_graph_metrics_modules_ext_rows,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.helpers import (
    filter_paths,
    get_source_root,
    is_test_path,
)
from codeintel.build.hamilton.naming import pipeline_node_name
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.options.graphs import GoidBuilderOptions, SymbolUsesOptions
from codeintel.build.hamilton.native.patterns import (
    IbisTableSaveSpec,
    IngestStep,
    SaverContext,
    TableSaveSpec,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_ibis_table,
    save_rows,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.optional_inputs import register_optional_inputs
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
)
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hamilton.validators import build_table_contract
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
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.core.validation.reporters import GraphValidationReporter
from codeintel.graphs.compute import goid as goid_compute
from codeintel.graphs.compute import symbols as symbols_compute
from codeintel.graphs.runtime import (
    GraphMetricsOptions,
    build_graph_runtime,
)
from codeintel.storage.gateway import DuckDBError, StorageGateway

if TYPE_CHECKING:
    from codeintel.graphs.compute.goid import GoidCrosswalkRow, GoidRow
log = logging.getLogger(__name__)
LOG = log

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, ir.Table)

GOIDS_TARGET_NAME = "goids"
SYMBOL_USES_TARGET_NAME = "symbol_uses"
CALL_GRAPH_VIEWS_TARGET_NAME = "call_graph_views"
GRAPH_METRICS_TARGET_NAME = "graph_metrics"
GRAPH_VALIDATION_TARGET_NAME = "graph_validation"

GOIDS_GOIDS_TABLE_KEY = "core.goids"
GOIDS_CROSSWALK_TABLE_KEY = "core.goid_crosswalk"
GOIDS_TABLE_KEYS = (GOIDS_GOIDS_TABLE_KEY, GOIDS_CROSSWALK_TABLE_KEY)
SCIP_OCCURRENCES_TABLE_KEY = "core.scip_occurrences"
SYMBOL_USE_EDGES_TABLE_KEY = "graph.symbol_use_edges"
SYMBOL_USES_TABLE_KEYS = (SYMBOL_USE_EDGES_TABLE_KEY,)

CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS = "graph.v_function_call_counts"
CALL_GRAPH_VIEWS_CALL_DEPTH_STATS = "graph.v_call_depth_stats"
CALL_GRAPH_VIEWS_TABLE_KEYS = (
    CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS,
    CALL_GRAPH_VIEWS_CALL_DEPTH_STATS,
)

GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.graph_metrics_functions"
GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY = "analytics.graph_metrics_functions_ext"
GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.graph_metrics_modules"
GRAPH_METRICS_MODULES_EXT_TABLE_KEY = "analytics.graph_metrics_modules_ext"
GRAPH_STATS_TABLE_KEY = "analytics.graph_stats"
GRAPH_METRICS_TABLE_KEYS = (
    GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
    GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
    GRAPH_METRICS_MODULES_TABLE_KEY,
    GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
    GRAPH_STATS_TABLE_KEY,
)

GRAPH_VALIDATION_TABLE_KEY = "analytics.graph_validation"
GRAPH_VALIDATION_TABLE_KEYS = (GRAPH_VALIDATION_TABLE_KEY,)

register_optional_inputs(SYMBOL_USES_TARGET_NAME, (SCIP_OCCURRENCES_TABLE_KEY,))

CALL_GRAPH_FUNCTION_CALL_COUNTS_NAMESPACE = "call_graph_function_call_counts"
CALL_GRAPH_DEPTH_STATS_NAMESPACE = "call_graph_call_depth_stats"

GOIDS_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=GOIDS_TARGET_NAME,
)
SYMBOL_USES_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=SYMBOL_USES_TARGET_NAME,
)
CALL_GRAPH_VIEWS_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=CALL_GRAPH_VIEWS_TARGET_NAME,
)
GRAPH_METRICS_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
)
GRAPH_VALIDATION_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=GRAPH_VALIDATION_TARGET_NAME,
)


@dataclass(frozen=True)
class GoidsToolOutput(ToolStepOutput):
    """Tool step output for GOID extraction."""

    goid_rows: tuple[tuple[object, ...], ...] = ()
    crosswalk_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class SymbolUsesToolOutput(ToolStepOutput):
    """Tool step output for symbol use extraction."""

    edge_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class GraphMetricsToolOutput(ToolStepOutput):
    """Tool step output for graph metrics computation."""

    functions_rows: tuple[tuple[object, ...], ...] = ()
    modules_rows: tuple[tuple[object, ...], ...] = ()
    functions_ext_rows: tuple[tuple[object, ...], ...] = ()
    modules_ext_rows: tuple[tuple[object, ...], ...] = ()
    graph_stats_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class GraphValidationIssue:
    """Structured issue detail for graph validation checks."""

    graph_name: str
    issue: str
    detail: str
    severity: str | None = None
    rel_path: str | None = None
    entity_id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GraphValidationToolOutput(ToolStepOutput):
    """Tool step output for graph validation checks."""

    rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class GoidsRunInputs:
    """Inputs required for GOID extraction."""

    modules: ir.Table
    modules_record: TargetRunRecord
    source_root: Path | None


@dataclass(frozen=True)
class SymbolUsesRunInputs:
    """Inputs required for symbol use extraction."""

    inputs: SymbolUsesInputs
    scip_record: TargetRunRecord
    modules_record: TargetRunRecord
    goids_record: TargetRunRecord


@dataclass(frozen=True)
class GraphValidationRunInputs:
    """Inputs required for graph validation checks."""

    inputs: GraphValidationInputs
    call_graph_record: TargetRunRecord
    import_graph_record: TargetRunRecord
    cfg_record: TargetRunRecord


# ---------------------------------------------------------------------------
# Input dataclasses for goids target
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
class GraphValidationInputs:
    """Input tables required for graph_validation checks."""

    call_graph_edges: ir.Table
    call_graph_nodes: ir.Table
    import_graph_edges: ir.Table
    import_modules: ir.Table
    cfg_edges: ir.Table
    cfg_blocks: ir.Table


@dataclass(frozen=True)
class CallGraphTables:
    """Call graph tables needed for validation checks."""

    edges: ir.Table
    nodes: ir.Table


@dataclass(frozen=True)
class ImportGraphTables:
    """Import graph tables needed for validation checks."""

    edges: ir.Table
    modules: ir.Table


@dataclass(frozen=True)
class CfgTables:
    """CFG tables needed for validation checks."""

    edges: ir.Table
    blocks: ir.Table


@tag_helper(domain="graphs")
def graph_validation__call_graph_tables(
    q__graph__call_graph_edges: ir.Table,
    q__graph__call_graph_nodes: ir.Table,
) -> CallGraphTables:
    """Bundle call graph tables for validation.

    Returns
    -------
    CallGraphTables
        Tables required for call graph integrity checks.
    """
    return CallGraphTables(
        edges=q__graph__call_graph_edges,
        nodes=q__graph__call_graph_nodes,
    )


@tag_helper(domain="graphs")
def graph_validation__import_graph_tables(
    q__graph__import_graph_edges: ir.Table,
    q__graph__import_modules: ir.Table,
) -> ImportGraphTables:
    """Bundle import graph tables for validation.

    Returns
    -------
    ImportGraphTables
        Tables required for import graph integrity checks.
    """
    return ImportGraphTables(
        edges=q__graph__import_graph_edges,
        modules=q__graph__import_modules,
    )


@tag_helper(domain="graphs")
def graph_validation__cfg_tables(
    q__graph__cfg_edges: ir.Table,
    q__graph__cfg_blocks: ir.Table,
) -> CfgTables:
    """Bundle CFG tables for validation.

    Returns
    -------
    CfgTables
        Tables required for CFG integrity checks.
    """
    return CfgTables(
        edges=q__graph__cfg_edges,
        blocks=q__graph__cfg_blocks,
    )


@tag_helper(domain="graphs")
def graph_validation__inputs(
    graph_validation__call_graph_tables: CallGraphTables,
    graph_validation__import_graph_tables: ImportGraphTables,
    graph_validation__cfg_tables: CfgTables,
) -> GraphValidationInputs:
    """Bundle tables required for graph_validation checks.

    Returns
    -------
    GraphValidationInputs
        Combined tables for validation checks.
    """
    return GraphValidationInputs(
        call_graph_edges=graph_validation__call_graph_tables.edges,
        call_graph_nodes=graph_validation__call_graph_tables.nodes,
        import_graph_edges=graph_validation__import_graph_tables.edges,
        import_modules=graph_validation__import_graph_tables.modules,
        cfg_edges=graph_validation__cfg_tables.edges,
        cfg_blocks=graph_validation__cfg_tables.blocks,
    )


# ---------------------------------------------------------------------------
# Helper functions for goids target
# ---------------------------------------------------------------------------




@tag_helper(domain="graphs", target=GOIDS_TARGET_NAME)
def goids__run_inputs(
    q__core__modules: ir.Table,
    t__modules: TargetRunRecord,
    goids__source_root: Path | None,
) -> GoidsRunInputs:
    """Bundle inputs for GOID extraction.

    Returns
    -------
    GoidsRunInputs
        Return value.

    """
    return GoidsRunInputs(
        modules=q__core__modules,
        modules_record=t__modules,
        source_root=goids__source_root,
    )


@tag_helper(domain="graphs", target=GOIDS_TARGET_NAME)
def goids__source_root(env: BuildEnv) -> Path | None:
    """Resolve repository root for GOID extraction.

    Returns
    -------
    Path | None
        Return value.

    """
    repo_root = env.snapshot.repo_root
    if repo_root is not None:
        return repo_root
    try:
        return get_source_root(env.gateway, env.snapshot.repo, env.snapshot.commit)
    except (OSError, RuntimeError, ValueError):
        return None


@tag_helper(domain="graphs")
def _get_tracked_files(q__core__modules: ir.Table, repo: str, commit: str) -> list[str]:
    """Get list of tracked Python files from core.modules.

    Parameters
    ----------
    q__core__modules
        Ibis table expression for core.modules.
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
        expr = (
            filter_by(
                q__core__modules,
                q__core__modules.repo == repo,
                q__core__modules.commit == commit,
            )
            .select(q__core__modules.path)
            .distinct()
            .order_by(q__core__modules.path)
        )
        df = expr.execute()
        return [str(path) for (path,) in df.itertuples(index=False, name=None)]
    except DuckDBError:
        return []


@tag_helper(domain="graphs")
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


@tag_helper(domain="graphs")
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


@tag_helper(domain="graphs")
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


@dataclass(frozen=True)
class SymbolUsesInputs:
    """Input tables required for symbol_uses extraction."""

    scip_occurrences: ir.Table
    modules: ir.Table
    goids: ir.Table


@tag_helper(domain="graphs")
def symbol_uses__inputs(
    q__core__scip_occurrences: ir.Table,
    q__core__modules: ir.Table,
    q__core__goids: ir.Table,
) -> SymbolUsesInputs:
    """Bundle tables required for symbol_uses extraction.

    Returns
    -------
    SymbolUsesInputs
        Tables required for symbol_uses extraction.
    """
    return SymbolUsesInputs(
        scip_occurrences=q__core__scip_occurrences,
        modules=q__core__modules,
        goids=q__core__goids,
    )




@tag_helper(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def symbol_uses__run_inputs(
    symbol_uses__inputs: SymbolUsesInputs,
    t__scip: TargetRunRecord,
    t__modules: TargetRunRecord,
    t__goids: TargetRunRecord,
) -> SymbolUsesRunInputs:
    """Bundle inputs for symbol use extraction.

    Returns
    -------
    SymbolUsesRunInputs
        Return value.

    """
    return SymbolUsesRunInputs(
        inputs=symbol_uses__inputs,
        scip_record=t__scip,
        modules_record=t__modules,
        goids_record=t__goids,
    )


@tag_helper(domain="graphs")
def _load_symbol_occurrences(
    q__core__scip_occurrences: ir.Table,
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
        expr = filter_by(
            q__core__scip_occurrences,
            q__core__scip_occurrences.repo == repo,
            q__core__scip_occurrences.commit == commit,
        ).select(
            q__core__scip_occurrences.symbol,
            q__core__scip_occurrences.rel_path,
            q__core__scip_occurrences.line,
            q__core__scip_occurrences.roles,
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


@tag_helper(domain="graphs")
def _load_module_map(
    q__core__modules: ir.Table,
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
        expr = filter_by(
            q__core__modules,
            q__core__modules.repo == repo,
            q__core__modules.commit == commit,
        ).select(q__core__modules.path, q__core__modules.module)
        rows = expr.execute()
        return {
            normalize_path(str(path)): str(module)
            for path, module in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper(domain="graphs")
def _load_path_to_goid_map(
    q__core__goids: ir.Table,
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
        expr = filter_by(
            q__core__goids,
            q__core__goids.repo == repo,
            q__core__goids.commit == commit,
            q__core__goids.kind == "module",
        ).select(q__core__goids.rel_path, q__core__goids.goid_h128)
        rows = expr.execute()
        return {
            normalize_path(str(rel_path)): int(goid)
            for rel_path, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper(domain="graphs")
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


@tag_helper(domain="graphs")
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
# Helper functions for graph_validation target
# ---------------------------------------------------------------------------


@tag_helper(domain="graphs")
def _validate_call_graph_integrity(
    q__graph__call_graph_edges: ir.Table,
    q__graph__call_graph_nodes: ir.Table,
    repo: str,
    commit: str,
) -> list[GraphValidationIssue]:
    """Validate call graph edge integrity.

    Returns
    -------
    list[GraphValidationIssue]
        Validation issues for call graph integrity.
    """
    issues: list[GraphValidationIssue] = []

    try:
        edges = q__graph__call_graph_edges
        nodes = q__graph__call_graph_nodes

        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        caller_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.caller_goid_h128, nodes.goid_h128)]
        )
        orphan_callers_expr = caller_join.filter(is_null(nodes.goid_h128)).count()
        orphan_callers = int(cast("SupportsInt", orphan_callers_expr.execute()))
        if orphan_callers > 0:
            issues.append(
                GraphValidationIssue(
                    graph_name="call_graph",
                    issue="orphan_caller_goids",
                    detail=(f"Found {orphan_callers} call graph edges with orphan caller GOIDs"),
                    severity="error",
                    entity_id="call_graph_edges",
                    metadata={"orphan_count": orphan_callers},
                )
            )

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

    return issues


@tag_helper(domain="graphs")
def _validate_import_graph_integrity(
    q__graph__import_graph_edges: ir.Table,
    q__graph__import_modules: ir.Table,
    repo: str,
    commit: str,
) -> list[GraphValidationIssue]:
    """Validate import graph integrity.

    Returns
    -------
    list[GraphValidationIssue]
        Validation issues for import graph integrity.
    """
    issues: list[GraphValidationIssue] = []

    try:
        edges = q__graph__import_graph_edges
        modules = q__graph__import_modules
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
            issues.append(
                GraphValidationIssue(
                    graph_name="import_graph",
                    issue="missing_source_modules",
                    detail=f"Found {orphan_src} import edges with missing source modules",
                    severity="error",
                    entity_id="import_graph_edges",
                    metadata={"orphan_count": orphan_src},
                )
            )

    except DuckDBError as exc:
        log.debug("validation: Could not validate import graph: %s", exc)

    return issues


@tag_helper(domain="graphs")
def _validate_cfg_integrity(
    q__graph__cfg_edges: ir.Table,
    q__graph__cfg_blocks: ir.Table,
    _repo: str,
    _commit: str,
) -> list[GraphValidationIssue]:
    """Validate CFG integrity.

    Returns
    -------
    list[GraphValidationIssue]
        Validation issues for CFG integrity.
    """
    issues: list[GraphValidationIssue] = []

    try:
        edges = q__graph__cfg_edges
        blocks = q__graph__cfg_blocks

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
            issues.append(
                GraphValidationIssue(
                    graph_name="cfg",
                    issue="orphan_cfg_edges",
                    detail=f"Found {orphan_edges} CFG edges with missing source blocks",
                    severity="error",
                    entity_id="cfg_edges",
                    metadata={"orphan_count": orphan_edges},
                )
            )

    except DuckDBError as exc:
        log.debug("validation: Could not validate CFG: %s", exc)

    return issues


# ---------------------------------------------------------------------------
# goids target - compute and materialize
# ---------------------------------------------------------------------------


def _coerce_goids_output(output: ToolStepOutput) -> GoidsToolOutput:
    if isinstance(output, GoidsToolOutput):
        return output
    return GoidsToolOutput(result=output.result)


@tag_tool(domain="graphs", target=GOIDS_TARGET_NAME)
def t__goids__run(
    env: BuildEnv,
    catalog: DagCatalog,
    goids__run_inputs: GoidsRunInputs,
) -> GoidsToolOutput:
    """Execute GOID extraction on repository modules.

    Returns
    -------
    GoidsToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=GOIDS_TARGET_NAME,
    )

    def _execute() -> GoidsToolOutput:
        if goids__run_inputs.modules_record.status == "skipped":
            return GoidsToolOutput(
                result=ExecutionResult.skip(
                    goids__run_inputs.modules_record.error or "Upstream modules target skipped"
                )
            )
        if goids__run_inputs.modules_record.status != "succeeded":
            return GoidsToolOutput(
                result=ExecutionResult.failed(
                    f"Upstream modules target failed: {goids__run_inputs.modules_record.error}"
                )
            )

        source_root = goids__run_inputs.source_root
        if source_root is None:
            return GoidsToolOutput(result=ExecutionResult.failed("GOID source root not resolved"))

        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = load_target_options(
            env,
            target_name=GOIDS_TARGET_NAME,
            options_type=GoidBuilderOptions,
        )

        tracked_files = filter_paths(
            _get_tracked_files(goids__run_inputs.modules, repo, commit),
            scope_paths=opts.scope_paths,
            include_tests=opts.include_tests,
        )

        if not tracked_files:
            log.info("goids: No tracked files found, skipping")
            return GoidsToolOutput(
                result=ExecutionResult.ok(
                    table_counts={
                        GOIDS_GOIDS_TABLE_KEY: 0,
                        GOIDS_CROSSWALK_TABLE_KEY: 0,
                    }
                )
            )

        now = datetime.now(UTC)
        all_goid_rows: list[GoidRow] = []
        all_crosswalk_rows: list[GoidCrosswalkRow] = []

        for rel_path in tracked_files:
            goid_rows, crosswalk_rows = _extract_entities_from_file(
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
            all_goid_rows.extend(goid_rows)
            all_crosswalk_rows.extend(crosswalk_rows)

        log.info(
            "goids: Extracted %d GOIDs and %d crosswalk entries from %d files",
            len(all_goid_rows),
            len(all_crosswalk_rows),
            len(tracked_files),
        )

        goid_rows = tuple(row.to_tuple() for row in all_goid_rows)
        crosswalk_rows = tuple(row.to_tuple() for row in all_crosswalk_rows)
        return GoidsToolOutput(
            result=ExecutionResult.ok(
                table_counts={
                    GOIDS_GOIDS_TABLE_KEY: len(goid_rows),
                    GOIDS_CROSSWALK_TABLE_KEY: len(crosswalk_rows),
                }
            ),
            goid_rows=goid_rows,
            crosswalk_rows=crosswalk_rows,
        )

    return _coerce_goids_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=GOIDS_TARGET_NAME)
def t__goids__ingest(
    t__goids__run: GoidsToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package GOID rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__goids__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "goids skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "goids failed",
                warnings=result.warnings,
            )
        )

    payload = {
        GOIDS_GOIDS_TABLE_KEY: t__goids__run.goid_rows,
        GOIDS_CROSSWALK_TABLE_KEY: t__goids__run.crosswalk_rows,
    }
    table_counts = {
        GOIDS_GOIDS_TABLE_KEY: len(t__goids__run.goid_rows),
        GOIDS_CROSSWALK_TABLE_KEY: len(t__goids__run.crosswalk_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(context=GOIDS_SAVE_CONTEXT, spec=TableSaveSpec(table_key=GOIDS_GOIDS_TABLE_KEY))
@tag_compute(domain="graphs", target=GOIDS_TARGET_NAME, target_="goids__goids_rows")
def goids__goids_rows(
    t__goids__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.goids.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__goids__ingest.result.skipped or not t__goids__ingest.result.success:
        return None
    payload = t__goids__ingest.payload
    if payload is None:
        msg = "Missing goids payload"
        raise ValueError(msg)
    rows = payload.get(GOIDS_GOIDS_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {GOIDS_GOIDS_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(context=GOIDS_SAVE_CONTEXT, spec=TableSaveSpec(table_key=GOIDS_CROSSWALK_TABLE_KEY))
@tag_compute(domain="graphs", target=GOIDS_TARGET_NAME, target_="goids__crosswalk_rows")
def goids__crosswalk_rows(
    t__goids__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for core.goid_crosswalk.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__goids__ingest.result.skipped or not t__goids__ingest.result.success:
        return None
    payload = t__goids__ingest.payload
    if payload is None:
        msg = "Missing goids payload"
        raise ValueError(msg)
    rows = payload.get(GOIDS_CROSSWALK_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {GOIDS_CROSSWALK_TABLE_KEY}"
        raise ValueError(msg)
    return rows


goids__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=GOIDS_TARGET_NAME,
    table_keys=GOIDS_TABLE_KEYS,
)


@tag_helper(domain="graphs", target=GOIDS_TARGET_NAME)
def goids__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for GOIDs.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=GOIDS_TARGET_NAME,
    )


@codeintel_target(domain="graphs", target=GOIDS_TARGET_NAME)
def t__goids(
    goids__finalize_context: ToolFinalizeContext,
    t__goids__run: GoidsToolOutput,
    t__goids__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    goids__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Resolve GOIDs and build crosswalks.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=goids__finalize_context,
        tool_step=t__goids__run,
        ingest_step=t__goids__ingest,
        artifact_materializations=None,
        table_materializations=goids__table_materializations,
    )


# ---------------------------------------------------------------------------
# symbol_uses target - compute and materialize
# ---------------------------------------------------------------------------


def _coerce_symbol_uses_output(output: ToolStepOutput) -> SymbolUsesToolOutput:
    if isinstance(output, SymbolUsesToolOutput):
        return output
    return SymbolUsesToolOutput(result=output.result)


@tag_tool(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def t__symbol_uses__run(
    env: BuildEnv,
    catalog: DagCatalog,
    symbol_uses__run_inputs: SymbolUsesRunInputs,
) -> SymbolUsesToolOutput:
    """Execute symbol use extraction from SCIP data.

    Returns
    -------
    SymbolUsesToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=SYMBOL_USES_TARGET_NAME,
    )

    def _execute() -> SymbolUsesToolOutput:
        for name, record in [
            ("scip", symbol_uses__run_inputs.scip_record),
            ("modules", symbol_uses__run_inputs.modules_record),
            ("goids", symbol_uses__run_inputs.goids_record),
        ]:
            if record.status == "skipped":
                return SymbolUsesToolOutput(
                    result=ExecutionResult.skip(record.error or f"Upstream {name} target skipped")
                )
            if record.status != "succeeded":
                return SymbolUsesToolOutput(
                    result=ExecutionResult.failed(f"Upstream {name} target failed: {record.error}")
                )

        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = load_target_options(
            env,
            target_name=SYMBOL_USES_TARGET_NAME,
            options_type=SymbolUsesOptions,
        )

        occurrences = _load_symbol_occurrences(
            symbol_uses__run_inputs.inputs.scip_occurrences,
            repo,
            commit,
        )

        if not occurrences:
            log.info("symbol_uses: No SCIP occurrences found, skipping")
            return SymbolUsesToolOutput(
                result=ExecutionResult.ok(table_counts={SYMBOL_USE_EDGES_TABLE_KEY: 0})
            )

        occurrences = _filter_symbol_occurrences(occurrences, options=opts)

        module_map = _load_module_map(symbol_uses__run_inputs.inputs.modules, repo, commit)
        path_to_goid = _load_path_to_goid_map(symbol_uses__run_inputs.inputs.goids, repo, commit)

        def_map = symbols_compute.build_def_map(occurrences)
        edges = symbols_compute.build_use_edges(
            occurrences,
            def_map=def_map,
            module_by_path=module_map,
        )

        enriched_edges = _enrich_edges_with_goids(edges, path_to_goid)
        rows = symbols_compute.edges_to_rows(enriched_edges)
        edge_rows = tuple(row.to_tuple() for row in rows)
        return SymbolUsesToolOutput(
            result=ExecutionResult.ok(table_counts={SYMBOL_USE_EDGES_TABLE_KEY: len(edge_rows)}),
            edge_rows=edge_rows,
        )

    return _coerce_symbol_uses_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def t__symbol_uses__ingest(
    t__symbol_uses__run: SymbolUsesToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package symbol use rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__symbol_uses__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "symbol_uses skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "symbol_uses failed",
                warnings=result.warnings,
            )
        )

    payload = {SYMBOL_USE_EDGES_TABLE_KEY: t__symbol_uses__run.edge_rows}
    table_counts = {SYMBOL_USE_EDGES_TABLE_KEY: len(t__symbol_uses__run.edge_rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=SYMBOL_USES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SYMBOL_USE_EDGES_TABLE_KEY),
)
@tag_compute(domain="graphs", target=SYMBOL_USES_TARGET_NAME, target_="symbol_uses__rows")
def symbol_uses__rows(
    t__symbol_uses__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.symbol_use_edges.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__symbol_uses__ingest.result.skipped or not t__symbol_uses__ingest.result.success:
        return None
    payload = t__symbol_uses__ingest.payload
    if payload is None:
        msg = "Missing symbol_uses payload"
        raise ValueError(msg)
    rows = payload.get(SYMBOL_USE_EDGES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {SYMBOL_USE_EDGES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


symbol_uses__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=SYMBOL_USES_TARGET_NAME,
    table_keys=SYMBOL_USES_TABLE_KEYS,
)


@tag_helper(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def symbol_uses__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for symbol uses.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=SYMBOL_USES_TARGET_NAME,
    )


@codeintel_target(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def t__symbol_uses(
    symbol_uses__finalize_context: ToolFinalizeContext,
    t__symbol_uses__run: SymbolUsesToolOutput,
    t__symbol_uses__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    symbol_uses__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Extract symbol definition-to-use edges.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=symbol_uses__finalize_context,
        tool_step=t__symbol_uses__run,
        ingest_step=t__symbol_uses__ingest,
        artifact_materializations=None,
        table_materializations=symbol_uses__table_materializations,
    )


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
    dtype = edges.caller_goid_h128.type().copy(nullable=True)
    caller_funcs: ir.Table = edges.select(
        caller_function_goid_h128=edges.caller_goid_h128.cast(dtype),
    ).distinct()
    callee_funcs: ir.Table = (
        filter_by(edges, not_null(edges.callee_goid_h128))
        .select(
            function_goid_h128=edges.callee_goid_h128.cast(dtype),
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


@save_ibis_table(
    context=CALL_GRAPH_VIEWS_SAVE_CONTEXT,
    spec=IbisTableSaveSpec(table_key=CALL_GRAPH_VIEWS_FUNCTION_CALL_COUNTS),
)
@pipe_input(
    step(_call_graph_views_filter_edges, env=source("env")).named(
        pipeline_node_name(
            CALL_GRAPH_FUNCTION_CALL_COUNTS_NAMESPACE,
            _call_graph_views_filter_edges.__name__,
        ),
    ),
    step(_call_graph_views_build_call_count_stats).named(
        pipeline_node_name(
            CALL_GRAPH_FUNCTION_CALL_COUNTS_NAMESPACE,
            _call_graph_views_build_call_count_stats.__name__,
        ),
    ),
    step(_call_graph_views_finalize_call_counts, env=source("env")).named(
        pipeline_node_name(
            CALL_GRAPH_FUNCTION_CALL_COUNTS_NAMESPACE,
            _call_graph_views_finalize_call_counts.__name__,
        ),
    ),
    namespace=None,
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


@save_ibis_table(
    context=CALL_GRAPH_VIEWS_SAVE_CONTEXT,
    spec=IbisTableSaveSpec(table_key=CALL_GRAPH_VIEWS_CALL_DEPTH_STATS),
)
@pipe_input(
    step(_call_graph_views_filter_edges, env=source("env")).named(
        pipeline_node_name(
            CALL_GRAPH_DEPTH_STATS_NAMESPACE,
            _call_graph_views_filter_edges.__name__,
        ),
    ),
    step(_call_graph_views_prepare_depth_tables).named(
        pipeline_node_name(
            CALL_GRAPH_DEPTH_STATS_NAMESPACE,
            _call_graph_views_prepare_depth_tables.__name__,
        ),
    ),
    step(_call_graph_views_finalize_depth_stats, env=source("env")).named(
        pipeline_node_name(
            CALL_GRAPH_DEPTH_STATS_NAMESPACE,
            _call_graph_views_finalize_depth_stats.__name__,
        ),
    ),
    namespace=None,
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


call_graph_views__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=CALL_GRAPH_VIEWS_TARGET_NAME,
    table_keys=CALL_GRAPH_VIEWS_TABLE_KEYS,
)


@codeintel_target(domain="graphs", target=CALL_GRAPH_VIEWS_TARGET_NAME)
def t__call_graph_views(
    env: BuildEnv,
    catalog: DagCatalog,
    call_graph_views__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize derived views over the call graph for analytics.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    LOG.info("Materializing call_graph_views to DuckDB")
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=CALL_GRAPH_VIEWS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=call_graph_views__table_materializations,
    )


# ---------------------------------------------------------------------------
# graph_metrics target - compute and materialize
# ---------------------------------------------------------------------------


@tag_helper(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def graph_metrics__gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for graph metrics.

    Returns
    -------
    StorageGateway
        Storage gateway from the build environment.
    """
    return env.gateway




def _coerce_graph_metrics_output(output: ToolStepOutput) -> GraphMetricsToolOutput:
    if isinstance(output, GraphMetricsToolOutput):
        return output
    return GraphMetricsToolOutput(result=output.result)


@tag_tool(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def t__graph_metrics__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__call_graph: TargetRunRecord,
    graph_metrics__gateway: StorageGateway,
) -> GraphMetricsToolOutput:
    """Compute graph metrics rows from call graph data.

    Returns
    -------
    GraphMetricsToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_METRICS_TARGET_NAME,
    )

    def _execute() -> GraphMetricsToolOutput:
        if t__call_graph.status == "skipped":
            return GraphMetricsToolOutput(
                result=ExecutionResult.skip(
                    t__call_graph.error or "Upstream call_graph target skipped"
                )
            )
        if t__call_graph.status != "succeeded":
            return GraphMetricsToolOutput(
                result=ExecutionResult.failed(
                    f"Upstream call_graph target failed: {t__call_graph.error}"
                )
            )

        try:
            log.info(
                "graph_metrics: Computing metrics for repo=%s commit=%s",
                env.snapshot.repo,
                env.snapshot.commit,
            )

            runtime_options = replace(
                load_graph_runtime_options(env, target_name=GRAPH_METRICS_TARGET_NAME),
                snapshot=env.snapshot,
                backend=GraphBackendConfig(use_gpu=True, backend="auto", strict=False),
            )
            runtime = build_graph_runtime(graph_metrics__gateway, runtime_options)
            filters = build_graph_metric_filters(graph_metrics__gateway, env.snapshot)

            metrics_rows = build_graph_metrics_rows(
                graph_metrics__gateway,
                env.snapshot,
                options=load_target_options(
                    env,
                    target_name=GRAPH_METRICS_TARGET_NAME,
                    options_type=GraphMetricsOptions,
                ),
                deps=GraphMetricsDeps(
                    catalog_provider=None,
                    runtime=runtime,
                    filters=filters,
                ),
            )
            functions_ext_rows = build_graph_metrics_functions_ext_rows(
                graph_metrics__gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                runtime=runtime,
                filters=filters,
            )
            modules_ext_rows = build_graph_metrics_modules_ext_rows(
                graph_metrics__gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                runtime=runtime,
                filters=filters,
            )
            graph_stats_rows = build_graph_stats_rows(
                graph_metrics__gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                runtime=runtime,
            )

            log.info(
                "graph_metrics: rows built functions=%d modules=%d functions_ext=%d "
                "modules_ext=%d stats=%d",
                len(metrics_rows.function_rows),
                len(metrics_rows.module_rows),
                len(functions_ext_rows),
                len(modules_ext_rows),
                len(graph_stats_rows),
            )

            functions_rows = tuple(
                row_to_tuple(GRAPH_METRICS_FUNCTIONS_TABLE_KEY, row)
                for row in metrics_rows.function_rows
            )
            modules_rows = tuple(
                row_to_tuple(GRAPH_METRICS_MODULES_TABLE_KEY, row)
                for row in metrics_rows.module_rows
            )
            functions_ext_rows_tuple = tuple(
                row_to_tuple(GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY, row)
                for row in functions_ext_rows
            )
            modules_ext_rows_tuple = tuple(
                row_to_tuple(GRAPH_METRICS_MODULES_EXT_TABLE_KEY, row) for row in modules_ext_rows
            )
            graph_stats_rows_tuple = tuple(graph_stats_rows)
            table_counts = {
                GRAPH_METRICS_FUNCTIONS_TABLE_KEY: len(functions_rows),
                GRAPH_METRICS_MODULES_TABLE_KEY: len(modules_rows),
                GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY: len(functions_ext_rows_tuple),
                GRAPH_METRICS_MODULES_EXT_TABLE_KEY: len(modules_ext_rows_tuple),
                GRAPH_STATS_TABLE_KEY: len(graph_stats_rows_tuple),
            }
            return GraphMetricsToolOutput(
                result=ExecutionResult.ok(table_counts=table_counts),
                functions_rows=functions_rows,
                modules_rows=modules_rows,
                functions_ext_rows=functions_ext_rows_tuple,
                modules_ext_rows=modules_ext_rows_tuple,
                graph_stats_rows=graph_stats_rows_tuple,
            )

        except (RuntimeError, ValueError, OSError) as exc:
            log.exception("Graph metrics computation failed")
            return GraphMetricsToolOutput(result=ExecutionResult.failed(str(exc)))

    return _coerce_graph_metrics_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def t__graph_metrics__ingest(
    t__graph_metrics__run: GraphMetricsToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package graph metrics rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__graph_metrics__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "graph_metrics skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "graph_metrics failed",
                warnings=result.warnings,
            )
        )

    payload = {
        GRAPH_METRICS_FUNCTIONS_TABLE_KEY: t__graph_metrics__run.functions_rows,
        GRAPH_METRICS_MODULES_TABLE_KEY: t__graph_metrics__run.modules_rows,
        GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY: t__graph_metrics__run.functions_ext_rows,
        GRAPH_METRICS_MODULES_EXT_TABLE_KEY: t__graph_metrics__run.modules_ext_rows,
        GRAPH_STATS_TABLE_KEY: t__graph_metrics__run.graph_stats_rows,
    }
    table_counts = {
        GRAPH_METRICS_FUNCTIONS_TABLE_KEY: len(t__graph_metrics__run.functions_rows),
        GRAPH_METRICS_MODULES_TABLE_KEY: len(t__graph_metrics__run.modules_rows),
        GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY: len(t__graph_metrics__run.functions_ext_rows),
        GRAPH_METRICS_MODULES_EXT_TABLE_KEY: len(t__graph_metrics__run.modules_ext_rows),
        GRAPH_STATS_TABLE_KEY: len(t__graph_metrics__run.graph_stats_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


def _graph_metrics_rows_payload(
    ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    table_key: str,
) -> tuple[tuple[object, ...], ...] | None:
    if ingest.result.skipped or not ingest.result.success:
        return None
    payload = ingest.payload
    if payload is None:
        msg = "Missing graph_metrics payload"
        raise ValueError(msg)
    rows = payload.get(table_key)
    if rows is None:
        msg = f"Missing rows for {table_key}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=GRAPH_METRICS_FUNCTIONS_TABLE_KEY),
)
@tag_compute(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
    target_="graph_metrics__functions_rows",
)
def graph_metrics__functions_rows(
    t__graph_metrics__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract function graph metrics rows for materialization.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    """
    return _graph_metrics_rows_payload(
        t__graph_metrics__ingest,
        GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
    )


@save_rows(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=GRAPH_METRICS_MODULES_TABLE_KEY),
)
@tag_compute(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
    target_="graph_metrics__modules_rows",
)
def graph_metrics__modules_rows(
    t__graph_metrics__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract module graph metrics rows for materialization.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    """
    return _graph_metrics_rows_payload(
        t__graph_metrics__ingest,
        GRAPH_METRICS_MODULES_TABLE_KEY,
    )


@save_rows(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY),
)
@tag_compute(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
    target_="graph_metrics__functions_ext_rows",
)
def graph_metrics__functions_ext_rows(
    t__graph_metrics__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract function extended graph metric rows for materialization.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    """
    return _graph_metrics_rows_payload(
        t__graph_metrics__ingest,
        GRAPH_METRICS_FUNCTIONS_EXT_TABLE_KEY,
    )


@save_rows(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=GRAPH_METRICS_MODULES_EXT_TABLE_KEY),
)
@tag_compute(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
    target_="graph_metrics__modules_ext_rows",
)
def graph_metrics__modules_ext_rows(
    t__graph_metrics__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract module extended graph metric rows for materialization.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    """
    return _graph_metrics_rows_payload(
        t__graph_metrics__ingest,
        GRAPH_METRICS_MODULES_EXT_TABLE_KEY,
    )


@save_rows(
    context=GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=GRAPH_STATS_TABLE_KEY),
)
@tag_compute(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
    target_="graph_metrics__stats_rows",
)
def graph_metrics__stats_rows(
    t__graph_metrics__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract graph stats rows for materialization.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    """
    return _graph_metrics_rows_payload(
        t__graph_metrics__ingest,
        GRAPH_STATS_TABLE_KEY,
    )


graph_metrics__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=GRAPH_METRICS_TARGET_NAME,
    table_keys=GRAPH_METRICS_TABLE_KEYS,
)


@tag_helper(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def graph_metrics__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for graph metrics.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_METRICS_TARGET_NAME,
    )


@codeintel_target(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
def t__graph_metrics(
    graph_metrics__finalize_context: ToolFinalizeContext,
    t__graph_metrics__run: GraphMetricsToolOutput,
    t__graph_metrics__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Compute graph topology metrics for functions and modules.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=graph_metrics__finalize_context,
        tool_step=t__graph_metrics__run,
        ingest_step=t__graph_metrics__ingest,
        artifact_materializations=None,
        table_materializations=graph_metrics__table_materializations,
    )


# ---------------------------------------------------------------------------
# graph_validation target - compute and materialize
# ---------------------------------------------------------------------------




@tag_helper(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def graph_validation__run_inputs(
    graph_validation__inputs: GraphValidationInputs,
    t__call_graph: TargetRunRecord,
    t__import_graph: TargetRunRecord,
    t__cfg: TargetRunRecord,
) -> GraphValidationRunInputs:
    """Bundle inputs for graph validation.

    Returns
    -------
    GraphValidationRunInputs
        Return value.

    """
    return GraphValidationRunInputs(
        inputs=graph_validation__inputs,
        call_graph_record=t__call_graph,
        import_graph_record=t__import_graph,
        cfg_record=t__cfg,
    )


def _coerce_graph_validation_output(output: ToolStepOutput) -> GraphValidationToolOutput:
    if isinstance(output, GraphValidationToolOutput):
        return output
    return GraphValidationToolOutput(result=output.result)


def _graph_validation_upstream_result(
    run_inputs: GraphValidationRunInputs,
) -> ExecutionResult | None:
    deps = [
        ("call_graph", run_inputs.call_graph_record),
        ("import_graph", run_inputs.import_graph_record),
        ("cfg", run_inputs.cfg_record),
    ]
    for name, record in deps:
        if record.status == "skipped":
            return ExecutionResult.skip(record.error or f"Upstream {name} target skipped")
        if record.status != "succeeded":
            return ExecutionResult.failed(f"Upstream {name} target failed: {record.error}")
    return None


def _collect_graph_validation_issues(
    run_inputs: GraphValidationRunInputs,
    *,
    repo: str,
    commit: str,
) -> list[GraphValidationIssue]:
    issues: list[GraphValidationIssue] = []
    issues.extend(
        _validate_call_graph_integrity(
            run_inputs.inputs.call_graph_edges,
            run_inputs.inputs.call_graph_nodes,
            repo,
            commit,
        )
    )
    issues.extend(
        _validate_import_graph_integrity(
            run_inputs.inputs.import_graph_edges,
            run_inputs.inputs.import_modules,
            repo,
            commit,
        )
    )
    issues.extend(
        _validate_cfg_integrity(
            run_inputs.inputs.cfg_edges,
            run_inputs.inputs.cfg_blocks,
            repo,
            commit,
        )
    )
    return issues


@tag_tool(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def t__graph_validation__run(
    env: BuildEnv,
    catalog: DagCatalog,
    graph_validation__run_inputs: GraphValidationRunInputs,
) -> GraphValidationToolOutput:
    """Run validation checks on all graph data.

    Returns
    -------
    GraphValidationToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_VALIDATION_TARGET_NAME,
    )

    def _execute() -> GraphValidationToolOutput:
        upstream_result = _graph_validation_upstream_result(graph_validation__run_inputs)
        if upstream_result is not None:
            return GraphValidationToolOutput(result=upstream_result)

        repo = env.snapshot.repo
        commit = env.snapshot.commit

        issues = _collect_graph_validation_issues(
            graph_validation__run_inputs,
            repo=repo,
            commit=commit,
        )

        for issue in issues:
            log.warning("graph_validation: %s", issue.detail)

        reporter = GraphValidationReporter(repo=repo, commit=commit)
        for issue in issues:
            extras: dict[str, object] = {}
            if issue.severity is not None:
                extras["severity"] = issue.severity
            if issue.rel_path is not None:
                extras["rel_path"] = issue.rel_path
            if issue.metadata is not None:
                extras["metadata"] = issue.metadata
            reporter.record(
                graph_name=issue.graph_name,
                issue=issue.issue,
                detail=issue.detail,
                entity_id=issue.entity_id,
                extras=extras or None,
            )

        rows = reporter.to_rows() if issues else ()
        warnings: tuple[str, ...] = ()
        if issues:
            warnings = (f"graph_validation: {len(issues)} issue(s) found",)

        log.info(
            "graph_validation: Completed with %d issues found for repo=%s commit=%s",
            len(issues),
            repo,
            commit,
        )

        return GraphValidationToolOutput(
            result=ExecutionResult.ok(
                table_counts={GRAPH_VALIDATION_TABLE_KEY: len(rows)},
                warnings=warnings,
            ),
            rows=rows,
        )

    return _coerce_graph_validation_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def t__graph_validation__ingest(
    t__graph_validation__run: GraphValidationToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package graph validation rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__graph_validation__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "graph_validation skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "graph_validation failed",
                warnings=result.warnings,
            )
        )

    payload = {GRAPH_VALIDATION_TABLE_KEY: t__graph_validation__run.rows}
    table_counts = {GRAPH_VALIDATION_TABLE_KEY: len(t__graph_validation__run.rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=GRAPH_VALIDATION_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=GRAPH_VALIDATION_TABLE_KEY),
)
@tag_compute(
    domain="graphs",
    target=GRAPH_VALIDATION_TARGET_NAME,
    target_="graph_validation__rows",
)
def graph_validation__rows(
    t__graph_validation__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.graph_validation.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__graph_validation__ingest.result.skipped or not t__graph_validation__ingest.result.success:
        return None
    payload = t__graph_validation__ingest.payload
    if payload is None:
        msg = "Missing graph_validation payload"
        raise ValueError(msg)
    rows = payload.get(GRAPH_VALIDATION_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {GRAPH_VALIDATION_TABLE_KEY}"
        raise ValueError(msg)
    return rows


graph_validation__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=GRAPH_VALIDATION_TARGET_NAME,
    table_keys=GRAPH_VALIDATION_TABLE_KEYS,
)


@tag_helper(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def graph_validation__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for graph validation.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_VALIDATION_TARGET_NAME,
    )


@codeintel_target(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
def t__graph_validation(
    graph_validation__finalize_context: ToolFinalizeContext,
    t__graph_validation__run: GraphValidationToolOutput,
    t__graph_validation__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    graph_validation__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Run graph integrity validation checks and persist findings.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=graph_validation__finalize_context,
        tool_step=t__graph_validation__run,
        ingest_step=t__graph_validation__ingest,
        artifact_materializations=None,
        table_materializations=graph_validation__table_materializations,
    )


__all__ = [
    "GoidExtractionInputs",
    "GraphMetricsToolOutput",
    "GraphValidationIssue",
    "GraphValidationToolOutput",
    "call_graph_depth_stats",
    "call_graph_function_call_counts",
    "graph_metrics__functions_ext_rows",
    "graph_metrics__functions_rows",
    "graph_metrics__modules_ext_rows",
    "graph_metrics__modules_rows",
    "graph_metrics__stats_rows",
    "t__call_graph_views",
    "t__goids",
    "t__goids__ingest",
    "t__goids__run",
    "t__graph_metrics",
    "t__graph_metrics__ingest",
    "t__graph_metrics__run",
    "t__graph_validation",
    "t__graph_validation__ingest",
    "t__graph_validation__run",
    "t__symbol_uses",
    "t__symbol_uses__ingest",
    "t__symbol_uses__run",
]
