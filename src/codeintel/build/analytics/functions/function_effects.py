"""Classify function side effects and purity."""

from __future__ import annotations

import ast
import logging
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.compute.row_builders import buffer_for_table
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.build.analytics.utilities.list_semantics import normalize_list_semantics
from codeintel.build.analytics.utilities.snapshot import SnapshotContext, snapshot_plan
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    bfs_distances_by_id,
    ensure_directed_store,
)
from codeintel.build.graphs.rx.build_from_edges import (
    BuildStoreOptions,
    EdgeBuildSpec,
    build_store_from_edge_tuples,
)
from codeintel.build.graphs.rx.policies import DEFAULT_NUMERIC_POLICY, weight_policy_for_kind
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
)
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges_for_join
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.query_results import coerce_int

if TYPE_CHECKING:
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)

CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"
CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"


def _default_io_apis() -> dict[str, list[str]]:
    return {
        "builtins": ["open", "print"],
        "pathlib": ["Path.open", "Path.write_text", "Path.write_bytes"],
        "logging": ["debug", "info", "warning", "error", "exception", "critical", "log"],
        "requests": ["get", "post", "put", "delete", "patch", "head", "options"],
        "httpx": ["get", "post", "put", "delete", "patch", "head", "options"],
    }


def _default_db_apis() -> dict[str, list[str]]:
    return {
        "sqlite3": ["connect"],
        "psycopg": ["connect"],
        "psycopg2": ["connect"],
        "asyncpg": ["connect", "create_pool"],
        "sqlalchemy": ["create_engine", "Session"],
    }


def _default_time_apis() -> dict[str, list[str]]:
    return {
        "time": ["sleep", "time"],
        "asyncio": ["sleep"],
        "datetime": ["datetime.now", "datetime.utcnow", "date.today"],
    }


def _default_random_apis() -> dict[str, list[str]]:
    return {
        "random": ["random", "randint", "choice", "randrange", "shuffle"],
        "secrets": ["token_hex", "token_urlsafe"],
        "uuid": ["uuid4", "uuid1"],
    }


def _default_threading_apis() -> dict[str, list[str]]:
    return {
        "threading": ["Thread", "Timer"],
        "multiprocessing": ["Process", "Pool"],
        "asyncio": ["create_task", "ensure_future", "gather"],
        "concurrent.futures": ["ThreadPoolExecutor", "ProcessPoolExecutor"],
    }


@dataclass(frozen=True)
class FunctionEffectsOptions:
    """Configuration options for function effects computation.

    Parameters
    ----------
    max_call_depth
        Maximum depth to trace transitive effects.
    require_all_callees_pure
        Whether all callees must be pure for function to be pure.
    io_apis
        Mapping of modules to I/O API function names.
    db_apis
        Mapping of modules to database API function names.
    time_apis
        Mapping of modules to time-related API function names.
    random_apis
        Mapping of modules to randomness API function names.
    threading_apis
        Mapping of modules to threading/async API function names.
    """

    max_call_depth: int = 3
    require_all_callees_pure: bool = True
    io_apis: dict[str, list[str]] = field(default_factory=_default_io_apis)
    db_apis: dict[str, list[str]] = field(default_factory=_default_db_apis)
    time_apis: dict[str, list[str]] = field(default_factory=_default_time_apis)
    random_apis: dict[str, list[str]] = field(default_factory=_default_random_apis)
    threading_apis: dict[str, list[str]] = field(default_factory=_default_threading_apis)


@dataclass(frozen=True)
class EffectAnalysis:
    """Direct effect flags derived from a function body."""

    uses_io: bool
    touches_db: bool
    uses_time: bool
    uses_randomness: bool
    modifies_globals: bool
    modifies_closure: bool
    spawns_threads_or_tasks: bool
    evidence: dict[str, list[dict[str, object]]]

    @property
    def direct_effectful(self) -> bool:
        """Return True when any direct side effect was detected."""
        return any(
            (
                self.uses_io,
                self.touches_db,
                self.uses_time,
                self.uses_randomness,
                self.modifies_globals,
                self.modifies_closure,
                self.spawns_threads_or_tasks,
            )
        )


@dataclass(frozen=True)
class FunctionEffectsInputs:
    """Optional inputs for function effects computation."""

    catalog_provider: FunctionCatalogProvider | None = None
    ast_map: dict[int, FunctionAst] | None = None
    missing_goids: set[int] | None = None
    worklist: pa.Table | pa.RecordBatchReader | None = None
    call_graph_edges: pa.Table | None = None
    call_graph_nodes: pa.Table | None = None
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


@dataclass(frozen=True)
class _EffectInputs:
    snapshot: SnapshotRef
    options: FunctionEffectsOptions
    catalog: FunctionCatalogProvider
    ast_map: dict[int, FunctionAst] | None = None
    missing_goids: set[int] | None = None
    worklist: pa.Table | pa.RecordBatchReader | None = None
    call_graph_edges: pa.Table | None = None
    call_graph_nodes: pa.Table | None = None
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


@dataclass(frozen=True)
class _EffectRowContext:
    snapshot: SnapshotRef
    now: datetime
    analyses: dict[int, EffectAnalysis]
    direct_flags: dict[int, bool]
    missing: set[int]
    transitive_hits: dict[int, set[int]]
    unresolved_calls: dict[int, int]


def _effects_payload(
    analysis: EffectAnalysis, transitive_targets: set[int] | None
) -> dict[str, list[dict[str, object]]]:
    """
    Build evidence payload including transitive effect lineage when present.

    Parameters
    ----------
    analysis :
        Effect analysis for a function.
    transitive_targets : set[int] | None
        Optional downstream functions carrying effects.

    Returns
    -------
    dict[str, object]
        Evidence payload augmented with transitive lineage when available.
    """
    payload = dict(analysis.evidence)
    if transitive_targets:
        payload["transitive_effects_via"] = [
            {
                "path": "",
                "lineno": None,
                "end_lineno": None,
                "snippet": "",
                "details": {"goid": target},
                "tags": ["transitive_effect"],
            }
            for target in normalize_list_semantics(transitive_targets)
        ]
    return payload


def build_function_effects_rows(
    snapshot: SnapshotRef,
    *,
    options: FunctionEffectsOptions | None = None,
    inputs: FunctionEffectsInputs | None = None,
) -> ColumnarRowBuffer:
    """
    Build function effects rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/ArrowDatasetSaver.

    Parameters
    ----------
    snapshot
        Snapshot reference (repo, commit, repo_root).
    options
        Configuration options for effects detection.
    inputs
        Optional inputs containing catalog, runtime, AST map, and missing GOIDs.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing effect rows ready for persistence.

    Raises
    ------
    ValueError
        If the catalog provider is missing from the inputs.
    """
    opts = options or FunctionEffectsOptions()
    input_opts = inputs or FunctionEffectsInputs()
    if input_opts.catalog_provider is None:
        msg = "FunctionEffectsInputs.catalog_provider is required."
        raise ValueError(msg)
    catalog = input_opts.catalog_provider

    effect_inputs = _EffectInputs(
        snapshot=snapshot,
        options=opts,
        catalog=catalog,
        ast_map=input_opts.ast_map,
        missing_goids=input_opts.missing_goids,
        worklist=input_opts.worklist,
        call_graph_edges=input_opts.call_graph_edges,
        call_graph_nodes=input_opts.call_graph_nodes,
        ctx=input_opts.ctx,
    )
    rows = _build_effect_rows(inputs=effect_inputs, now=datetime.now(tz=UTC))
    log.info(
        "function_effects populated: %d rows for %s@%s",
        rows.row_count,
        snapshot.repo,
        snapshot.commit,
    )
    return rows


def _resolve_effect_asts(inputs: _EffectInputs) -> tuple[dict[int, FunctionAst], set[int]]:
    if inputs.ast_map is not None:
        ast_by_goid = inputs.ast_map
        missing = inputs.missing_goids or set()
    else:
        ast_by_goid, missing = load_function_asts(
            FunctionAstLoadRequest(
                repo=inputs.snapshot.repo,
                commit=inputs.snapshot.commit,
                repo_root=inputs.snapshot.repo_root,
                catalog_provider=inputs.catalog,
                worklist=inputs.worklist,
            )
        )
    if missing:
        log.warning(
            "Missing AST for %d functions while computing effects: %s",
            len(missing),
            sorted(missing),
        )
    return ast_by_goid, missing


def _analysis_maps(
    ast_by_goid: dict[int, FunctionAst],
    options: FunctionEffectsOptions,
) -> tuple[dict[int, EffectAnalysis], dict[int, bool]]:
    analyses: dict[int, EffectAnalysis] = {
        goid: _analyze_function(info, options) for goid, info in ast_by_goid.items()
    }
    direct_flags: dict[int, bool] = {
        goid: analysis.direct_effectful for goid, analysis in analyses.items()
    }
    return analyses, direct_flags


def _default_analysis(goid: int, missing: set[int]) -> EffectAnalysis:
    if goid not in missing:
        return EffectAnalysis(
            uses_io=False,
            touches_db=False,
            uses_time=False,
            uses_randomness=False,
            modifies_globals=False,
            modifies_closure=False,
            spawns_threads_or_tasks=False,
            evidence={},
        )
    return EffectAnalysis(
        uses_io=False,
        touches_db=False,
        uses_time=False,
        uses_randomness=False,
        modifies_globals=False,
        modifies_closure=False,
        spawns_threads_or_tasks=False,
        evidence={
            "errors": [
                {
                    "path": "",
                    "lineno": None,
                    "end_lineno": None,
                    "snippet": "",
                    "details": {"kind": "missing_ast"},
                    "tags": ["error"],
                }
            ]
        },
    )


def _build_effect_row(goid: int, context: _EffectRowContext) -> dict[str, object]:
    analysis = context.analyses.get(goid) or _default_analysis(goid, context.missing)
    transitive_targets = context.transitive_hits.get(goid)
    is_pure = (
        not context.direct_flags.get(goid, False)
        and not transitive_targets
        and goid not in context.missing
    )
    purity_confidence = _purity_confidence(
        parsed=goid not in context.missing,
        unresolved_call_count=context.unresolved_calls.get(goid, 0),
    )
    return {
        "repo": context.snapshot.repo,
        "commit": context.snapshot.commit,
        "function_goid_h128": goid,
        "is_pure": is_pure,
        "uses_io": analysis.uses_io,
        "touches_db": analysis.touches_db,
        "uses_time": analysis.uses_time,
        "uses_randomness": analysis.uses_randomness,
        "modifies_globals": analysis.modifies_globals,
        "modifies_closure": analysis.modifies_closure,
        "spawns_threads_or_tasks": analysis.spawns_threads_or_tasks,
        "has_transitive_effects": bool(transitive_targets),
        "purity_confidence": purity_confidence,
        "extras": {"effects": _effects_payload(analysis, transitive_targets)},
        "created_at": context.now,
    }


def _build_effect_rows(
    inputs: _EffectInputs,
    now: datetime,
) -> ColumnarRowBuffer:
    ast_by_goid, missing = _resolve_effect_asts(inputs)
    analyses, direct_flags = _analysis_maps(ast_by_goid, inputs.options)
    all_goids = {span.goid for span in inputs.catalog.catalog().function_spans}
    call_graph = _call_graph_from_frames(
        inputs.call_graph_edges,
        inputs.call_graph_nodes,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        ctx=inputs.ctx,
    )
    transitive_hits = _compute_transitive_effects(
        call_graph,
        direct_flags,
        max_depth=inputs.options.max_call_depth,
    )
    unresolved_calls = _unresolved_call_counts_from_frame(
        inputs.call_graph_edges,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        ctx=inputs.ctx,
    )
    if unresolved_calls:
        log.warning(
            "Unresolved call edges detected for %d functions while computing effects: %s",
            len(unresolved_calls),
            sorted(unresolved_calls),
        )

    row_context = _EffectRowContext(
        snapshot=inputs.snapshot,
        now=now,
        analyses=analyses,
        direct_flags=direct_flags,
        missing=missing,
        transitive_hits=transitive_hits,
        unresolved_calls=unresolved_calls,
    )
    buffer = buffer_for_table("analytics.function_effects")
    for goid in all_goids:
        buffer.append(_build_effect_row(goid, row_context))
    return buffer


def _compute_transitive_effects(
    call_graph: GraphInput,
    direct_flags: dict[int, bool],
    *,
    max_depth: int,
) -> dict[int, set[int]]:
    transitive: dict[int, set[int]] = {}
    store = ensure_directed_store(call_graph)
    for node_id in store.node_ids():
        normalized = normalize_decimal_id(node_id)
        if normalized is None:
            continue
        if direct_flags.get(normalized):
            continue
        hits = _transitive_hits_for_node(
            store,
            direct_flags,
            node_id,
            max_depth,
        )
        if hits:
            transitive[normalized] = hits
    return transitive


def _transitive_hits_for_node(
    store: RxGraphStore,
    direct_flags: dict[int, bool],
    node_id: object,
    max_depth: int,
) -> set[int]:
    """Compute transitive effect hits for a single node.

    Returns
    -------
    set[int]
        GOIDs for reachable nodes with direct effects.
    """
    distances = bfs_distances_by_id(store, node_id, max_depth=max_depth)
    hits: set[int] = set()
    for target_id, depth in distances.items():
        if target_id == node_id or depth <= 0 or depth > max_depth:
            continue
        normalized = normalize_decimal_id(target_id)
        if normalized is None:
            continue
        if direct_flags.get(normalized):
            hits.add(normalized)
    return hits


def _unresolved_call_counts_from_frame(
    edges_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    if edges_frame is None or edges_frame.num_rows == 0:
        return counts
    aggregated = _unresolved_call_rowset(edges_frame, repo=repo, commit=commit, ctx=ctx)
    for row in iter_rows(aggregated, ("caller_goid_h128", "unresolved_call_count")):
        goid = normalize_decimal_id(row.get("caller_goid_h128"))
        if goid is None:
            continue
        count_value = row.get("unresolved_call_count")
        count = (
            coerce_int(count_value, ctx="unresolved_call_count") if count_value is not None else 0
        )
        if count:
            counts[goid] = count
    return counts


def _call_graph_from_frames(
    edges_frame: pa.Table | None,
    nodes_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> GraphInput:
    policy = weight_policy_for_kind(GraphKind.CALL_GRAPH)
    edge_rows, node_ids = _call_graph_edges_from_frame(
        edges_frame,
        repo=repo,
        commit=commit,
        ctx=ctx,
    )
    node_attrs, node_ids_from_nodes = _call_graph_nodes_from_frame(
        nodes_frame,
        repo=repo,
        commit=commit,
        ctx=ctx,
    )
    node_ids.update(node_ids_from_nodes)
    if not edge_rows and not node_ids:
        return RxGraphStore.directed(
            weight_policy=policy,
            numeric_policy=DEFAULT_NUMERIC_POLICY,
        )
    spec = EdgeBuildSpec(
        directed=True,
        weight_policy=policy,
        numeric_policy=DEFAULT_NUMERIC_POLICY,
    )
    options = BuildStoreOptions(
        stable_nodes=True,
        node_ids=node_ids or None,
        node_attrs=node_attrs or None,
        node_hint=len(node_ids) if node_ids else None,
        edge_hint=len(edge_rows),
    )
    return build_store_from_edge_tuples(edge_rows, spec=spec, options=options)


def _call_graph_edges_from_frame(
    edges_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> tuple[list[tuple[int, int, float]], set[Hashable]]:
    edge_rows: list[tuple[int, int, float]] = []
    node_ids: set[Hashable] = set()
    if edges_frame is None or edges_frame.num_rows == 0:
        return edge_rows, node_ids
    rowset = _call_graph_rowset(edges_frame, repo=repo, commit=commit, ctx=ctx)
    exploded = explode_edges_for_join(
        rowset,
        spec=ExplodeSpec(
            src_col="caller_goid_h128",
            dst_list_col="callee_goid_h128",
            null_list_policy="empty",
            null_child_policy="drop",
            error_context_cols=("caller_goid_h128",),
        ),
        allowed_columns=("caller_goid_h128", "callee_goid_h128"),
    )
    for caller_raw, callee_raw in iter_tuples(
        exploded.good.to_reader(),
        columns=("caller_goid_h128", "callee_goid_h128"),
    ):
        caller = normalize_decimal_id(caller_raw)
        callee = normalize_decimal_id(callee_raw)
        if caller is None or callee is None:
            continue
        caller_id = int(caller)
        callee_id = int(callee)
        node_ids.add(caller_id)
        node_ids.add(callee_id)
        edge_rows.append((caller_id, callee_id, 1.0))
    return edge_rows, node_ids


def _call_graph_nodes_from_frame(
    nodes_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> tuple[dict[Hashable, dict[str, object]], set[Hashable]]:
    node_attrs: dict[Hashable, dict[str, object]] = {}
    node_ids: set[Hashable] = set()
    if nodes_frame is None or "goid_h128" not in nodes_frame.column_names:
        return node_attrs, node_ids
    plan = snapshot_plan(
        nodes_frame,
        columns=("goid_h128", "kind"),
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CALL_GRAPH_NODES_TABLE_KEY,
        ),
    )
    plan = plan.filter(E.is_valid("goid_h128"))
    table = _materialize_plan(plan, table_key=CALL_GRAPH_NODES_TABLE_KEY, ctx=ctx)
    for row in iter_rows(table, ("goid_h128", "kind")):
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is None:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        node_id = int(goid)
        node_attrs[node_id] = attrs
        node_ids.add(node_id)
    return node_attrs, node_ids


def _call_graph_rowset(
    edges_frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    required = ("caller_goid_h128", "callee_goid_h128")
    missing = [name for name in required if name not in edges_frame.column_names]
    if missing:
        msg = f"Missing call graph edge columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(
        edges_frame,
        columns=required,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CALL_GRAPH_EDGES_TABLE_KEY,
        ),
    )
    plan = plan.filter(
        E.and_(
            E.is_valid("caller_goid_h128"),
            E.is_valid("callee_goid_h128"),
            E.field("callee_goid_h128") != E.scalar(-1),
        )
    )
    filtered = _materialize_plan(plan, table_key=CALL_GRAPH_EDGES_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("caller_goid_h128",),
            aggregates=[("callee_goid_h128", "list", None, "callee_goid_h128")],
            pre_sort_keys=(
                ("caller_goid_h128", "ascending"),
                ("callee_goid_h128", "ascending"),
            ),
        ),
        ctx=resolve_columnar_context(ctx),
    )


def _unresolved_call_rowset(
    edges_frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    required = ("caller_goid_h128", "callee_goid_h128")
    missing = [name for name in required if name not in edges_frame.column_names]
    if missing:
        msg = f"Missing call graph edge columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(
        edges_frame,
        columns=required,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CALL_GRAPH_EDGES_TABLE_KEY,
        ),
    )
    plan = plan.filter(
        E.and_(
            E.is_valid("caller_goid_h128"),
            E.or_(
                E.is_null("callee_goid_h128"),
                E.field("callee_goid_h128") == E.scalar(-1),
            ),
        )
    )
    filtered = _materialize_plan(plan, table_key=CALL_GRAPH_EDGES_TABLE_KEY, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("caller_goid_h128",),
            aggregates=[("caller_goid_h128", "count", None, "unresolved_call_count")],
        ),
        ctx=resolve_columnar_context(ctx),
    )


def _purity_confidence(*, parsed: bool, unresolved_call_count: int) -> float:
    if not parsed:
        return 0.0
    penalty = min(unresolved_call_count * 0.1, 0.7)
    return max(0.0, 1.0 - penalty)


def _analyze_function(func: FunctionAst, options: FunctionEffectsOptions) -> EffectAnalysis:
    """Analyze a function AST for side effects.

    Parameters
    ----------
    func
        Parsed function AST.
    options
        Detection options controlling which APIs are flagged.

    Returns
    -------
    EffectAnalysis
        Detected side-effect flags and supporting evidence.
    """
    visitor = _EffectVisitor(options, rel_path=func.rel_path, lines=func.lines)
    visitor.visit(func.node)
    if visitor.modifies_globals:
        visitor.record_scope_change("globals", func.start_line)
    if visitor.modifies_closure:
        visitor.record_scope_change("nonlocals", func.start_line)
    return EffectAnalysis(
        uses_io=visitor.uses_io,
        touches_db=visitor.touches_db,
        uses_time=visitor.uses_time,
        uses_randomness=visitor.uses_randomness,
        modifies_globals=visitor.modifies_globals,
        modifies_closure=visitor.modifies_closure,
        spawns_threads_or_tasks=visitor.spawns_threads_or_tasks,
        evidence=visitor.evidence_payload,
    )


class _EffectVisitor(ast.NodeVisitor):
    """Lightweight AST visitor to spot side-effectful operations."""

    def __init__(self, options: FunctionEffectsOptions, *, rel_path: str, lines: list[str]) -> None:
        self.options = options
        self.uses_io = False
        self.touches_db = False
        self.uses_time = False
        self.uses_randomness = False
        self.modifies_globals = False
        self.modifies_closure = False
        self.spawns_threads_or_tasks = False
        self._rel_path = rel_path
        self._lines = lines
        self._evidence: dict[str, EvidenceCollector] = {}

    def visit_Global(self, node: ast.Global) -> None:
        self.modifies_globals = True
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.modifies_closure = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = call_name(node.func)
        if name is None:
            self.generic_visit(node)
            return
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", lineno)

        if _matches_api(name, self.options.io_apis):
            self.uses_io = True
            self._record_call("io_calls", name, lineno, end_lineno)
        if _matches_api(name, self.options.db_apis):
            self.touches_db = True
            self._record_call("db_calls", name, lineno, end_lineno)
        if _matches_api(name, self.options.time_apis):
            self.uses_time = True
            self._record_call("time_calls", name, lineno, end_lineno)
        if _matches_api(name, self.options.random_apis):
            self.uses_randomness = True
            self._record_call("random_calls", name, lineno, end_lineno)
        if _matches_api(name, self.options.threading_apis):
            self.spawns_threads_or_tasks = True
            self._record_call("thread_calls", name, lineno, end_lineno)

        self.generic_visit(node)

    @property
    def evidence_payload(self) -> dict[str, list[dict[str, object]]]:
        """Return JSON-serializable evidence grouped by category."""
        return {
            kind: collector.to_dicts()
            for kind, collector in self._evidence.items()
            if collector.samples
        }

    def _record_call(
        self, kind: str, name: str, lineno: int | None, end_lineno: int | None
    ) -> None:
        collector = self._evidence.setdefault(kind, EvidenceCollector())
        snippet = snippet_from_lines(self._lines, lineno, end_lineno)
        collector.add_sample(
            path=self._rel_path,
            line_span=(lineno, end_lineno),
            snippet=snippet,
            details={"call": name, "category": kind},
            tags=(kind,),
        )

    def record_scope_change(self, kind: str, lineno: int | None) -> None:
        collector = self._evidence.setdefault(kind, EvidenceCollector())
        snippet = snippet_from_lines(self._lines, lineno, lineno)
        collector.add_sample(
            path=self._rel_path,
            line_span=(lineno, lineno),
            snippet=snippet,
            details={"category": kind},
            tags=(kind,),
        )


def _matches_api(target: str, patterns: dict[str, list[str]]) -> bool:
    simple = target.rsplit(".", maxsplit=1)[-1]
    for module, funcs in patterns.items():
        if (target == module or target.startswith(f"{module}.")) and (
            not funcs or simple in funcs or target in funcs
        ):
            return True
        if simple in funcs:
            return True
    return False


def _materialize_plan(
    plan: Plan,
    *,
    table_key: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    resolved_ctx = resolve_columnar_context(ctx)
    result = run_pipeline(
        plan=ExecutionPlan.from_plan(plan),
        finalize=finalize_spec_for_table(table_key, mode="tolerant"),
        options=PipelineRunOptions(ctx=resolved_ctx),
    )
    return result.good
