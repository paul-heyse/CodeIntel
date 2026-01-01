"""Classify function side effects and purity."""

from __future__ import annotations

import ast
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import networkx as nx
import polars as pl

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_int

if TYPE_CHECKING:
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)


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
    call_graph_edges: pl.DataFrame | None = None
    call_graph_nodes: pl.DataFrame | None = None


@dataclass(frozen=True)
class _EffectInputs:
    snapshot: SnapshotRef
    options: FunctionEffectsOptions
    catalog: FunctionCatalogProvider
    ast_map: dict[int, FunctionAst] | None = None
    missing_goids: set[int] | None = None
    call_graph_edges: pl.DataFrame | None = None
    call_graph_nodes: pl.DataFrame | None = None


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
            for target in sorted(transitive_targets)
        ]
    return payload


def build_function_effects_rows(
    snapshot: SnapshotRef,
    *,
    options: FunctionEffectsOptions | None = None,
    inputs: FunctionEffectsInputs | None = None,
) -> list[dict[str, object]]:
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
    list[dict[str, object]]
        Effect rows ready for persistence.

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
        call_graph_edges=input_opts.call_graph_edges,
        call_graph_nodes=input_opts.call_graph_nodes,
    )
    rows = _build_effect_rows(inputs=effect_inputs, now=datetime.now(tz=UTC))
    log.info(
        "function_effects populated: %d rows for %s@%s",
        len(rows),
        snapshot.repo,
        snapshot.commit,
    )
    return rows


def _build_effect_rows(
    inputs: _EffectInputs,
    now: datetime,
) -> list[dict[str, object]]:
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
            )
        )
    if missing:
        log.warning(
            "Missing AST for %d functions while computing effects: %s",
            len(missing),
            sorted(missing),
        )
    all_goids = {span.goid for span in inputs.catalog.catalog().function_spans}
    analyses: dict[int, EffectAnalysis] = {
        goid: _analyze_function(info, inputs.options) for goid, info in ast_by_goid.items()
    }
    direct_flags: dict[int, bool] = {
        goid: analysis.direct_effectful for goid, analysis in analyses.items()
    }

    call_graph = _call_graph_from_frames(
        inputs.call_graph_edges,
        inputs.call_graph_nodes,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
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
    )
    if unresolved_calls:
        log.warning(
            "Unresolved call edges detected for %d functions while computing effects: %s",
            len(unresolved_calls),
            sorted(unresolved_calls),
        )

    rows: list[dict[str, object]] = []
    for goid in all_goids:
        analysis = analyses.get(
            goid,
            EffectAnalysis(
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
                }
                if goid in missing
                else {},
            ),
        )
        transitive_targets = transitive_hits.get(goid)
        is_pure = not direct_flags.get(goid) and not transitive_targets and goid not in missing
        purity_confidence = _purity_confidence(
            parsed=goid not in missing,
            unresolved_call_count=unresolved_calls.get(goid, 0),
        )

        rows.append(
            {
                "repo": inputs.snapshot.repo,
                "commit": inputs.snapshot.commit,
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
                "effects_json": json.dumps(_effects_payload(analysis, transitive_targets)),
                "created_at": now,
            }
        )
    return rows


def _compute_transitive_effects(
    call_graph: nx.DiGraph, direct_flags: dict[int, bool], *, max_depth: int
) -> dict[int, set[int]]:
    transitive: dict[int, set[int]] = {}
    for node_value in call_graph.nodes:
        node = cast("int", node_value)
        if direct_flags.get(node):
            continue
        hits: set[int] = set()
        visited: set[int] = {node}
        queue: deque[tuple[int, int]] = deque([(node, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for succ_value in call_graph.successors(current):
                succ = cast("int", succ_value)
                if succ in visited:
                    continue
                visited.add(succ)
                if direct_flags.get(succ):
                    hits.add(succ)
                queue.append((succ, depth + 1))
            if hits:
                break
        if hits:
            transitive[node] = hits
    return transitive


def _unresolved_call_counts_from_frame(
    edges_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    if edges_frame is None or edges_frame.is_empty():
        return counts
    frame = edges_frame
    if "repo" in frame.columns:
        frame = frame.filter(pl.col("repo") == repo)
    if "commit" in frame.columns:
        frame = frame.filter(pl.col("commit") == commit)
    if "callee_goid_h128" not in frame.columns:
        return counts
    unresolved = frame.filter(
        pl.col("callee_goid_h128").is_null() | (pl.col("callee_goid_h128") == -1)
    )
    if unresolved.is_empty() or "caller_goid_h128" not in unresolved.columns:
        return counts
    grouped = unresolved.group_by("caller_goid_h128").len()
    for row in grouped.iter_rows(named=True):
        goid = normalize_decimal_id(row.get("caller_goid_h128"))
        if goid is None:
            continue
        counts[goid] = coerce_int(row.get("len"), ctx="unresolved_count")
    return counts


def _edge_weight_value(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _filter_edges_frame(
    edges_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> pl.DataFrame | None:
    if edges_frame is None or edges_frame.is_empty():
        return None
    frame = edges_frame
    if "repo" in frame.columns:
        frame = frame.filter(pl.col("repo") == repo)
    if "commit" in frame.columns:
        frame = frame.filter(pl.col("commit") == commit)
    return frame


def _add_call_edges(graph: nx.DiGraph, frame: pl.DataFrame) -> None:
    for row in frame.iter_rows(named=True):
        caller = normalize_decimal_id(row.get("caller_goid_h128"))
        callee = normalize_decimal_id(row.get("callee_goid_h128"))
        if caller is None or callee is None:
            continue
        if graph.has_edge(caller, callee):
            attrs = graph[caller][callee]
            attrs["weight"] = _edge_weight_value(attrs.get("weight")) + 1
        else:
            graph.add_edge(caller, callee, weight=1)


def _add_call_nodes(graph: nx.DiGraph, nodes_frame: pl.DataFrame | None) -> None:
    if nodes_frame is None or nodes_frame.is_empty():
        return
    for row in nodes_frame.iter_rows(named=True):
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is None:
            continue
        if goid in graph:
            continue
        attrs: dict[str, object] = {}
        kind = row.get("kind")
        if kind is not None:
            attrs["kind"] = str(kind)
        graph.add_node(goid, **attrs)


def _call_graph_from_frames(
    edges_frame: pl.DataFrame | None,
    nodes_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    frame = _filter_edges_frame(edges_frame, repo=repo, commit=commit)
    if frame is None:
        return graph
    _add_call_edges(graph, frame)
    _add_call_nodes(graph, nodes_frame)
    return graph


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
