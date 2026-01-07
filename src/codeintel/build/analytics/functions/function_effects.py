"""Classify function side effects and purity."""

from __future__ import annotations

import ast
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import networkx as nx
import pyarrow as pa

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.build.graphs.builders import EdgeWeightPolicy, build_call_graph_from_rows
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.data_models.ids import normalize_decimal_id

if TYPE_CHECKING:
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)
_EDGE_WEIGHT_POLICY = EdgeWeightPolicy()


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
    call_graph_edges: pa.Table | None = None
    call_graph_nodes: pa.Table | None = None


@dataclass(frozen=True)
class _EffectInputs:
    snapshot: SnapshotRef
    options: FunctionEffectsOptions
    catalog: FunctionCatalogProvider
    ast_map: dict[int, FunctionAst] | None = None
    missing_goids: set[int] | None = None
    call_graph_edges: pa.Table | None = None
    call_graph_nodes: pa.Table | None = None


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
                "effects_json": _effects_payload(analysis, transitive_targets),
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
    edges_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    if edges_frame is None or edges_frame.num_rows == 0:
        return counts
    if "callee_goid_h128" not in edges_frame.column_names:
        return counts
    has_repo = "repo" in edges_frame.column_names
    has_commit = "commit" in edges_frame.column_names
    has_caller = "caller_goid_h128" in edges_frame.column_names
    if not has_caller:
        return counts
    for row in iter_rows(edges_frame):
        if has_repo and row.get("repo") != repo:
            continue
        if has_commit and row.get("commit") != commit:
            continue
        callee = row.get("callee_goid_h128")
        if callee is not None and callee != -1:
            continue
        goid = normalize_decimal_id(row.get("caller_goid_h128"))
        if goid is None:
            continue
        counts[goid] = counts.get(goid, 0) + 1
    return counts


def _filter_edges_rows(
    edges_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    if edges_frame is None or edges_frame.num_rows == 0:
        return []
    scope = SnapshotScope(repo=repo, commit=commit)
    filtered = scope.filter_arrow_table(edges_frame, require_columns=True)
    if filtered.num_rows == 0:
        return []
    return list(iter_rows(filtered))


def _call_graph_from_frames(
    edges_frame: pa.Table | None,
    nodes_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> nx.DiGraph:
    rows = _filter_edges_rows(edges_frame, repo=repo, commit=commit)
    if not rows:
        return nx.DiGraph()
    node_rows = iter_rows(nodes_frame) if nodes_frame is not None else None
    return build_call_graph_from_rows(rows, node_rows, policy=_EDGE_WEIGHT_POLICY)


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
