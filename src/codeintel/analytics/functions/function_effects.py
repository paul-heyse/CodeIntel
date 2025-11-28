"""Classify function side effects and purity."""

from __future__ import annotations

import ast
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import networkx as nx

from codeintel.analytics.ast_utils import call_name, snippet_from_lines
from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.evidence import EvidenceCollector
from codeintel.analytics.function_ast_cache import (
    FunctionAst,
    FunctionAstLoadRequest,
    load_function_asts,
)
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graph_service import normalize_decimal_id
from codeintel.config import FunctionEffectsStepConfig
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.ingestion.common import run_batch
from codeintel.storage.gateway import DuckDBConnection, DuckDBError, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

log = logging.getLogger(__name__)


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
class _EffectInputs:
    gateway: StorageGateway
    cfg: FunctionEffectsStepConfig
    catalog: FunctionCatalogProvider
    context: AnalyticsContext | None
    runtime: GraphRuntime


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


def compute_function_effects(
    gateway: StorageGateway,
    cfg: FunctionEffectsStepConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
    context: AnalyticsContext | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> None:
    """
    Populate `analytics.function_effects` for the target repo/commit.

    Parameters
    ----------
    gateway:
        Storage gateway for DuckDB.
    cfg:
        Function effects configuration.
    catalog_provider:
        Optional function catalog to reuse across steps.
    context:
        Optional shared analytics context to reuse catalog, ASTs, and call graph.
    runtime:
        Optional shared graph runtime to reuse an existing engine/backend wiring.
    """
    ensure_schema(gateway.con, "analytics.function_effects")

    catalog = (
        context.catalog
        if context is not None
        else catalog_provider
        or FunctionCatalogService.from_db(gateway, repo=cfg.repo, commit=cfg.commit)
    )
    active_runtime = resolve_graph_runtime(
        gateway,
        cfg.snapshot,
        runtime,
        context=context,
    )

    inputs = _EffectInputs(
        gateway=gateway,
        cfg=cfg,
        catalog=catalog,
        context=context,
        runtime=active_runtime,
    )
    rows = _build_effect_rows(inputs=inputs, now=datetime.now(tz=UTC))

    run_batch(
        gateway,
        "analytics.function_effects",
        rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    log.info("function_effects populated: %d rows for %s@%s", len(rows), cfg.repo, cfg.commit)


def _build_effect_rows(
    inputs: _EffectInputs,
    now: datetime,
) -> list[tuple[object, ...]]:
    if inputs.context is not None:
        ast_by_goid = inputs.context.function_ast_map
        missing = inputs.context.missing_function_goids
    else:
        ast_by_goid, missing = load_function_asts(
            inputs.gateway,
            FunctionAstLoadRequest(
                repo=inputs.cfg.repo,
                commit=inputs.cfg.commit,
                repo_root=inputs.cfg.repo_root,
                catalog_provider=inputs.catalog,
            ),
        )
    all_goids = {span.goid for span in inputs.catalog.catalog().function_spans}
    analyses: dict[int, EffectAnalysis] = {
        goid: _analyze_function(info, inputs.cfg) for goid, info in ast_by_goid.items()
    }
    direct_flags: dict[int, bool] = {
        goid: analysis.direct_effectful for goid, analysis in analyses.items()
    }

    call_graph = inputs.runtime.ensure_call_graph()
    transitive_hits = _compute_transitive_effects(
        call_graph,
        direct_flags,
        max_depth=inputs.cfg.max_call_depth,
    )
    unresolved_calls = _unresolved_call_counts(
        inputs.gateway.con, inputs.cfg.repo, inputs.cfg.commit
    )

    rows: list[tuple[object, ...]] = []
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
            (
                inputs.cfg.repo,
                inputs.cfg.commit,
                goid,
                is_pure,
                analysis.uses_io,
                analysis.touches_db,
                analysis.uses_time,
                analysis.uses_randomness,
                analysis.modifies_globals,
                analysis.modifies_closure,
                analysis.spawns_threads_or_tasks,
                bool(transitive_targets),
                purity_confidence,
                _effects_payload(analysis, transitive_targets),
                now,
            )
        )
    return rows


def _compute_transitive_effects(
    call_graph: nx.DiGraph, direct_flags: dict[int, bool], *, max_depth: int
) -> dict[int, set[int]]:
    transitive: dict[int, set[int]] = {}
    for node in call_graph.nodes:
        if direct_flags.get(node):
            continue
        hits: set[int] = set()
        visited: set[int] = {node}
        queue: deque[tuple[int, int]] = deque([(node, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for succ in call_graph.successors(current):
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


def _unresolved_call_counts(con: DuckDBConnection, repo: str, commit: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    try:
        rows: Iterable[tuple[int, int]] = con.execute(
            """
            SELECT caller_goid_h128::BIGINT, COUNT(*) AS unresolved_count
            FROM graph.call_graph_edges
            WHERE repo = ? AND commit = ?
              AND callee_goid_h128 IS NULL
            GROUP BY caller_goid_h128
            """,
            [repo, commit],
        ).fetchall()
    except DuckDBError:
        return counts
    for raw_goid, count in rows:
        goid = normalize_decimal_id(raw_goid)
        if goid is None:
            continue
        counts[goid] = int(count)
    return counts


def _purity_confidence(*, parsed: bool, unresolved_call_count: int) -> float:
    if not parsed:
        return 0.0
    penalty = min(unresolved_call_count * 0.1, 0.7)
    return max(0.0, 1.0 - penalty)


def _analyze_function(func: FunctionAst, cfg: FunctionEffectsStepConfig) -> EffectAnalysis:
    visitor = _EffectVisitor(cfg, rel_path=func.rel_path, lines=func.lines)
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

    def __init__(self, cfg: FunctionEffectsStepConfig, *, rel_path: str, lines: list[str]) -> None:
        self.cfg = cfg
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

        if _matches_api(name, self.cfg.io_apis):
            self.uses_io = True
            self._record_call("io_calls", name, lineno, end_lineno)
        if _matches_api(name, self.cfg.db_apis):
            self.touches_db = True
            self._record_call("db_calls", name, lineno, end_lineno)
        if _matches_api(name, self.cfg.time_apis):
            self.uses_time = True
            self._record_call("time_calls", name, lineno, end_lineno)
        if _matches_api(name, self.cfg.random_apis):
            self.uses_randomness = True
            self._record_call("random_calls", name, lineno, end_lineno)
        if _matches_api(name, self.cfg.threading_apis):
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
