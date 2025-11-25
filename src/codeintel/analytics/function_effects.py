"""Classify function side effects and purity."""

from __future__ import annotations

import ast
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import networkx as nx

from codeintel.analytics.function_ast_cache import FunctionAst, load_function_asts
from codeintel.config.models import FunctionEffectsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.graphs.nx_views import _normalize_decimal, load_call_graph
from codeintel.ingestion.common import run_batch
from codeintel.storage.gateway import StorageGateway

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


def compute_function_effects(
    gateway: StorageGateway,
    cfg: FunctionEffectsConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
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
    """
    ensure_schema(gateway.con, "analytics.function_effects")

    catalog = catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    rows = _build_effect_rows(
        gateway=gateway,
        cfg=cfg,
        catalog=catalog,
        now=datetime.now(tz=UTC),
    )

    run_batch(
        gateway,
        "analytics.function_effects",
        rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    log.info("function_effects populated: %d rows for %s@%s", len(rows), cfg.repo, cfg.commit)


def _build_effect_rows(
    *,
    gateway: StorageGateway,
    cfg: FunctionEffectsConfig,
    catalog: FunctionCatalogProvider,
    now: datetime,
) -> list[tuple[object, ...]]:
    ast_by_goid, missing = load_function_asts(
        gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog_provider=catalog,
    )
    all_goids = {span.goid for span in catalog.catalog().function_spans}
    direct_flags: dict[int, bool] = dict.fromkeys(all_goids, False)
    analyses: dict[int, EffectAnalysis] = {}

    for goid, info in ast_by_goid.items():
        analysis = _analyze_function(info, cfg)
        analyses[goid] = analysis
        direct_flags[goid] = analysis.direct_effectful

    transitive_hits = _compute_transitive_effects(
        load_call_graph(gateway, cfg.repo, cfg.commit),
        direct_flags,
        max_depth=cfg.max_call_depth,
    )
    unresolved_calls = _unresolved_call_counts(gateway.con, cfg.repo, cfg.commit)

    rows: list[tuple[object, ...]] = []
    for goid in all_goids:
        unknown = goid in missing
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
                evidence={"errors": [{"kind": "missing_ast"}]} if unknown else {},
            ),
        )
        direct_effectful = direct_flags.get(goid, False)
        transitive = bool(transitive_hits.get(goid))
        is_pure = not direct_effectful and not transitive and not unknown
        purity_confidence = _purity_confidence(
            parsed=goid not in missing,
            unresolved_call_count=unresolved_calls.get(goid, 0),
        )

        effects_json = dict(analysis.evidence)
        if transitive_hits.get(goid):
            effects_json["transitive_effects_via"] = [
                {"goid": target} for target in sorted(transitive_hits[goid])
            ]

        rows.append(
            (
                cfg.repo,
                cfg.commit,
                goid,
                is_pure,
                analysis.uses_io,
                analysis.touches_db,
                analysis.uses_time,
                analysis.uses_randomness,
                analysis.modifies_globals,
                analysis.modifies_closure,
                analysis.spawns_threads_or_tasks,
                transitive,
                purity_confidence,
                effects_json,
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


def _unresolved_call_counts(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[int, int]:
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
    except duckdb.Error:
        return counts
    for raw_goid, count in rows:
        goid = _normalize_decimal(raw_goid)
        if goid is None:
            continue
        counts[goid] = int(count)
    return counts


def _purity_confidence(*, parsed: bool, unresolved_call_count: int) -> float:
    if not parsed:
        return 0.0
    penalty = min(unresolved_call_count * 0.1, 0.7)
    return max(0.0, 1.0 - penalty)


def _analyze_function(func: FunctionAst, cfg: FunctionEffectsConfig) -> EffectAnalysis:
    visitor = _EffectVisitor(cfg)
    visitor.visit(func.node)
    evidence = visitor.evidence
    if visitor.modifies_globals:
        evidence.setdefault("globals", []).append({"line": func.start_line})
    if visitor.modifies_closure:
        evidence.setdefault("nonlocals", []).append({"line": func.start_line})
    return EffectAnalysis(
        uses_io=visitor.uses_io,
        touches_db=visitor.touches_db,
        uses_time=visitor.uses_time,
        uses_randomness=visitor.uses_randomness,
        modifies_globals=visitor.modifies_globals,
        modifies_closure=visitor.modifies_closure,
        spawns_threads_or_tasks=visitor.spawns_threads_or_tasks,
        evidence=evidence,
    )


class _EffectVisitor(ast.NodeVisitor):
    """Lightweight AST visitor to spot side-effectful operations."""

    def __init__(self, cfg: FunctionEffectsConfig) -> None:
        self.cfg = cfg
        self.uses_io = False
        self.touches_db = False
        self.uses_time = False
        self.uses_randomness = False
        self.modifies_globals = False
        self.modifies_closure = False
        self.spawns_threads_or_tasks = False
        self.evidence: dict[str, list[dict[str, object]]] = {}

    def visit_Global(self, node: ast.Global) -> None:
        self.modifies_globals = True
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.modifies_closure = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name is None:
            self.generic_visit(node)
            return
        lineno = getattr(node, "lineno", None)

        if _matches_api(name, self.cfg.io_apis):
            self.uses_io = True
            self.evidence.setdefault("io_calls", []).append({"name": name, "line": lineno})
        if _matches_api(name, self.cfg.db_apis):
            self.touches_db = True
            self.evidence.setdefault("db_calls", []).append({"name": name, "line": lineno})
        if _matches_api(name, self.cfg.time_apis):
            self.uses_time = True
            self.evidence.setdefault("time_calls", []).append({"name": name, "line": lineno})
        if _matches_api(name, self.cfg.random_apis):
            self.uses_randomness = True
            self.evidence.setdefault("random_calls", []).append({"name": name, "line": lineno})
        if _matches_api(name, self.cfg.threading_apis):
            self.spawns_threads_or_tasks = True
            self.evidence.setdefault("thread_calls", []).append({"name": name, "line": lineno})

        self.generic_visit(node)


def _dotted_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value)
        if base:
            return f"{base}.{expr.attr}"
        return expr.attr
    if isinstance(expr, ast.Call):
        return _dotted_name(expr.func)
    return None


def _matches_api(call_name: str, patterns: dict[str, list[str]]) -> bool:
    simple = call_name.rsplit(".", maxsplit=1)[-1]
    for module, funcs in patterns.items():
        if (call_name == module or call_name.startswith(f"{module}.")) and (
            not funcs or simple in funcs or call_name in funcs
        ):
            return True
        if simple in funcs:
            return True
    return False
