"""Track config key usage at the function level with call-chain context."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import networkx as nx

from codeintel.analytics.ast_utils import call_name, snippet_from_lines
from codeintel.analytics.context import (
    AnalyticsContext,
    AnalyticsContextConfig,
    ensure_analytics_context,
)
from codeintel.analytics.evidence import EvidenceCollector
from codeintel.config.models import ConfigDataFlowConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.analytics.function_ast_cache import FunctionAst

log = logging.getLogger(__name__)

LOGGER_METHODS = {
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "critical",
    "log",
}
ENV_HELPERS = {"os.getenv", "environ.get", "decouple.config", "settings.get_env"}


@dataclass
class ConfigUsageResult:
    """Accumulated usage kinds and evidence for a config key."""

    kinds: set[str] = field(default_factory=set)
    evidence: dict[str, EvidenceCollector] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigFlowArtifacts:
    """Shared datasets used during config data flow analysis."""

    entrypoints: set[int]
    call_graph: nx.DiGraph
    ast_by_goid: dict[int, FunctionAst]
    refs_by_path: dict[str, list[tuple[str, str]]]


class ConfigUsageVisitor(ast.NodeVisitor):
    """Detect how a specific config key is used inside a function."""

    def __init__(
        self,
        *,
        config_key: str,
        config_path: str,
        rel_path: str,
        lines: list[str],
        max_examples: int,
    ) -> None:
        self.config_key = config_key
        self.config_path = config_path
        self.lines = lines
        self.max_examples = max_examples
        self.result = ConfigUsageResult()
        self._in_condition = False
        self._rel_path = rel_path

    def visit_If(self, node: ast.If) -> None:
        """Track conditional branches involving config keys."""
        self._visit_test(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        """Handle while conditions referencing config keys."""
        self._visit_test(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Inspect ternary expressions for config reads."""
        self._visit_test(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect writes to config-like mappings."""
        if self._targets_config_key(node.targets):
            self._record("write", node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Capture augmented assignments on config mappings."""
        if self._targets_config_key([node.target]):
            self._record("write", node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Record config reads and logging that mention the tracked key."""
        target_name = call_name(node.func) or ""
        if (
            target_name.endswith(("getenv", "environ.get", "get_env")) or target_name in ENV_HELPERS
        ) and self._first_arg_matches(node.args):
            self._record(self._kind_for_context("read"), node.lineno)
        if ("config" in target_name or "settings" in target_name) and self._first_arg_matches(
            node.args
        ):
            self._record(self._kind_for_context("read"), node.lineno)
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in {"get", "setdefault"} and self._first_arg_matches(node.args):
                kind = "write" if attr_name == "setdefault" else self._kind_for_context("read")
                self._record(kind, node.lineno)
            if attr_name == "update" and self._args_reference_key(node.args):
                self._record("write", node.lineno)
        if self._is_logger_call(target_name) and self._args_reference_key(node.args):
            self._record("logging", node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Handle direct subscript reads of config mappings."""
        if self._subscript_matches(node):
            self._record(self._kind_for_context("read"), node.lineno)
        self.generic_visit(node)

    def _visit_test(self, test: ast.AST) -> None:
        previous = self._in_condition
        self._in_condition = True
        self.visit(test)
        self._in_condition = previous

    def _kind_for_context(self, default: str) -> str:
        return "conditional_branch" if self._in_condition else default

    def _matches_key(self, node: ast.AST | None) -> bool:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (str, int, float, bool)):
                return str(value) == self.config_key or value == self.config_key
        return False

    def _first_arg_matches(self, args: Sequence[ast.AST]) -> bool:
        if not args:
            return False
        return self._matches_key(args[0])

    def _subscript_matches(self, node: ast.Subscript) -> bool:
        if isinstance(node.slice, ast.Tuple):
            return any(self._matches_key(elt) for elt in node.slice.elts)
        return self._matches_key(getattr(node, "slice", None))

    def _targets_config_key(self, targets: Sequence[ast.expr]) -> bool:
        for target in targets:
            if isinstance(target, ast.Subscript) and self._subscript_matches(target):
                return True
        return False

    def _args_reference_key(self, args: Sequence[ast.AST]) -> bool:
        for arg in args:
            if self._matches_key(arg):
                return True
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and self.config_key in arg.value
            ):
                return True
            if isinstance(arg, ast.Subscript) and self._subscript_matches(arg):
                return True
            if isinstance(arg, ast.Dict):
                for key in arg.keys:
                    if self._matches_key(key):
                        return True
        return False

    @staticmethod
    def _is_logger_call(target_name: str) -> bool:
        return any(target_name.endswith(method) for method in LOGGER_METHODS)

    def _record(self, kind: str, lineno: int | None) -> None:
        if lineno is None:
            lineno = 0
        self.result.kinds.add(kind)
        collector = self.result.evidence.setdefault(
            kind, EvidenceCollector(max_samples=self.max_examples)
        )
        snippet = snippet_from_lines(self.lines, lineno, lineno)
        collector.add_sample(
            path=self._rel_path,
            line_span=(lineno, lineno),
            snippet=snippet,
            details={
                "config_key": self.config_key,
                "config_path": self.config_path,
                "usage_kind": kind,
            },
        )


def _coerce_paths(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        parsed = raw
    if not isinstance(parsed, (list, tuple)):
        return []
    return [normalize_rel_path(path) for path in parsed]


def _config_references(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[str, list[tuple[str, str]]]:
    rows = con.execute(
        """
        SELECT config_path, key, reference_paths
        FROM analytics.config_values
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    refs: dict[str, list[tuple[str, str]]] = {}
    for config_path, key, reference_paths in rows:
        for rel_path in _coerce_paths(reference_paths):
            refs.setdefault(rel_path, []).append((str(key), str(config_path)))
    return refs


def _entrypoints(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> set[int]:
    rows = con.execute(
        """
        SELECT handler_goid_h128
        FROM analytics.entrypoints
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    return {int(row[0]) for row in rows if row[0] is not None}


def _call_chains(
    graph: nx.DiGraph,
    entrypoints: set[int],
    target: int,
    *,
    max_paths: int,
    max_length: int,
) -> list[list[int]]:
    if target not in graph:
        graph.add_node(target)
    paths: list[list[int]] = []
    for entry in entrypoints:
        if entry not in graph:
            continue
        try:
            for path in nx.all_simple_paths(graph, entry, target, cutoff=max_length):
                paths.append([int(node) for node in path])
                if len(paths) >= max_paths:
                    return paths
        except nx.NetworkXNoPath:
            continue
    if not paths:
        paths.append([target])
    return paths[:max_paths]


def _call_chain_id(
    repo: str, commit: str, config_key: str, usage_kind: str, path: list[int]
) -> str:
    raw = f"{repo}:{commit}:{config_key}:{usage_kind}:{'->'.join(str(node) for node in path)}"
    return hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def compute_config_data_flow(
    gateway: StorageGateway,
    cfg: ConfigDataFlowConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    """
    Populate analytics.config_data_flow with config usage per function.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    cfg
        Config data flow analytics configuration.
    context
        Optional shared analytics context to reuse catalog, call graph, and ASTs.
    """
    con = gateway.con
    ensure_schema(con, "analytics.config_data_flow")
    con.execute(
        "DELETE FROM analytics.config_data_flow WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    refs_by_path = _config_references(con, cfg.repo, cfg.commit)
    if not refs_by_path:
        log.info(
            "No config references found for %s@%s; skipping config flow analysis",
            cfg.repo,
            cfg.commit,
        )
        return

    shared_context = ensure_analytics_context(
        gateway,
        cfg=AnalyticsContextConfig(
            repo=cfg.repo,
            commit=cfg.commit,
            repo_root=cfg.repo_root,
        ),
        context=context,
    )

    entrypoints = _entrypoints(con, cfg.repo, cfg.commit)
    call_graph = shared_context.call_graph
    ast_by_goid = shared_context.function_ast_map
    missing = shared_context.missing_function_goids
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during config data flow analysis",
            len(missing),
        )

    artifacts = ConfigFlowArtifacts(
        entrypoints=entrypoints,
        call_graph=call_graph,
        ast_by_goid=ast_by_goid,
        refs_by_path=refs_by_path,
    )
    now = datetime.now(tz=UTC)
    rows_to_insert = _build_config_flow_rows(
        artifacts=artifacts,
        cfg=cfg,
        now=now,
    )

    if rows_to_insert:
        con.executemany(
            """
            INSERT INTO analytics.config_data_flow (
                repo, commit, config_key, config_path,
                function_goid_h128, usage_kind, evidence_json,
                call_chain_id, call_chain_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
    log.info(
        "config_data_flow populated: %d rows for %s@%s",
        len(rows_to_insert),
        cfg.repo,
        cfg.commit,
    )


def _build_config_flow_rows(
    *,
    artifacts: ConfigFlowArtifacts,
    cfg: ConfigDataFlowConfig,
    now: datetime,
) -> list[tuple[object, ...]]:
    rows_to_insert: list[tuple[object, ...]] = []
    for goid, func_ast in artifacts.ast_by_goid.items():
        rel_path = normalize_rel_path(func_ast.rel_path)
        config_refs = artifacts.refs_by_path.get(rel_path, [])
        if not config_refs:
            continue
        lines = func_ast.lines
        for config_key, config_path in config_refs:
            visitor = ConfigUsageVisitor(
                config_key=config_key,
                config_path=config_path,
                rel_path=rel_path,
                lines=lines,
                max_examples=cfg.max_paths_per_usage,
            )
            visitor.visit(func_ast.node)
            if not visitor.result.kinds:
                continue
            chains = _call_chains(
                artifacts.call_graph,
                artifacts.entrypoints,
                goid,
                max_paths=cfg.max_paths_per_usage,
                max_length=cfg.max_path_length,
            )
            for usage_kind in sorted(visitor.result.kinds):
                collector = visitor.result.evidence.get(usage_kind)
                evidence = collector.to_dicts() if collector is not None else []
                for chain in chains:
                    chain_id = _call_chain_id(cfg.repo, cfg.commit, config_key, usage_kind, chain)
                    rows_to_insert.append(
                        (
                            cfg.repo,
                            cfg.commit,
                            config_key,
                            config_path,
                            goid,
                            usage_kind,
                            json.dumps(evidence) if evidence else None,
                            chain_id,
                            json.dumps(chain),
                            now,
                        )
                    )
    return rows_to_insert
