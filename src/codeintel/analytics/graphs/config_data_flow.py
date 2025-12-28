"""Track config key usage at the function level with call-chain context."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import networkx as nx
from networkx.exception import NetworkXNoPath

from codeintel.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.core.hashing import sha256_short
from codeintel.core.paths import normalize_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway


CONFIG_DATA_FLOW_COLS = (
    "repo",
    "commit",
    "config_key",
    "config_path",
    "function_goid_h128",
    "usage_kind",
    "evidence",
    "call_chain_id",
    "call_chain",
    "created_at",
)

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
    return [normalize_path(path) for path in parsed]


def _config_references(
    con: DuckDBConnection, repo: str, commit: str
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


def _entrypoints(con: DuckDBConnection, repo: str, commit: str) -> set[int]:
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
                paths.append([int(str(node)) for node in path])
                if len(paths) >= max_paths:
                    return paths
        except NetworkXNoPath:
            continue
    if not paths:
        paths.append([target])
    return paths[:max_paths]


def _call_chain_id(
    repo: str, commit: str, config_key: str, usage_kind: str, path: list[int]
) -> str:
    raw = f"{repo}:{commit}:{config_key}:{usage_kind}:{'->'.join(str(node) for node in path)}"
    return sha256_short(raw, length=16, used_for_security=False)


@dataclass(frozen=True)
class ConfigDataFlowResult:
    """Result from config data flow computation.

    Attributes
    ----------
    rows
        Tuple rows for analytics.config_data_flow table, or None if skipped.
    """

    rows: tuple[tuple[object, ...], ...] | None


def compute_config_data_flow_result(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    call_graph: nx.DiGraph,
    ast_by_goid: dict[int, FunctionAst],
    missing_goids: set[int] | None = None,
) -> ConfigDataFlowResult:
    """Compute config data flow rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/DuckDBRelationSaver.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Repository and commit identifiers.
    call_graph
        Call graph for the repository snapshot.
    ast_by_goid
        Mapping of function GOID to parsed AST data.
    missing_goids
        Optional set of function GOIDs that could not be parsed.

    Returns
    -------
    ConfigDataFlowResult
        Container with config data flow rows.
    """
    con = gateway.con
    refs_by_path = _config_references(con, snapshot.repo, snapshot.commit)
    if not refs_by_path:
        log.info(
            "No config references found for %s@%s; skipping config flow analysis",
            snapshot.repo,
            snapshot.commit,
        )
        return ConfigDataFlowResult(rows=None)

    entrypoints = _entrypoints(con, snapshot.repo, snapshot.commit)
    missing = missing_goids or set()
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
        snapshot=snapshot,
        now=now,
    )
    return ConfigDataFlowResult(rows=tuple(rows_to_insert) if rows_to_insert else None)


def _build_config_flow_rows(
    *,
    artifacts: ConfigFlowArtifacts,
    snapshot: SnapshotRef,
    now: datetime,
) -> list[tuple[object, ...]]:
    max_paths_per_usage = 5
    max_path_length = 10
    rows_to_insert: list[tuple[object, ...]] = []
    for goid, func_ast in artifacts.ast_by_goid.items():
        rel_path = normalize_path(func_ast.rel_path)
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
                max_examples=max_paths_per_usage,
            )
            visitor.visit(func_ast.node)
            if not visitor.result.kinds:
                continue
            chains = _call_chains(
                artifacts.call_graph,
                artifacts.entrypoints,
                goid,
                max_paths=max_paths_per_usage,
                max_length=max_path_length,
            )
            for usage_kind in sorted(visitor.result.kinds):
                collector = visitor.result.evidence.get(usage_kind)
                evidence = collector.to_dicts() if collector is not None else []
                for chain in chains:
                    chain_id = _call_chain_id(
                        snapshot.repo, snapshot.commit, config_key, usage_kind, chain
                    )
                    rows_to_insert.append(
                        (
                            snapshot.repo,
                            snapshot.commit,
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
