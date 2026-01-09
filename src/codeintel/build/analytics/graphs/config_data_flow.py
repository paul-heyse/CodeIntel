"""Track config key usage at the function level with call-chain context."""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import rustworkx as rx

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.compute.row_builders import rows_to_tuples_for_table
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.build.graphs.rx.algos import GraphInput, ensure_directed_store
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_masks import FilterExprContext
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.hashing import sha256_short
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.row_models import columns_for_table_key

if TYPE_CHECKING:
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef


CONFIG_DATA_FLOW_TABLE_KEY = "analytics.config_data_flow"


def _columns_for_table(table_key: str) -> tuple[str, ...]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return tuple(columns)


CONFIG_DATA_FLOW_COLS = _columns_for_table(CONFIG_DATA_FLOW_TABLE_KEY)

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
    call_graph: GraphInput
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


def _coerce_paths(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    elif isinstance(raw, (list, tuple)):
        parsed = raw
    else:
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [normalize_path(path) for path in parsed if isinstance(path, str)]


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _filter_table_by_scope(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
) -> pa.Table:
    context = FilterExprContext(repo=repo, commit=commit)
    return context.apply(table)


def _config_references_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
) -> dict[str, list[tuple[str, str]]]:
    refs: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        config_path = row.get("config_path")
        key = row.get("key")
        if config_path is None or key is None:
            continue
        extras = row.get("extras")
        reference_paths = extras.get("reference_paths") if isinstance(extras, Mapping) else None
        for rel_path in _coerce_paths(reference_paths):
            refs.setdefault(rel_path, []).append((str(key), str(config_path)))
    return refs


def _entrypoints_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
) -> set[int]:
    entrypoints: set[int] = set()
    for row in rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        handler_goid = normalize_decimal_id(row.get("handler_goid_h128"))
        if handler_goid is not None:
            entrypoints.add(handler_goid)
    return entrypoints


def _call_chains(
    graph: GraphInput,
    entrypoints: set[int],
    target: int,
    *,
    max_paths: int,
    max_length: int,
) -> list[list[int]]:
    store = ensure_directed_store(graph)
    target_idx = store.id_to_index.get(target)
    if target_idx is None:
        return [[target]]
    paths: list[list[int]] = []
    directed_graph = cast("rx.PyDiGraph", store.graph)
    for entry in sorted(entrypoints):
        entry_idx = store.id_to_index.get(entry)
        if entry_idx is None:
            continue
        try:
            for path in rx.digraph_all_simple_paths(
                directed_graph,
                entry_idx,
                target_idx,
                cutoff=max_length,
            ):
                ids = [int(str(store.index_to_id[idx])) for idx in path]
                paths.append(ids)
                if len(paths) >= max_paths:
                    break
        except (rx.InvalidNode, rx.NoPathFound, rx.NullGraph):
            continue
    if not paths:
        return [[target]]
    paths.sort(key=lambda path: stable_key(tuple(path)))
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


@dataclass(frozen=True)
class ConfigDataFlowInputs:
    """Inputs required to compute config data flow rows."""

    snapshot: SnapshotRef
    config_value_rows: Sequence[Mapping[str, object]] | pa.Table
    entrypoint_rows: Sequence[Mapping[str, object]] | pa.Table
    call_graph: GraphInput
    ast_by_goid: dict[int, FunctionAst]
    missing_goids: set[int] | None = None


def compute_config_data_flow_result(inputs: ConfigDataFlowInputs) -> ConfigDataFlowResult:
    """Compute config data flow rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/ArrowDatasetSaver.

    Parameters
    ----------
    inputs
        Config data flow inputs derived from DAG-provided tables.

    Returns
    -------
    ConfigDataFlowResult
        Container with config data flow rows.
    """
    config_rows = _rows_from_tabular(
        inputs.config_value_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
    )
    refs_by_path = _config_references_from_rows(
        config_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
    )
    if not refs_by_path:
        log.info(
            "No config references found for %s@%s; skipping config flow analysis",
            inputs.snapshot.repo,
            inputs.snapshot.commit,
        )
        return ConfigDataFlowResult(rows=None)

    entrypoint_rows = _rows_from_tabular(
        inputs.entrypoint_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
    )
    entrypoints = _entrypoints_from_rows(
        entrypoint_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
    )
    missing = inputs.missing_goids or set()
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during config data flow analysis",
            len(missing),
        )

    artifacts = ConfigFlowArtifacts(
        entrypoints=entrypoints,
        call_graph=inputs.call_graph,
        ast_by_goid=inputs.ast_by_goid,
        refs_by_path=refs_by_path,
    )
    now = datetime.now(tz=UTC)
    rows_to_insert = _build_config_flow_rows(
        artifacts=artifacts,
        snapshot=inputs.snapshot,
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
    row_dicts: list[dict[str, object]] = []
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
                    row_dicts.append(
                        {
                            "repo": snapshot.repo,
                            "commit": snapshot.commit,
                            "config_key": config_key,
                            "config_path": config_path,
                            "function_goid_h128": goid,
                            "usage_kind": usage_kind,
                            "call_chain_id": chain_id,
                            "extras": {
                                "evidence": evidence if evidence else None,
                                "call_chain": list(chain),
                            },
                            "created_at": now,
                        }
                    )
    return rows_to_tuples_for_table(CONFIG_DATA_FLOW_TABLE_KEY, row_dicts)


def _rows_from_tabular(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    if isinstance(rows, pa.Table):
        table = _filter_table_by_scope(cast("pa.Table", rows), repo=repo, commit=commit)
        return [dict(row) for row in iter_rows(table)]
    return [dict(row) for row in rows]
