"""Track config key usage at the function level with call-chain context."""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.compute.row_builders import rows_to_tuples_for_table
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.build.analytics.utilities.snapshot import SnapshotContext, snapshot_plan
from codeintel.build.graphs.rx.algos import (
    GraphInput,
    ensure_directed_store,
    simple_paths_by_id,
)
from codeintel.build.tabular.expr_vocab import E
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.explode_ops import (
    ExplodeSpec,
    explode_edges,
    explode_edges_for_join,
)
from codeintel.core.columnar.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.hashing import sha256_short
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.row_models import columns_for_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef


CONFIG_DATA_FLOW_TABLE_KEY = "analytics.config_data_flow"
CONFIG_REFERENCES_TABLE_KEY = "analytics.config_references"
ENTRYPOINTS_TABLE_KEY = "analytics.entrypoints"
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


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


def _config_reference_rowset(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    required = {"config_path", "key", "extras"}
    missing = [name for name in required if name not in table.column_names]
    if missing:
        msg = f"Missing config reference columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(
        table,
        columns=None,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CONFIG_REFERENCES_TABLE_KEY,
        ),
    )
    plan = plan.filter(E.and_(E.is_valid("config_path"), E.is_valid("key")))
    plan = plan.project(
        {
            "config_path": E.field("config_path"),
            "key": E.field("key"),
            "reference_paths": E.field(("extras", "reference_paths")),
        }
    )
    filtered = _materialize_plan(plan, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("config_path", "key"),
            aggregates=[("reference_paths", "list", None, "reference_paths")],
        ),
        ctx=resolve_columnar_context(ctx),
    )


def _entrypoint_rowset(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    if "handler_goid_h128" not in table.column_names:
        msg = "Missing entrypoint column: handler_goid_h128"
        raise ValueError(msg)
    plan = snapshot_plan(
        table,
        columns=("handler_goid_h128",),
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=ENTRYPOINTS_TABLE_KEY,
        ),
    )
    plan = plan.filter(E.is_valid("handler_goid_h128"))
    filtered = _materialize_plan(plan, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("handler_goid_h128",),
            aggregates=(),
        ),
        ctx=resolve_columnar_context(ctx),
    )


def _table_from_rows(
    rows: Sequence[Mapping[str, object]] | pa.Table,
) -> pa.Table:
    if isinstance(rows, pa.Table):
        return rows
    rows_list = list(rows)
    if not rows_list:
        return pa.Table.from_pylist([])
    return pa.Table.from_pylist(rows_list)


def _config_reference_edges_table(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    table = _table_from_rows(rows)
    if table.num_rows == 0:
        return pa.Table.from_pylist([])
    rowset = _config_reference_rowset(table, repo=repo, commit=commit, ctx=ctx)
    if rowset.num_rows == 0:
        return rowset
    exploded = explode_edges_for_join(
        rowset,
        spec=ExplodeSpec(
            src_col="reference_paths",
            dst_list_col="reference_path",
            repeat_cols=("config_path", "key"),
            null_list_policy="empty",
            null_child_policy="drop",
            error_context_cols=("config_path", "key"),
        ),
        allowed_columns=("config_path", "key", "reference_path"),
    )
    return exploded.good


def _config_references_by_path(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[str, list[tuple[str, str]]]:
    edges = _config_reference_edges_table(rows, repo=repo, commit=commit, ctx=ctx)
    if edges.num_rows == 0:
        return {}
    grouped = grouped_rollup_table(
        edges,
        spec=GroupedRollupSpec(
            keys=("reference_path",),
            aggregates=[
                ("config_path", "list", None, "config_paths"),
                ("key", "list", None, "keys"),
            ],
            pre_sort_keys=(
                ("reference_path", "ascending"),
                ("config_path", "ascending"),
                ("key", "ascending"),
            ),
        ),
        ctx=resolve_columnar_context(ctx),
    )
    exploded = explode_edges(
        grouped,
        spec=ExplodeSpec(
            src_col="reference_path",
            dst_list_col="config_paths",
            aligned_list_cols=("keys",),
            null_list_policy="empty",
            null_child_policy="drop",
            error_context_cols=("reference_path",),
        ),
    )
    refs: dict[str, list[tuple[str, str]]] = {}
    for reference_path, config_path, key in iter_tuples(
        exploded.good.to_reader(),
        columns=("reference_path", "config_paths", "keys"),
    ):
        if not isinstance(reference_path, str) or not reference_path.strip():
            continue
        if config_path is None or key is None:
            continue
        rel_path = normalize_path(reference_path)
        refs.setdefault(rel_path, []).append((str(key), str(config_path)))
    return refs


def _entrypoint_ids_from_tabular(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> set[int]:
    table = _table_from_rows(rows)
    if table.num_rows == 0:
        return set()
    rowset = _entrypoint_rowset(table, repo=repo, commit=commit, ctx=ctx)
    entrypoints: set[int] = set()
    for (handler_goid,) in iter_tuples(
        rowset.to_reader(),
        columns=("handler_goid_h128",),
    ):
        goid = normalize_decimal_id(handler_goid)
        if goid is not None:
            entrypoints.add(goid)
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
    ordered_entrypoints: list[int] = []
    for node_id in store.node_ids():
        entry_id = normalize_decimal_id(node_id)
        if entry_id is None:
            continue
        if entry_id in entrypoints:
            ordered_entrypoints.append(int(entry_id))
    for entry in ordered_entrypoints:
        entry_idx = store.id_to_index.get(entry)
        if entry_idx is None:
            continue
        for path in simple_paths_by_id(
            store,
            entry,
            target,
            cutoff=max_length,
            limit=max_paths - len(paths),
        ):
            ids = [int(str(node)) for node in path]
            paths.append(ids)
            if len(paths) >= max_paths:
                break
    if not paths:
        return [[target]]
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
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


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
    refs_by_path = _config_references_by_path(
        inputs.config_value_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        ctx=inputs.ctx,
    )
    if not refs_by_path:
        log.info(
            "No config references found for %s@%s; skipping config flow analysis",
            inputs.snapshot.repo,
            inputs.snapshot.commit,
        )
        return ConfigDataFlowResult(rows=None)

    entrypoints = _entrypoint_ids_from_tabular(
        inputs.entrypoint_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        ctx=inputs.ctx,
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


def build_config_data_flow_table(inputs: ConfigDataFlowInputs) -> pa.Table:
    """Build a config data flow table from computed rows.

    Returns
    -------
    pa.Table
        Config data flow table aligned to the contract schema.
    """
    result = compute_config_data_flow_result(inputs)
    if result.rows is None:
        return empty_table_for_table(CONFIG_DATA_FLOW_TABLE_KEY)
    table, _ = table_for_rows(CONFIG_DATA_FLOW_TABLE_KEY, result.rows)
    return table


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
            for usage_kind in visitor.result.kinds:
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


def _materialize_plan(
    plan: Plan,
    *,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    execution_ctx = resolve_execution_context(resolve_columnar_context(ctx))
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(_INTERNAL_PLAN_TABLE_KEY, mode="tolerant"),
    )
    return result.good
