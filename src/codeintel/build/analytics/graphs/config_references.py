"""Extract config key references from Python modules."""

from __future__ import annotations

import ast
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.compute.row_builders import buffer_for_table
from codeintel.build.analytics.functions.parsing import parse_python_file
from codeintel.build.analytics.parsing.worklists import build_module_ast_worklist
from codeintel.build.analytics.utilities.list_semantics import normalize_list_semantics
from codeintel.build.analytics.utilities.snapshot import SnapshotContext, snapshot_plan
from codeintel.build.tabular.expr_vocab import E
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import table_to_reader
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.explode_ops import ExplodeSpec, explode_edges
from codeintel.core.columnar.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.paths import normalize_path, safe_relpath

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.columnar.rows import ColumnarRowBuffer
    from codeintel.core.parsing import ParsedModule

log = logging.getLogger(__name__)

CONFIG_REFERENCES_TABLE_KEY = "analytics.config_references"
CONFIG_VALUES_TABLE_KEY = "analytics.config_values"
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"


@dataclass(frozen=True, slots=True)
class ConfigReferenceInputs:
    """Inputs required to compute config reference rows."""

    snapshot: SnapshotRef
    config_value_rows: Sequence[Mapping[str, object]] | pa.Table
    module_rows: Sequence[Mapping[str, object]] | pa.Table
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


@dataclass(frozen=True, slots=True)
class _ConfigKeyEntry:
    config_path: str
    key: str


@dataclass(slots=True)
class _ReferenceAccumulator:
    paths: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)


def compute_config_reference_rows(inputs: ConfigReferenceInputs) -> ColumnarRowBuffer:
    """Compute config reference rows from config values and module ASTs.

    Parameters
    ----------
    inputs
        Config reference inputs with snapshot, config values, and module rows.

    Returns
    -------
    ColumnarRowBuffer
        Buffer containing analytics.config_references rows.
    """
    entries, keys = _config_entries_from_tabular(
        inputs.config_value_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        ctx=inputs.ctx,
    )
    if not entries or not keys:
        return buffer_for_table(CONFIG_REFERENCES_TABLE_KEY)

    modules_by_path = _modules_by_path_from_tabular(
        inputs.module_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
        repo_root=inputs.snapshot.repo_root,
        ctx=inputs.ctx,
    )
    references = _reference_map(
        keys=keys,
        modules_by_path=modules_by_path,
        repo_root=inputs.snapshot.repo_root,
    )
    now = datetime.now(tz=UTC)
    buffer = buffer_for_table(CONFIG_REFERENCES_TABLE_KEY)
    for entry in entries:
        accumulator = references.get(entry.key)
        paths = normalize_list_semantics(accumulator.paths) if accumulator else []
        modules = normalize_list_semantics(accumulator.modules) if accumulator else []
        buffer.append(
            {
                "repo": inputs.snapshot.repo,
                "commit": inputs.snapshot.commit,
                "config_path": entry.config_path,
                "key": entry.key,
                "extras": {
                    "reference_paths": paths,
                    "reference_modules": modules,
                },
                "reference_count": len(paths),
                "created_at": now,
            }
        )
    return buffer


def _config_entries_from_tabular(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> tuple[list[_ConfigKeyEntry], set[str]]:
    if isinstance(rows, pa.Table):
        source = rows
    else:
        if not rows:
            return [], set()
        source = pa.Table.from_pylist(list(rows))
    entry_table = _config_entry_rowset(source, repo=repo, commit=commit, ctx=ctx)
    return _config_entries_from_table(entry_table)


def _config_entry_rowset(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    required = {"config_path", "key"}
    missing = [name for name in required if name not in table.column_names]
    if missing:
        msg = f"Missing config value columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(
        table,
        columns=None,
        context=SnapshotContext(
            repo=repo,
            commit=commit,
            ctx=ctx,
            table_key=CONFIG_VALUES_TABLE_KEY,
        ),
    )
    plan = plan.filter(E.and_(E.is_valid("config_path"), E.is_valid("key")))
    filtered = _materialize_plan(plan, ctx=ctx)
    return grouped_rollup_table(
        filtered,
        spec=GroupedRollupSpec(
            keys=("config_path", "key"),
            aggregates=(),
            pre_sort_keys=(
                ("config_path", "ascending"),
                ("key", "ascending"),
            ),
        ),
        ctx=resolve_columnar_context(ctx),
    )


def _config_entries_from_table(
    table: pa.Table,
) -> tuple[list[_ConfigKeyEntry], set[str]]:
    entries: list[_ConfigKeyEntry] = []
    keys: set[str] = set()
    reader = table_to_reader(table, batch_size=None)
    for config_path, key in iter_tuples(
        reader,
        columns=("config_path", "key"),
    ):
        if config_path is None or key is None:
            continue
        normalized_path = normalize_path(str(config_path).strip())
        key_value = str(key).strip()
        if not normalized_path or not key_value:
            continue
        entries.append(_ConfigKeyEntry(config_path=normalized_path, key=key_value))
        keys.add(key_value)
    return entries, keys


def _list_values(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _modules_by_path(
    rows: Sequence[Mapping[str, object]],
    *,
    repo_root: Path,
) -> dict[str, set[str]]:
    modules_by_path: dict[str, set[str]] = {}
    for row in rows:
        path = row.get("path")
        module = row.get("module")
        language = row.get("language")
        if language not in {None, "python"}:
            continue
        if not isinstance(path, str) or not path.strip():
            continue
        if not isinstance(module, str) or not module.strip():
            continue
        rel_path = _normalize_module_path(path, repo_root=repo_root)
        if not rel_path:
            continue
        modules_by_path.setdefault(rel_path, set()).add(module.strip())
    return modules_by_path


def _modules_by_path_from_tabular(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
    repo_root: Path,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[str, set[str]]:
    if isinstance(rows, pa.Table):
        worklist = build_module_ast_worklist(rows, repo=repo, commit=commit, ctx=ctx)
        if not {"path", "modules"}.issubset(worklist.column_names):
            return {}
        return _modules_by_path_from_table(worklist, repo_root=repo_root)
    return _modules_by_path(rows, repo_root=repo_root)


def _modules_by_path_from_table(
    table: pa.Table,
    *,
    repo_root: Path,
) -> dict[str, set[str]]:
    modules_by_path: dict[str, set[str]] = {}
    exploded = explode_edges(
        table,
        spec=ExplodeSpec(
            src_col="path",
            dst_list_col="modules",
            null_list_policy="empty",
            null_child_policy="drop",
            error_context_cols=("path",),
        ),
    )
    grouped = grouped_rollup_table(
        exploded.good,
        spec=GroupedRollupSpec(
            keys=("path",),
            aggregates=[("modules", "list", None, "modules")],
            pre_sort_keys=(("path", "ascending"), ("modules", "ascending")),
        ),
        ctx=None,
    )
    reader = table_to_reader(grouped, batch_size=None)
    for path, module_list in iter_tuples(reader, columns=("path", "modules")):
        if not isinstance(path, str) or not path.strip():
            continue
        rel_path = _normalize_module_path(path, repo_root=repo_root)
        if not rel_path:
            continue
        modules = normalize_list_semantics(_list_values(module_list))
        if not modules:
            continue
        modules_by_path.setdefault(rel_path, set()).update(modules)
    return modules_by_path


def _normalize_module_path(path: str, *, repo_root: Path) -> str:
    rel_path = safe_relpath(path, repo_root)
    return normalize_path(rel_path)


def _reference_map(
    *,
    keys: set[str],
    modules_by_path: Mapping[str, set[str]],
    repo_root: Path,
) -> dict[str, _ReferenceAccumulator]:
    references: dict[str, _ReferenceAccumulator] = {}
    parsed_cache: dict[str, ParsedModule | None] = {}
    for rel_path, module_names in modules_by_path.items():
        matched = _keys_in_module(
            rel_path=rel_path,
            keys=keys,
            repo_root=repo_root,
            cache=parsed_cache,
        )
        if not matched:
            continue
        for key in matched:
            accumulator = references.setdefault(key, _ReferenceAccumulator())
            accumulator.paths.add(rel_path)
            accumulator.modules.update(module_names)
    return references


def _keys_in_module(
    *,
    rel_path: str,
    keys: set[str],
    repo_root: Path,
    cache: dict[str, ParsedModule | None],
) -> set[str]:
    parsed = _parsed_module_for_path(rel_path, repo_root=repo_root, cache=cache)
    if parsed is None:
        return set()
    literals = _string_literals(parsed.module_ast)
    if not literals:
        return set()
    return literals.intersection(keys)


def _parsed_module_for_path(
    rel_path: str,
    *,
    repo_root: Path,
    cache: dict[str, ParsedModule | None],
) -> ParsedModule | None:
    cached = cache.get(rel_path)
    if rel_path in cache:
        return cached
    abs_path = (repo_root / rel_path).resolve()
    try:
        parsed = parse_python_file(abs_path)
    except (OSError, ValueError) as exc:
        log.debug("Skipping config reference parse for %s: %s", abs_path, exc)
        cache[rel_path] = None
        return None
    cache[rel_path] = parsed
    return parsed


def _string_literals(module_ast: ast.AST) -> set[str]:
    literals: set[str] = set()
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value:
                literals.add(value)
    return literals


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


__all__ = [
    "ConfigReferenceInputs",
    "compute_config_reference_rows",
]
