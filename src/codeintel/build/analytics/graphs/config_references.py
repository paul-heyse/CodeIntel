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
from codeintel.build.analytics.utilities.snapshot import snapshot_plan
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import materialize_plan
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.rows import ColumnarRowBuffer
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.paths import normalize_path, safe_relpath

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.parsing import ParsedModule

log = logging.getLogger(__name__)

CONFIG_REFERENCES_TABLE_KEY = "analytics.config_references"


@dataclass(frozen=True, slots=True)
class ConfigReferenceInputs:
    """Inputs required to compute config reference rows."""

    snapshot: SnapshotRef
    config_value_rows: Sequence[Mapping[str, object]] | pa.Table
    module_rows: Sequence[Mapping[str, object]] | pa.Table
    ctx: ExecutionContext | None = None


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
        paths = sorted(accumulator.paths) if accumulator else []
        modules = sorted(accumulator.modules) if accumulator else []
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
    ctx: ExecutionContext | None,
) -> tuple[list[_ConfigKeyEntry], set[str]]:
    if isinstance(rows, pa.Table):
        entry_table = _config_entry_rowset(rows, repo=repo, commit=commit, ctx=ctx)
        return _config_entries_from_table(entry_table)
    filtered: list[dict[str, object]] = []
    for row in rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        filtered.append(dict(row))
    return _config_entries(filtered)


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _config_entry_rowset(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> pa.Table:
    required = {"config_path", "key"}
    missing = [name for name in required if name not in table.column_names]
    if missing:
        msg = f"Missing config value columns: {missing}"
        raise ValueError(msg)
    plan = snapshot_plan(
        table,
        repo=repo,
        commit=commit,
        columns=("config_path", "key"),
        ctx=ctx,
    )
    plan = plan.filter(E.and_(E.is_valid("config_path"), E.is_valid("key")))
    plan = plan.order_by(
        sort_keys=[
            ("config_path", "ascending"),
            ("key", "ascending"),
        ]
    )
    plan = plan.aggregate(
        keys=[E.field("config_path")],
        aggregates=[("key", "list", None, "keys")],
    )
    return materialize_plan(plan, use_threads=True)


def _config_entries(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[_ConfigKeyEntry], set[str]]:
    entries: list[_ConfigKeyEntry] = []
    keys: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for row in rows:
        config_path = row.get("config_path")
        key = row.get("key")
        if config_path is None or key is None:
            continue
        config_path_value = str(config_path).strip()
        key_value = str(key).strip()
        if not config_path_value or not key_value:
            continue
        normalized_path = normalize_path(config_path_value)
        entry_key = (normalized_path, key_value)
        if entry_key in seen:
            continue
        entries.append(_ConfigKeyEntry(config_path=normalized_path, key=key_value))
        keys.add(key_value)
        seen.add(entry_key)
    return entries, keys


def _config_entries_from_table(
    table: pa.Table,
) -> tuple[list[_ConfigKeyEntry], set[str]]:
    entries: list[_ConfigKeyEntry] = []
    keys: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for config_path, key_list in iter_tuples(
        table.to_reader(),
        columns=("config_path", "keys"),
    ):
        if config_path is None:
            continue
        normalized_path = normalize_path(str(config_path).strip())
        if not normalized_path:
            continue
        for key in _list_values(key_list):
            key_value = str(key).strip()
            if not key_value:
                continue
            entry_key = (normalized_path, key_value)
            if entry_key in seen:
                continue
            entries.append(_ConfigKeyEntry(config_path=normalized_path, key=key_value))
            keys.add(key_value)
            seen.add(entry_key)
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
    ctx: ExecutionContext | None,
) -> dict[str, set[str]]:
    if isinstance(rows, pa.Table):
        table = _module_rowset(rows, repo=repo, commit=commit, ctx=ctx)
        return _modules_by_path_from_table(table, repo_root=repo_root)
    return _modules_by_path(rows, repo_root=repo_root)


def _module_rowset(
    table: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | None,
) -> pa.Table:
    required = {"path", "module"}
    missing = [name for name in required if name not in table.column_names]
    if missing:
        msg = f"Missing module columns: {missing}"
        raise ValueError(msg)
    columns = ["path", "module"]
    if "language" in table.column_names:
        columns.append("language")
    plan = snapshot_plan(
        table,
        repo=repo,
        commit=commit,
        columns=tuple(columns),
        ctx=ctx,
    )
    filters = [E.is_valid("path"), E.is_valid("module")]
    if "language" in table.column_names:
        filters.append(E.or_(E.is_null("language"), E.field("language") == E.scalar("python")))
    plan = plan.filter(E.and_(*filters))
    plan = plan.order_by(
        sort_keys=[
            ("path", "ascending"),
            ("module", "ascending"),
        ]
    )
    plan = plan.aggregate(
        keys=[E.field("path")],
        aggregates=[("module", "list", None, "modules")],
    )
    return materialize_plan(plan, use_threads=True)


def _modules_by_path_from_table(
    table: pa.Table,
    *,
    repo_root: Path,
) -> dict[str, set[str]]:
    modules_by_path: dict[str, set[str]] = {}
    for path, module_list in iter_tuples(table.to_reader(), columns=("path", "modules")):
        if not isinstance(path, str) or not path.strip():
            continue
        rel_path = _normalize_module_path(path, repo_root=repo_root)
        if not rel_path:
            continue
        for module in _list_values(module_list):
            module_value = str(module).strip()
            if not module_value:
                continue
            modules_by_path.setdefault(rel_path, set()).add(module_value)
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


__all__ = [
    "ConfigReferenceInputs",
    "compute_config_reference_rows",
]
