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

from codeintel.build.analytics.functions.parsing import parse_python_file
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_masks import FilterExprContext
from codeintel.core.paths import normalize_path, safe_relpath

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.parsing import ParsedModule

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfigReferenceInputs:
    """Inputs required to compute config reference rows."""

    snapshot: SnapshotRef
    config_value_rows: Sequence[Mapping[str, object]] | pa.Table
    module_rows: Sequence[Mapping[str, object]] | pa.Table


@dataclass(frozen=True, slots=True)
class _ConfigKeyEntry:
    config_path: str
    key: str


@dataclass(slots=True)
class _ReferenceAccumulator:
    paths: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)


def compute_config_reference_rows(inputs: ConfigReferenceInputs) -> list[dict[str, object]]:
    """Compute config reference rows from config values and module ASTs.

    Parameters
    ----------
    inputs
        Config reference inputs with snapshot, config values, and module rows.

    Returns
    -------
    list[dict[str, object]]
        Rows for analytics.config_references.
    """
    config_rows = _rows_from_tabular(
        inputs.config_value_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
    )
    if not config_rows:
        return []

    entries, keys = _config_entries(config_rows)
    if not entries or not keys:
        return []

    module_rows = _rows_from_tabular(
        inputs.module_rows,
        repo=inputs.snapshot.repo,
        commit=inputs.snapshot.commit,
    )
    modules_by_path = _modules_by_path(module_rows, repo_root=inputs.snapshot.repo_root)
    references = _reference_map(
        keys=keys,
        modules_by_path=modules_by_path,
        repo_root=inputs.snapshot.repo_root,
    )
    now = datetime.now(tz=UTC)
    rows: list[dict[str, object]] = []
    for entry in entries:
        accumulator = references.get(entry.key)
        paths = sorted(accumulator.paths) if accumulator else []
        modules = sorted(accumulator.modules) if accumulator else []
        rows.append(
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
    return rows


def _rows_from_tabular(
    rows: Sequence[Mapping[str, object]] | pa.Table,
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    if isinstance(rows, pa.Table):
        table = _filter_table_by_scope(rows, repo=repo, commit=commit)
        return [dict(row) for row in iter_rows(table)]

    filtered: list[dict[str, object]] = []
    for row in rows:
        if not _matches_optional_scope(row.get("repo"), repo):
            continue
        if not _matches_optional_scope(row.get("commit"), commit):
            continue
        filtered.append(dict(row))
    return filtered


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
