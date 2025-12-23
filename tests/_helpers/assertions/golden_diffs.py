"""Golden diff formatters for high-signal pytest failures.

These helpers are intentionally formatter-only (no pytest dependency), so
they can be used from either:

- plain `assert` statements (by raising AssertionError yourself), or
- `pytest.fail(...)` messages.

Primary use case (modules-first migration)
------------------------------------------

When migrating tests toward exercising the Hamilton-derived build outputs,
modules tests often compare:

- expected module paths (filesystem truth) vs.
- actual persisted module inventory (core.modules / core.repo_map).

Plain list equality diffs can be noisy. These formatters intentionally surface:

- missing vs extra items (set-style)
- and (for module maps) path changes for the same module key.
- duplicates within expected/actual collections (useful for detecting double writes).

Note: `core.modules` exposes a `path -> module` mapping. The module-map diff
helpers below expect `module -> path`, so invert the mapping (or use the
`module_map_from_path_map(...)` helper) before diffing.
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codeintel.core.paths import normalize_path

DEFAULT_DIFF_LIMIT = 40
"""Maximum number of entries shown per diff section by default."""


@runtime_checkable
class HasModuleAndPath(Protocol):
    """Protocol for module records used in tests.

    Any object with `module_name` and `rel_path` fields is supported.
    """

    module_name: str
    rel_path: str


def _normalize_path(value: object) -> str:
    """Normalize path-ish values to a stable POSIX string.

    Returns
    -------
    str
        Normalized path string.
    """
    return normalize_path(str(value))


def _sorted(
    values: Iterable[str],
) -> list[str]:
    """Deterministically sort values.

    Returns
    -------
    list[str]
        Sorted values.
    """
    return sorted(values)


def _find_duplicates(values: Iterable[str]) -> list[tuple[str, int]]:
    """Collect duplicate values with counts.

    Returns
    -------
    list[tuple[str, int]]
        Sorted list of (value, count) entries where count > 1.
    """
    counts = Counter(values)
    duplicates = [(value, count) for value, count in counts.items() if count > 1]
    return sorted(duplicates, key=lambda item: item[0])


@dataclass(frozen=True)
class MissingExtraOptions:
    """Options for missing/extra diff formatting."""

    noun: str = "items"
    context: str | None = None
    limit: int = DEFAULT_DIFF_LIMIT


@dataclass(frozen=True)
class ModuleMapDiffOptions:
    """Options for module map diff formatting."""

    context: str = "module inventory"
    limit: int = DEFAULT_DIFF_LIMIT
    include_path_section: bool = True
    include_unified_diff: bool = False


@dataclass(frozen=True)
class UnifiedDiffOptions:
    """Options for unified diff formatting."""

    fromfile: str = "expected"
    tofile: str = "actual"
    context_lines: int = 3
    limit_lines: int = 200


def _append_value_section(
    lines: list[str],
    *,
    header: str,
    prefix: str,
    values: list[str],
    limit: int,
) -> None:
    if not values:
        return
    lines.append(f"  {header} ({len(values)}):")
    lines.extend([f"    {prefix} {value}" for value in values[:limit]])
    if len(values) > limit:
        lines.append(f"    ... ({len(values) - limit} more)")


def _append_duplicate_section(
    lines: list[str],
    *,
    header: str,
    prefix: str,
    values: list[tuple[str, int]],
    limit: int,
) -> None:
    if not values:
        return
    lines.append(f"  {header} ({len(values)}):")
    lines.extend([f"    {prefix} {value} ({count}x)" for value, count in values[:limit]])
    if len(values) > limit:
        lines.append(f"    ... ({len(values) - limit} more)")


def _coerce_module_map(
    mapping: Mapping[str, str] | Iterable[HasModuleAndPath],
) -> dict[str, str]:
    if isinstance(mapping, Mapping):
        return {str(key): _normalize_path(value) for key, value in mapping.items()}
    return module_map_from_records(mapping)


def _collect_module_changes(
    expected_map: Mapping[str, str],
    actual_map: Mapping[str, str],
) -> tuple[list[str], list[str], list[str]]:
    expected_keys = set(expected_map)
    actual_keys = set(actual_map)
    missing_keys = _sorted(expected_keys - actual_keys)
    extra_keys = _sorted(actual_keys - expected_keys)
    common_keys = expected_keys & actual_keys
    changed_keys = _sorted([key for key in common_keys if expected_map[key] != actual_map[key]])
    return missing_keys, extra_keys, changed_keys


def format_missing_extra(
    expected: Iterable[str],
    actual: Iterable[str],
    *,
    options: MissingExtraOptions | None = None,
) -> str:
    """Format a compact missing/extra diff for two string collections.

    Parameters
    ----------
    expected
        Expected collection.
    actual
        Actual collection.
    options
        Formatting options for labels and limits.

    Returns
    -------
    str
        Formatted diff string including missing/extra and duplicates sections.
    """
    resolved = options or MissingExtraOptions()
    expected_list = [_normalize_path(value) for value in expected]
    actual_list = [_normalize_path(value) for value in actual]
    exp_set = set(expected_list)
    act_set = set(actual_list)
    missing = _sorted(exp_set - act_set)
    extra = _sorted(act_set - exp_set)
    expected_dupes = _find_duplicates(expected_list)
    actual_dupes = _find_duplicates(actual_list)

    header_ctx = f"{resolved.context}: " if resolved.context else ""
    lines: list[str] = [
        (
            f"{header_ctx}{resolved.noun} mismatch "
            f"(expected={len(expected_list)} actual={len(actual_list)})"
        ),
    ]

    if not missing and not extra and not expected_dupes and not actual_dupes:
        lines.append("  (no set differences)")
        return "\n".join(lines)

    _append_value_section(
        lines,
        header="missing",
        prefix="-",
        values=missing,
        limit=resolved.limit,
    )
    _append_value_section(
        lines,
        header="extra",
        prefix="+",
        values=extra,
        limit=resolved.limit,
    )
    _append_duplicate_section(
        lines,
        header=f"duplicate expected {resolved.noun}",
        prefix="-",
        values=expected_dupes,
        limit=resolved.limit,
    )
    _append_duplicate_section(
        lines,
        header=f"duplicate actual {resolved.noun}",
        prefix="+",
        values=actual_dupes,
        limit=resolved.limit,
    )

    return "\n".join(lines)


def format_unified_diff(
    expected: Iterable[str],
    actual: Iterable[str],
    *,
    options: UnifiedDiffOptions | None = None,
) -> str:
    """Render a unified diff (git-style) for two sorted collections.

    Parameters
    ----------
    expected
        Expected collection.
    actual
        Actual collection.
    options
        Formatting options for diff headers and truncation.

    Returns
    -------
    str
        Unified diff string.
    """
    resolved = options or UnifiedDiffOptions()
    exp_lines = _sorted([_normalize_path(value) for value in expected])
    act_lines = _sorted([_normalize_path(value) for value in actual])

    diff_iter = difflib.unified_diff(
        exp_lines,
        act_lines,
        fromfile=resolved.fromfile,
        tofile=resolved.tofile,
        lineterm="",
        n=resolved.context_lines,
    )
    diff = list(diff_iter)
    if not diff:
        return "(no diff)"
    if len(diff) > resolved.limit_lines:
        truncated = len(diff) - resolved.limit_lines
        diff = [
            *diff[: resolved.limit_lines],
            f"... (diff truncated; {truncated} more lines)",
        ]
    return "\n".join(diff)


def module_map_from_records(records: Iterable[HasModuleAndPath]) -> dict[str, str]:
    """Convert module records into a `module -> rel_path` mapping.

    Parameters
    ----------
    records
        Iterable of module records with module_name and rel_path fields.

    Returns
    -------
    dict[str, str]
        Mapping of module name to normalized relative path.
    """
    return {record.module_name: _normalize_path(record.rel_path) for record in records}


def module_map_from_path_map(module_path_map: Mapping[str, str]) -> dict[str, str]:
    """Invert a `path -> module` mapping into `module -> path`.

    Parameters
    ----------
    module_path_map
        Mapping of relative paths to module names (e.g., core.modules).

    Returns
    -------
    dict[str, str]
        Mapping of module name to normalized relative path.
    """
    return {str(module): _normalize_path(path) for path, module in module_path_map.items()}


def format_module_map_diff(
    expected: Mapping[str, str] | Iterable[HasModuleAndPath],
    actual: Mapping[str, str] | Iterable[HasModuleAndPath],
    *,
    options: ModuleMapDiffOptions | None = None,
) -> str:
    """Format a high-signal diff for module maps.

    Expects `module -> path` mappings. If you have a `path -> module` map
    (e.g., core.modules), invert it with `module_map_from_path_map(...)`
    before calling this formatter.

    Emits:
    - missing modules (by key)
    - extra modules (by key)
    - path changes (same module key, different path)
    - optionally: a separate missing/extra diff for paths (useful when module naming differs)
    - optionally: a unified diff of `module -> path` lines

    Parameters
    ----------
    expected
        Expected module mapping or module-record iterable.
    actual
        Actual module mapping or module-record iterable.
    options
        Formatting options for headers and sections.

    Returns
    -------
    str
        Formatted diff string for module inventory mismatches.
    """
    resolved = options or ModuleMapDiffOptions()
    exp_map = _coerce_module_map(expected)
    act_map = _coerce_module_map(actual)
    missing_keys, extra_keys, changed_keys = _collect_module_changes(exp_map, act_map)
    lines: list[str] = [
        (f"{resolved.context} mismatch (modules expected={len(exp_map)} actual={len(act_map)})"),
    ]

    if not missing_keys and not extra_keys and not changed_keys:
        lines.append("  (no module-key differences)")
    else:
        _append_value_section(
            lines,
            header="missing modules",
            prefix="-",
            values=[f"{key} -> {exp_map[key]}" for key in missing_keys],
            limit=resolved.limit,
        )
        _append_value_section(
            lines,
            header="extra modules",
            prefix="+",
            values=[f"{key} -> {act_map[key]}" for key in extra_keys],
            limit=resolved.limit,
        )
        _append_value_section(
            lines,
            header="path changes",
            prefix="~",
            values=[f"{key}: {exp_map[key]} -> {act_map[key]}" for key in changed_keys],
            limit=resolved.limit,
        )

    if resolved.include_path_section:
        exp_paths = list(exp_map.values())
        act_paths = list(act_map.values())
        exp_paths_set = set(exp_paths)
        act_paths_set = set(act_paths)
        if (
            exp_paths_set != act_paths_set
            or _find_duplicates(exp_paths)
            or _find_duplicates(act_paths)
        ):
            lines.append("")
            lines.append(
                format_missing_extra(
                    exp_paths,
                    act_paths,
                    options=MissingExtraOptions(
                        noun="paths",
                        context=resolved.context,
                        limit=resolved.limit,
                    ),
                )
            )

    if resolved.include_unified_diff:
        lines.append("")
        exp_lines = [f"{key} -> {value}" for key, value in sorted(exp_map.items())]
        act_lines = [f"{key} -> {value}" for key, value in sorted(act_map.items())]
        lines.append(
            format_unified_diff(
                exp_lines,
                act_lines,
                options=UnifiedDiffOptions(
                    fromfile=f"{resolved.context} (expected)",
                    tofile=f"{resolved.context} (actual)",
                ),
            )
        )

    return "\n".join(lines)


__all__ = [
    "DEFAULT_DIFF_LIMIT",
    "HasModuleAndPath",
    "MissingExtraOptions",
    "ModuleMapDiffOptions",
    "UnifiedDiffOptions",
    "format_missing_extra",
    "format_module_map_diff",
    "format_unified_diff",
    "module_map_from_path_map",
    "module_map_from_records",
]
