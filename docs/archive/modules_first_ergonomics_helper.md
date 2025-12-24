
Here’s a concrete, repo-ready implementation of that “modules-first ergonomics” helper, designed to **mirror production behavior** as closely as possible by *reusing the same code paths your `modules` target uses* and by accepting the same ingestion options:

* `build_scan_profile(...)` (src/ vs repo-root, ignore dirs, scope paths)
* `FilesystemDiscoveryAdapter.discover_modules(...)` (same module naming, including `__init__.py -> pkg.__init__`)
* `filter_modules(...)` (test-file detection, generated-file detection, max-size filtering)

---

## Add this helper

### Recommended location

Create a new helper module:

* **`tests/_helpers/modules_expectations.py`**

This keeps it small, focused, and reusable across ingestion + orchestration tests without further bloating `tests/_helpers/ingestion.py`.

### Implementation

```python
# tests/_helpers/modules_expectations.py
"""Helpers for expressing module-inventory expectations from real repo trees.

These helpers intentionally reuse production scanning/filtering code to keep
tests realistic and aligned with the `modules` Hamilton target behavior.
"""

from __future__ import annotations

from pathlib import Path

from codeintel.build.hamilton.helpers import build_scan_profile, filter_modules
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter


def modules_expected_from_repo_tree(
    repo_root: Path,
    *,
    options: ModuleIngestOptions | None = None,
) -> dict[str, str]:
    """Compute expected module inventory (rel_path -> module_name) from repo tree.

    This mirrors the production `modules` target discovery pipeline:

    - `default_code_profile` via `build_scan_profile`:
        * prefers `repo_root/src` when it exists, otherwise uses `repo_root`
        * honors the same ignore dirs (e.g., .git, .venv, __pycache__, node_modules, ...)
        * adds ignores for `tests/` and `generated/` depending on options
        * respects `scope_paths` when provided
    - `FilesystemDiscoveryAdapter.discover_modules`:
        * same SourceScanner behavior + glob matching
        * same repo_relpath normalization (POSIX rel paths)
        * same module naming semantics (`pkg/__init__.py` -> `pkg.__init__`)
    - `filter_modules`:
        * same include_tests behavior (directory + naming conventions)
        * same generated-file heuristics
        * same file-size filtering

    Parameters
    ----------
    repo_root:
        Root directory of the repository.
    options:
        Module ingestion options. Defaults to ModuleIngestOptions(), which mirrors
        production defaults (including include_tests=True). For production-only
        expectations, pass ModuleIngestOptions(include_tests=False).

    Returns
    -------
    dict[str, str]
        Mapping of relative POSIX file path -> module name.
        This matches the shape returned by `load_module_map(...)` (core.modules).
    """
    resolved = options or ModuleIngestOptions()
    profile = build_scan_profile(repo_root, resolved)

    discovered = FilesystemDiscoveryAdapter.discover_modules(repo_root, profile)
    filtered = filter_modules(discovered, resolved)

    # Deterministic ordering for easy diffs in assertion failures.
    module_map = {record.rel_path: record.module_name for record in filtered}
    return dict(sorted(module_map.items(), key=lambda kv: kv[0]))
```

If you're diffing `core.modules` (path -> module), invert it first:
`module_map_from_path_map(load_module_map(...))` before calling
`format_module_map_diff(...)`.

---

## How to use it immediately (example: remove hard-coded expectations)

Your current `tests/ingestion/test_module_inventory.py` has a hand-written list:

```python
expected = ["src/pkg/a.py", "src/pkg/b.py"]
```

You can replace that with computed expectations:

```python
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

# ...

expected_map = modules_expected_from_repo_tree(
    ctx.snapshot.repo_root,
    options=ModuleIngestOptions(include_tests=True),
)
expected_paths = sorted(expected_map.keys())

rel_paths = sorted(record.rel_path for record in records)
if rel_paths != expected_paths:
    pytest.fail(f"Unexpected module paths {rel_paths}, expected {expected_paths}")
```

And you can also compare the *actual persisted* module_map directly:

```python
# module_map comes from core.modules (path -> module)
if module_map != expected_map:
    pytest.fail(f"module_map differs from scan expectations.\nGot: {module_map}\nExp: {expected_map}")
```

> Note: I used `ModuleIngestOptions(include_tests=True)` in the example above purely to
> show it’s explicit. For your default `module_inventory_context` (src-layout, no tests),
> it won’t matter either way. For “production-code-only” expectations, pass
> `ModuleIngestOptions(include_tests=False)`.

---

## Optional tiny add-on (if you want even cleaner call sites)

If you find yourself frequently wanting “just paths” for assertions, you can keep the canonical helper as-is and add this trivial wrapper *in the same file*:

```python
def module_paths_expected_from_repo_tree(
    repo_root: Path,
    *,
    options: ModuleIngestOptions | None = None,
) -> list[str]:
    """Compute sorted expected module paths from a repository tree."""
    return sorted(modules_expected_from_repo_tree(repo_root, options=options).keys())
```

But I’d keep the main helper returning the `path -> module` mapping because it composes perfectly with:

* `load_module_map(...)` comparisons
* repo_map inversion (`{module: path for path, module in expected_map.items()}`) when validating `core.repo_map`

---

## Optional production-profile variant

If you want to mirror the *active* ingestion profile (e.g., loaded from config or CLI),
add a tiny wrapper that uses the same target options loader as the Hamilton target:

```python
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.ingestion.ingest_targets import MODULES_TARGET_NAME
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.hamilton.options_loading import load_target_options


def modules_expected_from_env(env: BuildEnv) -> dict[str, str]:
    """Compute expected modules using active config/CLI target options."""
    options = load_target_options(
        env,
        target_name=MODULES_TARGET_NAME,
        options_type=ModuleIngestOptions,
    )
    return modules_expected_from_repo_tree(env.snapshot.repo_root, options=options)
```

# golden diff helper #

Below is a **small “golden failure diff” formatter** you can drop into your pytest helpers so “modules-first” assertions print **clean missing/extra (and moved) items**, instead of noisy list equality dumps.

It’s intentionally **formatter-only** (no pytest import), so you can use it from either `pytest.fail(...)` or `raise AssertionError(...)`.

---

## 1) Add `tests/_helpers/assertions/golden_diffs.py`

```python
"""
Golden diff formatters for high-signal pytest failures.

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
`module_map_from_path_map(...)` helper in the snippet) before diffing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
import difflib
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
    *,
    sort_key: Callable[[str], object] | None = None,
) -> list[str]:
    """Deterministically sort values with an optional key.

    Returns
    -------
    list[str]
        Sorted values.
    """
    if sort_key is None:
        return sorted(values)
    return sorted(values, key=sort_key)


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


def format_missing_extra(
    expected: Iterable[str],
    actual: Iterable[str],
    *,
    noun: str = "items",
    context: str | None = None,
    limit: int = DEFAULT_DIFF_LIMIT,
) -> str:
    """Format a compact missing/extra diff for two string collections.

    Parameters
    ----------
    expected
        Expected collection.
    actual
        Actual collection.
    noun
        Label for diff entries.
    context
        Optional context prefix.
    limit
        Maximum number of entries per diff section.

    Returns
    -------
    str
        Formatted diff string including missing/extra and duplicates sections.
    """
    expected_list = [_normalize_path(value) for value in expected]
    actual_list = [_normalize_path(value) for value in actual]
    exp_set = set(expected_list)
    act_set = set(actual_list)
    missing = _sorted(exp_set - act_set)
    extra = _sorted(act_set - exp_set)
    expected_dupes = _find_duplicates(expected_list)
    actual_dupes = _find_duplicates(actual_list)

    header_ctx = f"{context}: " if context else ""
    lines: list[str] = [
        f"{header_ctx}{noun} mismatch (expected={len(expected_list)} actual={len(actual_list)})",
    ]

    if not missing and not extra and not expected_dupes and not actual_dupes:
        lines.append("  (no set differences)")
        return "\n".join(lines)

    if missing:
        lines.append(f"  missing ({len(missing)}):")
        for item in missing[:limit]:
            lines.append(f"    - {item}")
        if len(missing) > limit:
            lines.append(f"    ... ({len(missing) - limit} more)")

    if extra:
        lines.append(f"  extra ({len(extra)}):")
        for item in extra[:limit]:
            lines.append(f"    + {item}")
        if len(extra) > limit:
            lines.append(f"    ... ({len(extra) - limit} more)")

    if expected_dupes:
        lines.append(f"  duplicate expected {noun} ({len(expected_dupes)}):")
        for item, count in expected_dupes[:limit]:
            lines.append(f"    - {item} ({count}x)")
        if len(expected_dupes) > limit:
            lines.append(f"    ... ({len(expected_dupes) - limit} more)")

    if actual_dupes:
        lines.append(f"  duplicate actual {noun} ({len(actual_dupes)}):")
        for item, count in actual_dupes[:limit]:
            lines.append(f"    + {item} ({count}x)")
        if len(actual_dupes) > limit:
            lines.append(f"    ... ({len(actual_dupes) - limit} more)")

    return "\n".join(lines)


def format_unified_diff(
    expected: Iterable[str],
    actual: Iterable[str],
    *,
    fromfile: str = "expected",
    tofile: str = "actual",
    context_lines: int = 3,
    limit_lines: int = 200,
) -> str:
    """Render a unified diff (git-style) for two sorted collections.

    Parameters
    ----------
    expected
        Expected collection.
    actual
        Actual collection.
    fromfile
        Name of the "expected" file in the diff header.
    tofile
        Name of the "actual" file in the diff header.
    context_lines
        Number of context lines in the unified diff.
    limit_lines
        Maximum number of diff lines to include before truncation.

    Returns
    -------
    str
        Unified diff string.
    """
    exp_lines = _sorted([_normalize_path(value) for value in expected])
    act_lines = _sorted([_normalize_path(value) for value in actual])

    diff_iter = difflib.unified_diff(
        exp_lines,
        act_lines,
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
        n=context_lines,
    )
    diff = list(diff_iter)
    if not diff:
        return "(no diff)"
    if len(diff) > limit_lines:
        truncated = len(diff) - limit_lines
        diff = diff[:limit_lines] + [f"... (diff truncated; {truncated} more lines)"]
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
    context: str = "module inventory",
    limit: int = DEFAULT_DIFF_LIMIT,
    include_path_section: bool = True,
    include_unified_diff: bool = False,
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
    context
        Context label for diff headers.
    limit
        Maximum number of entries per diff section.
    include_path_section
        Whether to add a missing/extra diff for paths.
    include_unified_diff
        Whether to append a unified diff of module-to-path lines.

    Returns
    -------
    str
        Formatted diff string for module inventory mismatches.
    """
    exp_map = (
        {k: _normalize_path(v) for k, v in expected.items()}
        if isinstance(expected, Mapping)
        else module_map_from_records(expected)
    )
    act_map = (
        {k: _normalize_path(v) for k, v in actual.items()}
        if isinstance(actual, Mapping)
        else module_map_from_records(actual)
    )

    exp_keys = set(exp_map)
    act_keys = set(act_map)
    missing_keys = _sorted(exp_keys - act_keys)
    extra_keys = _sorted(act_keys - exp_keys)
    common_keys = exp_keys & act_keys
    changed_keys = _sorted(k for k in common_keys if exp_map.get(k) != act_map.get(k))

    lines: list[str] = [
        f"{context} mismatch (modules expected={len(exp_keys)} actual={len(act_keys)})",
    ]

    if not missing_keys and not extra_keys and not changed_keys:
        lines.append("  (no module-key differences)")
    else:
        if missing_keys:
            lines.append(f"  missing modules ({len(missing_keys)}):")
            for key in missing_keys[:limit]:
                lines.append(f"    - {key} -> {exp_map[key]}")
            if len(missing_keys) > limit:
                lines.append(f"    ... ({len(missing_keys) - limit} more)")

        if extra_keys:
            lines.append(f"  extra modules ({len(extra_keys)}):")
            for key in extra_keys[:limit]:
                lines.append(f"    + {key} -> {act_map[key]}")
            if len(extra_keys) > limit:
                lines.append(f"    ... ({len(extra_keys) - limit} more)")

        if changed_keys:
            lines.append(f"  path changes ({len(changed_keys)}):")
            for key in changed_keys[:limit]:
                lines.append(f"    ~ {key}: {exp_map[key]} -> {act_map[key]}")
            if len(changed_keys) > limit:
                lines.append(f"    ... ({len(changed_keys) - limit} more)")

    if include_path_section:
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
                    noun="paths",
                    context=context,
                    limit=limit,
                )
            )

    if include_unified_diff:
        lines.append("")
        exp_lines = [f"{key} -> {value}" for key, value in sorted(exp_map.items())]
        act_lines = [f"{key} -> {value}" for key, value in sorted(act_map.items())]
        lines.append(
            format_unified_diff(
                exp_lines,
                act_lines,
                fromfile=f"{context} (expected)",
                tofile=f"{context} (actual)",
            )
        )

    return "\n".join(lines)


__all__ = [
    "DEFAULT_DIFF_LIMIT",
    "HasModuleAndPath",
    "format_missing_extra",
    "format_module_map_diff",
    "format_unified_diff",
    "module_map_from_path_map",
    "module_map_from_records",
]
```

---

## 2) Re-export in `tests/_helpers/assertions/__init__.py` (recommended)

Add:

```python
from tests._helpers.assertions.golden_diffs import (
    DEFAULT_DIFF_LIMIT,
    format_missing_extra,
    format_module_map_diff,
    format_unified_diff,
    module_map_from_path_map,
    module_map_from_records,
)
```

…and include these in `__all__` so tests can just do:

```python
from tests._helpers.assertions import format_module_map_diff
```

---

## 3) Example usage in “modules-first” tests

### A) Path list diff (simple + clean)

```python
import pytest
from tests._helpers.assertions import format_missing_extra

if rel_paths != expected_paths:
    pytest.fail(
        format_missing_extra(
            expected_paths,
            rel_paths,
            noun="module paths",
            context="module inventory",
        )
    )
```

### B) Module map diff (modules + paths + moved keys)

```python
import pytest
from tests._helpers.assertions import format_module_map_diff, module_map_from_records

expected_map = {"src.pkg.a": "src/pkg/a.py", "src.pkg.b": "src/pkg/b.py"}
actual_map = module_map_from_records(records)  # records are ModuleRecord-like

if actual_map != expected_map:
    pytest.fail(
        format_module_map_diff(
            expected_map,
            actual_map,
            context="core.repo_map.modules",
            include_path_section=True,
            include_unified_diff=True,   # optional “git-style” section at bottom
        )
    )
```

---

### Where this fits with your earlier “ModulesAssertions” mini-helper

If you already added/plan to add something like `assert_modules_equal(...)`, the best move is:

* keep `assert_modules_equal(...)` as the “public” assertion helper
* have it call `format_module_map_diff(...)` when it fails

That gives you **concise tests** and **high-signal failures**.

If you want, I can also show the tiny `assert_modules_equal(...)` wrapper wired to this formatter (so individual tests never have to write the `if ...: pytest.fail(...)` boilerplate).
