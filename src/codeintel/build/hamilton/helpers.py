"""Shared helper utilities for Hamilton native modules.

Provides common functionality used across multiple native Hamilton implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.duckdb_types import DuckDBError
from codeintel.ingestion.infrastructure.scanning import ScanProfile, default_code_profile
from codeintel.ingestion.ports.discovery import ModuleRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
    from codeintel.core.gateway import BuildGateway

__all__ = [
    "build_scan_profile",
    "filter_mapping",
    "filter_modules",
    "filter_paths",
    "get_module_paths_from_env",
    "get_source_root",
    "is_test_path",
    "paths_to_modules",
]

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Path Filtering Utilities
# -----------------------------------------------------------------------------


def is_test_path(path: str) -> bool:
    """Check whether a path appears to be a test file.

    Use common Python test file naming conventions to detect test files:
    - Files in a ``tests/`` directory
    - Files ending in ``_test.py``
    - Files containing ``/test_`` in the path
    - Files starting with ``test_``

    Parameters
    ----------
    path
        Relative file path to check.

    Returns
    -------
    bool
        True if the path matches test file patterns.

    Examples
    --------
    >>> is_test_path("tests/test_module.py")
    True
    >>> is_test_path("src/module.py")
    False
    >>> is_test_path("test_utils.py")
    True
    """
    lowered = path.lower()
    return (
        "tests/" in lowered
        or lowered.endswith("_test.py")
        or "/test_" in lowered
        or lowered.startswith("test_")
    )


def filter_paths(
    paths: Iterable[str],
    *,
    scope_paths: list[str] | None = None,
    include_tests: bool = True,
) -> list[str]:
    """Filter paths by scope and test inclusion.

    Provide a unified path filtering mechanism for targets that need to
    restrict processing to specific directories and optionally exclude test files.

    Parameters
    ----------
    paths
        Paths to filter.
    scope_paths
        Optional list of path prefixes to include. If None or empty,
        all paths are included.
    include_tests
        Whether to include test files. Uses ``is_test_path()`` for detection.
        Defaults to True.

    Returns
    -------
    list[str]
        Filtered list of paths.

    Examples
    --------
    Filter to a specific directory:

    >>> filter_paths(["src/a.py", "lib/b.py"], scope_paths=["src/"])
    ['src/a.py']

    Exclude test files:

    >>> filter_paths(["src/main.py", "tests/test_main.py"], include_tests=False)
    ['src/main.py']
    """
    result = list(paths)

    if scope_paths:
        prefixes = tuple(scope_paths)
        result = [path for path in result if path.startswith(prefixes)]

    if not include_tests:
        result = [path for path in result if not is_test_path(path)]

    return result


def filter_mapping[T](
    mapping: Mapping[str, T],
    *,
    scope_paths: list[str] | None = None,
    include_tests: bool = True,
) -> dict[str, T]:
    """Filter a path-keyed mapping by scope and test inclusion.

    Provide a unified filtering mechanism for dict-based data structures
    where keys are relative file paths.

    Parameters
    ----------
    mapping
        Mapping with relative paths as keys.
    scope_paths
        Optional list of path prefixes to include. If None or empty,
        all paths are included.
    include_tests
        Whether to include test paths. Uses ``is_test_path()`` for detection.
        Defaults to True.

    Returns
    -------
    dict[str, T]
        Filtered mapping containing only matching entries.

    Examples
    --------
    Filter module map by scope:

    >>> modules = {"src/a.py": "a", "lib/b.py": "b"}
    >>> filter_mapping(modules, scope_paths=["src/"])
    {'src/a.py': 'a'}

    Exclude test files:

    >>> paths = {"src/main.py": 1, "tests/test_main.py": 2}
    >>> filter_mapping(paths, include_tests=False)
    {'src/main.py': 1}
    """
    result = dict(mapping)

    if scope_paths:
        prefixes = tuple(scope_paths)
        result = {k: v for k, v in result.items() if k.startswith(prefixes)}

    if not include_tests:
        result = {k: v for k, v in result.items() if not is_test_path(k)}

    return result


# -----------------------------------------------------------------------------
# Source Root and Gateway Utilities
# -----------------------------------------------------------------------------


def get_source_root(
    gateway: BuildGateway,
    repo: str,
    commit: str,
    *,
    fallback: Path | None = None,
) -> Path:
    """Retrieve source root from core.snapshots with fallback.

    Look up the source root for the given repository snapshot from the
    core.snapshots table. Return a fallback path if not found.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    repo
        Repository identifier.
    commit
        Commit SHA.
    fallback
        Fallback path if not found. Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        Absolute path to the source root.
    """
    try:
        row = gateway.execute(
            "SELECT source_root FROM core.snapshots WHERE repo = ? AND commit = ? LIMIT 1",
            [repo, commit],
        ).fetchone()
        if row is not None:
            value = row[0]
            if value:
                return Path(str(value))
    except DuckDBError as exc:
        log.debug("get_source_root: Could not get source root: %s", exc)
    return fallback or Path.cwd()


def get_module_paths_from_env(env: BuildEnv) -> list[str]:
    """Fetch module paths from BuildEnv gateway.

    Query the core.modules table to retrieve module paths for the current
    repository and commit.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.

    Returns
    -------
    list[str]
        Module paths from storage; empty when unavailable.
    """
    try:
        reader = env.gateway.execute(
            "SELECT path FROM core.modules WHERE repo = ? AND commit = ?",
            [env.snapshot.repo, env.snapshot.commit],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        paths: list[str] = []
        for batch in reader:
            values = [str(value) for value in batch.column(0).to_pylist() if value is not None]
            paths.extend(values)
    except (RuntimeError, OSError, DuckDBError) as exc:
        log.warning("gateway error fetching module paths: %s", exc)
        return []
    else:
        return paths


# -----------------------------------------------------------------------------
# Module Ingestion Utilities
# -----------------------------------------------------------------------------


def paths_to_modules(paths: Sequence[str], repo_root: Path) -> list[ModuleRecord]:
    """Convert relative paths to ModuleRecord objects with metadata.

    Parameters
    ----------
    paths
        Sequence of relative file paths.
    repo_root
        Root directory of the repository.

    Returns
    -------
    list[ModuleRecord]
        Module metadata for each provided path in order.
    """
    total = len(paths)
    return [
        ModuleRecord(
            rel_path=path,
            module_name=path.replace("/", ".").removesuffix(".py"),
            file_path=repo_root / path,
            index=i + 1,
            total=total,
        )
        for i, path in enumerate(paths)
    ]


def build_scan_profile(repo_root: Path, options: ModuleIngestOptions) -> ScanProfile:
    """Build a scan profile that respects configured scope and ignore settings.

    Parameters
    ----------
    repo_root
        Root directory of the repository.
    options
        Module ingestion options with scope and filter settings.

    Returns
    -------
    ScanProfile
        Scan profile with option-driven roots and ignores applied.
    """
    base_profile = default_code_profile(repo_root)

    ignore_dirs = list(base_profile.ignore_dirs)
    if not options.include_tests and "tests" not in ignore_dirs:
        ignore_dirs.append("tests")
    if not options.include_generated:
        for name in ("generated", ".generated"):
            if name not in ignore_dirs:
                ignore_dirs.append(name)

    source_roots = base_profile.source_roots
    if options.scope_paths:
        resolved_roots: list[Path] = []
        for scope in options.scope_paths:
            scope_path = Path(scope)
            resolved = scope_path if scope_path.is_absolute() else repo_root / scope_path
            if resolved.is_file():
                resolved = resolved.parent
            if resolved.is_dir() and resolved not in resolved_roots:
                resolved_roots.append(resolved)
        if resolved_roots:
            source_roots = tuple(resolved_roots)

    return ScanProfile(
        repo_root=repo_root,
        source_roots=source_roots,
        include_globs=base_profile.include_globs,
        ignore_dirs=tuple(ignore_dirs),
        log_every=base_profile.log_every,
        log_interval=base_profile.log_interval,
    )


def _is_generated_path(rel_path: str) -> bool:
    """Return True when the path looks like a generated artifact.

    Parameters
    ----------
    rel_path
        Relative path to check.

    Returns
    -------
    bool
        True when path appears generated.
    """
    path = Path(rel_path)
    lower_parts = [part.lower() for part in path.parts]
    if any(part in {"generated", ".generated"} for part in lower_parts):
        return True
    filename = path.name.lower()
    return filename.endswith(("_generated.py", ".generated.py", "_pb2.py", "_pb2.pyi"))


def _is_in_scope(rel_path: str, scope_paths: Sequence[str]) -> bool:
    """Check whether a path is contained within any configured scope prefix.

    Parameters
    ----------
    rel_path
        Relative path to check.
    scope_paths
        List of scope path prefixes.

    Returns
    -------
    bool
        True when the path resides under a configured scope.
    """
    if not scope_paths:
        return True
    rel_parts = Path(rel_path).parts
    for scope in scope_paths:
        scope_parts = Path(scope).parts
        if scope_parts and rel_parts[: len(scope_parts)] == scope_parts:
            return True
    return False


def filter_modules(
    modules: Sequence[ModuleRecord],
    options: ModuleIngestOptions,
) -> list[ModuleRecord]:
    """Apply scope, test, generated, and size filters to discovered modules.

    Parameters
    ----------
    modules
        Modules to filter.
    options
        Module ingestion options with filter settings.

    Returns
    -------
    list[ModuleRecord]
        Modules that satisfy configured options.
    """
    filtered: list[ModuleRecord] = []
    for module in modules:
        rel_path = module.rel_path
        if options.scope_paths and not _is_in_scope(rel_path, options.scope_paths):
            continue
        if not options.include_tests and is_test_path(rel_path):
            continue
        if not options.include_generated and _is_generated_path(rel_path):
            continue
        if options.max_file_size_kb > 0:
            try:
                size_bytes = module.file_path.stat().st_size
            except OSError:
                continue
            if size_bytes > options.max_file_size_kb * 1024:
                continue
        filtered.append(module)
    return filtered
