"""Helpers for expressing module-inventory expectations from real repo trees.

These helpers intentionally reuse production scanning/filtering code to keep
tests realistic and aligned with the `modules` Hamilton target behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import build_scan_profile, filter_modules
from codeintel.build.hamilton.native.ingestion.ingest_targets import MODULES_TARGET_NAME
from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.ingestion.adapters.filesystem_discovery import FilesystemDiscoveryAdapter


class RepoFixtureLike(Protocol):
    """Protocol for repo fixtures that expose module expectations."""

    module_map: Mapping[str, str]


class RepoFixtureSpecLike(Protocol):
    """Protocol for repo fixture specifications."""

    repo_root: Path | None
    module_map: Mapping[str, str] | None


type RepoRootLike = Path | RepoFixtureLike | RepoFixtureSpecLike


def _module_map_from_fixture(repo_root: RepoRootLike) -> dict[str, str] | None:
    if isinstance(repo_root, Path):
        return None

    module_map = getattr(repo_root, "module_map", None)
    if module_map is None:
        return None
    return dict(cast("Mapping[str, str]", module_map))


def _resolve_repo_root(repo_root: RepoRootLike) -> Path:
    if isinstance(repo_root, Path):
        return repo_root

    repo_root_value = getattr(repo_root, "repo_root", None)
    if repo_root_value is None:
        message = "RepoFixtureSpec.repo_root is required when module_map is missing"
        raise ValueError(message)
    if isinstance(repo_root_value, Path):
        return repo_root_value
    message = "repo_root must be a Path or RepoFixtureSpec"
    raise TypeError(message)


def modules_expected_from_repo_tree(
    repo_root: RepoRootLike,
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
        Root directory or a repo fixture/spec that already carries module expectations.
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
    fixture_map = _module_map_from_fixture(repo_root)
    if fixture_map is not None:
        return dict(sorted(fixture_map.items(), key=lambda kv: kv[0]))

    resolved_root = _resolve_repo_root(repo_root)
    resolved = options or ModuleIngestOptions()
    profile = build_scan_profile(resolved_root, resolved)

    discovered = FilesystemDiscoveryAdapter.discover_modules(resolved_root, profile)
    filtered = filter_modules(discovered, resolved)

    module_map = {record.rel_path: record.module_name for record in filtered}
    return dict(sorted(module_map.items(), key=lambda kv: kv[0]))


def module_paths_expected_from_repo_tree(
    repo_root: RepoRootLike,
    *,
    options: ModuleIngestOptions | None = None,
) -> list[str]:
    """Compute sorted expected module paths from a repository tree.

    Returns
    -------
    list[str]
        Sorted module paths derived from the repo tree.
    """
    return sorted(modules_expected_from_repo_tree(repo_root, options=options).keys())


def modules_expected_from_env(env: BuildEnv) -> dict[str, str]:
    """Compute expected modules using active config/CLI target options.

    Parameters
    ----------
    env:
        Build environment with config and snapshot metadata.

    Returns
    -------
    dict[str, str]
        Mapping of relative POSIX file path -> module name.
    """
    options = load_target_options(
        env,
        target_name=MODULES_TARGET_NAME,
        options_type=ModuleIngestOptions,
    )
    return modules_expected_from_repo_tree(env.snapshot.repo_root, options=options)


__all__ = [
    "module_paths_expected_from_repo_tree",
    "modules_expected_from_env",
    "modules_expected_from_repo_tree",
]
