"""CLI golden snapshot tests for Hamilton Phase 2.

This module provides parametrized snapshot tests driven by a YAML/JSON manifest.
Each test case executes a CLI command and compares output against golden files.

Usage
-----
Run all snapshot tests:
    pytest -m cli_snapshot

Update snapshots:
    pytest -m cli_snapshot --update-cli-snapshots

Filter by tags:
    pytest -m cli_snapshot --cli-snapshot-tags pr14,graph

Filter by pattern:
    pytest -m cli_snapshot --cli-snapshot-pattern "pr14_*"

List available cases:
    pytest -m cli_snapshot --list-cli-snapshots
"""

from __future__ import annotations

import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.build.hamilton.snapshots._manifest import SnapshotCase, load_snapshot_manifest
from tests.build.hamilton.snapshots._runner import execute_and_assert_snapshot

if TYPE_CHECKING:
    from tests.build.hamilton.snapshots._manifest import SnapshotManifest


def _default_snapshots_dir() -> Path:
    """Return the default snapshots directory path.

    Returns
    -------
    Path
        Path to the snapshots directory.
    """
    return Path(__file__).parent / "snapshots"


def _manifest_path(config: pytest.Config) -> Path:
    """Determine manifest path from config or default.

    Parameters
    ----------
    config
        Pytest configuration object.

    Returns
    -------
    Path
        Path to the manifest file.
    """
    override = config.getoption("--cli-snapshot-manifest")
    if override:
        return Path(override)

    snapshots_dir = _default_snapshots_dir()
    manifest_yaml = snapshots_dir / "manifest.yaml"
    manifest_json = snapshots_dir / "manifest.json"
    return manifest_yaml if manifest_yaml.exists() else manifest_json


@lru_cache(maxsize=8)
def _load_manifest_cached(manifest_path_str: str) -> SnapshotManifest:
    """Load and cache manifest from path string.

    Parameters
    ----------
    manifest_path_str
        String path to manifest (for hashable caching).

    Returns
    -------
    SnapshotManifest
        Loaded manifest.
    """
    path = Path(manifest_path_str)
    return load_snapshot_manifest(path)


def _parse_csv_opt(value: str | None) -> set[str]:
    """Parse comma-separated option value to set.

    Parameters
    ----------
    value
        Comma-separated string or None.

    Returns
    -------
    set[str]
        Set of trimmed non-empty values.
    """
    if not value:
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def _parse_patterns(value: str | None) -> list[str]:
    """Parse comma-separated glob patterns.

    Parameters
    ----------
    value
        Comma-separated pattern string or None.

    Returns
    -------
    list[str]
        List of trimmed non-empty patterns.
    """
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def _select_cases(
    *,
    cases: tuple[SnapshotCase, ...],
    tags: set[str],
    patterns: list[str],
) -> list[SnapshotCase]:
    """Filter cases by tags and patterns.

    Parameters
    ----------
    cases
        All cases from manifest.
    tags
        Tag filter set (empty means no filter).
    patterns
        Glob patterns for case names (empty means no filter).

    Returns
    -------
    list[SnapshotCase]
        Filtered cases.
    """
    selected: list[SnapshotCase] = []
    for c in cases:
        # Tag filtering - case must have at least one matching tag
        if tags:
            case_tags = set(c.tags)
            if case_tags.isdisjoint(tags):
                continue

        # Pattern filtering - case name must match at least one pattern
        if patterns and not any(fnmatch.fnmatch(c.name, pat) for pat in patterns):
            continue

        selected.append(c)
    return selected


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize snapshot cases dynamically from the manifest.

    This enables per-case reporting while respecting config overrides.

    Parameters
    ----------
    metafunc
        Pytest metafunc for parametrization.
    """
    if "snapshot_case" not in metafunc.fixturenames:
        return

    manifest_path = _manifest_path(metafunc.config)
    if not manifest_path.exists():
        # No manifest - skip parametrization
        return

    manifest = _load_manifest_cached(str(manifest_path))

    tag_filter = _parse_csv_opt(metafunc.config.getoption("--cli-snapshot-tags"))
    patterns = _parse_patterns(metafunc.config.getoption("--cli-snapshot-pattern"))

    selected = _select_cases(cases=manifest.cases, tags=tag_filter, patterns=patterns)
    metafunc.parametrize(
        "snapshot_case",
        selected,
        ids=[c.name for c in selected],
    )


@pytest.fixture(scope="session")
def cli_snapshot_context(request: pytest.FixtureRequest) -> tuple[Path, SnapshotManifest]:
    """Shared context (manifest + snapshot dir) for parametrized cases.

    Parameters
    ----------
    request
        Pytest fixture request.

    Returns
    -------
    tuple[Path, SnapshotManifest]
        Snapshots directory and loaded manifest.
    """
    snapshots_dir = _default_snapshots_dir()
    manifest_path = _manifest_path(request.config)
    manifest = _load_manifest_cached(str(manifest_path))
    return snapshots_dir, manifest


@pytest.mark.cli_snapshot
def test_cli_snapshot(
    snapshot_case: SnapshotCase,
    cli_snapshot_context: tuple[Path, SnapshotManifest],
    request: pytest.FixtureRequest,
) -> None:
    """Execute CLI command and compare output to golden snapshot.

    Parameters
    ----------
    snapshot_case
        Test case from manifest.
    cli_snapshot_context
        Shared context with manifest and directory.
    request
        Pytest fixture request for config access.
    """
    snapshots_dir, manifest = cli_snapshot_context
    update = bool(request.config.getoption("--update-cli-snapshots"))

    execute_and_assert_snapshot(
        manifest=manifest,
        snapshots_dir=snapshots_dir,
        case=snapshot_case,
        update=update,
    )


__all__ = [
    "cli_snapshot_context",
    "pytest_generate_tests",
    "test_cli_snapshot",
]
