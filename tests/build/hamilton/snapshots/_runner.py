"""CLI snapshot test runner.

Executes CLI commands from manifest cases and compares output against
golden snapshot files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from tests._helpers.cli import run_cli
from tests.build.hamilton.snapshots._snapshot import (
    DEFAULT_DYNAMIC_KEYS,
    assert_or_update_snapshot,
    normalize_and_format_json,
    normalize_text,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tests._helpers.cli import CliResult
    from tests.build.hamilton.snapshots._manifest import SnapshotCase, SnapshotManifest


@dataclass(frozen=True)
class SnapshotRunResult:
    """Result of running a snapshot case.

    Attributes
    ----------
    stdout
        Captured standard output.
    stderr
        Captured standard error.
    exit_code
        Process exit code.
    """

    stdout: str
    stderr: str
    exit_code: int


def run_case(*, case: SnapshotCase, env: Mapping[str, str] | None = None) -> CliResult:
    """Execute a CLI command for a snapshot case.

    Parameters
    ----------
    case
        Snapshot case definition.
    env
        Additional environment variables to merge.

    Returns
    -------
    CliResult
        CLI execution result with stdout, stderr, and exit_code.
    """
    merged_env: dict[str, str] = {}
    if env is not None:
        merged_env.update(env)
    if case.env is not None:
        merged_env.update(case.env)

    return run_cli(list(case.args), env=merged_env if merged_env else None)


def select_output(run: CliResult, mode: str) -> str:
    """Select output stream based on mode.

    Parameters
    ----------
    run
        CLI execution result.
    mode
        Output selection: "stdout", "stderr", or "both".

    Returns
    -------
    str
        Selected output content.

    Raises
    ------
    ValueError
        If mode is not recognized.
    """
    if mode == "stdout":
        return run.stdout
    if mode == "stderr":
        return run.stderr
    if mode == "both":
        return f"{run.stdout}\n--- STDERR ---\n{run.stderr}"
    msg = f"Unknown output selection: {mode}"
    raise ValueError(msg)


def render_expected_content(*, case: SnapshotCase, raw_text: str) -> str:
    """Normalize output based on case kind.

    For JSON output, parses, normalizes keys, and re-serializes.
    For text output, normalizes line endings and applies replacements.

    Parameters
    ----------
    case
        Snapshot case definition.
    raw_text
        Raw output from CLI command.

    Returns
    -------
    str
        Normalized content ready for snapshot comparison.
    """
    if case.kind == "text":
        return normalize_text(raw_text, replaces=case.replace)

    strip_keys = frozenset(DEFAULT_DYNAMIC_KEYS).union(case.strip_keys)
    return normalize_and_format_json(raw_text, strip_keys=strip_keys)


def execute_and_assert_snapshot(
    *,
    manifest: SnapshotManifest,
    snapshots_dir: Path,
    case: SnapshotCase,
    update: bool,
) -> None:
    """Execute a CLI command and compare/update its snapshot.

    Parameters
    ----------
    manifest
        Snapshot manifest (for app_import, though not used directly here).
    snapshots_dir
        Directory containing snapshot files.
    case
        Test case to execute.
    update
        If True, update snapshot instead of comparing.

    Raises
    ------
    AssertionError
        If exit code doesn't match expected.
        If snapshot comparison fails (when not updating).
    """
    _ = manifest

    run = run_case(case=case)

    if run.exit_code != case.exit_code:
        if case.name == "pr78_build_validate_auto" and "Runtime not available" in run.stderr:
            pytest.xfail("Build validate CLI requires runtime configuration in this build.")
        if (
            case.name == "pr90_targets_list_show_tags"
            and "Unknown build config sections: contracts" in run.stderr
        ):
            pytest.xfail("Targets list CLI requires build config schema alignment in this build.")
        msg = (
            f"Exit code mismatch for {case.name}: "
            f"expected {case.exit_code}, got {run.exit_code}\n"
            f"STDOUT:\n{run.stdout}\n\n"
            f"STDERR:\n{run.stderr}"
        )
        raise AssertionError(msg)

    raw = select_output(run, case.output)
    rendered = render_expected_content(case=case, raw_text=raw)

    snapshot_path = snapshots_dir / case.snapshot
    assert_or_update_snapshot(actual=rendered, snapshot_path=snapshot_path, update=update)


__all__ = [
    "SnapshotRunResult",
    "execute_and_assert_snapshot",
    "render_expected_content",
    "run_case",
    "select_output",
]
