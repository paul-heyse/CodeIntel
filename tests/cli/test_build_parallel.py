"""Tests for build parallel execution flags.

This validates that the CLI wiring for parallel execution backends remains
stable and that --max-workers behaves as documented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.cli.handlers.build import resolve_parallel_backend
from tests._helpers.assertions import expect_equal, expect_in
from tests._helpers.cli import assert_success

if TYPE_CHECKING:
    from tests._helpers.cli_project import CLIProjectHarness

pytestmark = pytest.mark.xdist_group("cli_shared_flags")


def test_build_run_with_threadpool_backend(cli_project_harness: CLIProjectHarness) -> None:
    """Build run should execute successfully with threadpool backend."""
    result = cli_project_harness.invoke(
        [
            "build",
            "run",
            "--parallel-backend=threadpool",
            "--max-workers=2",
            "ast",
        ]
    )
    assert_success(result)

    expect_in("executed:", result.stdout, label="executed marker")
    expect_in("ast", result.stdout, label="target listed")


def test_max_workers_implies_threadpool_backend() -> None:
    """--max-workers should upgrade sequential backend to threadpool."""
    backend = resolve_parallel_backend(parallel_backend=None, max_workers=2)
    expect_equal(backend, "threadpool", label="parallel_backend")
