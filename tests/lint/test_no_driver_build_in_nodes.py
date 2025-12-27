"""Ensure runtime/driver construction is blocked in DAG nodes."""

from __future__ import annotations

from pathlib import Path

from tools.lint_no_driver_build_in_nodes import main


def test_no_driver_build_in_nodes() -> None:
    """Ensure the node driver-build lint passes for the repo."""
    repo_root = Path(__file__).resolve().parents[2]
    exit_code = main([str(repo_root)])
    assert exit_code == 0
