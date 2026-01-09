"""Ensure raw pyarrow.compute imports are blocked in build/ingestion nodes."""

from __future__ import annotations

from pathlib import Path

from tools.lint_no_raw_pyarrow_compute_in_nodes import main


def test_no_raw_pyarrow_compute_in_nodes() -> None:
    """Ensure raw pyarrow.compute imports are disallowed in nodes."""
    repo_root = Path(__file__).resolve().parents[2]
    exit_code = main([str(repo_root)])
    assert exit_code == 0
