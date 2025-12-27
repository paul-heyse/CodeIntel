"""Ensure manual tag scans are blocked by lint."""

from __future__ import annotations

from pathlib import Path

from tools.lint_no_manual_tag_scans import main


def test_no_manual_tag_scans() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    exit_code = main([str(repo_root)])
    assert exit_code == 0
