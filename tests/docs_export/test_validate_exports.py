"""Tests for validate_exports tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.docs_export.validate_exports import main


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_validate_jsonl_happy_path(tmp_path: Path) -> None:
    """Validator should return 0 for conforming JSONL rows."""
    data_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": 1,
                "callee_goid_h128": 2,
                "callsite_path": "a.py",
                "language": "python",
            }
        ],
    )
    exit_code = main(["--schema", "call_graph_edges", str(data_path)])
    if exit_code != 0:
        message = f"Expected success, got exit code {exit_code}"
        pytest.fail(message)


def test_validate_jsonl_failure(tmp_path: Path) -> None:
    """Validator should fail when required fields are missing."""
    data_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "commit": "c",
                "caller_goid_h128": 1,
                "callee_goid_h128": 2,
            }
        ],
    )
    exit_code = main(["--schema", "call_graph_edges", str(data_path)])
    if exit_code == 0:
        message = "Expected validation failure for missing repo field"
        pytest.fail(message)
