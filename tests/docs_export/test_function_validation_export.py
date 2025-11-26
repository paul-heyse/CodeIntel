"""Ensure function_validation exports include the new dataset."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.pipeline.export.export_jsonl import export_all_jsonl
from codeintel.pipeline.export.export_parquet import export_all_parquet
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import FunctionValidationRow, insert_function_validation


def test_function_validation_export(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """Export writes function_validation artifacts when rows exist."""
    gateway = fresh_gateway

    insert_function_validation(
        gateway,
        [
            FunctionValidationRow(
                repo="demo/repo",
                commit="deadbeef",
                function_goid_h128=123,
                rel_path="mod.py",
                qualname="pkg.mod.func",
                issue="span_not_found",
                detail="Span 1-2",
                created_at=datetime.now(UTC),
            )
        ],
    )

    jsonl_dir = tmp_path / "jsonl"
    parquet_dir = tmp_path / "parquet"
    export_all_jsonl(gateway, jsonl_dir)
    export_all_parquet(gateway, parquet_dir)

    jsonl_file = jsonl_dir / "function_validation.jsonl"
    parquet_file = parquet_dir / "function_validation.parquet"
    if not jsonl_file.exists():
        pytest.fail("function_validation.jsonl not exported")
    if not parquet_file.exists():
        pytest.fail("function_validation.parquet not exported")

    with jsonl_file.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines:
        pytest.fail("exported JSONL is empty")
    row = lines[0]
    for key, expected in {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "rel_path": "mod.py",
        "qualname": "pkg.mod.func",
        "issue": "span_not_found",
        "detail": "Span 1-2",
    }.items():
        if row.get(key) != expected:
            pytest.fail(f"unexpected value for {key}: {row.get(key)}")
    if "created_at" not in row:
        pytest.fail("created_at missing in export row")
