"""Ensure require_normalized_macros enforcement works for exports."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.pipeline.export.export_jsonl import export_jsonl_for_table
from codeintel.pipeline.export.export_parquet import export_parquet_for_table
from codeintel.storage.gateway import StorageGateway


def test_require_macros_allows_macro_backed_tables(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Macro-backed datasets export successfully when enforcement is enabled."""
    output = tmp_path / "function_metrics.jsonl"
    export_jsonl_for_table(
        fresh_gateway,
        "analytics.function_metrics",
        output,
        require_normalized_macros=True,
    )
    if not output.exists():
        pytest.fail("Expected JSONL export output to be written")


def test_require_macros_rejects_dataset_rows_only(
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Dataset_rows-only tables are rejected when macros are required."""
    with pytest.raises(ValueError, match="No normalized macro"):
        export_jsonl_for_table(
            fresh_gateway,
            "core.goids",
            tmp_path / "goids.jsonl",
            require_normalized_macros=True,
        )
    with pytest.raises(ValueError, match="No normalized macro"):
        export_parquet_for_table(
            fresh_gateway,
            "core.goids",
            tmp_path / "goids.parquet",
            require_normalized_macros=True,
        )
