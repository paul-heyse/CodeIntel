"""Conformance CLI and helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from codeintel.storage.conformance import run_conformance
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
from codeintel.storage.schemas import apply_all_schemas


def test_conformance_passes_with_empty_db(tmp_path: Path) -> None:
    """Conformance should succeed when the catalog is freshly bootstrapped."""
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    bootstrap_metadata_datasets(con)
    report = run_conformance(con, schema_base_dir=tmp_path, sample_rows=False)
    if not report.ok:
        pytest.fail(f"Unexpected contract issues: {[issue.message for issue in report.issues]}")
