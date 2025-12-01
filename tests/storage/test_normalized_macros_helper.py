"""Helper to detect drift between normalized macro mapping and DuckDB definitions."""

from __future__ import annotations

import re

import pytest

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.metadata_bootstrap import (
    DATASET_ROWS_ONLY,
    METADATA_SCHEMA_DDL,
    validate_normalized_macro_schemas,
)
from codeintel.storage.metadata_bootstrap import NORMALIZED_MACROS as BOOTSTRAP_MACROS
from codeintel.storage.sql_helpers import safe_macro_call

pytestmark = pytest.mark.smoke


def test_normalized_macros_defined(fresh_gateway: StorageGateway) -> None:
    """Ensure every macro referenced in NORMALIZED_MACROS exists in DuckDB."""
    _ = fresh_gateway  # Gateway ensures bootstrap runs.

    ddl_text = "\n".join(METADATA_SCHEMA_DDL)
    defined = {
        match.group(1).lower()
        for match in re.finditer(
            r"CREATE\s+OR\s+REPLACE\s+MACRO\s+([\w\.]+)", ddl_text, re.IGNORECASE
        )
    }
    missing: list[str] = []
    for macro in sorted(set(NORMALIZED_MACROS.values())):
        macro_lower = macro.lower()
        macro_name = macro_lower.split(".")[-1]
        if macro_lower not in defined and macro_name not in defined:
            missing.append(macro)
    if missing:
        message = "Missing normalized macros: " + ", ".join(missing)
        pytest.fail(message)


def test_normalized_macros_match_expected_sets() -> None:
    """Catch drift when adding datasets without macros or allowlisting explicitly."""
    datasets = set(TABLE_SCHEMAS)
    macro_backed = set(NORMALIZED_MACROS)
    dataset_rows_only = set(DATASET_ROWS_ONLY)
    unexpected_dataset_rows = datasets - macro_backed - dataset_rows_only
    if unexpected_dataset_rows:
        message = "Datasets missing normalized macros or allowlist entries: " + ", ".join(
            sorted(unexpected_dataset_rows)
        )
        pytest.fail(message)


def test_normalized_macro_schema_validation(fresh_gateway: StorageGateway) -> None:
    """Ensure schema validation helper raises on drift (no drift expected)."""
    con = fresh_gateway.con
    # Provide a sanity check that the helper executes successfully.
    validate_normalized_macro_schemas(con)
    # Keep export mapping aligned with bootstrap mapping.
    if set(BOOTSTRAP_MACROS) != set(NORMALIZED_MACROS):
        pytest.fail("Export and bootstrap macro mappings diverged")


def test_dataset_rows_only_tables_parse(fresh_gateway: StorageGateway) -> None:
    """Ensure dataset_rows-only tables at least parse with zero-row selects."""
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key in DATASET_ROWS_ONLY:
        try:
            sql, params = safe_macro_call(
                "metadata.dataset_rows", [table_key, 0, 0], allowed={"metadata.dataset_rows"}
            )
            con.sql(sql, params=params).fetchall()
        except (DuckDBError, RuntimeError, ValueError) as exc:
            failures.append(f"{table_key}: {exc}")
    if failures:
        pytest.fail("; ".join(failures))
