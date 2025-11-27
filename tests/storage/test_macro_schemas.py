"""Validate normalized macro output schemas align with table schemas."""

from __future__ import annotations

import pytest

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import StorageGateway


def _canonical_type(type_str: str) -> str:
    upper = type_str.upper()
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


@pytest.mark.smoke
def test_macro_schemas_match_table_definitions(fresh_gateway: StorageGateway) -> None:
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        schema = TABLE_SCHEMAS[table_key]
        rel = con.sql(
            f"SELECT * FROM {macro}(?, ?, ?)",  # noqa: S608 - trusted macro name
            params=[table_key, 0, 0],
        )
        actual: dict[str, str] = {}
        for name, dtype in zip(rel.columns, rel.dtypes, strict=False):
            if name.endswith("_1"):
                continue
            actual[name] = _canonical_type(str(dtype))

        expected = {col.name: _canonical_type(col.type) for col in schema.columns}

        missing = expected.keys() - actual.keys()
        if missing:
            failures.append(f"{table_key}: missing columns {sorted(missing)}")
            continue
        for col_name, expected_type in expected.items():
            actual_type = actual[col_name]
            if expected_type in {"TIMESTAMP", "DATE"} and actual_type == "VARCHAR":
                continue
            if actual_type != expected_type:
                failures.append(f"{table_key}.{col_name}: {actual_type} != {expected_type}")
    if failures:
        raise AssertionError("; ".join(failures))
