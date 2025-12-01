"""Validate normalized macro output schemas align with table schemas."""

from __future__ import annotations

import pytest

from codeintel.config.dataset_contract import DATASET_CONTRACTS_BY_TABLE_KEY
from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql_helpers import safe_macro_call


def _canonical_type(type_str: str) -> str:
    upper = type_str.upper()
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper


@pytest.mark.smoke
def test_macro_schemas_match_table_definitions(fresh_gateway: StorageGateway) -> None:
    """
    Normalized macro outputs should align with DatasetContract schema definitions.

    Raises
    ------
    AssertionError
        If any macro output schema deviates from the contract schema.
    """
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None or contract.schema is None:
            failures.append(f"{table_key}: no contract schema found")
            continue
        schema = contract.schema
        sql, params = safe_macro_call(
            macro, [table_key, 0, 0], allowed=set(NORMALIZED_MACROS.values())
        )
        rel = con.sql(sql, params=params)
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
