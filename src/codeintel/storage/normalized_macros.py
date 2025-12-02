"""Generate stub normalized macros from TABLE_SCHEMAS for quick authoring.

Usage
-----
python -m codeintel.storage.normalized_macros [table_key ...]

Emits DuckDB macro DDL to stdout. Casting rules are intentionally simple:
- Columns containing ``goid_h128`` are cast to BIGINT.
- TIMESTAMP/DATE columns are cast to VARCHAR to keep exports stable.
- Other columns are selected as-is.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import NamedTuple

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY

DEFAULT_LIMIT = 9_223_372_036_854_775_807


class RenderedMacro(NamedTuple):
    """Container for macro name and rendered SQL."""

    macro_name: str
    ddl: str


def _cast_expression(col_name: str, duckdb_type: str) -> str:
    """
    Cast a column to match the declared TABLE_SCHEMAS type.

    Returns
    -------
    str
        SQL expression casting the column to the expected type.
    """
    upper_type = duckdb_type.upper()
    col_upper = col_name.upper()
    if "GOID_H128" in col_upper:
        return f"CAST(ds.{col_name} AS BIGINT) AS {col_name}"
    if "TIMESTAMPTZ" in upper_type or "TIMESTAMP WITH TIME ZONE" in upper_type:
        return f"CAST(ds.{col_name} AS TIMESTAMPTZ) AS {col_name}"
    if upper_type.startswith("TIMESTAMP"):
        return f"CAST(ds.{col_name} AS TIMESTAMP) AS {col_name}"
    if upper_type == "DATE":
        return f"CAST(ds.{col_name} AS DATE) AS {col_name}"
    return f"ds.{col_name}"


def render_macro(table_key: str, *, default_limit: int = DEFAULT_LIMIT) -> RenderedMacro:
    """
    Render a normalized macro DDL for the given table key.

    Parameters
    ----------
    table_key
        Fully qualified table key (schema.table) present in TABLE_SCHEMAS.
    default_limit
        Optional default limit used in the macro signature.

    Returns
    -------
    RenderedMacro
        Name and SQL DDL for the macro.

    Raises
    ------
    KeyError
        If the table key is unknown or lacks a declared schema in DatasetContract.
    """
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    if contract is None or contract.schema is None:
        message = f"Unknown table key or missing schema for normalized macro: {table_key}"
        raise KeyError(message)
    schema = contract.schema
    _, short_name = table_key.split(".", maxsplit=1)
    macro_name = "metadata.normalized_" + short_name
    select_list = ",\n        ".join(
        _cast_expression(column.name, column.type) for column in schema.columns
    )
    ddl_lines = [
        "CREATE OR REPLACE MACRO ",
        macro_name,
        "(\n    table_key TEXT,\n    row_limit BIGINT := ",
        str(default_limit),
        ",\n    row_offset BIGINT := 0\n) AS TABLE\nSELECT\n        ",
        select_list,
        "\nFROM metadata.dataset_rows(table_key, row_limit, row_offset) ds;",
    ]
    ddl = "".join(ddl_lines).strip()
    return RenderedMacro(macro_name=macro_name, ddl=ddl)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render stub normalized macros.")
    parser.add_argument(
        "tables",
        nargs="*",
        help="Optional list of table keys to render (defaults to all in TABLE_SCHEMAS).",
    )
    return parser.parse_args()


def _iter_tables(selected: Iterable[str]) -> list[str]:
    """
    Return selected table keys or all known schema keys when none provided.

    Returns
    -------
    list[str]
        Ordered list of table keys to render.
    """
    if selected:
        return list(selected)
    return sorted(
        table_key
        for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
        if contract.schema is not None and not contract.is_view
    )


def main() -> None:
    """Render normalized macro DDL for the requested tables to stdout."""
    args = _parse_args()
    for table_key in _iter_tables(args.tables):
        macro = render_macro(table_key)
        # Emit to stdout; callers can redirect as needed.
        sys.stdout.write(macro.ddl)
        sys.stdout.write("\n\n")


if __name__ == "__main__":
    main()
