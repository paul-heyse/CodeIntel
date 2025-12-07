"""Generate typed insert methods for table accessors from TABLE_SCHEMAS.

This script reads the TABLE_SCHEMAS registry and generates typed insert method
mixins for each accessor class (CoreTables, GraphTables, AnalyticsTables).

Usage
-----
Generate the file:
    uv run python -m tools.generate_accessor_inserts > src/codeintel/storage/gateway/_generated_inserts.py

Check if file is up-to-date (for CI):
    uv run python -m tools.generate_accessor_inserts --check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

from codeintel.config.datasets import TABLE_SCHEMAS

LOG = logging.getLogger(__name__)

# DuckDB type to Python type mapping
DUCKDB_TO_PYTHON: Final[dict[str, str]] = {
    "BOOLEAN": "bool",
    "INTEGER": "int",
    "BIGINT": "int",
    "DOUBLE": "float",
    "DECIMAL": "float",
    "DECIMAL(38,0)": "float",
    "VARCHAR": "str",
    "JSON": "str",
    "TIMESTAMP": "str",
    "TIMESTAMPTZ": "str",
}

# Tables that have special handling in accessors.py and should NOT be generated
SKIP_TABLES: Final[set[str]] = {
    # insert_modules has row normalization logic
    "core.modules",
    # insert_symbol_use_edges has row normalization logic (5->7 columns)
    "graph.symbol_use_edges",
}

# Schema prefixes to mixin class mapping
SCHEMA_TO_MIXIN: Final[dict[str, str]] = {
    "core": "CoreTableInsertsMixin",
    "graph": "GraphTableInsertsMixin",
    "analytics": "AnalyticsTableInsertsMixin",
}

# Threshold for multi-line tuple type formatting
MULTILINE_TYPE_THRESHOLD: Final[int] = 8

# Threshold for truncating column descriptions in docstrings
COLUMN_DESC_TRUNCATE_THRESHOLD: Final[int] = 10
COLUMN_DESC_PREVIEW_COUNT: Final[int] = 8


def duckdb_type_to_python(duckdb_type: str, *, nullable: bool) -> str:
    """Convert a DuckDB type to Python type annotation.

    Parameters
    ----------
    duckdb_type
        DuckDB column type (e.g., "VARCHAR", "BIGINT").
    nullable
        Whether the column allows NULL values.

    Returns
    -------
    str
        Python type annotation string.
    """
    python_type = DUCKDB_TO_PYTHON.get(duckdb_type, "object")
    if nullable:
        return f"{python_type} | None"
    return python_type


def generate_method_name(table_key: str) -> str:
    """Generate the insert method name for a table.

    Parameters
    ----------
    table_key
        Fully qualified table name (schema.table).

    Returns
    -------
    str
        Method name like "insert_goids".
    """
    _, table_name = table_key.split(".", maxsplit=1)
    return f"insert_{table_name}"


def generate_tuple_type(table_key: str) -> str:
    """Generate the tuple type annotation for a table's rows.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    str
        Tuple type annotation like "tuple[int, str, str | None, ...]".
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return "tuple[object, ...]"

    types = [duckdb_type_to_python(col.type, nullable=col.nullable) for col in schema.columns]

    # Format the tuple type with line breaks for readability if long
    if len(types) > MULTILINE_TYPE_THRESHOLD:
        # Multi-line format
        type_str = ",\n                ".join(types)
        return f"tuple[\n                {type_str},\n            ]"
    return f"tuple[{', '.join(types)}]"


def generate_docstring_params(table_key: str) -> str:
    """Generate the Parameters section of a docstring.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    str
        Docstring Parameters section text.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return "        rows\n            Iterable of row tuples."

    col_names = [col.name for col in schema.columns]
    # Truncate if too many columns
    col_desc = (
        ", ".join(col_names[:COLUMN_DESC_PREVIEW_COUNT]) + ", ..."
        if len(col_names) > COLUMN_DESC_TRUNCATE_THRESHOLD
        else ", ".join(col_names)
    )

    return f"        rows\n            Iterable of ({col_desc})."


def generate_insert_method(table_key: str) -> str:
    """Generate a single insert method for a table.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    str
        Generated Python method code.
    """
    method_name = generate_method_name(table_key)
    tuple_type = generate_tuple_type(table_key)
    params_doc = generate_docstring_params(table_key)

    return f'''    def {method_name}(
        self,
        rows: Iterable[{tuple_type}],
    ) -> None:
        """Insert rows into {table_key}.

        Parameters
        ----------
{params_doc}
        """
        self._insert_rows("{table_key}", rows)
'''


def generate_mixin_class(schema_prefix: str, table_keys: list[str]) -> str:
    """Generate a mixin class with insert methods for a schema.

    Parameters
    ----------
    schema_prefix
        Schema prefix (e.g., "core", "graph", "analytics").
    table_keys
        List of table keys belonging to this schema.

    Returns
    -------
    str
        Generated Python class code.
    """
    mixin_name = SCHEMA_TO_MIXIN[schema_prefix]
    methods: list[str] = []

    for table_key in sorted(table_keys):
        if table_key in SKIP_TABLES:
            continue
        methods.append(generate_insert_method(table_key))

    methods_str = "\n".join(methods)

    return f'''class {mixin_name}:
    """Generated insert methods for {schema_prefix} schema tables.

    This class provides typed insert methods that delegate to the
    BaseTableAccessor._insert_rows method. Methods are generated from
    TABLE_SCHEMAS to ensure type safety.

    Note: Some tables have special handling (e.g., insert_modules,
    insert_symbol_use_edges) and are defined manually in accessors.py.
    """

    # Type annotation for _insert_rows inherited from BaseTableAccessor
    _insert_rows: Any

{methods_str}'''


def generate_full_module() -> str:
    """Generate the complete _generated_inserts.py module.

    Returns
    -------
    str
        Complete Python module source code.
    """
    # Group tables by schema prefix
    tables_by_schema: dict[str, list[str]] = {
        "core": [],
        "graph": [],
        "analytics": [],
    }

    for table_key in TABLE_SCHEMAS:
        schema_prefix, _ = table_key.split(".", maxsplit=1)
        if schema_prefix in tables_by_schema:
            tables_by_schema[schema_prefix].append(table_key)

    # Generate mixin classes
    mixin_classes = [
        generate_mixin_class(schema_prefix, tables_by_schema[schema_prefix])
        for schema_prefix in ["core", "graph", "analytics"]
        if tables_by_schema[schema_prefix]
    ]

    mixin_code = "\n\n".join(mixin_classes)

    return f'''"""Generated insert methods for table accessors.

AUTO-GENERATED by tools/generate_accessor_inserts.py
Do not edit manually. Regenerate with:
    python -m tools.generate_accessor_inserts > src/codeintel/storage/gateway/_generated_inserts.py
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "AnalyticsTableInsertsMixin",
    "CoreTableInsertsMixin",
    "GraphTableInsertsMixin",
]


{mixin_code}
'''


def main() -> int:
    """Run the code generator.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate typed insert methods for table accessors"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if generated file is up-to-date instead of generating",
    )
    args = parser.parse_args()

    generated_code = generate_full_module()

    if args.check:
        # Check if existing file matches
        output_path = Path("src/codeintel/storage/gateway/_generated_inserts.py")
        if not output_path.exists():
            LOG.error("Generated file does not exist")
            return 1

        existing_code = output_path.read_text(encoding="utf-8")
        if existing_code != generated_code:
            LOG.error("Generated file is out of date. Regenerate with:")
            LOG.error(
                "  uv run python -m tools.generate_accessor_inserts "
                "> src/codeintel/storage/gateway/_generated_inserts.py"
            )
            return 1

        LOG.info("Generated file is up-to-date")
        return 0

    # Output generated code to stdout
    sys.stdout.write(generated_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
