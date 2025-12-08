"""Generate typed insert methods for table accessors from TABLE_SCHEMAS.

This script reads the TABLE_SCHEMAS registry and generates typed insert method
mixins for each accessor class (CoreTables, GraphTables, AnalyticsTables).

Usage
-----
Generate registry and row models:
    uv run python -m tools.generate_accessor_inserts --registry-output src/codeintel/storage/gateway/registry_generated.py --rows-output-dir src/codeintel/storage/gateway/rows

Check if files are up-to-date (for CI):
    uv run python -m tools.generate_accessor_inserts --check --registry-output src/codeintel/storage/gateway/registry_generated.py --rows-output-dir src/codeintel/storage/gateway/rows
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
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
SKIP_TABLES: Final[set[str]] = set()

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


@dataclass(frozen=True)
class ColumnSpec:
    """Lightweight column specification for generated metadata."""

    name: str
    duckdb_type: str
    python_type: str
    nullable: bool


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


def _build_column_specs(table_key: str) -> list[ColumnSpec]:
    """Build ColumnSpec entries for a given table key.

    Returns
    -------
    list[ColumnSpec]
        Column specifications derived from TABLE_SCHEMAS.
    """
    schema = TABLE_SCHEMAS.get(table_key)
    if schema is None:
        return []
    return [
        ColumnSpec(
            name=col.name,
            duckdb_type=col.type,
            python_type=duckdb_type_to_python(col.type, nullable=col.nullable),
            nullable=col.nullable,
        )
        for col in schema.columns
    ]


def generate_registry_module() -> str:
    """Generate a registry mapping table ids to metadata.

    Returns
    -------
    str
        Python module source containing TABLE_REGISTRY.
    """
    lines: list[str] = [
        '"""Generated table registry for insert helpers.',
        "",
        "AUTO-GENERATED by tools/generate_accessor_inserts.py",
        'Do not edit manually."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
        "",
        "TableMeta = dict[str, object]",
        "TABLE_REGISTRY: Final[dict[str, TableMeta]] = {",
    ]

    for table_key in sorted(TABLE_SCHEMAS):
        if table_key in SKIP_TABLES:
            continue
        specs = _build_column_specs(table_key)
        columns_repr = ", ".join(f'"{c.name}"' for c in specs)
        lines.append(f'    "{table_key}": {{')
        lines.append(f'        "table": "{table_key}",')
        lines.append(f'        "columns": [{columns_repr}],')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def generate_row_models_module(schema_prefix: str, table_keys: Iterable[str]) -> str:
    """Generate TypedDict row models for a schema prefix.

    Returns
    -------
    str
        Python module source containing row model definitions.
    """
    lines: list[str] = [
        '"""Generated row models for insert helpers."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TypedDict",
        "",
        "__all__ = [",
    ]
    model_names = sorted(
        f"{schema_prefix.title()}{table_key.split('.', maxsplit=1)[1].title().replace('_', '')}Row"
        for table_key in sorted(table_keys)
        if table_key not in SKIP_TABLES
    )
    lines.extend(f'    "{name}",' for name in model_names)
    lines.append("]")
    lines.append("")
    lines.append("")

    first_class = True

    for table_key in sorted(table_keys):
        if table_key in SKIP_TABLES:
            continue
        specs = _build_column_specs(table_key)
        if not specs:
            continue
        _, table_name = table_key.split(".", maxsplit=1)
        model_name = f"{schema_prefix.title()}{table_name.title().replace('_', '')}Row"
        if not first_class:
            lines.append("")
            lines.append("")
        first_class = False
        lines.append(f"class {model_name}(TypedDict):")
        lines.append(f'    """Row model for {table_key}."""')
        lines.append("")
        lines.extend(f"    {col.name}: {col.python_type}" for col in specs)

    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--registry-output",
        type=Path,
        help="Optional path to write the generated table registry module",
    )
    parser.add_argument(
        "--rows-output-dir",
        type=Path,
        help="Optional directory to write generated row model modules (one per schema)",
    )
    args = parser.parse_args()

    registry_code = generate_registry_module() if args.registry_output else None
    rows_by_schema: dict[str, str] = {}
    if args.rows_output_dir:
        tables_by_schema: dict[str, list[str]] = {"core": [], "graph": [], "analytics": []}
        for table_key in TABLE_SCHEMAS:
            schema_prefix, _ = table_key.split(".", maxsplit=1)
            if schema_prefix in tables_by_schema:
                tables_by_schema[schema_prefix].append(table_key)
        for schema_prefix, table_keys in tables_by_schema.items():
            if not table_keys:
                continue
            rows_by_schema[schema_prefix] = generate_row_models_module(schema_prefix, table_keys)

    if args.check:
        return _run_check(
            registry_code=registry_code,
            registry_output=args.registry_output,
            rows_by_schema=rows_by_schema,
            rows_output_dir=args.rows_output_dir,
        )

    if args.registry_output and registry_code is not None:
        args.registry_output.parent.mkdir(parents=True, exist_ok=True)
        args.registry_output.write_text(registry_code, encoding="utf-8")
        LOG.info("Wrote registry module to %s", args.registry_output)
    if args.rows_output_dir:
        args.rows_output_dir.mkdir(parents=True, exist_ok=True)
        for schema_prefix, module_code in rows_by_schema.items():
            target = args.rows_output_dir / f"{schema_prefix}.py"
            target.write_text(module_code, encoding="utf-8")
            LOG.info("Wrote row models for %s to %s", schema_prefix, target)
    return 0


def _run_check(
    *,
    registry_code: str | None,
    registry_output: Path | None,
    rows_by_schema: dict[str, str],
    rows_output_dir: Path | None,
) -> int:
    """Check whether generated artifacts are up-to-date.

    Returns
    -------
    int
        0 when all artifacts match, 1 otherwise.
    """
    if (
        registry_output
        and registry_code is not None
        and _is_registry_outdated(registry_output, registry_code)
    ):
        return 1
    if rows_output_dir and _rows_outdated(rows_output_dir, rows_by_schema):
        return 1

    LOG.info("Generated files are up-to-date")
    return 0


def _is_registry_outdated(registry_output: Path, registry_code: str) -> bool:
    """Check the generated registry module for staleness.

    Returns
    -------
    bool
        True when the registry on disk differs from the generated content.
    """
    if not registry_output.exists():
        LOG.error("Registry output does not exist at %s", registry_output)
        return True
    existing_registry = registry_output.read_text(encoding="utf-8")
    if existing_registry != registry_code:
        LOG.error("Registry output is out of date at %s", registry_output)
        return True
    return False


def _rows_outdated(rows_output_dir: Path, rows_by_schema: dict[str, str]) -> bool:
    """Check generated row model modules for staleness.

    Returns
    -------
    bool
        True when any row model module is missing or differs from generated content.
    """
    outdated = False
    for schema_prefix, module_code in rows_by_schema.items():
        target = rows_output_dir / f"{schema_prefix}.py"
        if not target.exists():
            LOG.error("Row model output does not exist at %s", target)
            outdated = True
            continue
        existing_rows = target.read_text(encoding="utf-8")
        if existing_rows != module_code:
            LOG.error("Row model output is out of date at %s", target)
            outdated = True
    return outdated


if __name__ == "__main__":
    sys.exit(main())
