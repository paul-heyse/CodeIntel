"""DuckDB metadata cache tables for Iceberg catalogs."""

from __future__ import annotations

from codeintel.core.schemas.primitives import Column, Index, TableSchema

ICEBERG_TABLES_TABLE = TableSchema(
    schema="metadata",
    name="iceberg_tables",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("identifier", "VARCHAR", nullable=False),
        Column("location", "VARCHAR", nullable=False),
        Column("current_snapshot_id", "BIGINT"),
        Column("current_schema_id", "INTEGER"),
        Column("current_spec_id", "INTEGER"),
        Column("current_sort_order_id", "INTEGER"),
        Column("properties", "MAP(VARCHAR, VARCHAR)"),
        Column(
            "refs",
            "MAP(VARCHAR, STRUCT(snapshot_id BIGINT, ref_type VARCHAR, max_ref_age_ms BIGINT))",
        ),
        Column("last_updated_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("table_key",),
    indexes=(Index("idx_iceberg_tables_snapshot", ("current_snapshot_id",)),),
)

ICEBERG_SCHEMAS_TABLE = TableSchema(
    schema="metadata",
    name="iceberg_schemas",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("schema_id", "INTEGER", nullable=False),
        Column(
            "fields",
            "LIST(STRUCT(field_id INTEGER, name VARCHAR, type VARCHAR, required BOOLEAN, doc VARCHAR, parent_id INTEGER))",
        ),
        Column("name_mapping_json", "JSON"),
        Column("schema_json", "JSON"),
    ],
    primary_key=("table_key", "schema_id"),
    indexes=(Index("idx_iceberg_schemas_table_key", ("table_key",)),),
)

ICEBERG_PARTITION_SPECS_TABLE = TableSchema(
    schema="metadata",
    name="iceberg_partition_specs",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("spec_id", "INTEGER", nullable=False),
        Column(
            "fields",
            "LIST(STRUCT(field_id INTEGER, name VARCHAR, transform VARCHAR, source_id INTEGER))",
        ),
    ],
    primary_key=("table_key", "spec_id"),
    indexes=(Index("idx_iceberg_specs_table_key", ("table_key",)),),
)

ICEBERG_SORT_ORDERS_TABLE = TableSchema(
    schema="metadata",
    name="iceberg_sort_orders",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("order_id", "INTEGER", nullable=False),
        Column(
            "fields",
            "LIST(STRUCT(field_id INTEGER, transform VARCHAR, direction VARCHAR, null_order VARCHAR))",
        ),
    ],
    primary_key=("table_key", "order_id"),
    indexes=(Index("idx_iceberg_sort_orders_table_key", ("table_key",)),),
)

ICEBERG_SNAPSHOTS_TABLE = TableSchema(
    schema="metadata",
    name="iceberg_snapshots",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("snapshot_id", "BIGINT", nullable=False),
        Column("parent_snapshot_id", "BIGINT"),
        Column("committed_at", "TIMESTAMPTZ", nullable=False),
        Column("operation", "VARCHAR"),
        Column("summary", "MAP(VARCHAR, VARCHAR)"),
        Column("manifest_list_path", "VARCHAR"),
    ],
    primary_key=("table_key", "snapshot_id"),
    indexes=(Index("idx_iceberg_snapshots_table_key", ("table_key",)),),
)

ICEBERG_ARROW_SCHEMA_TABLE = TableSchema(
    schema="metadata",
    name="iceberg_arrow_schema",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),
        Column("schema_id", "INTEGER", nullable=False),
        Column("arrow_schema_ipc", "BLOB"),
        Column("arrow_schema_json", "JSON"),
    ],
    primary_key=("table_key", "schema_id"),
    indexes=(Index("idx_iceberg_arrow_schema_table_key", ("table_key",)),),
)

ICEBERG_METADATA_TABLES: tuple[TableSchema, ...] = (
    ICEBERG_TABLES_TABLE,
    ICEBERG_SCHEMAS_TABLE,
    ICEBERG_PARTITION_SPECS_TABLE,
    ICEBERG_SORT_ORDERS_TABLE,
    ICEBERG_SNAPSHOTS_TABLE,
    ICEBERG_ARROW_SCHEMA_TABLE,
)

__all__ = [
    "ICEBERG_ARROW_SCHEMA_TABLE",
    "ICEBERG_METADATA_TABLES",
    "ICEBERG_PARTITION_SPECS_TABLE",
    "ICEBERG_SCHEMAS_TABLE",
    "ICEBERG_SNAPSHOTS_TABLE",
    "ICEBERG_SORT_ORDERS_TABLE",
    "ICEBERG_TABLES_TABLE",
]
