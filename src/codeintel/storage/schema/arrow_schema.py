"""Arrow schema rendering helpers for storage boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.storage.schema.duckdb_contracts import contract_schema_for_table_key

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def arrow_schema_for_table_key(
    con: DuckDBPyConnection,
    *,
    table_key: str,
    repo: str | None = None,
    commit: str | None = None,
    pii_by_column: Mapping[str, str] | None = None,
) -> pa.Schema | None:
    """Render a PyArrow schema enriched with metadata for a table key.

    Parameters
    ----------
    con
        DuckDB connection used to load registry metadata and lineage.
    table_key
        Fully qualified table key (schema.table).
    repo
        Optional repository identifier for lineage lookups.
    commit
        Optional commit hash for lineage lookups.
    pii_by_column
        Optional mapping of column name to PII classification labels.

    Returns
    -------
    pa.Schema | None
        Rendered PyArrow schema with metadata, or None if the table is unknown.
    """
    return contract_schema_for_table_key(
        con=con,
        table_key=table_key,
        repo=repo,
        commit=commit,
        pii_by_column=pii_by_column,
    )


@dataclass(frozen=True, slots=True)
class RegistryArrowSchemaProvider:
    """Arrow schema provider backed by registry metadata."""

    con: DuckDBPyConnection
    repo: str | None = None
    commit: str | None = None

    def get_arrow_schema(self, table_key: str) -> pa.Schema | None:
        """Return Arrow schema for the table key.

        Returns
        -------
        pa.Schema | None
            Arrow schema for the table when available.
        """
        return arrow_schema_for_table_key(
            self.con,
            table_key=table_key,
            repo=self.repo,
            commit=self.commit,
        )


def arrow_schema_hash(schema: pa.Schema) -> str | None:
    """Return the CodeIntel schema hash embedded in Arrow metadata.

    Parameters
    ----------
    schema
        PyArrow schema to inspect.

    Returns
    -------
    str | None
        Schema hash when present, otherwise None.
    """
    return _schema_metadata_value(schema, "codeintel.schema_hash")


def arrow_schema_digest(schema: pa.Schema) -> str | None:
    """Return the schema digest embedded in Arrow metadata.

    Parameters
    ----------
    schema
        PyArrow schema to inspect.

    Returns
    -------
    str | None
        Schema digest when present, otherwise None.
    """
    return _schema_metadata_value(schema, "codeintel.schema_digest")


def _schema_metadata_value(schema: pa.Schema, key: str) -> str | None:
    metadata = schema.metadata
    if not metadata:
        return None
    raw = metadata.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")


__all__ = [
    "RegistryArrowSchemaProvider",
    "arrow_schema_digest",
    "arrow_schema_for_table_key",
    "arrow_schema_hash",
]
