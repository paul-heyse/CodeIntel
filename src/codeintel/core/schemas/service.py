"""Canonical schema service for resolving schema artifacts.

This module provides the SchemaService abstraction that unifies schema lookups
across table schemas, JSON schemas, row bindings, and optional dataset schemas.
The service lives in ``codeintel.core.schemas`` so higher-level layers can
depend on it without importing build-specific modules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.json_schema_gen import json_schema_from_table_schema
from codeintel.core.schemas.primitives import TableSchema
from codeintel.core.schemas.provider import SchemaProvider
from codeintel.core.schemas.row_models import GeneratedRowBinding, row_binding_for_table_schema
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pyarrow as pa


class DatasetSchemaLike(Protocol):
    """Protocol for build-owned DatasetSchema metadata."""

    name: str
    json_schema: dict[str, Any] | None
    ddl_schema: TableSchema | None


class DatasetSchemaProvider(Protocol):
    """Protocol for resolving DatasetSchema-like objects."""

    def get_dataset_schema(self, table_key: str) -> DatasetSchemaLike | None:
        """Return a DatasetSchema for the table key, or None when unavailable.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        DatasetSchemaLike | None
            Dataset schema metadata if available.
        """
        ...

    def iter_dataset_schemas(self) -> Iterable[DatasetSchemaLike]:
        """Iterate all dataset schemas known to the provider.

        Returns
        -------
        Iterable[DatasetSchemaLike]
            All dataset schemas provided by the implementation.
        """
        ...


class ArrowSchemaProvider(Protocol):
    """Protocol for resolving Arrow schemas by table key."""

    def get_arrow_schema(self, table_key: str) -> pa.Schema | None:
        """Return the Arrow schema for a table key."""
        ...


def _default_row_binding_factory(table_schema: TableSchema) -> GeneratedRowBinding:
    return row_binding_for_table_schema(table_schema=table_schema)


@dataclass(frozen=True)
class SchemaRecord:
    """Aggregated schema artifacts for a single table key."""

    table_key: str
    table_schema: TableSchema | None
    dataset_schema: DatasetSchemaLike | None
    json_schema: dict[str, Any] | None
    json_schema_id: str | None
    json_schema_digest: str | None
    row_binding: GeneratedRowBinding | None
    schema_hash: str | None


@dataclass(frozen=True)
class SchemaService:
    """Canonical schema service for resolving related schema artifacts."""

    table_provider: SchemaProvider
    dataset_provider: DatasetSchemaProvider | None = None
    arrow_provider: ArrowSchemaProvider | None = None
    json_schema_factory: Callable[[TableSchema, str | None], dict[str, Any]] | None = None
    row_binding_factory: Callable[[TableSchema], GeneratedRowBinding] = _default_row_binding_factory
    _table_cache: dict[str, TableSchema | None] = field(default_factory=dict, repr=False)
    _dataset_cache: dict[str, DatasetSchemaLike | None] = field(default_factory=dict, repr=False)
    _json_cache: dict[str, dict[str, Any] | None] = field(default_factory=dict, repr=False)
    _arrow_cache: dict[str, pa.Schema | None] = field(default_factory=dict, repr=False)
    _row_cache: dict[str, GeneratedRowBinding | None] = field(default_factory=dict, repr=False)
    _record_cache: dict[str, SchemaRecord] = field(default_factory=dict, repr=False)

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return a TableSchema for a table key, or None if unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema | None
            Resolved table schema when known.
        """
        if table_key in self._table_cache:
            return self._table_cache[table_key]
        schema = self.table_provider.get_table_schema(table_key)
        self._table_cache[table_key] = schema
        return schema

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return a TableSchema for a table key, raising if unknown.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TableSchema
            Resolved table schema.

        Raises
        ------
        KeyError
            If the table key is not registered.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known TableSchema values.

        Returns
        -------
        Iterable[TableSchema]
            Table schemas known to the provider.
        """
        return self.table_provider.iter_table_schemas()

    def get_dataset_schema(self, table_key: str) -> DatasetSchemaLike | None:
        """Return a DatasetSchema-like object when available.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        DatasetSchemaLike | None
            Dataset schema metadata if the provider supplies it.
        """
        if self.dataset_provider is None:
            return None
        if table_key in self._dataset_cache:
            return self._dataset_cache[table_key]
        schema = self.dataset_provider.get_dataset_schema(table_key)
        self._dataset_cache[table_key] = schema
        return schema

    def iter_dataset_schemas(self) -> Iterable[DatasetSchemaLike]:
        """Iterate dataset schemas when a provider is configured.

        Returns
        -------
        Iterable[DatasetSchemaLike]
            Dataset schemas supplied by the provider, or empty if absent.
        """
        if self.dataset_provider is None:
            return ()
        return self.dataset_provider.iter_dataset_schemas()

    def get_row_binding(self, table_key: str) -> GeneratedRowBinding | None:
        """Return a generated row binding for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        GeneratedRowBinding | None
            Generated row binding for the table schema when available.
        """
        if table_key in self._row_cache:
            return self._row_cache[table_key]
        schema = self.get_table_schema(table_key)
        if schema is None:
            self._row_cache[table_key] = None
            return None
        binding = self.row_binding_factory(schema)
        self._row_cache[table_key] = binding
        return binding

    def require_row_binding(self, table_key: str) -> GeneratedRowBinding:
        """Return a generated row binding or raise if schema is missing.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        GeneratedRowBinding
            Generated row binding for the table schema.

        Raises
        ------
        KeyError
            If the table schema is not registered.
        """
        binding = self.get_row_binding(table_key)
        if binding is None:
            msg = f"Unknown table schema for row binding: {table_key}"
            raise KeyError(msg)
        return binding

    def get_arrow_schema(self, table_key: str) -> pa.Schema | None:
        """Return a PyArrow schema rendered from the TableSchema.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        pa.Schema | None
            PyArrow schema derived from the TableSchema when available.
        """
        if table_key in self._arrow_cache:
            return self._arrow_cache[table_key]
        if self.arrow_provider is not None:
            arrow_schema = self.arrow_provider.get_arrow_schema(table_key)
            if arrow_schema is not None:
                self._arrow_cache[table_key] = arrow_schema
                return arrow_schema
        table_schema = self.get_table_schema(table_key)
        if table_schema is None:
            self._arrow_cache[table_key] = None
            return None
        arrow_schema = arrow_schema_from_table_schema(table_schema=table_schema)
        self._arrow_cache[table_key] = arrow_schema
        return arrow_schema

    def get_json_schema(self, table_key: str) -> dict[str, Any] | None:
        """Return a generated JSON Schema for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        dict[str, Any] | None
            JSON Schema dictionary for the table schema when available.
        """
        if table_key in self._json_cache:
            return self._json_cache[table_key]
        schema_id = f"urn:codeintel:schema:{table_key}"
        dataset_schema = self.get_dataset_schema(table_key)
        if dataset_schema is not None and dataset_schema.json_schema is not None:
            json_schema = dict(dataset_schema.json_schema)
            json_schema.setdefault("$id", schema_id)
            self._json_cache[table_key] = json_schema
            return json_schema

        schema = self.get_table_schema(table_key)
        if schema is None:
            self._json_cache[table_key] = None
            return None
        factory = self.json_schema_factory or _default_json_schema_factory
        json_schema = factory(schema, schema_id)
        self._json_cache[table_key] = json_schema
        return json_schema

    def compute_json_schema_digest(self, table_key: str) -> str | None:
        """Compute a stable digest for a generated JSON schema.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        str | None
            Digest of the JSON schema, or None when the schema is missing.
        """
        schema = self.get_json_schema(table_key)
        if schema is None:
            return None
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get_record(self, table_key: str) -> SchemaRecord:
        """Return a SchemaRecord aggregating all schema artifacts.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        SchemaRecord
            Aggregated schema record for the table key.
        """
        if table_key in self._record_cache:
            return self._record_cache[table_key]
        table_schema = self.get_table_schema(table_key)
        dataset_schema = self.get_dataset_schema(table_key)
        json_schema = self.get_json_schema(table_key)
        json_schema_id = f"urn:codeintel:schema:{table_key}" if json_schema is not None else None
        digest = self.compute_json_schema_digest(table_key) if json_schema is not None else None
        row_binding = self.get_row_binding(table_key)
        schema_hash_value = schema_hash(table_schema) if table_schema is not None else None
        record = SchemaRecord(
            table_key=table_key,
            table_schema=table_schema,
            dataset_schema=dataset_schema,
            json_schema=json_schema,
            json_schema_id=json_schema_id,
            json_schema_digest=digest,
            row_binding=row_binding,
            schema_hash=schema_hash_value,
        )
        self._record_cache[table_key] = record
        return record

    def clear_caches(self) -> None:
        """Clear all cached schema artifacts."""
        self._table_cache.clear()
        self._dataset_cache.clear()
        self._json_cache.clear()
        self._arrow_cache.clear()
        self._row_cache.clear()
        self._record_cache.clear()


def _default_json_schema_factory(schema: TableSchema, schema_id: str | None) -> dict[str, Any]:
    """Generate JSON Schema from a TableSchema.

    Parameters
    ----------
    schema
        TableSchema to convert into JSON Schema.
    schema_id
        Optional ``$id`` identifier for the schema.

    Returns
    -------
    dict[str, Any]
        JSON Schema dictionary.
    """
    return json_schema_from_table_schema(schema, schema_id=schema_id)


class _SchemaServiceHolder(SingletonHolder["SchemaService"]):
    """Singleton holder for SchemaService."""

    @classmethod
    def set(cls, service: SchemaService) -> None:
        with cls._lock:
            cls._instance = service


def set_schema_service(service: SchemaService) -> None:
    """Register a global SchemaService instance.

    Parameters
    ----------
    service
        SchemaService instance to register.
    """
    _SchemaServiceHolder.set(service)


def get_schema_service() -> SchemaService:
    """Return the registered SchemaService instance.

    Returns
    -------
    SchemaService
        Registered SchemaService singleton.

    Raises
    ------
    RuntimeError
        If the SchemaService has not been configured.
    """
    service = _SchemaServiceHolder.get_or_none()
    if service is None:
        msg = "SchemaService has not been configured"
        raise RuntimeError(msg)
    return service


def clear_schema_service() -> None:
    """Clear the registered SchemaService."""
    _SchemaServiceHolder.reset()


__all__ = [
    "ArrowSchemaProvider",
    "DatasetSchemaLike",
    "DatasetSchemaProvider",
    "SchemaRecord",
    "SchemaService",
    "clear_schema_service",
    "get_schema_service",
    "set_schema_service",
]
