"""SQLAlchemy-free DuckDB-backed Iceberg catalog implementation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

import duckdb
import sqlglot
from pyiceberg.catalog import (
    METADATA_LOCATION,
    URI,
    Catalog,
    MetastoreCatalog,
    PropertiesUpdateSummary,
)
from pyiceberg.exceptions import (
    CommitFailedException,
    NamespaceAlreadyExistsError,
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchPropertyException,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io import load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import FromInputFile
from pyiceberg.table import CommitTableResponse, StagedTable, Table, TableProperties
from pyiceberg.table.locations import load_location_provider
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.table.update import TableRequirement, TableUpdate
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties, TableVersion
from pyiceberg.types import strtobool
from sqlglot import exp

from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.sqlglot_tools import render_sql_duckdb

if TYPE_CHECKING:
    import pyarrow as pa

DEFAULT_INIT_CATALOG_TABLES = "true"
NAMESPACE_MINIMAL_PROPERTIES = {"exists": "true"}
_MAX_CREATE_TABLE_ARGS = 4
_TABLE_VERSION_V1: TableVersion = 1
_TABLE_VERSION_V2: TableVersion = 2
_TABLE_VERSION_V3: TableVersion = 3
_ICEBERG_TABLES = "iceberg_tables"
_ICEBERG_NAMESPACE_PROPERTIES = "iceberg_namespace_properties"

_CATALOG_TABLE_DDL: dict[str, str] = {
    _ICEBERG_TABLES: (
        "CREATE TABLE IF NOT EXISTS iceberg_tables ("
        "catalog_name VARCHAR NOT NULL, "
        "table_namespace VARCHAR NOT NULL, "
        "table_name VARCHAR NOT NULL, "
        "metadata_location VARCHAR, "
        "previous_metadata_location VARCHAR, "
        "PRIMARY KEY (catalog_name, table_namespace, table_name)"
        ")"
    ),
    _ICEBERG_NAMESPACE_PROPERTIES: (
        "CREATE TABLE IF NOT EXISTS iceberg_namespace_properties ("
        "catalog_name VARCHAR NOT NULL, "
        "namespace VARCHAR NOT NULL, "
        "property_key VARCHAR NOT NULL, "
        "property_value VARCHAR NOT NULL, "
        "PRIMARY KEY (catalog_name, namespace, property_key)"
        ")"
    ),
}


@dataclass(frozen=True, slots=True)
class _CreateTableOptions:
    location: str | None
    partition_spec: PartitionSpec
    sort_order: SortOrder
    properties: Properties


def _duckdb_database_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "duckdb":
        msg = f"Unsupported DuckDB catalog URI: {uri!r}"
        raise ValueError(msg)
    raw_path = f"{parsed.netloc}{parsed.path}"
    if not raw_path:
        msg = "DuckDB catalog URI must include a database path."
        raise ValueError(msg)
    if raw_path in {":memory:", "/:memory:"}:
        return ":memory:"
    return str(Path(raw_path).expanduser())


def _ensure_duckdb_parent(database: str) -> None:
    if database == ":memory:":
        return
    path = Path(database)
    parent = path.parent
    if parent == Path():
        return
    parent.mkdir(parents=True, exist_ok=True)


def _format_version_from_properties(properties: Properties) -> TableVersion:
    raw = properties.get(
        TableProperties.FORMAT_VERSION,
        TableProperties.DEFAULT_FORMAT_VERSION,
    )
    if isinstance(raw, int):
        return _coerce_table_version(raw)
    if isinstance(raw, str):
        raw_value = raw.strip()
        if not raw_value:
            return TableProperties.DEFAULT_FORMAT_VERSION
        try:
            parsed = int(raw_value)
        except ValueError:
            return TableProperties.DEFAULT_FORMAT_VERSION
        return _coerce_table_version(parsed)
    return TableProperties.DEFAULT_FORMAT_VERSION


def _coerce_table_version(value: int) -> TableVersion:
    if value == _TABLE_VERSION_V1:
        return _TABLE_VERSION_V1
    if value == _TABLE_VERSION_V2:
        return _TABLE_VERSION_V2
    if value == _TABLE_VERSION_V3:
        return _TABLE_VERSION_V3
    return TableProperties.DEFAULT_FORMAT_VERSION


def _resolve_create_option(
    args: Sequence[object],
    kwargs: Mapping[str, object],
    *,
    index: int,
    name: str,
    default: object,
) -> object:
    if index < len(args):
        if name in kwargs:
            msg = f"{name} was provided both positionally and as a keyword argument."
            raise TypeError(msg)
        return args[index]
    return kwargs.get(name, default)


def _parse_create_table_options(
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> _CreateTableOptions:
    if len(args) > _MAX_CREATE_TABLE_ARGS:
        msg = (
            f"create_table accepts at most {_MAX_CREATE_TABLE_ARGS} positional arguments: "
            "location, partition_spec, sort_order, properties."
        )
        raise TypeError(msg)
    allowed_keys = {"location", "partition_spec", "sort_order", "properties"}
    extra = set(kwargs) - allowed_keys
    if extra:
        msg = f"Unexpected keyword arguments: {sorted(extra)!r}"
        raise TypeError(msg)
    location = _resolve_create_option(args, kwargs, index=0, name="location", default=None)
    partition_spec = _resolve_create_option(
        args,
        kwargs,
        index=1,
        name="partition_spec",
        default=UNPARTITIONED_PARTITION_SPEC,
    )
    sort_order = _resolve_create_option(
        args,
        kwargs,
        index=2,
        name="sort_order",
        default=UNSORTED_SORT_ORDER,
    )
    properties = _resolve_create_option(
        args,
        kwargs,
        index=3,
        name="properties",
        default=EMPTY_DICT,
    )
    if properties is None:
        properties = EMPTY_DICT
    return _CreateTableOptions(
        location=cast("str | None", location),
        partition_spec=cast("PartitionSpec", partition_spec),
        sort_order=cast("SortOrder", sort_order),
        properties=cast("Properties", properties),
    )


@dataclass(slots=True)
class DuckDBCatalogSession:
    """Connection manager for DuckDB-backed Iceberg catalog operations."""

    database: str
    read_only: bool = False
    connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Return a DuckDB connection for catalog operations.

        Returns
        -------
        duckdb.DuckDBPyConnection
            Active DuckDB connection for catalog operations.
        """
        if self.connection is None:
            _ensure_duckdb_parent(self.database)
            self.connection = duckdb.connect(self.database, read_only=self.read_only)
        return self.connection

    def ensure_tables(self) -> None:
        """Ensure catalog metadata tables exist."""
        con = self.connect()
        con.execute(_catalog_table_ddl_sql(_ICEBERG_TABLES))
        con.execute(_catalog_table_ddl_sql(_ICEBERG_NAMESPACE_PROPERTIES))

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.connection is None:
            return
        self.connection.close()
        self.connection = None


class DuckDBCatalog(MetastoreCatalog):
    """DuckDB-backed Iceberg catalog using SQLGlot for SQL generation."""

    def __init__(self, name: str, **properties: str) -> None:
        """Create a DuckDB-backed Iceberg catalog.

        Raises
        ------
        NoSuchPropertyException
            If the catalog URI is missing.
        """
        super().__init__(name, **properties)

        if not (uri_prop := self.properties.get(URI)):
            msg = "DuckDB catalog URI is required"
            raise NoSuchPropertyException(msg)

        database = _duckdb_database_from_uri(uri_prop)
        init_catalog_tables = strtobool(
            self.properties.get("init_catalog_tables", DEFAULT_INIT_CATALOG_TABLES)
        )
        self._session = DuckDBCatalogSession(database=database)

        if init_catalog_tables:
            self._session.ensure_tables()

    def create_table(
        self,
        identifier: str | Identifier,
        schema: Schema | pa.Schema,
        *args: object,
        **kwargs: object,
    ) -> Table:
        """Create an Iceberg table in the DuckDB catalog.

        Parameters
        ----------
        identifier
            Table identifier.
        schema
            Iceberg or Arrow schema for the new table.
        *args
            Optional positional args: location, partition_spec, sort_order, properties.
        **kwargs
            Optional keyword args: location, partition_spec, sort_order, properties.

        Returns
        -------
        Table
            Loaded Iceberg table after registration.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        TableAlreadyExistsError
            If the table already exists.
        """
        options = _parse_create_table_options(args, kwargs)
        properties = options.properties
        schema = self._convert_schema_if_needed(
            schema,
            _format_version_from_properties(properties),
        )
        namespace_identifier = Catalog.namespace_from(identifier)
        table_name = Catalog.table_name_from(identifier)
        if not self._namespace_exists(namespace_identifier):
            msg = f"Namespace does not exist: {namespace_identifier}"
            raise NoSuchNamespaceError(msg)

        namespace = Catalog.namespace_to_string(namespace_identifier)
        location = self._resolve_table_location(options.location, namespace, table_name)
        location_provider = load_location_provider(
            table_location=location,
            table_properties=properties,
        )
        metadata_location = location_provider.new_table_metadata_file_location()
        metadata = new_table_metadata(
            location=location,
            schema=schema,
            partition_spec=options.partition_spec,
            sort_order=options.sort_order,
            properties=properties,
        )
        io = load_file_io(properties=self.properties, location=metadata_location)
        self._write_metadata(metadata, io, metadata_location)

        with self._transaction() as con:
            try:
                con.execute(
                    _insert_iceberg_tables_sql(),
                    [self.name, namespace, table_name, metadata_location, None],
                )
            except duckdb.ConstraintException as exc:
                msg = f"Table {namespace}.{table_name} already exists"
                raise TableAlreadyExistsError(msg) from exc

        return self.load_table(identifier)

    def register_table(self, identifier: str | Identifier, metadata_location: str) -> Table:
        """Register an existing Iceberg table metadata location.

        Parameters
        ----------
        identifier
            Table identifier.
        metadata_location
            Location of the Iceberg metadata JSON file.

        Returns
        -------
        Table
            Loaded Iceberg table after registration.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        TableAlreadyExistsError
            If the table already exists.
        """
        namespace_tuple = Catalog.namespace_from(identifier)
        namespace = Catalog.namespace_to_string(namespace_tuple)
        table_name = Catalog.table_name_from(identifier)
        if not self._namespace_exists(namespace):
            msg = f"Namespace does not exist: {namespace}"
            raise NoSuchNamespaceError(msg)

        with self._transaction() as con:
            try:
                con.execute(
                    _insert_iceberg_tables_sql(),
                    [self.name, namespace, table_name, metadata_location, None],
                )
            except duckdb.ConstraintException as exc:
                msg = f"Table {namespace}.{table_name} already exists"
                raise TableAlreadyExistsError(msg) from exc

        return self.load_table(identifier)

    def load_table(self, identifier: str | Identifier) -> Table:
        """Load table metadata and return the Iceberg table.

        Parameters
        ----------
        identifier
            Table identifier.

        Returns
        -------
        Table
            Loaded Iceberg table.

        Raises
        ------
        NoSuchTableError
            If the table does not exist.
        """
        namespace_tuple = Catalog.namespace_from(identifier)
        namespace = Catalog.namespace_to_string(namespace_tuple)
        table_name = Catalog.table_name_from(identifier)
        con = self._session.connect()
        row = con.execute(
            _select_iceberg_table_sql(),
            [self.name, namespace, table_name],
        ).fetchone()
        if row is None:
            msg = f"Table does not exist: {namespace}.{table_name}"
            raise NoSuchTableError(msg)
        return self._convert_row_to_table(row)

    def drop_table(self, identifier: str | Identifier) -> None:
        """Drop a table from the catalog.

        Parameters
        ----------
        identifier
            Table identifier.

        Raises
        ------
        NoSuchTableError
            If the table does not exist.
        """
        namespace_tuple = Catalog.namespace_from(identifier)
        namespace = Catalog.namespace_to_string(namespace_tuple)
        table_name = Catalog.table_name_from(identifier)
        con = self._session.connect()
        row = con.execute(
            _delete_iceberg_table_sql(),
            [self.name, namespace, table_name],
        ).fetchone()
        if row is None:
            msg = f"Table does not exist: {namespace}.{table_name}"
            raise NoSuchTableError(msg)

    def rename_table(
        self,
        from_identifier: str | Identifier,
        to_identifier: str | Identifier,
    ) -> Table:
        """Rename an Iceberg table.

        Parameters
        ----------
        from_identifier
            Current table identifier.
        to_identifier
            New table identifier.

        Returns
        -------
        Table
            Loaded Iceberg table for the new identifier.

        Raises
        ------
        NoSuchNamespaceError
            If the destination namespace does not exist.
        NoSuchTableError
            If the source table does not exist.
        TableAlreadyExistsError
            If the destination table already exists.
        """
        from_namespace_tuple = Catalog.namespace_from(from_identifier)
        from_namespace = Catalog.namespace_to_string(from_namespace_tuple)
        from_table_name = Catalog.table_name_from(from_identifier)
        to_namespace_tuple = Catalog.namespace_from(to_identifier)
        to_namespace = Catalog.namespace_to_string(to_namespace_tuple)
        to_table_name = Catalog.table_name_from(to_identifier)
        if not self._namespace_exists(to_namespace):
            msg = f"Namespace does not exist: {to_namespace}"
            raise NoSuchNamespaceError(msg)

        with self._transaction() as con:
            try:
                row = con.execute(
                    _rename_iceberg_table_sql(),
                    [to_namespace, to_table_name, self.name, from_namespace, from_table_name],
                ).fetchone()
            except duckdb.ConstraintException as exc:
                msg = f"Table {to_namespace}.{to_table_name} already exists"
                raise TableAlreadyExistsError(msg) from exc
            if row is None:
                msg = f"Table does not exist: {from_table_name}"
                raise NoSuchTableError(msg)

        return self.load_table(to_identifier)

    def commit_table(
        self,
        table: Table,
        requirements: tuple[TableRequirement, ...],
        updates: tuple[TableUpdate, ...],
    ) -> CommitTableResponse:
        """Commit updates to a table with optimistic concurrency checks.

        Parameters
        ----------
        table
            Table to update.
        requirements
            Update requirements for optimistic concurrency.
        updates
            Updates to apply to the table metadata.

        Returns
        -------
        CommitTableResponse
            Commit response containing metadata and location.

        Raises
        ------
        CommitFailedException
            If the table metadata changed during the commit.
        TableAlreadyExistsError
            If the table is created concurrently.
        """
        table_identifier = table.name()
        namespace_tuple = Catalog.namespace_from(table_identifier)
        namespace = Catalog.namespace_to_string(namespace_tuple)
        table_name = Catalog.table_name_from(table_identifier)

        try:
            current_table = self.load_table(table_identifier)
        except NoSuchTableError:
            current_table = None

        updated_staged_table = self._update_and_stage_table(
            current_table,
            table.name(),
            requirements,
            updates,
        )
        if current_table and updated_staged_table.metadata == current_table.metadata:
            return CommitTableResponse.model_construct(
                metadata=current_table.metadata,
                metadata_location=current_table.metadata_location,
            )
        self._write_metadata(
            metadata=updated_staged_table.metadata,
            io=updated_staged_table.io,
            metadata_path=updated_staged_table.metadata_location,
        )

        with self._transaction() as con:
            if current_table:
                row = con.execute(
                    _commit_iceberg_table_sql(),
                    [
                        updated_staged_table.metadata_location,
                        current_table.metadata_location,
                        self.name,
                        namespace,
                        table_name,
                        current_table.metadata_location,
                    ],
                ).fetchone()
                if row is None:
                    msg = (
                        "Table has been updated by another process: "
                        f"{namespace}.{table_name}"
                    )
                    raise CommitFailedException(msg)
            else:
                try:
                    con.execute(
                        _insert_iceberg_tables_sql(),
                        [
                            self.name,
                            namespace,
                            table_name,
                            updated_staged_table.metadata_location,
                            None,
                        ],
                    )
                except duckdb.ConstraintException as exc:
                    msg = f"Table {namespace}.{table_name} already exists"
                    raise TableAlreadyExistsError(msg) from exc

        return CommitTableResponse.model_construct(
            metadata=updated_staged_table.metadata,
            metadata_location=updated_staged_table.metadata_location,
        )

    def create_namespace(
        self,
        namespace: str | Identifier,
        properties: Properties = EMPTY_DICT,
    ) -> None:
        """Create a namespace in the catalog.

        Parameters
        ----------
        namespace
            Namespace identifier.
        properties
            Optional namespace properties.

        Raises
        ------
        NamespaceAlreadyExistsError
            If the namespace already exists.
        """
        if self._namespace_exists(namespace):
            msg = f"Namespace {namespace} already exists"
            raise NamespaceAlreadyExistsError(msg)
        namespace_str = Catalog.namespace_to_string(namespace, NoSuchNamespaceError)
        create_properties = properties or NAMESPACE_MINIMAL_PROPERTIES
        rows = [
            (self.name, namespace_str, key, value) for key, value in create_properties.items()
        ]
        with self._transaction() as con:
            try:
                con.executemany(_insert_namespace_properties_sql(), rows)
            except duckdb.ConstraintException as exc:
                msg = f"Namespace {namespace_str} already exists"
                raise NamespaceAlreadyExistsError(msg) from exc

    def drop_namespace(self, namespace: str | Identifier) -> None:
        """Drop a namespace if it is empty.

        Parameters
        ----------
        namespace
            Namespace identifier.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        NamespaceNotEmptyError
            If the namespace has tables.
        """
        if not self._namespace_exists(namespace):
            msg = f"Namespace does not exist: {namespace}"
            raise NoSuchNamespaceError(msg)
        namespace_str = Catalog.namespace_to_string(namespace)
        if tables := self.list_tables(namespace):
            msg = (
                f"Namespace {namespace_str} is not empty. "
                f"{len(tables)} tables exist."
            )
            raise NamespaceNotEmptyError(msg)
        con = self._session.connect()
        con.execute(_delete_namespace_sql(), [self.name, namespace_str])

    def list_tables(self, namespace: str | Identifier) -> list[Identifier]:
        """List tables under the provided namespace.

        Parameters
        ----------
        namespace
            Namespace identifier.

        Returns
        -------
        list[Identifier]
            Table identifiers within the namespace.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        """
        if namespace and not self._namespace_exists(namespace):
            msg = f"Namespace does not exist: {namespace}"
            raise NoSuchNamespaceError(msg)
        namespace_str = Catalog.namespace_to_string(namespace)
        con = self._session.connect()
        rows = con.execute(
            _list_tables_sql(),
            [self.name, namespace_str],
        ).fetchall()
        return [
            (*Catalog.identifier_to_tuple(str(row[0])), str(row[1])) for row in rows
        ]

    def list_namespaces(self, namespace: str | Identifier = ()) -> list[Identifier]:
        """List child namespaces under the provided namespace.

        Parameters
        ----------
        namespace
            Parent namespace identifier.

        Returns
        -------
        list[Identifier]
            Child namespace identifiers.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        """
        if namespace and not self._namespace_exists(namespace):
            msg = f"Namespace does not exist: {namespace}"
            raise NoSuchNamespaceError(msg)
        namespace_tuple = Catalog.identifier_to_tuple(namespace)
        namespace_str = (
            Catalog.namespace_to_string(namespace, NoSuchNamespaceError)
            if namespace
            else None
        )
        con = self._session.connect()
        table_rows = con.execute(
            _list_namespaces_from_tables_sql(with_prefix=namespace_str is not None),
            _namespace_query_params(self.name, namespace_str),
        ).fetchall()
        prop_rows = con.execute(
            _list_namespaces_from_props_sql(with_prefix=namespace_str is not None),
            _namespace_query_params(self.name, namespace_str),
        ).fetchall()

        namespaces = {Catalog.identifier_to_tuple(str(row[0])) for row in table_rows}
        namespaces.update({Catalog.identifier_to_tuple(str(row[0])) for row in prop_rows})

        sub_namespaces_level_length = len(namespace_tuple) + 1
        candidates = {
            ns[:sub_namespaces_level_length]
            for ns in namespaces
            if len(ns) >= sub_namespaces_level_length
            and ns[: sub_namespaces_level_length - 1] == namespace_tuple
        }
        return sorted(candidates)

    def list_views(self, namespace: str | Identifier) -> list[Identifier]:
        """DuckDB catalog does not support views.

        Raises
        ------
        NotImplementedError
            DuckDB catalog does not support views.
        """
        raise NotImplementedError

    def view_exists(self, identifier: str | Identifier) -> bool:
        """DuckDB catalog does not support views.

        Raises
        ------
        NotImplementedError
            DuckDB catalog does not support views.
        """
        raise NotImplementedError

    def load_namespace_properties(self, namespace: str | Identifier) -> Properties:
        """Return properties for the provided namespace.

        Parameters
        ----------
        namespace
            Namespace identifier.

        Returns
        -------
        Properties
            Namespace properties.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        """
        namespace_str = Catalog.namespace_to_string(namespace)
        if not self._namespace_exists(namespace):
            msg = f"Namespace {namespace_str} does not exists"
            raise NoSuchNamespaceError(msg)
        con = self._session.connect()
        rows = con.execute(
            _list_namespace_properties_sql(),
            [self.name, namespace_str],
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def update_namespace_properties(
        self,
        namespace: str | Identifier,
        removals: set[str] | None = None,
        updates: Properties = EMPTY_DICT,
    ) -> PropertiesUpdateSummary:
        """Update namespace properties and return a summary.

        Parameters
        ----------
        namespace
            Namespace identifier.
        removals
            Property keys to remove.
        updates
            Property values to set or update.

        Returns
        -------
        PropertiesUpdateSummary
            Summary of applied namespace property updates.

        Raises
        ------
        NoSuchNamespaceError
            If the namespace does not exist.
        """
        namespace_str = Catalog.namespace_to_string(namespace)
        if not self._namespace_exists(namespace):
            msg = f"Namespace {namespace_str} does not exists"
            raise NoSuchNamespaceError(msg)

        current_properties = self.load_namespace_properties(namespace=namespace)
        summary, _ = self._get_updated_props_and_update_summary(
            current_properties=current_properties,
            removals=removals,
            updates=updates,
        )

        with self._transaction() as con:
            if removals:
                delete_rows = [(self.name, namespace_str, key) for key in removals]
                con.executemany(_delete_namespace_property_sql(), delete_rows)
            if updates:
                delete_rows = [(self.name, namespace_str, key) for key in updates]
                con.executemany(_delete_namespace_property_sql(), delete_rows)
                insert_rows = [
                    (self.name, namespace_str, key, value) for key, value in updates.items()
                ]
                con.executemany(_insert_namespace_properties_sql(), insert_rows)

        return summary

    def drop_view(self, identifier: str | Identifier) -> None:
        """DuckDB catalog does not support views.

        Raises
        ------
        NotImplementedError
            DuckDB catalog does not support views.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close any catalog connections."""
        self._session.close()

    def _create_staged_table(
        self,
        identifier: str | Identifier,
        schema: Schema | pa.Schema,
        *args: object,
        **kwargs: object,
    ) -> StagedTable:
        """Build a staged table without committing catalog metadata.

        Parameters
        ----------
        identifier
            Table identifier.
        schema
            Iceberg or Arrow schema for the table.
        *args
            Optional positional args: location, partition_spec, sort_order, properties.
        **kwargs
            Optional keyword args: location, partition_spec, sort_order, properties.

        Returns
        -------
        StagedTable
            Staged table with metadata persisted to storage.
        """
        options = _parse_create_table_options(args, kwargs)
        properties = options.properties
        schema = self._convert_schema_if_needed(
            schema,
            _format_version_from_properties(properties),
        )
        namespace_identifier = Catalog.namespace_from(identifier)
        table_name = Catalog.table_name_from(identifier)
        namespace = Catalog.namespace_to_string(namespace_identifier, NoSuchNamespaceError)

        location = self._resolve_table_location(options.location, namespace, table_name)
        provider = load_location_provider(location, properties)
        metadata_location = provider.new_table_metadata_file_location()
        metadata = new_table_metadata(
            location=location,
            schema=schema,
            partition_spec=options.partition_spec,
            sort_order=options.sort_order,
            properties=properties,
        )
        io = self._load_file_io(properties=properties, location=metadata_location)
        return StagedTable(
            identifier=(*namespace_identifier, table_name),
            metadata=metadata,
            metadata_location=metadata_location,
            io=io,
            catalog=self,
        )

    def _convert_row_to_table(self, row: Sequence[object]) -> Table:
        """Convert a catalog row into a loaded Iceberg table.

        Parameters
        ----------
        row
            Row containing metadata location, namespace, and table name.

        Returns
        -------
        Table
            Loaded Iceberg table.

        Raises
        ------
        NoSuchTableError
            If required table metadata is missing.
        """
        metadata_location = row[0]
        table_namespace = row[1]
        table_name = row[2]
        if not metadata_location:
            msg = f"Table property {METADATA_LOCATION} is missing"
            raise NoSuchTableError(msg)
        if not table_namespace:
            msg = "Table property table_namespace is missing"
            raise NoSuchTableError(msg)
        if not table_name:
            msg = "Table property table_name is missing"
            raise NoSuchTableError(msg)
        metadata_location_str = str(metadata_location)
        io = load_file_io(properties=self.properties, location=metadata_location_str)
        file = io.new_input(metadata_location_str)
        metadata = FromInputFile.table_metadata(file)
        identifier = (*Catalog.identifier_to_tuple(str(table_namespace)), str(table_name))
        return Table(
            identifier=identifier,
            metadata=metadata,
            metadata_location=metadata_location_str,
            io=self._load_file_io(metadata.properties, metadata_location_str),
            catalog=self,
        )

    def _namespace_exists(self, identifier: str | Identifier) -> bool:
        """Return True when a namespace exists in the catalog.

        Parameters
        ----------
        identifier
            Namespace identifier.

        Returns
        -------
        bool
            True when the namespace exists.

        """
        namespace_tuple = Catalog.identifier_to_tuple(identifier)
        namespace = Catalog.namespace_to_string(namespace_tuple, NoSuchNamespaceError)
        namespace_prefix = f"{namespace}."

        con = self._session.connect()
        table_row = con.execute(
            _namespace_exists_in_tables_sql(),
            [self.name, namespace, namespace_prefix],
        ).fetchone()
        if table_row is not None:
            return True
        prop_row = con.execute(
            _namespace_exists_in_props_sql(),
            [self.name, namespace, namespace_prefix],
        ).fetchone()
        return prop_row is not None

    @contextmanager
    def _transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Run catalog operations inside a DuckDB transaction.

        Yields
        ------
        duckdb.DuckDBPyConnection
            Connection bound to the open transaction.

        """
        con = self._session.connect()
        con.execute("BEGIN")
        try:
            yield con
        except Exception:
            con.execute("ROLLBACK")
            raise
        else:
            con.execute("COMMIT")


@lru_cache(maxsize=2)
def _catalog_table_ddl_sql(table_name: str) -> str:
    if table_name not in _CATALOG_TABLE_DDL:
        msg = f"Unknown catalog table: {table_name}"
        raise ValueError(msg)
    sql = _CATALOG_TABLE_DDL[table_name]
    parsed = sqlglot.parse_one(sql, read=DUCKDB_DIALECT)
    if not isinstance(parsed, exp.Create):
        msg = "Generated catalog DDL did not produce a CREATE statement"
        raise TypeError(msg)
    return render_sql_duckdb(parsed)


def _table_expr(table_name: str) -> exp.Table:
    return exp.Table(this=exp.to_identifier(table_name))


@lru_cache(maxsize=1)
def _insert_iceberg_tables_sql() -> str:
    columns = (
        "catalog_name",
        "table_namespace",
        "table_name",
        "metadata_location",
        "previous_metadata_location",
    )
    statement = exp.Insert(
        this=exp.Schema(
            this=_table_expr(_ICEBERG_TABLES),
            expressions=[exp.to_identifier(column) for column in columns],
        ),
        expression=exp.Values(
            expressions=[exp.Tuple(expressions=[exp.Placeholder() for _ in columns])]
        ),
    )
    return render_sql_duckdb(statement)


@lru_cache(maxsize=1)
def _insert_namespace_properties_sql() -> str:
    columns = ("catalog_name", "namespace", "property_key", "property_value")
    statement = exp.Insert(
        this=exp.Schema(
            this=_table_expr(_ICEBERG_NAMESPACE_PROPERTIES),
            expressions=[exp.to_identifier(column) for column in columns],
        ),
        expression=exp.Values(
            expressions=[exp.Tuple(expressions=[exp.Placeholder() for _ in columns])]
        ),
    )
    return render_sql_duckdb(statement)


@lru_cache(maxsize=1)
def _select_iceberg_table_sql() -> str:
    table = _table_expr(_ICEBERG_TABLES)
    query = (
        exp.select(
            exp.column("metadata_location"),
            exp.column("table_namespace"),
            exp.column("table_name"),
        )
        .from_(table)
        .where(
            exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_name"), expression=exp.Placeholder()),
            )
        )
    )
    return render_sql_duckdb(query)


@lru_cache(maxsize=1)
def _delete_iceberg_table_sql() -> str:
    statement = exp.Delete(
        this=_table_expr(_ICEBERG_TABLES),
        where=exp.Where(
            this=exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_name"), expression=exp.Placeholder()),
            )
        ),
        returning=exp.Returning(expressions=[exp.Literal.number(1)]),
    )
    return render_sql_duckdb(statement)


@lru_cache(maxsize=1)
def _rename_iceberg_table_sql() -> str:
    statement = exp.Update(
        this=_table_expr(_ICEBERG_TABLES),
        expressions=[
            exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
            exp.EQ(this=exp.column("table_name"), expression=exp.Placeholder()),
        ],
        where=exp.Where(
            this=exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_name"), expression=exp.Placeholder()),
            )
        ),
        returning=exp.Returning(expressions=[exp.Literal.number(1)]),
    )
    return render_sql_duckdb(statement)


@lru_cache(maxsize=1)
def _commit_iceberg_table_sql() -> str:
    statement = exp.Update(
        this=_table_expr(_ICEBERG_TABLES),
        expressions=[
            exp.EQ(this=exp.column("metadata_location"), expression=exp.Placeholder()),
            exp.EQ(this=exp.column("previous_metadata_location"), expression=exp.Placeholder()),
        ],
        where=exp.Where(
            this=exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("metadata_location"), expression=exp.Placeholder()),
            )
        ),
        returning=exp.Returning(expressions=[exp.Literal.number(1)]),
    )
    return render_sql_duckdb(statement)


@lru_cache(maxsize=1)
def _list_tables_sql() -> str:
    table = _table_expr(_ICEBERG_TABLES)
    query = (
        exp.select(
            exp.column("table_namespace"),
            exp.column("table_name"),
        )
        .from_(table)
        .where(
            exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
            )
        )
        .order_by(exp.column("table_name"))
    )
    return render_sql_duckdb(query)


@lru_cache(maxsize=1)
def _list_namespace_properties_sql() -> str:
    table = _table_expr(_ICEBERG_NAMESPACE_PROPERTIES)
    query = (
        exp.select(
            exp.column("property_key"),
            exp.column("property_value"),
        )
        .from_(table)
        .where(
            exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("namespace"), expression=exp.Placeholder()),
            )
        )
        .order_by(exp.column("property_key"))
    )
    return render_sql_duckdb(query)


@lru_cache(maxsize=1)
def _delete_namespace_property_sql() -> str:
    statement = exp.Delete(
        this=_table_expr(_ICEBERG_NAMESPACE_PROPERTIES),
        where=exp.Where(
            this=exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("namespace"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("property_key"), expression=exp.Placeholder()),
            )
        ),
    )
    return render_sql_duckdb(statement)


@lru_cache(maxsize=1)
def _delete_namespace_sql() -> str:
    statement = exp.Delete(
        this=_table_expr(_ICEBERG_NAMESPACE_PROPERTIES),
        where=exp.Where(
            this=exp.and_(
                exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
                exp.EQ(this=exp.column("namespace"), expression=exp.Placeholder()),
            )
        ),
    )
    return render_sql_duckdb(statement)


def _starts_with_expr(column: str) -> exp.Expression:
    return exp.Anonymous(this="starts_with", expressions=[exp.column(column), exp.Placeholder()])


@lru_cache(maxsize=1)
def _namespace_exists_in_tables_sql() -> str:
    table = _table_expr(_ICEBERG_TABLES)
    predicate = exp.and_(
        exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
        exp.or_(
            exp.EQ(this=exp.column("table_namespace"), expression=exp.Placeholder()),
            _starts_with_expr("table_namespace"),
        ),
    )
    query = exp.select(exp.Literal.number(1)).from_(table).where(predicate).limit(1)
    return render_sql_duckdb(query)


@lru_cache(maxsize=1)
def _namespace_exists_in_props_sql() -> str:
    table = _table_expr(_ICEBERG_NAMESPACE_PROPERTIES)
    predicate = exp.and_(
        exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder()),
        exp.or_(
            exp.EQ(this=exp.column("namespace"), expression=exp.Placeholder()),
            _starts_with_expr("namespace"),
        ),
    )
    query = exp.select(exp.Literal.number(1)).from_(table).where(predicate).limit(1)
    return render_sql_duckdb(query)


@lru_cache(maxsize=2)
def _list_namespaces_from_tables_sql(*, with_prefix: bool) -> str:
    table = _table_expr(_ICEBERG_TABLES)
    predicate = exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder())
    if with_prefix:
        predicate = exp.and_(predicate, _starts_with_expr("table_namespace"))
    query = exp.select(exp.column("table_namespace")).from_(table).where(predicate)
    return render_sql_duckdb(query)


@lru_cache(maxsize=2)
def _list_namespaces_from_props_sql(*, with_prefix: bool) -> str:
    table = _table_expr(_ICEBERG_NAMESPACE_PROPERTIES)
    predicate = exp.EQ(this=exp.column("catalog_name"), expression=exp.Placeholder())
    if with_prefix:
        predicate = exp.and_(predicate, _starts_with_expr("namespace"))
    query = exp.select(exp.column("namespace")).from_(table).where(predicate)
    return render_sql_duckdb(query)


def _namespace_query_params(catalog_name: str, namespace: str | None) -> list[str]:
    if namespace is None:
        return [catalog_name]
    return [catalog_name, namespace]


__all__ = ["DuckDBCatalog", "DuckDBCatalogSession"]
