"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for semantic layer queries, used by
both FastAPI routes and MCP tools.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.core.schemas.hashing import schema_hash
from codeintel.serving.meta.service import build_kernel_meta_payload
from codeintel.serving.search.models import SearchQueryResponse, SearchResult
from codeintel.serving.semantic.fingerprints import (
    SemanticQueryFingerprintInput,
    fingerprint_search,
    fingerprint_semantic_query,
)
from codeintel.serving.semantic.models import (
    SemanticExplainResponse,
    SemanticQueryResponse,
)
from codeintel.serving.semantic.query_builder import SemanticQueryPlan, build_query
from codeintel.serving.semantic.templates import DbApiTemplate
from codeintel.serving.snapshot.models import ServingSnapshotIdentity
from codeintel.storage.queries.safe import UnsafeSqlError, assert_single_select_statement
from codeintel.storage.serving.search_index import fts_index_available

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.db.manager import ServingDBManager, ServingSnapshotContext
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.models import (
        FilterSpec,
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticViewSpec,
    )
    from codeintel.serving.semantic.templates import BoundQuery, DbApiQuery
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.warehouse import Warehouse

LOG = logging.getLogger(__name__)

_SEARCH_TABLE_SCHEMA = "docs"
_SEARCH_TABLE_NAME = "search_documents"
_SEARCH_TABLE_KEY = "docs.search_documents"


class UnknownViewIdError(KeyError):
    """Raise when a semantic view identifier cannot be resolved."""

    def __init__(self, view_id: str) -> None:
        super().__init__(view_id)
        self.view_id = view_id


class UnknownColumnsError(ValueError):
    """Raise when a request selects columns not allowed by a semantic view."""

    def __init__(self, *, unknown: tuple[str, ...], allowed: tuple[str, ...]) -> None:
        unknown_list = list(unknown)
        allowed_sorted = sorted(allowed)
        message = f"Unknown columns requested: {unknown_list}. Allowed columns: {allowed_sorted}"
        super().__init__(message)
        self.unknown = unknown
        self.allowed = allowed


@dataclass(frozen=True, slots=True)
class _ResolvedViewContext:
    pointer: ServingSnapshotPointer
    view: SemanticViewSpec
    inventory: SchemaInventory
    allowed_columns: list[str]
    column_types: dict[str, ColumnType] | None


@dataclass(frozen=True, slots=True)
class _PlanInputs:
    columns: list[str]
    filters: list[FilterSpec]
    order_by: list[str]
    offset: int


_SQL_SEARCH_FTS = """
	SELECT kind, name, module, rel_path, ref_goid_h128, score
	FROM (
	    SELECT
        kind,
        name,
        module,
        rel_path,
        ref_goid_h128,
        fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
) ranked
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_FTS_KINDS = """
SELECT kind, name, module, rel_path, ref_goid_h128, score
FROM (
    SELECT
        kind,
        name,
        module,
        rel_path,
        ref_goid_h128,
        fts_docs_search_documents.match_bm25(doc_id, ?) AS score
    FROM docs.search_documents
    WHERE kind = ANY(?)
) ranked
WHERE score IS NOT NULL
ORDER BY score DESC
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_LIKE = """
SELECT kind, name, module, rel_path, ref_goid_h128, NULL AS score
FROM docs.search_documents
WHERE (
    COALESCE(text, '') ILIKE '%' || ? || '%'
    OR COALESCE(name, '') ILIKE '%' || ? || '%'
    OR COALESCE(module, '') ILIKE '%' || ? || '%'
)
ORDER BY kind, name
LIMIT ? OFFSET ?
"""

_SQL_SEARCH_LIKE_KINDS = """
SELECT kind, name, module, rel_path, ref_goid_h128, NULL AS score
FROM docs.search_documents
WHERE (
    COALESCE(text, '') ILIKE '%' || ? || '%'
    OR COALESCE(name, '') ILIKE '%' || ? || '%'
    OR COALESCE(module, '') ILIKE '%' || ? || '%'
)
AND kind = ANY(?)
ORDER BY kind, name
LIMIT ? OFFSET ?
"""

_SEARCH_QUERY_FTS = DbApiTemplate(sql=_SQL_SEARCH_FTS)
_SEARCH_QUERY_FTS_KINDS = DbApiTemplate(sql=_SQL_SEARCH_FTS_KINDS)
_SEARCH_QUERY_LIKE = DbApiTemplate(sql=_SQL_SEARCH_LIKE)
_SEARCH_QUERY_LIKE_KINDS = DbApiTemplate(sql=_SQL_SEARCH_LIKE_KINDS)


def _sanitize_float_nan(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _sanitize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: _sanitize_float_nan(v) for k, v in row.items()} for row in rows]


def _format_explain_rows(rows: Sequence[Sequence[object]]) -> str:
    plan_lines: list[str] = []
    for row in rows:
        if not row:
            continue
        plan_lines.append(str(row[1] if len(row) > 1 else row[0]))
    return "\n".join(plan_lines)


@dataclass
class SemanticQueryKernel:
    """Unified query kernel for semantic layer access.

    Parameters
    ----------
    db
        Database manager for connection access.
    """

    db: ServingDBManager
    settings: ServingSettings

    def _snapshot_context(self, pointer: ServingSnapshotPointer) -> ServingSnapshotContext:
        """Return the cached snapshot context for a pointer.

        Parameters
        ----------
        pointer
            Snapshot pointer describing the active artifact paths.

        Returns
        -------
        ServingSnapshotContext
            Cached registry/inventory/buildspec for the snapshot.
        """
        return self.db.snapshot_context(pointer)

    def _resolve_allowed_columns(
        self,
        *,
        view: SemanticViewSpec,
        inventory: SchemaInventory,
    ) -> list[str]:
        """Resolve allowed columns for a view, enforcing schema manifest when enabled.

        Parameters
        ----------
        view
            Semantic view specification.
        inventory
            Schema inventory loaded from the current snapshot.

        Returns
        -------
        list[str]
            Allowed column names in result order.

        Raises
        ------
        ValueError
            If the view's table is missing from the manifest or exposes unknown columns in strict mode.
        """
        schema = inventory.get(view.table_key)
        if schema is None:
            msg = f"View table_key not present in schema manifest: {view.table_key}"
            raise ValueError(msg)

        schema_cols = [c.name for c in schema.columns]
        if not view.columns:
            return schema_cols

        unknown = sorted(set(view.columns) - set(schema_cols))
        mode = self.settings.schema_enforcement.lower()
        if unknown and mode == "strict":
            msg = f"Semantic view {view.id} exposes unknown columns: {unknown}"
            raise ValueError(msg)
        if unknown and mode == "warn":
            LOG.warning(
                "serving.semantic.columns.unknown view_id=%s table_key=%s unknown=%s",
                view.id,
                view.table_key,
                unknown,
            )
            return [c for c in view.columns if c in schema_cols]
        if mode == "off":
            return list(view.columns)

        return list(view.columns)

    def _execute_sql(
        self,
        *,
        warehouse: Warehouse,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, object]]:
        engine = self.settings.result_engine.lower()
        backend = warehouse.gateway.policy
        result = backend.execute_sql(sql, params=params)

        if engine == "polars" and pl is not None:
            df_pl = result.pl()
            return _sanitize_rows(df_pl.to_dicts())

        if engine == "polars" and pl is None:
            LOG.warning("polars not installed; falling back to pandas result extraction")

        df_pd = result.df()
        sanitized = df_pd.astype("object").where(pd.notna(df_pd), None)
        return sanitized.to_dict(orient="records")

    @staticmethod
    def _snapshot_dict(pointer: ServingSnapshotPointer) -> ServingSnapshotIdentity:
        """Return a stable snapshot identity model for responses.

        Returns
        -------
        ServingSnapshotIdentity
            Snapshot identity model.
        """
        return ServingSnapshotIdentity.from_pointer(pointer)

    @staticmethod
    def _schema_hash_for_table_key(
        *,
        inventory: SchemaInventory,
        table_key: str,
    ) -> str | None:
        schema = inventory.get(table_key)
        if schema is None:
            return None
        return schema_hash(schema)

    def _fingerprint_semantic_plan(
        self,
        *,
        pointer: ServingSnapshotPointer,
        view_id: str,
        plan: SemanticQueryPlan,
        inventory: SchemaInventory,
    ) -> tuple[str, str | None]:
        filter_dicts = [f.model_dump(mode="json") for f in plan.filters]
        schema_hash_value = self._schema_hash_for_table_key(inventory=inventory, table_key=plan.table_key)
        inputs = SemanticQueryFingerprintInput(
            snapshot=self._snapshot_dict(pointer).model_dump(mode="json"),
            view_id=view_id,
            table_key=plan.table_key,
            select=plan.columns,
            order_by=plan.order_by,
            filters=filter_dicts,
            limit=plan.limit,
            offset=plan.offset,
            schema_hash=schema_hash_value,
        )
        query_hash = fingerprint_semantic_query(inputs)
        return query_hash, schema_hash_value

    def _execute_semantic_plan(
        self,
        *,
        warehouse: Warehouse,
        plan: SemanticQueryPlan,
        column_types: dict[str, ColumnType] | None,
    ) -> list[dict[str, object]]:
        ibis_con = warehouse.gateway.ibis.con
        built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
        return self._execute_bound_query(warehouse=warehouse, query=built)

    def _execute_bound_query(self, *, warehouse: Warehouse, query: BoundQuery) -> list[dict[str, object]]:
        ibis_con = warehouse.gateway.ibis.con
        try:
            sql = query.compile_sql(ibis_con)
            assert_single_select_statement(sql)
            return self._execute_sql(warehouse=warehouse, sql=sql)
        except UnsafeSqlError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=query.temp_tables)

    def _execute_dbapi_query(self, *, warehouse: Warehouse, query: DbApiQuery) -> list[dict[str, object]]:
        try:
            assert_single_select_statement(query.sql)
        except UnsafeSqlError as exc:
            raise ValueError(str(exc)) from exc
        return self._execute_sql(warehouse=warehouse, sql=query.sql, params=query.params)

    def _resolve_view_context(self, *, pointer: ServingSnapshotPointer, view_id: str) -> _ResolvedViewContext:
        context = self._snapshot_context(pointer)
        inventory = context.inventory
        view = context.registry.by_id(view_id)
        allowed_columns = self._resolve_allowed_columns(view=view, inventory=inventory)
        column_types = _column_types_for_view(view=view, inventory=inventory)
        return _ResolvedViewContext(
            pointer=pointer,
            view=view,
            inventory=inventory,
            allowed_columns=allowed_columns,
            column_types=column_types,
        )

    def _resolve_view_context_for_export(
        self, *, pointer: ServingSnapshotPointer, view_id: str
    ) -> _ResolvedViewContext:
        try:
            return self._resolve_view_context(pointer=pointer, view_id=view_id)
        except KeyError as exc:
            raise UnknownViewIdError(view_id) from exc

    @staticmethod
    def _plan_from_inputs(*, ctx: _ResolvedViewContext, inputs: _PlanInputs, limit: int) -> SemanticQueryPlan:
        return SemanticQueryPlan(
            table_key=ctx.view.table_key,
            columns=inputs.columns,
            allowed_columns=frozenset(ctx.allowed_columns),
            filters=inputs.filters,
            order_by=inputs.order_by,
            limit=limit,
            offset=inputs.offset,
        )

    @staticmethod
    def _query_plan_inputs(*, ctx: _ResolvedViewContext, request: SemanticQueryRequest) -> tuple[_PlanInputs, int]:
        columns = request.select if request.select else ctx.allowed_columns
        effective_limit = request.limit if request.limit else ctx.view.defaults.limit
        effective_order = request.order_by if request.order_by else ctx.view.defaults.order_by
        inputs = _PlanInputs(
            columns=columns,
            filters=request.filters,
            order_by=effective_order,
            offset=request.offset,
        )
        return inputs, effective_limit

    @staticmethod
    def _export_plan_inputs(*, ctx: _ResolvedViewContext, request: SemanticExportRequest) -> tuple[_PlanInputs, int]:
        if request.select:
            unknown = [col for col in request.select if col not in ctx.allowed_columns]
            if unknown:
                raise UnknownColumnsError(
                    unknown=tuple(unknown),
                    allowed=tuple(ctx.allowed_columns),
                )
            columns = list(request.select)
        else:
            columns = ctx.allowed_columns

        inputs = _PlanInputs(
            columns=columns,
            filters=request.filters,
            order_by=request.order_by,
            offset=request.offset,
        )
        return inputs, request.limit

    @staticmethod
    def _compile_safe_sql(
        *,
        warehouse: Warehouse,
        plan: SemanticQueryPlan,
        column_types: dict[str, ColumnType] | None,
        cleanup_temp_tables: bool,
    ) -> tuple[str, tuple[str, ...]]:
        ibis_con = warehouse.gateway.ibis.con
        built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
        try:
            compiled = built.compile_sql(ibis_con)
            assert_single_select_statement(compiled)
        except UnsafeSqlError as exc:
            _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=built.temp_tables)
            raise ValueError(str(exc)) from exc
        except Exception:
            _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=built.temp_tables)
            raise

        if cleanup_temp_tables:
            _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=built.temp_tables)

        return compiled, built.temp_tables

    def catalog(self) -> dict[str, object]:
        """List all available semantic views.

        Returns
        -------
        dict[str, object]
            Catalog response with version, snapshot, and views.
        """
        pointer = self.db.current_pointer()
        context = self._snapshot_context(pointer)
        registry = context.registry

        return {
            "version": registry.version,
            "snapshot": self._snapshot_dict(pointer),
            "views": [
                {
                    "id": v.id,
                    "table_key": v.table_key,
                    "entity": v.entity,
                    "grain": v.grain,
                    "description": v.description,
                    "column_count": len(v.columns),
                }
                for v in registry.views
                if not v.deprecated
            ],
        }

    def describe(self, view_id: str) -> dict[str, object]:
        """Describe a single semantic view.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        dict[str, object]
            View description with schema details.
        """
        pointer = self.db.current_pointer()
        context = self._snapshot_context(pointer)
        registry = context.registry
        inventory = context.inventory

        view = registry.by_id(view_id)
        table_schema = inventory.get(view.table_key)

        column_types: dict[str, str] = {}
        if table_schema is not None:
            column_types = {c.name: c.type for c in table_schema.columns}

        return {
            "id": view.id,
            "table_key": view.table_key,
            "kind": view.kind,
            "entity": view.entity,
            "grain": view.grain,
            "description": view.description,
            "primary_key": view.primary_key,
            "columns": view.columns,
            "column_types": column_types,
            "joins": view.joins,
            "defaults": view.defaults.model_dump(mode="json"),
            "deprecated": view.deprecated,
            "replaced_by": view.replaced_by,
            "snapshot": self._snapshot_dict(pointer),
        }

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        """Execute a semantic view query.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.

        Returns
        -------
        SemanticQueryResponse
            Query results.
        """
        with self.db.connect() as (warehouse, pointer):
            resolved = self._resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._query_plan_inputs(ctx=resolved, request=request)
            plan = self._plan_from_inputs(ctx=resolved, inputs=inputs, limit=effective_limit + 1)
            rows = self._execute_semantic_plan(
                warehouse=warehouse,
                plan=plan,
                column_types=resolved.column_types,
            )

        truncated = len(rows) > effective_limit
        if truncated:
            rows = rows[:effective_limit]

        fingerprint_plan = self._plan_from_inputs(ctx=resolved, inputs=inputs, limit=effective_limit)
        query_hash, schema_hash_value = self._fingerprint_semantic_plan(
            pointer=resolved.pointer,
            view_id=resolved.view.id,
            plan=fingerprint_plan,
            inventory=resolved.inventory,
        )

        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=inputs.columns,
            rows=rows,
            truncated=truncated,
            snapshot=self._snapshot_dict(resolved.pointer),
            query_hash=query_hash,
            schema_hash=schema_hash_value,
        )

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse:
        """Return compiled SQL and DuckDB EXPLAIN plan for a semantic query.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.

        Returns
        -------
        SemanticExplainResponse
            Explain output including compiled SQL and plan text.
        """
        with self.db.connect() as (warehouse, pointer):
            resolved = self._resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._query_plan_inputs(ctx=resolved, request=request)
            plan = self._plan_from_inputs(ctx=resolved, inputs=inputs, limit=effective_limit)
            compiled, _temp_tables = self._compile_safe_sql(
                warehouse=warehouse,
                plan=plan,
                column_types=resolved.column_types,
                cleanup_temp_tables=True,
            )

            raw_rows = warehouse.gateway.policy.execute_sql(f"EXPLAIN {compiled}").fetchall()
            plan_text = _format_explain_rows(raw_rows)

            return SemanticExplainResponse(
                view_id=request.view_id,
                sql=compiled,
                plan=plan_text,
                snapshot=self._snapshot_dict(pointer),
            )

    def compile_query_sql(self, request: SemanticQueryRequest) -> str:
        """Compile a semantic query to SQL without executing it.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.

        Returns
        -------
        str
            Compiled SQL string (validated select-only).
        """
        with self.db.connect() as (warehouse, pointer):
            resolved = self._resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._query_plan_inputs(ctx=resolved, request=request)
            plan = self._plan_from_inputs(ctx=resolved, inputs=inputs, limit=effective_limit + 1)
            compiled, _temp_tables = self._compile_safe_sql(
                warehouse=warehouse,
                plan=plan,
                column_types=resolved.column_types,
                cleanup_temp_tables=True,
            )
            return compiled

    def meta(self) -> dict[str, object]:
        """Return serving metadata for /meta endpoint and tools.

        Returns
        -------
        dict[str, object]
            Comprehensive serving metadata.
        """
        return build_kernel_meta_payload(self.db)

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse:
        """Search code metadata using `docs.search_documents` (FTS when available).

        Parameters
        ----------
        request
            Search request parameters.

        Returns
        -------
        SearchQueryResponse
            Search results with stable ranking when the FTS index is available.
        """
        engine = self.settings.result_engine.lower()

        with self.db.connect() as (warehouse, pointer):
            backend = warehouse.gateway.policy
            if not backend.table_exists(schema=_SEARCH_TABLE_SCHEMA, table=_SEARCH_TABLE_NAME):
                return SearchQueryResponse(
                    query=request.query,
                    results=[],
                    truncated=False,
                    snapshot=self._snapshot_dict(pointer),
                    engine=engine,
                    query_hash=fingerprint_search(
                        snapshot=self._snapshot_dict(pointer).model_dump(mode="json"),
                        query=request.query,
                        kinds=request.kinds,
                        limit=request.limit,
                        offset=request.offset,
                    ),
                )

            fts_available = fts_index_available(warehouse.gateway.con, table_key=_SEARCH_TABLE_KEY)

            query_limit = request.limit + 1
            if fts_available and request.kinds:
                query = _SEARCH_QUERY_FTS_KINDS.bind(
                    [request.query, request.kinds, query_limit, request.offset]
                )
            elif fts_available:
                query = _SEARCH_QUERY_FTS.bind([request.query, query_limit, request.offset])
            elif request.kinds:
                query = _SEARCH_QUERY_LIKE_KINDS.bind(
                    [
                        request.query,
                        request.query,
                        request.query,
                        request.kinds,
                        query_limit,
                        request.offset,
                    ]
                )
            else:
                query = _SEARCH_QUERY_LIKE.bind(
                    [request.query, request.query, request.query, query_limit, request.offset]
                )

            rows = self._execute_dbapi_query(warehouse=warehouse, query=query)

        truncated = len(rows) > request.limit
        if truncated:
            rows = rows[: request.limit]

        results = [SearchResult.model_validate(row) for row in rows]
        query_hash = fingerprint_search(
            snapshot=self._snapshot_dict(pointer).model_dump(mode="json"),
            query=request.query,
            kinds=request.kinds,
            limit=request.limit,
            offset=request.offset,
        )
        return SearchQueryResponse(
            query=request.query,
            results=results,
            truncated=truncated,
            snapshot=self._snapshot_dict(pointer),
            engine=engine,
            query_hash=query_hash,
        )

    def export_fingerprint(self, request: SemanticExportRequest) -> tuple[str, str | None]:
        """Return a stable fingerprint for an export request.

        Parameters
        ----------
        request
            Export request with filters, selection, and pagination.

        Returns
        -------
        tuple[str, str | None]
            (query_hash, schema_hash) for the export request.
        """
        pointer = self.db.current_pointer()
        resolved = self._resolve_view_context_for_export(pointer=pointer, view_id=request.view_id)
        inputs, effective_limit = self._export_plan_inputs(ctx=resolved, request=request)
        fingerprint_plan = self._plan_from_inputs(ctx=resolved, inputs=inputs, limit=effective_limit)
        return self._fingerprint_semantic_plan(
            pointer=pointer,
            view_id=resolved.view.id,
            plan=fingerprint_plan,
            inventory=resolved.inventory,
        )

    def export_rows(self, request: SemanticExportRequest) -> Generator[dict[str, object]]:
        """Yield rows for streaming export (memory-efficient).

        Unlike query(), this method yields rows one at a time to support
        large result sets without buffering everything in memory.

        Parameters
        ----------
        request
            Export request with filters, selection, and pagination.

        Yields
        ------
        dict[str, object]
            Row dictionary for each result row.
        """
        with self.db.connect_export() as (warehouse, pointer):
            sql, columns, temp_tables = self._compile_export_query(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )

            try:
                result = warehouse.gateway.policy.execute_sql(sql)
                reader = result.fetch_record_batch(self.settings.export_batch_size)
                for batch in reader:
                    payload = batch.to_pydict()
                    row_count = batch.num_rows
                    for idx in range(row_count):
                        yield {name: payload[name][idx] for name in columns}
            finally:
                _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=temp_tables)

    def export_sql(self, request: SemanticExportRequest) -> str:
        """Return the compiled SQL for an export request.

        Parameters
        ----------
        request
            Export request with filters, selection, and pagination.

        Returns
        -------
        str
            Compiled DuckDB SQL for the export.
        """
        with self.db.connect_export() as (warehouse, pointer):
            sql, _columns, temp_tables = self._compile_export_query(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )
            try:
                return sql
            finally:
                _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=temp_tables)

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an export result to a Parquet file.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            sql, _columns, temp_tables = self._compile_export_query(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )
            try:
                result = warehouse.gateway.policy.execute_sql(sql)
                reader = result.fetch_record_batch(self.settings.export_batch_size)
                rows_written = 0
                writer = pq.ParquetWriter(str(output_path), reader.schema)
                try:
                    for batch in reader:
                        rows_written += batch.num_rows
                        writer.write_table(pa.Table.from_batches([batch], schema=reader.schema))
                finally:
                    writer.close()
                return rows_written
            finally:
                _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=temp_tables)

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an export result to an Arrow IPC file.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            sql, _columns, temp_tables = self._compile_export_query(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )
            try:
                result = warehouse.gateway.policy.execute_sql(sql)
                reader = result.fetch_record_batch(self.settings.export_batch_size)
                rows_written = 0
                with output_path.open("wb") as handle, pa.ipc.new_file(
                    handle,
                    reader.schema,
                ) as writer:
                    for batch in reader:
                        rows_written += batch.num_rows
                        writer.write_batch(batch)
                return rows_written
            finally:
                _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=temp_tables)

    def _compile_export_query(
        self,
        *,
        warehouse: Warehouse,
        pointer: ServingSnapshotPointer,
        request: SemanticExportRequest,
    ) -> tuple[str, list[str], tuple[str, ...]]:
        resolved = self._resolve_view_context_for_export(pointer=pointer, view_id=request.view_id)
        inputs, effective_limit = self._export_plan_inputs(ctx=resolved, request=request)
        plan = self._plan_from_inputs(ctx=resolved, inputs=inputs, limit=effective_limit)
        sql, temp_tables = self._compile_safe_sql(
            warehouse=warehouse,
            plan=plan,
            column_types=resolved.column_types,
            cleanup_temp_tables=False,
        )
        return sql, inputs.columns, temp_tables


def _column_types_for_view(
    *,
    view: SemanticViewSpec,
    inventory: SchemaInventory,
) -> dict[str, ColumnType] | None:
    table_schema = inventory.get(view.table_key)
    if table_schema is None:
        return None
    return {col.name: col.type for col in table_schema.columns}


def _cleanup_temp_tables(*, con: object, temp_tables: tuple[str, ...]) -> None:
    unregister = getattr(con, "unregister", None)
    if not callable(unregister):
        return
    for table_name in temp_tables:
        unregister(table_name)


__all__ = ["SemanticQueryKernel"]
