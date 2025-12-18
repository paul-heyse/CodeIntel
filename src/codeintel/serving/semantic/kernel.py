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

from codeintel.serving.search.models import SearchQueryResponse, SearchResult
from codeintel.serving.semantic.models import (
    SemanticExplainResponse,
    SemanticQueryResponse,
)
from codeintel.serving.semantic.query_builder import SemanticQueryPlan, build_query
from codeintel.storage.serving.search_index import fts_index_available

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.db.manager import ServingDBManager, ServingSnapshotContext
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.models import (
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticViewSpec,
    )
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

    def _execute_semantic_plan(
        self,
        *,
        warehouse: Warehouse,
        plan: SemanticQueryPlan,
        column_types: dict[str, ColumnType] | None,
    ) -> list[dict[str, object]]:
        ibis_con = warehouse.gateway.ibis.con
        built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
        try:
            sql = ibis_con.compile(built.expr)
            return self._execute_sql(warehouse=warehouse, sql=sql)
        finally:
            _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=built.temp_tables)

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
            "snapshot": {"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
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
            "snapshot": {"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
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
            context = self._snapshot_context(pointer)
            view = context.registry.by_id(request.view_id)
            column_types = _column_types_for_view(view=view, inventory=context.inventory)
            allowed_columns = self._resolve_allowed_columns(view=view, inventory=context.inventory)
            columns = request.select if request.select else allowed_columns

            effective_limit = request.limit if request.limit else view.defaults.limit
            effective_order = request.order_by if request.order_by else view.defaults.order_by

            query_limit = effective_limit + 1
            plan = SemanticQueryPlan(
                table_key=view.table_key,
                columns=columns,
                allowed_columns=frozenset(allowed_columns),
                filters=request.filters,
                order_by=effective_order,
                limit=query_limit,
                offset=request.offset,
            )

            rows = self._execute_semantic_plan(
                warehouse=warehouse,
                plan=plan,
                column_types=column_types,
            )

        truncated = len(rows) > effective_limit
        if truncated:
            rows = rows[:effective_limit]

        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=columns,
            rows=rows,
            truncated=truncated,
            snapshot={"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
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
            context = self._snapshot_context(pointer)
            view = context.registry.by_id(request.view_id)

            allowed_columns = self._resolve_allowed_columns(view=view, inventory=context.inventory)
            columns = request.select if request.select else allowed_columns

            effective_limit = request.limit if request.limit else view.defaults.limit
            effective_order = request.order_by if request.order_by else view.defaults.order_by

            plan = SemanticQueryPlan(
                table_key=view.table_key,
                columns=columns,
                allowed_columns=frozenset(allowed_columns),
                filters=request.filters,
                order_by=effective_order,
                limit=effective_limit,
                offset=request.offset,
            )

            ibis_con = warehouse.gateway.ibis.con
            column_types = _column_types_for_view(view=view, inventory=context.inventory)
            built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
            try:
                compiled = ibis_con.compile(built.expr)
            finally:
                _cleanup_temp_tables(con=warehouse.gateway.con, temp_tables=built.temp_tables)

            raw_rows = warehouse.gateway.policy.execute_sql(f"EXPLAIN {compiled}").fetchall()
            plan_text = _format_explain_rows(raw_rows)

        return SemanticExplainResponse(
            view_id=request.view_id,
            sql=compiled,
            plan=plan_text,
            snapshot={"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
        )

    def meta(self) -> dict[str, object]:
        """Return serving metadata for /meta endpoint and tools.

        Returns
        -------
        dict[str, object]
            Comprehensive serving metadata.
        """
        pointer = self.db.current_pointer()
        context = self._snapshot_context(pointer)
        registry = context.registry
        spec = context.buildspec

        tables = sum(1 for d in spec.datasets if not d.table_key.startswith("docs.v_"))
        views = sum(1 for d in spec.datasets if d.table_key.startswith("docs.v_"))

        return {
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
            "published_at": pointer.published_at.isoformat(),
            "semantic_layer_version": pointer.semantic_layer_version,
            "buildspec_hash": spec.buildspec_hash,
            "buildspec_version": spec.spec_version,
            "duckdb": {"db_path": str(pointer.db_path), "read_only": True},
            "semantic_views": [
                {"id": v.id, "table_key": v.table_key, "entity": v.entity, "grain": v.grain}
                for v in registry.views
                if not v.deprecated
            ],
            "datasets": [
                {"table_key": dataset.table_key, "schema_hash": dataset.schema_hash}
                for dataset in spec.datasets
            ],
            "targets": [
                {
                    "name": t.name,
                    "domain": t.domain,
                    "impl_kind": t.impl_kind,
                    "deps": list(t.deps),
                    "outputs": list(t.outputs),
                    "artifacts": [
                        {"name": artifact.name, "kind": artifact.kind} for artifact in t.artifacts
                    ],
                }
                for t in spec.targets
            ],
            "schema_inventory": {"tables": tables, "views": views},
        }

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
                    snapshot={
                        "repo": pointer.repo,
                        "commit": pointer.commit,
                        "run_id": pointer.run_id,
                    },
                    engine=engine,
                )

            fts_available = fts_index_available(warehouse.gateway.con, table_key=_SEARCH_TABLE_KEY)

            query_limit = request.limit + 1
            if fts_available and request.kinds:
                sql = _SQL_SEARCH_FTS_KINDS
                params: list[object] = [request.query, request.kinds, query_limit, request.offset]
            elif fts_available:
                sql = _SQL_SEARCH_FTS
                params = [request.query, query_limit, request.offset]
            elif request.kinds:
                sql = _SQL_SEARCH_LIKE_KINDS
                params = [
                    request.query,
                    request.query,
                    request.query,
                    request.kinds,
                    query_limit,
                    request.offset,
                ]
            else:
                sql = _SQL_SEARCH_LIKE
                params = [request.query, request.query, request.query, query_limit, request.offset]

            rows = self._execute_sql(warehouse=warehouse, sql=sql, params=params)

        truncated = len(rows) > request.limit
        if truncated:
            rows = rows[: request.limit]

        results = [SearchResult.model_validate(row) for row in rows]
        return SearchQueryResponse(
            query=request.query,
            results=results,
            truncated=truncated,
            snapshot={"repo": pointer.repo, "commit": pointer.commit, "run_id": pointer.run_id},
            engine=engine,
        )

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
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

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> None:
        """Write an export result to a Parquet file via DuckDB COPY."""
        with self.db.connect_export() as (warehouse, pointer):
            sql, _columns, temp_tables = self._compile_export_query(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )
            try:
                escaped = str(output_path).replace("'", "''")
                copy_sql = f"COPY ({sql}) TO '{escaped}' (FORMAT PARQUET)"
                warehouse.gateway.policy.execute_sql(copy_sql)
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
        context = self._snapshot_context(pointer)
        try:
            view = context.registry.by_id(request.view_id)
        except KeyError as exc:
            raise UnknownViewIdError(request.view_id) from exc

        allowed_columns = self._resolve_allowed_columns(view=view, inventory=context.inventory)
        if request.select:
            unknown = [col for col in request.select if col not in allowed_columns]
            if unknown:
                raise UnknownColumnsError(
                    unknown=tuple(unknown),
                    allowed=tuple(allowed_columns),
                )
            columns = list(request.select)
        else:
            columns = allowed_columns

        plan = SemanticQueryPlan(
            table_key=view.table_key,
            columns=columns,
            allowed_columns=frozenset(allowed_columns),
            filters=request.filters,
            order_by=request.order_by,
            limit=request.limit,
            offset=request.offset,
        )

        ibis_con = warehouse.gateway.ibis.con
        column_types = _column_types_for_view(view=view, inventory=context.inventory)
        built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
        sql = ibis_con.compile(built.expr)
        return sql, columns, built.temp_tables


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
