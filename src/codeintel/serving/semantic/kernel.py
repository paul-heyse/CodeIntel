"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for semantic layer queries, used by
both FastAPI routes and MCP tools.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.serving.meta.models import ServingKernelMetaResponse
from codeintel.serving.meta.service import build_kernel_meta_payload
from codeintel.serving.search.engine import (
    SEARCH_TABLE_NAME,
    SEARCH_TABLE_SCHEMA,
    build_search_query,
    is_fts_available,
)
from codeintel.serving.search.models import SearchQueryResponse, SearchResult
from codeintel.serving.semantic.fingerprints import (
    SemanticQueryFingerprintInput,
    fingerprint_search,
    fingerprint_semantic_query,
    sqlglot_canonical_sha256,
)
from codeintel.serving.semantic.models import (
    ColumnLineageRef,
    SemanticCatalogResponse,
    SemanticCatalogView,
    SemanticExplainResponse,
    SemanticQueryResponse,
    SemanticViewDescriptionResponse,
)
from codeintel.serving.semantic.planner import (
    SemanticQueryPlanner,
    cleanup_temp_tables_if_needed,
)
from codeintel.serving.semantic.query_builder import SemanticQueryPlan, build_query
from codeintel.serving.snapshot.models import ServingSnapshotIdentity
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.metadata import load_derived_lineage_columns
from codeintel.storage.queries.safe import (
    SqlIngressPolicy,
    UnsafeSqlError,
    assert_select_perimeter,
)

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from codeintel.core.schemas.primitives import ColumnType
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.models import (
        SemanticExportRequest,
        SemanticQueryRequest,
    )
    from codeintel.serving.semantic.planner import ResolvedViewContext
    from codeintel.serving.semantic.templates import BoundQuery, DbApiQuery
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.warehouse import Warehouse

LOG = logging.getLogger(__name__)


class UnknownViewIdError(KeyError):
    """Raise when a semantic view identifier cannot be resolved."""

    def __init__(self, view_id: str) -> None:
        super().__init__(view_id)
        self.view_id = view_id


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
    _planner: SemanticQueryPlanner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize planner after dataclass construction."""
        self._planner = SemanticQueryPlanner(db=self.db, settings=self.settings)

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

    def _fingerprint_semantic_plan(
        self,
        *,
        pointer: ServingSnapshotPointer,
        view_id: str,
        plan: SemanticQueryPlan,
        inventory: SchemaInventory,
    ) -> tuple[str, str | None]:
        filter_dicts = [f.model_dump(mode="json") for f in plan.filters]
        schema_hash_value = self._planner.schema_hash_for_table_key(
            inventory=inventory, table_key=plan.table_key
        )
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
    ) -> tuple[list[dict[str, object]], str]:
        ibis_con = warehouse.gateway.ibis.con
        built = build_query(ibis_con=ibis_con, plan=plan, column_types=column_types)
        return self._execute_bound_query(warehouse=warehouse, query=built)

    def _execute_bound_query(
        self, *, warehouse: Warehouse, query: BoundQuery
    ) -> tuple[list[dict[str, object]], str]:
        ibis_con = warehouse.gateway.ibis.con
        try:
            sql = query.compile_sql(ibis_con)
            assert_select_perimeter(sql, policy=SqlIngressPolicy())
            rows = self._execute_sql(warehouse=warehouse, sql=sql)
            return rows, sql
        except UnsafeSqlError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            cleanup_temp_tables_if_needed(
                con=warehouse.gateway.con,
                temp_tables=query.temp_tables,
            )

    def _execute_dbapi_query(
        self, *, warehouse: Warehouse, query: DbApiQuery
    ) -> list[dict[str, object]]:
        try:
            assert_select_perimeter(query.sql, policy=SqlIngressPolicy())
        except UnsafeSqlError as exc:
            raise ValueError(str(exc)) from exc
        return self._execute_sql(warehouse=warehouse, sql=query.sql, params=query.params)

    def _resolve_view_context_for_export(
        self, *, pointer: ServingSnapshotPointer, view_id: str
    ) -> ResolvedViewContext:
        try:
            return self._planner.resolve_view_context(pointer=pointer, view_id=view_id)
        except KeyError as exc:
            raise UnknownViewIdError(view_id) from exc

    def catalog(self) -> SemanticCatalogResponse:
        """List all available semantic views.

        Returns
        -------
        dict[str, object]
            Catalog response with version, snapshot, and views.
        """
        pointer = self.db.current_pointer()
        context = self._planner.snapshot_context(pointer)
        registry = context.registry

        views = [
            SemanticCatalogView(
                id=v.id,
                table_key=v.table_key,
                entity=v.entity,
                grain=v.grain,
                description=v.description,
                column_count=len(v.columns),
            )
            for v in registry.views
            if not v.deprecated
        ]
        return SemanticCatalogResponse(
            version=registry.version,
            snapshot=self._snapshot_dict(pointer),
            views=views,
        )

    def describe(self, view_id: str) -> SemanticViewDescriptionResponse:
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
        context = self._planner.snapshot_context(pointer)
        registry = context.registry
        inventory = context.inventory

        view = registry.by_id(view_id)
        table_schema = inventory.get(view.table_key)

        column_types: dict[str, str] = {}
        if table_schema is not None:
            column_types = {c.name: c.type for c in table_schema.columns}

        lineage: dict[str, list[ColumnLineageRef]] = {}
        try:
            with self.db.connect() as (warehouse, _):
                if warehouse.gateway.policy.table_exists(
                    schema="metadata",
                    table="derived_lineage_columns",
                ):
                    raw_lineage = load_derived_lineage_columns(
                        warehouse.gateway.con,
                        repo=pointer.repo,
                        commit=pointer.commit,
                        downstream_table=view.table_key.lower(),
                    )
                    lineage = {
                        downstream: [
                            ColumnLineageRef(table_key=table_key, column=column)
                            for table_key, column in refs
                        ]
                        for downstream, refs in raw_lineage.items()
                    }
        except DuckDBError:
            LOG.debug(
                "Unable to load lineage for view=%s repo=%s commit=%s",
                view.table_key,
                pointer.repo,
                pointer.commit,
            )

        return SemanticViewDescriptionResponse(
            id=view.id,
            table_key=view.table_key,
            kind=view.kind,
            entity=view.entity,
            grain=view.grain,
            description=view.description,
            primary_key=view.primary_key,
            columns=view.columns,
            column_types=column_types,
            joins=view.joins,
            defaults=view.defaults,
            deprecated=view.deprecated,
            replaced_by=view.replaced_by,
            snapshot=self._snapshot_dict(pointer),
            lineage=lineage,
        )

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
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_query(
                ctx=resolved, request=request
            )
            plan = self._planner.build_plan(ctx=resolved, inputs=inputs, limit=effective_limit + 1)
            rows, compiled_sql = self._execute_semantic_plan(
                warehouse=warehouse,
                plan=plan,
                column_types=resolved.column_types,
            )

        truncated = len(rows) > effective_limit
        if truncated:
            rows = rows[:effective_limit]

        fingerprint_plan = self._planner.build_plan(
            ctx=resolved, inputs=inputs, limit=effective_limit
        )
        query_hash, schema_hash_value = self._fingerprint_semantic_plan(
            pointer=resolved.pointer,
            view_id=resolved.view.id,
            plan=fingerprint_plan,
            inventory=resolved.inventory,
        )

        sql_fingerprint = sqlglot_canonical_sha256(compiled_sql) if compiled_sql else None

        return SemanticQueryResponse(
            view_id=request.view_id,
            columns=inputs.columns,
            rows=rows,
            truncated=truncated,
            snapshot=self._snapshot_dict(resolved.pointer),
            query_hash=query_hash,
            schema_hash=schema_hash_value,
            sql_fingerprint=sql_fingerprint,
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
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_query(
                ctx=resolved, request=request
            )
            plan = self._planner.build_plan(ctx=resolved, inputs=inputs, limit=effective_limit)
            compiled, _temp_tables = self._planner.compile_plan(
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
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_query(
                ctx=resolved, request=request
            )
            plan = self._planner.build_plan(ctx=resolved, inputs=inputs, limit=effective_limit + 1)
            compiled, _temp_tables = self._planner.compile_plan(
                warehouse=warehouse,
                plan=plan,
                column_types=resolved.column_types,
                cleanup_temp_tables=True,
            )
            return compiled

    def meta(self) -> ServingKernelMetaResponse:
        """Return serving metadata for /meta endpoint and tools.

        Returns
        -------
        ServingKernelMetaResponse
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
            if not backend.table_exists(schema=SEARCH_TABLE_SCHEMA, table=SEARCH_TABLE_NAME):
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
            query = build_search_query(
                request, fts_available=is_fts_available(warehouse.gateway.con)
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
        inputs, effective_limit = self._planner.plan_inputs_for_export(
            ctx=resolved, request=request
        )
        fingerprint_plan = self._planner.build_plan(
            ctx=resolved, inputs=inputs, limit=effective_limit
        )
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
            sql, columns, temp_tables = self._planner.compile_export(
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
                cleanup_temp_tables_if_needed(
                    con=warehouse.gateway.con,
                    temp_tables=temp_tables,
                )

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
            sql, _columns, temp_tables = self._planner.compile_export(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )
            try:
                return sql
            finally:
                cleanup_temp_tables_if_needed(
                    con=warehouse.gateway.con,
                    temp_tables=temp_tables,
                )

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an export result to a Parquet file.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            sql, _columns, temp_tables = self._planner.compile_export(
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
                cleanup_temp_tables_if_needed(
                    con=warehouse.gateway.con,
                    temp_tables=temp_tables,
                )

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an export result to an Arrow IPC file.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            sql, _columns, temp_tables = self._planner.compile_export(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
            )
            try:
                result = warehouse.gateway.policy.execute_sql(sql)
                reader = result.fetch_record_batch(self.settings.export_batch_size)
                rows_written = 0
                with (
                    output_path.open("wb") as handle,
                    pa.ipc.new_file(
                        handle,
                        reader.schema,
                    ) as writer,
                ):
                    for batch in reader:
                        rows_written += batch.num_rows
                        writer.write_batch(batch)
                return rows_written
            finally:
                cleanup_temp_tables_if_needed(
                    con=warehouse.gateway.con,
                    temp_tables=temp_tables,
                )


__all__ = ["SemanticQueryKernel"]
