"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for semantic layer queries, used by
both FastAPI routes and MCP tools.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.serving.errors import LineageMetadataMissingError, SearchIndexMissingError
from codeintel.serving.meta.models import ServingKernelMetaResponse
from codeintel.serving.meta.service import build_kernel_meta_payload
from codeintel.serving.search.engine import (
    SEARCH_TABLE_NAME,
    SEARCH_TABLE_SCHEMA,
    build_search_query,
    is_fts_available,
)
from codeintel.serving.search.models import SearchQueryResponse, SearchResult
from codeintel.serving.semantic.engines.duckdb_engine import DuckDBQueryEngine
from codeintel.serving.semantic.engines.polars_engine import PolarsQueryEngine
from codeintel.serving.semantic.engines.protocol import EngineContext, QueryExplain
from codeintel.serving.semantic.engines.registry import QueryEngineRegistry, build_engine_registry
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
from codeintel.serving.semantic.planner import SemanticQueryPlanner
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.snapshot.models import ServingSnapshotIdentity
from codeintel.storage.constants import META_CATALOG_NAME
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

    from codeintel.serving.db.manager import ServingDBManager, ServingSnapshotContext
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.models import (
        SemanticExportRequest,
        SemanticQueryRequest,
    )
    from codeintel.serving.semantic.planner import ResolvedViewContext
    from codeintel.serving.semantic.templates import DbApiQuery
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
    _engine_registry: QueryEngineRegistry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize planner after dataclass construction."""
        self._planner = SemanticQueryPlanner(db=self.db, settings=self.settings)
        self._engine_registry = build_engine_registry(
            (
                PolarsQueryEngine(),
                DuckDBQueryEngine(),
            )
        )

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
        records = [
            {str(key): value for key, value in record.items()}
            for record in df_pd.astype("object").to_dict(orient="records")
        ]
        return _sanitize_rows(records)

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
        spec: SemanticQuerySpec,
        inventory: SchemaInventory,
    ) -> tuple[str, str | None]:
        filter_dicts = [f.model_dump(mode="json") for f in spec.filters]
        schema_hash_value = self._planner.schema_hash_for_table_key(
            inventory=inventory, table_key=spec.table_key
        )
        inputs = SemanticQueryFingerprintInput(
            snapshot=self._snapshot_dict(pointer).model_dump(mode="json"),
            view_id=view_id,
            table_key=spec.table_key,
            select=spec.columns,
            order_by=spec.order_by,
            filters=filter_dicts,
            limit=spec.limit,
            offset=spec.offset,
            schema_hash=schema_hash_value,
        )
        query_hash = fingerprint_semantic_query(inputs)
        return query_hash, schema_hash_value

    def _engine_context(
        self,
        *,
        pointer: ServingSnapshotPointer,
        context: ServingSnapshotContext,
        warehouse: Warehouse | None,
    ) -> EngineContext:
        ctx_registry = context.registry
        ctx_inventory = context.inventory
        ctx_dataset_manifests = context.dataset_manifests
        ctx_view_registry = context.view_registry
        return EngineContext(
            pointer=pointer,
            inventory=ctx_inventory,
            registry=ctx_registry,
            dataset_manifests=ctx_dataset_manifests,
            view_registry=ctx_view_registry,
            settings=self.settings,
            warehouse=warehouse,
        )

    def _rows_from_table(self, table: pa.Table) -> list[dict[str, object]]:
        engine = self.settings.result_engine.lower()
        if engine == "polars" and pl is not None:
            df_pl = pl.from_arrow(table)
            if isinstance(df_pl, pl.Series):
                df_pl = df_pl.to_frame()
            return _sanitize_rows(df_pl.to_dicts())

        if engine == "pandas":
            df_pd = table.to_pandas()
            records = [
                {str(key): value for key, value in record.items()}
                for record in df_pd.astype("object").to_dict(orient="records")
            ]
            return _sanitize_rows(records)

        return _sanitize_rows(table.to_pylist())

    def _execute_dbapi_query(
        self, *, warehouse: Warehouse, query: DbApiQuery
    ) -> list[dict[str, object]]:
        try:
            assert_select_perimeter(query.sql, policy=SqlIngressPolicy())
        except UnsafeSqlError as exc:
            raise ValueError(str(exc)) from exc
        else:
            return self._execute_sql(warehouse=warehouse, sql=query.sql, params=query.params)

    def _execute_engine_plan(
        self,
        *,
        spec: SemanticQuerySpec,
        ctx: EngineContext,
    ) -> tuple[list[dict[str, object]], QueryExplain]:
        engine = self._engine_registry.select(
            preference=self.settings.query_engine,
            spec=spec,
            ctx=ctx,
        )
        plan = engine.compile(spec, ctx=ctx)
        try:
            table = plan.to_table()
            rows = self._rows_from_table(table)
            explain = plan.explain()
        finally:
            plan.cleanup()
        return rows, explain

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
        SemanticViewDescriptionResponse
            View description with schema details.

        Raises
        ------
        LineageMetadataMissingError
            If lineage metadata tables are unavailable in the snapshot database.
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
        with self.db.connect() as (warehouse, _):
            if not warehouse.gateway.policy.table_exists(
                schema="metadata",
                table="derived_lineage_columns",
                catalog=META_CATALOG_NAME,
            ):
                raise LineageMetadataMissingError(
                    table=f"{META_CATALOG_NAME}.metadata.derived_lineage_columns"
                )
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
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit + 1)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=self._planner.snapshot_context(pointer),
                warehouse=warehouse,
            )
            rows, explain = self._execute_engine_plan(spec=spec, ctx=engine_ctx)

        truncated = len(rows) > effective_limit
        if truncated:
            rows = rows[:effective_limit]

        fingerprint_spec = self._planner.build_spec(
            ctx=resolved, inputs=inputs, limit=effective_limit
        )
        query_hash, schema_hash_value = self._fingerprint_semantic_plan(
            pointer=resolved.pointer,
            view_id=resolved.view.id,
            spec=fingerprint_spec,
            inventory=resolved.inventory,
        )

        sql_fingerprint = sqlglot_canonical_sha256(explain.sql) if explain.sql is not None else None
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
            snapshot_context = self._planner.snapshot_context(pointer)
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_query(
                ctx=resolved, request=request
            )
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select_prefer(
                ["duckdb"],
                spec=spec,
                ctx=engine_ctx,
            )
            plan = engine.compile(spec, ctx=engine_ctx)
            try:
                explain = plan.explain()
            finally:
                plan.cleanup()
            return SemanticExplainResponse(
                view_id=request.view_id,
                sql=explain.sql or "",
                plan=explain.plan or "",
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
            snapshot_context = self._planner.snapshot_context(pointer)
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_query(
                ctx=resolved, request=request
            )
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit + 1)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select_prefer(
                ["duckdb"],
                spec=spec,
                ctx=engine_ctx,
            )
            plan = engine.compile(spec, ctx=engine_ctx)
            try:
                explain = plan.explain()
            finally:
                plan.cleanup()
            return explain.sql or ""

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

        Raises
        ------
        SearchIndexMissingError
            If the search index table is missing from the snapshot database.
        """
        engine = self.settings.result_engine.lower()

        with self.db.connect() as (warehouse, pointer):
            backend = warehouse.gateway.policy
            if not backend.table_exists(schema=SEARCH_TABLE_SCHEMA, table=SEARCH_TABLE_NAME):
                raise SearchIndexMissingError
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
        fingerprint_spec = self._planner.build_spec(
            ctx=resolved, inputs=inputs, limit=effective_limit
        )
        return self._fingerprint_semantic_plan(
            pointer=pointer,
            view_id=resolved.view.id,
            spec=fingerprint_spec,
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
            snapshot_context = self._planner.snapshot_context(pointer)
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_export(
                ctx=resolved, request=request
            )
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select(
                preference=self.settings.query_engine,
                spec=spec,
                ctx=engine_ctx,
            )
            plan = engine.compile(spec, ctx=engine_ctx)
            try:
                reader = plan.to_reader(batch_size=self.settings.export_batch_size)
                for batch in reader:
                    payload = batch.to_pydict()
                    row_count = batch.num_rows
                    for idx in range(row_count):
                        yield {name: payload[name][idx] for name in inputs.columns}
            finally:
                plan.cleanup()

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
            snapshot_context = self._planner.snapshot_context(pointer)
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_export(
                ctx=resolved, request=request
            )
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select_prefer(
                ["duckdb"],
                spec=spec,
                ctx=engine_ctx,
            )
            plan = engine.compile(spec, ctx=engine_ctx)
            try:
                return plan.explain().sql or ""
            finally:
                plan.cleanup()

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an export result to a Parquet file.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            snapshot_context = self._planner.snapshot_context(pointer)
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_export(
                ctx=resolved, request=request
            )
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select(
                preference=self.settings.query_engine,
                spec=spec,
                ctx=engine_ctx,
            )
            plan = engine.compile(spec, ctx=engine_ctx)
            try:
                reader = plan.to_reader(batch_size=self.settings.export_batch_size)
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
                plan.cleanup()

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        """Write an export result to an Arrow IPC file.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            snapshot_context = self._planner.snapshot_context(pointer)
            resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
            inputs, effective_limit = self._planner.plan_inputs_for_export(
                ctx=resolved, request=request
            )
            spec = self._planner.build_spec(ctx=resolved, inputs=inputs, limit=effective_limit)
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select(
                preference=self.settings.query_engine,
                spec=spec,
                ctx=engine_ctx,
            )
            plan = engine.compile(spec, ctx=engine_ctx)
            try:
                reader = plan.to_reader(batch_size=self.settings.export_batch_size)
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
                plan.cleanup()


__all__ = ["SemanticQueryKernel"]
