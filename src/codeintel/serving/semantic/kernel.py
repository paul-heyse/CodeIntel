"""Semantic query kernel - unified API for HTTP and MCP.

The kernel provides a single entry point for semantic layer queries, used by
both FastAPI routes and MCP tools.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq
from sqlglot.errors import SqlglotError

from codeintel.core.columnar import align_reader_to_contract, extras_policy_from_schema
from codeintel.core.exports import (
    apply_ipc_metadata,
    build_ipc_write_options,
    default_ipc_write_options,
    iter_ipc_stream,
)
from codeintel.core.schemas.hashing import schema_digest
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
from codeintel.serving.semantic.engines.protocol import EngineContext, ExecutablePlan, QueryExplain
from codeintel.serving.semantic.engines.registry import QueryEngineRegistry, build_engine_registry
from codeintel.serving.semantic.fingerprints import (
    SemanticQueryFingerprintInput,
    fingerprint_search,
    fingerprint_semantic_query,
    sqlglot_canonical_sha256,
)
from codeintel.serving.semantic.guardrails import (
    warn_contract_metadata_mismatch,
    warn_contract_metadata_missing,
    warn_missing_contract_schema,
)
from codeintel.serving.semantic.models import (
    ColumnLineageRef,
    QueryScanMetrics,
    SemanticCatalogResponse,
    SemanticCatalogView,
    SemanticExplainResponse,
    SemanticQueryResponse,
    SemanticViewDescriptionResponse,
)
from codeintel.serving.semantic.planner import SemanticQueryPlanner
from codeintel.serving.semantic.query_ast import ServingQuery, build_serving_query
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import SqlglotQueryBuilderError
from codeintel.serving.snapshot.models import ServingSnapshotIdentity
from codeintel.storage.constants import DUCKDB_DIALECT, META_CATALOG_NAME
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.metadata import load_derived_lineage_columns
from codeintel.storage.queries.safe import (
    SqlIngressPolicy,
    UnsafeSqlError,
    assert_select_perimeter,
)
from codeintel.storage.schema import arrow_schema_for_table_key
from codeintel.storage.sqlglot_tools import (
    extract_column_lineage_duckdb,
    extract_table_keys_duckdb,
    semantic_diff_sql_duckdb,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from duckdb import DuckDBPyConnection

    from codeintel.serving.db.manager import ServingDBManager, ServingSnapshotContext
    from codeintel.serving.db.pointer import ServingSnapshotPointer
    from codeintel.serving.operations.cancellation import CancelCheck
    from codeintel.serving.search.models import SearchQueryRequest
    from codeintel.serving.semantic.datasets import DatasetManifestIndex
    from codeintel.serving.semantic.inventory import SchemaInventory
    from codeintel.serving.semantic.models import (
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticViewSpec,
    )
    from codeintel.serving.semantic.planner import ResolvedViewContext
    from codeintel.serving.semantic.templates import DbApiQuery
    from codeintel.serving.settings import ServingSettings
    from codeintel.storage.warehouse import Warehouse


class UnknownViewIdError(KeyError):
    """Raise when a semantic view identifier cannot be resolved."""

    def __init__(self, view_id: str) -> None:
        super().__init__(view_id)
        self.view_id = view_id


MIN_COLUMN_LINEAGE_PARTS = 2

LOG = logging.getLogger(__name__)


def _sanitize_float_nan(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _sanitize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: _sanitize_float_nan(v) for k, v in row.items()} for row in rows]


def _records_from_batch(batch: pa.RecordBatch) -> list[dict[str, object]]:
    columns = batch.schema.names
    arrays = [batch.column(idx) for idx in range(batch.num_columns)]
    return [
        {name: arrays[idx][row_idx].as_py() for idx, name in enumerate(columns)}
        for row_idx in range(batch.num_rows)
    ]


def _column_lineage_refs(entries: Iterable[str]) -> list[ColumnLineageRef]:
    refs: list[ColumnLineageRef] = []
    for entry in entries:
        parts = [part for part in entry.split(".") if part]
        if len(parts) < MIN_COLUMN_LINEAGE_PARTS:
            continue
        table_key = ".".join(parts[:-1])
        refs.append(ColumnLineageRef(table_key=table_key, column=parts[-1]))
    refs.sort(key=lambda ref: (ref.table_key, ref.column))
    return refs


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _coerce_optional_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return None


@dataclass(frozen=True, slots=True)
class _IpcMetadataInput:
    pointer: ServingSnapshotPointer
    table_key: str
    view_id: str
    query_hash: str
    schema_hash: str | None
    schema_digest: str | None
    engine: str | None


@dataclass(frozen=True, slots=True)
class _SemanticQueryStream:
    stream: Iterator[bytes]
    engine: str | None
    query_hash: str | None
    schema_hash: str | None
    schema_digest: str | None
    scan_metrics: QueryScanMetrics | None
    batch_size: int | None

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        return next(self.stream)


@dataclass(frozen=True, slots=True)
class _IpcQueryPlan:
    snapshot_context: ServingSnapshotContext
    resolved: ResolvedViewContext
    serving_query: ServingQuery
    engine_name: str
    scan_metrics: QueryScanMetrics | None
    batch_size: int
    query_hash: str
    schema_hash: str | None
    schema_digest: str | None


@dataclass(frozen=True, slots=True)
class _IpcExportPlan:
    plan: ExecutablePlan
    reader: pa.RecordBatchReader
    metadata: dict[str, object]


def _build_ipc_metadata(input_data: _IpcMetadataInput) -> dict[str, object]:
    metadata: dict[str, object] = {
        "codeintel.table_key": input_data.table_key,
        "codeintel.snapshot_id": input_data.pointer.run_id,
        "codeintel.repo": input_data.pointer.repo,
        "codeintel.commit": input_data.pointer.commit,
        "codeintel.view_id": input_data.view_id,
        "codeintel.query_hash": input_data.query_hash,
    }
    if input_data.schema_hash is not None:
        metadata["codeintel.schema_hash"] = input_data.schema_hash
    if input_data.schema_digest is not None:
        metadata["codeintel.schema_digest"] = input_data.schema_digest
    if input_data.engine is not None:
        metadata["codeintel.query_engine"] = input_data.engine
    return metadata


def _contract_schema_for_table(
    con: DuckDBPyConnection,
    *,
    table_key: str,
    pointer: ServingSnapshotPointer,
) -> pa.Schema | None:
    try:
        return arrow_schema_for_table_key(
            con,
            table_key=table_key,
            repo=pointer.repo,
            commit=pointer.commit,
        )
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


def _check_contract_metadata(
    *,
    contract_schema: pa.Schema,
    table_key: str,
    schema_hash_value: str | None,
    schema_digest_value: str | None,
) -> None:
    metadata = _decode_arrow_metadata(contract_schema.metadata)
    contract_hash = metadata.get("codeintel.schema_hash")
    contract_digest = metadata.get("codeintel.schema_digest")
    if schema_hash_value and isinstance(contract_hash, str) and contract_hash != schema_hash_value:
        warn_contract_metadata_mismatch(
            table_key=table_key,
            field="schema_hash",
            expected=schema_hash_value,
            actual=contract_hash,
        )
    if (
        schema_digest_value
        and isinstance(contract_digest, str)
        and contract_digest != schema_digest_value
    ):
        warn_contract_metadata_mismatch(
            table_key=table_key,
            field="schema_digest",
            expected=schema_digest_value,
            actual=contract_digest,
        )
    if "codeintel.schema_contract_version" not in metadata:
        warn_contract_metadata_missing(table_key=table_key, field="schema_contract_version")


def _decode_arrow_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, object]:
    if not metadata:
        return {}
    decoded: dict[str, object] = {}
    for key, value in metadata.items():
        key_str = key.decode("utf-8")
        value_str = value.decode("utf-8")
        try:
            decoded[key_str] = json.loads(value_str)
        except json.JSONDecodeError:
            decoded[key_str] = value_str
    return decoded


def _ipc_write_options(settings: ServingSettings) -> pa.ipc.IpcWriteOptions:
    if not settings.ipc_enable_options:
        return default_ipc_write_options()
    return build_ipc_write_options(
        compression=settings.ipc_compression or "zstd",
        use_threads=settings.ipc_use_threads,
        unify_dictionaries=settings.ipc_unify_dictionaries,
        metadata_version=settings.ipc_metadata_version or "V5",
    )


def _resolve_view_columns(
    *,
    view: SemanticViewSpec,
    inventory: SchemaInventory,
) -> list[str]:
    if view.columns_dynamic:
        schema = inventory.get(view.table_key)
        if schema is None:
            return list(view.columns)
        return [column.name for column in schema.columns]
    if view.columns:
        return list(view.columns)
    schema = inventory.get(view.table_key)
    if schema is None:
        return list(view.columns)
    return [column.name for column in schema.columns]


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
        backend = warehouse.gateway.policy
        result = backend.execute_sql(sql, params=params)
        reader = result.fetch_record_batch(self.settings.export_batch_size)
        return _sanitize_rows(self._rows_from_reader(reader, cancel_check=None))

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
        ast_hash = self._ast_hash_for_spec(spec=spec)
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
            ast_hash=ast_hash,
        )
        query_hash = fingerprint_semantic_query(inputs)
        return query_hash, schema_hash_value

    @staticmethod
    def _schema_digest_for_table_key(*, inventory: SchemaInventory, table_key: str) -> str | None:
        schema = inventory.get(table_key)
        if schema is None:
            return None
        return schema_digest(schema)

    @staticmethod
    def _ast_hash_for_spec(*, spec: SemanticQuerySpec) -> str | None:
        try:
            serving_query = build_serving_query(spec=spec)
            sql = serving_query.ast.sql(dialect=DUCKDB_DIALECT)
            return sqlglot_canonical_sha256(sql)
        except (SqlglotError, SqlglotQueryBuilderError, TypeError, ValueError):
            return None

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

    def _scan_metrics_for_table_key(
        self,
        *,
        table_key: str,
        dataset_manifests: DatasetManifestIndex,
    ) -> QueryScanMetrics | None:
        if not self.settings.dataset_scan_metrics_enabled:
            return None
        entry = dataset_manifests.get(table_key)
        if entry is None:
            return None
        stats = entry.manifest.stats or {}
        row_count = entry.manifest.row_count
        if row_count is None:
            row_count = _coerce_optional_int(stats.get("rows_from_metadata"))
        file_count = _coerce_optional_int(stats.get("file_count"))
        if file_count is None and entry.manifest.files:
            file_count = len(entry.manifest.files)
        total_bytes = _coerce_optional_int(stats.get("total_bytes"))
        return QueryScanMetrics(
            row_count=row_count,
            file_count=file_count,
            total_bytes=total_bytes,
        )

    @staticmethod
    def _rows_from_reader(
        reader: pa.RecordBatchReader,
        *,
        cancel_check: CancelCheck | None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for batch in reader:
            _raise_if_cancelled(cancel_check)
            rows.extend(_records_from_batch(batch))
        return rows

    def _execute_dbapi_query(
        self, *, warehouse: Warehouse, query: DbApiQuery
    ) -> list[dict[str, object]]:
        try:
            assert_select_perimeter(query.sql, policy=SqlIngressPolicy())
        except UnsafeSqlError as exc:
            raise ValueError(str(exc)) from exc
        else:
            return self._execute_sql(warehouse=warehouse, sql=query.sql, params=query.params)

    @staticmethod
    def _log_ast_diff(
        *,
        query: ServingQuery,
        explain: QueryExplain,
        engine_name: str,
    ) -> None:
        if not LOG.isEnabledFor(logging.INFO):
            return
        if explain.sql is None:
            return
        try:
            requested_sql = query.ast.sql(dialect=DUCKDB_DIALECT)
            diff = semantic_diff_sql_duckdb(requested_sql, explain.sql)
        except SqlglotError:
            return
        if not diff:
            return
        LOG.info(
            "semantic_query_diff",
            extra={
                "engine": engine_name,
                "view_id": query.spec.view_id,
                "table_key": query.spec.table_key,
                "diff": diff,
            },
        )

    def _execute_engine_plan(
        self,
        *,
        query: ServingQuery,
        ctx: EngineContext,
        cancel_check: CancelCheck | None,
    ) -> tuple[list[dict[str, object]], QueryExplain, str]:
        engine = self._engine_registry.select(
            preference=self.settings.query_engine,
            query=query,
            ctx=ctx,
        )
        engine_name = engine.name.lower()
        plan = engine.compile(query, ctx=ctx)
        try:
            _raise_if_cancelled(cancel_check)
            reader = plan.to_reader(batch_size=self.settings.export_batch_size)
            rows = _sanitize_rows(self._rows_from_reader(reader, cancel_check=cancel_check))
            explain = plan.explain()
            self._log_ast_diff(
                query=query,
                explain=explain,
                engine_name=engine_name,
            )
        finally:
            plan.cleanup()
        return rows, explain, engine_name

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
        inventory = context.inventory

        views = [
            SemanticCatalogView(
                id=v.id,
                table_key=v.table_key,
                entity=v.entity,
                grain=v.grain,
                description=v.description,
                column_count=len(_resolve_view_columns(view=v, inventory=inventory)),
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
        resolved_columns = _resolve_view_columns(view=view, inventory=inventory)

        column_types: dict[str, str] = {}
        if table_schema is not None:
            allowed = set(resolved_columns)
            column_types = {c.name: c.type for c in table_schema.columns if c.name in allowed}

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
            columns=resolved_columns,
            column_types=column_types,
            joins=view.joins,
            defaults=view.defaults,
            snapshot=self._snapshot_dict(pointer),
            lineage=lineage,
        )

    def query(
        self, request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> SemanticQueryResponse:
        """Execute a semantic view query.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.
        cancel_check
            Optional cancellation hook invoked during query execution.

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
            serving_query = self._planner.build_query(
                ctx=resolved,
                inputs=inputs,
                limit=effective_limit + 1,
            )
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=self._planner.snapshot_context(pointer),
                warehouse=warehouse,
            )
            scan_metrics = self._scan_metrics_for_table_key(
                table_key=serving_query.spec.table_key,
                dataset_manifests=engine_ctx.dataset_manifests,
            )
            batch_size = self.settings.export_batch_size
            rows, explain, engine = self._execute_engine_plan(
                query=serving_query,
                ctx=engine_ctx,
                cancel_check=cancel_check,
            )

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
            engine=engine,
            snapshot=self._snapshot_dict(resolved.pointer),
            query_hash=query_hash,
            schema_hash=schema_hash_value,
            scan_metrics=scan_metrics,
            batch_size=batch_size,
            sql_fingerprint=sql_fingerprint,
        )

    def _plan_ipc_stream(self, request: SemanticQueryRequest) -> _IpcQueryPlan:
        pointer = self.db.current_pointer()
        snapshot_context = self._planner.snapshot_context(pointer)
        resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
        inputs, effective_limit = self._planner.plan_inputs_for_query(ctx=resolved, request=request)
        serving_query = self._planner.build_query(
            ctx=resolved,
            inputs=inputs,
            limit=effective_limit,
        )
        query_hash, schema_hash_value = self._fingerprint_semantic_plan(
            pointer=resolved.pointer,
            view_id=resolved.view.id,
            spec=serving_query.spec,
            inventory=resolved.inventory,
        )
        return _IpcQueryPlan(
            snapshot_context=snapshot_context,
            resolved=resolved,
            serving_query=serving_query,
            engine_name=self._engine_registry.select(
                preference=self.settings.query_engine,
                query=serving_query,
                ctx=self._engine_context(
                    pointer=pointer,
                    context=snapshot_context,
                    warehouse=cast("Warehouse", object()),
                ),
            ).name.lower(),
            scan_metrics=self._scan_metrics_for_table_key(
                table_key=serving_query.spec.table_key,
                dataset_manifests=snapshot_context.dataset_manifests,
            ),
            batch_size=self.settings.export_batch_size,
            query_hash=query_hash,
            schema_hash=schema_hash_value,
            schema_digest=self._schema_digest_for_table_key(
                inventory=resolved.inventory,
                table_key=resolved.view.table_key,
            ),
        )

    def query_ipc_stream(
        self, request: SemanticQueryRequest, *, cancel_check: CancelCheck | None = None
    ) -> Iterable[bytes]:
        """Execute a semantic query and return Arrow IPC stream bytes.

        Parameters
        ----------
        request
            Query request with filters, selection, and pagination.
        cancel_check
            Optional cancellation hook invoked during query execution.

        Returns
        -------
        Iterable[bytes]
            Iterable of Arrow IPC stream bytes.
        """
        plan = self._plan_ipc_stream(request)

        def _stream() -> Generator[bytes]:
            with self.db.connect() as (warehouse, live_pointer):
                engine_ctx = self._engine_context(
                    pointer=live_pointer,
                    context=plan.snapshot_context,
                    warehouse=warehouse,
                )
                engine = self._engine_registry.select(
                    preference=self.settings.query_engine,
                    query=plan.serving_query,
                    ctx=engine_ctx,
                )
                compiled_plan = engine.compile(plan.serving_query, ctx=engine_ctx)
                try:
                    contract_schema = _contract_schema_for_table(
                        warehouse.gateway.con,
                        table_key=plan.resolved.view.table_key,
                        pointer=live_pointer,
                    )
                    if contract_schema is None:
                        warn_missing_contract_schema(table_key=plan.resolved.view.table_key)
                    else:
                        _check_contract_metadata(
                            contract_schema=contract_schema,
                            table_key=plan.resolved.view.table_key,
                            schema_hash_value=plan.schema_hash,
                            schema_digest_value=plan.schema_digest,
                        )
                    _raise_if_cancelled(cancel_check)
                    reader = compiled_plan.to_reader(batch_size=plan.batch_size)
                    if contract_schema is not None:
                        reader = align_reader_to_contract(
                            reader,
                            contract_schema,
                            extras_policy=extras_policy_from_schema(contract_schema),
                        )
                    metadata = _build_ipc_metadata(
                        _IpcMetadataInput(
                            pointer=live_pointer,
                            table_key=plan.resolved.view.table_key,
                            view_id=plan.resolved.view.id,
                            query_hash=plan.query_hash,
                            schema_hash=plan.schema_hash,
                            schema_digest=plan.schema_digest,
                            engine=engine.name.lower(),
                        )
                    )
                    write_options = _ipc_write_options(self.settings)
                    yield from iter_ipc_stream(
                        reader,
                        metadata=metadata,
                        options=write_options,
                        cancel_check=cancel_check,
                    )
                finally:
                    compiled_plan.cleanup()

        return _SemanticQueryStream(
            stream=_stream(),
            engine=plan.engine_name,
            query_hash=plan.query_hash,
            schema_hash=plan.schema_hash,
            schema_digest=plan.schema_digest,
            scan_metrics=plan.scan_metrics,
            batch_size=plan.batch_size,
        )

    def _build_ipc_export_plan(
        self,
        *,
        warehouse: Warehouse,
        pointer: ServingSnapshotPointer,
        request: SemanticExportRequest,
        cancel_check: CancelCheck | None,
    ) -> _IpcExportPlan:
        snapshot_context = self._planner.snapshot_context(pointer)
        resolved = self._planner.resolve_view_context(pointer=pointer, view_id=request.view_id)
        inputs, effective_limit = self._planner.plan_inputs_for_export(
            ctx=resolved,
            request=request,
        )
        serving_query = self._planner.build_query(
            ctx=resolved,
            inputs=inputs,
            limit=effective_limit,
        )
        engine_ctx = self._engine_context(
            pointer=pointer,
            context=snapshot_context,
            warehouse=warehouse,
        )
        engine = self._engine_registry.select(
            preference=self.settings.query_engine,
            query=serving_query,
            ctx=engine_ctx,
        )
        plan = engine.compile(serving_query, ctx=engine_ctx)
        try:
            query_hash, schema_hash_value = self._fingerprint_semantic_plan(
                pointer=resolved.pointer,
                view_id=resolved.view.id,
                spec=serving_query.spec,
                inventory=resolved.inventory,
            )
            schema_digest_value = self._schema_digest_for_table_key(
                inventory=resolved.inventory,
                table_key=resolved.view.table_key,
            )
            contract_schema = _contract_schema_for_table(
                warehouse.gateway.con,
                table_key=resolved.view.table_key,
                pointer=pointer,
            )
            if contract_schema is None:
                warn_missing_contract_schema(table_key=resolved.view.table_key)
            else:
                _check_contract_metadata(
                    contract_schema=contract_schema,
                    table_key=resolved.view.table_key,
                    schema_hash_value=schema_hash_value,
                    schema_digest_value=schema_digest_value,
                )
            _raise_if_cancelled(cancel_check)
            reader = plan.to_reader(batch_size=self.settings.export_batch_size)
            if contract_schema is not None:
                reader = align_reader_to_contract(
                    reader,
                    contract_schema,
                    extras_policy=extras_policy_from_schema(contract_schema),
                )
            metadata = _build_ipc_metadata(
                _IpcMetadataInput(
                    pointer=resolved.pointer,
                    table_key=resolved.view.table_key,
                    view_id=resolved.view.id,
                    query_hash=query_hash,
                    schema_hash=schema_hash_value,
                    schema_digest=schema_digest_value,
                    engine=engine.name.lower(),
                )
            )
            return _IpcExportPlan(
                plan=plan,
                reader=reader,
                metadata=metadata,
            )
        except Exception:
            plan.cleanup()
            raise

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
            serving_query = self._planner.build_query(
                ctx=resolved,
                inputs=inputs,
                limit=effective_limit,
            )
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select_prefer(
                ["duckdb"],
                query=serving_query,
                ctx=engine_ctx,
            )
            plan = engine.compile(serving_query, ctx=engine_ctx)
            try:
                explain = plan.explain()
            finally:
                plan.cleanup()
            table_keys: list[str] = []
            column_lineage: dict[str, list[ColumnLineageRef]] = {}
            try:
                ast_sql = serving_query.ast.sql(dialect=DUCKDB_DIALECT)
                table_keys = sorted(extract_table_keys_duckdb(ast_sql))
                raw_lineage = extract_column_lineage_duckdb(ast_sql)
                column_lineage = {
                    column: _column_lineage_refs(entries) for column, entries in raw_lineage.items()
                }
            except (SqlglotError, TypeError, ValueError):
                table_keys = []
                column_lineage = {}
            return SemanticExplainResponse(
                view_id=request.view_id,
                sql=explain.sql or "",
                plan=explain.plan or "",
                snapshot=self._snapshot_dict(pointer),
                table_keys=table_keys,
                column_lineage=column_lineage,
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
            serving_query = self._planner.build_query(
                ctx=resolved,
                inputs=inputs,
                limit=effective_limit + 1,
            )
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select_prefer(
                ["duckdb"],
                query=serving_query,
                ctx=engine_ctx,
            )
            plan = engine.compile(serving_query, ctx=engine_ctx)
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

    def export_rows(
        self,
        request: SemanticExportRequest,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> Generator[dict[str, object]]:
        """Yield rows for streaming export (memory-efficient).

        Unlike query(), this method yields rows one at a time to support
        large result sets without buffering everything in memory.

        Parameters
        ----------
        request
            Export request with filters, selection, and pagination.
        cancel_check
            Optional cancellation hook invoked during export.

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
            serving_query = self._planner.build_query(
                ctx=resolved,
                inputs=inputs,
                limit=effective_limit,
            )
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select(
                preference=self.settings.query_engine,
                query=serving_query,
                ctx=engine_ctx,
            )
            plan = engine.compile(serving_query, ctx=engine_ctx)
            try:
                _raise_if_cancelled(cancel_check)
                reader = plan.to_reader(batch_size=self.settings.export_batch_size)
                for batch in reader:
                    _raise_if_cancelled(cancel_check)
                    columns = batch.schema.names
                    column_index = {name: idx for idx, name in enumerate(columns)}
                    indices = [column_index[name] for name in inputs.columns]
                    arrays = [batch.column(idx) for idx in range(batch.num_columns)]
                    for row_idx in range(batch.num_rows):
                        yield {
                            inputs.columns[col_idx]: arrays[array_idx][row_idx].as_py()
                            for col_idx, array_idx in enumerate(indices)
                        }
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
            serving_query = self._planner.build_query(
                ctx=resolved,
                inputs=inputs,
                limit=effective_limit,
            )
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select_prefer(
                ["duckdb"],
                query=serving_query,
                ctx=engine_ctx,
            )
            plan = engine.compile(serving_query, ctx=engine_ctx)
            try:
                return plan.explain().sql or ""
            finally:
                plan.cleanup()

    def export_to_parquet(
        self,
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        """Write an export result to a Parquet file.

        Parameters
        ----------
        request
            Export request with filters, selection, and pagination.
        output_path
            Output path for the Parquet file.
        cancel_check
            Optional cancellation hook invoked during export.

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
            serving_query = self._planner.build_query(
                ctx=resolved,
                inputs=inputs,
                limit=effective_limit,
            )
            engine_ctx = self._engine_context(
                pointer=pointer,
                context=snapshot_context,
                warehouse=warehouse,
            )
            engine = self._engine_registry.select(
                preference=self.settings.query_engine,
                query=serving_query,
                ctx=engine_ctx,
            )
            plan = engine.compile(serving_query, ctx=engine_ctx)
            try:
                _raise_if_cancelled(cancel_check)
                reader = plan.to_reader(batch_size=self.settings.export_batch_size)
                rows_written = 0
                writer = pq.ParquetWriter(str(output_path), reader.schema)
                try:
                    for batch in reader:
                        _raise_if_cancelled(cancel_check)
                        rows_written += batch.num_rows
                        writer.write_table(pa.Table.from_batches([batch], schema=reader.schema))
                finally:
                    writer.close()
                return rows_written
            finally:
                plan.cleanup()

    def export_to_arrow_ipc(
        self,
        request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: CancelCheck | None = None,
    ) -> int:
        """Write an export result to an Arrow IPC file.

        Parameters
        ----------
        request
            Export request with filters, selection, and pagination.
        output_path
            Output path for the Arrow IPC file.
        cancel_check
            Optional cancellation hook invoked during export.

        Returns
        -------
        int
            Number of rows written.
        """
        with self.db.connect_export() as (warehouse, pointer):
            export_plan = self._build_ipc_export_plan(
                warehouse=warehouse,
                pointer=pointer,
                request=request,
                cancel_check=cancel_check,
            )
            try:
                _raise_if_cancelled(cancel_check)
                schema = apply_ipc_metadata(export_plan.reader.schema, export_plan.metadata)
                write_options = _ipc_write_options(self.settings)
                rows_written = 0
                with (
                    output_path.open("wb") as handle,
                    pa.ipc.new_stream(
                        handle,
                        schema,
                        options=write_options,
                    ) as writer,
                ):
                    for batch in export_plan.reader:
                        _raise_if_cancelled(cancel_check)
                        rows_written += batch.num_rows
                        writer.write_batch(batch)
                return rows_written
            finally:
                export_plan.plan.cleanup()


__all__ = ["SemanticQueryKernel"]
