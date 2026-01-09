"""Typed warehouse API for storage.

The warehouse is the intended single I/O boundary for build + serving. It owns:
- read/exists/count primitives (snapshot-aware where applicable)
- table materialization (snapshot-scoped replace) and view creation
- contract-aware metadata capture (schema hash/version) and optional profiling artifacts

Implementation intentionally composes existing primitives (`StorageGateway`,
`DuckDBPolicyBackend`) so callers can adopt incrementally.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pyarrow as pa
import sqlglot.expressions as exp
from duckdb import ExplainType
from sqlglot import parse_one
from sqlglot.errors import ParseError

from codeintel.core.columnar import (
    ColumnarStream,
    align_reader_to_contract,
    coerce_arrow_reader,
    coerce_arrow_table,
    deep_cast_table_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.conversion import reader_to_table, table_to_reader
from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.kernels import SortKey, hash_struct_ordinal, stable_sort_indices
from codeintel.core.filters import FilterSpecInput
from codeintel.core.queries.filter_compiler import (
    FilterCompilerError,
    compile_filter_predicates,
    duckdb_filter_expression,
)
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.resolution import resolve_table_schema
from codeintel.core.storage import StorageContext
from codeintel.core.validation.mode import ContractValidationMode
from codeintel.core.validation.schema_constraints import (
    list_alignment_specs_for_table_key,
    schema_metadata_errors,
)
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE, DUCKDB_DIALECT
from codeintel.storage.duckdb_explain import normalize_explain_output
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.query_results import coerce_int
from codeintel.storage.schema.duckdb_contracts import (
    ContractSchemaOptions,
    contract_schema_for_table_key,
    table_schema_for_table_key,
)
from codeintel.storage.snapshot_scoping import RepoCommitScope
from codeintel.storage.staging import registered_temp_relation
from codeintel.storage.upsert import UpsertSpec
from codeintel.storage.validation.columnar import (
    ColumnarValidationContext,
    TableValidationError,
    ValidationMode,
    validate_record_batch_reader,
    validate_table,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.storage.duckdb_types import DuckDBConnection
    from codeintel.storage.gateway import StorageGateway

from codeintel.storage.duckdb_types import (
    DuckDBCatalogException,
    DuckDBError,
    DuckDBRelation,
    Expression,
)

WriteMode = Literal["append", "replace", "upsert"]
ReplaceScope = Literal["snapshot", "table"]

type TabularInput = DuckDBRelation | pa.Table | pa.RecordBatchReader | ColumnarStream | object

_PROFILE_DIR_ENV = "CODEINTEL_WAREHOUSE_PROFILING_DIR"
_HASH_ORDINAL_MODULUS = 2**31 - 1

log = logging.getLogger(__name__)
_SNAPSHOT_FILTER_COLUMNS = frozenset({"commit", "repo"})


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Result of a materialization operation.

    Attributes
    ----------
    table_key
        Fully qualified destination table key.
    repo
        Repository slug for snapshot identity, when applicable.
    commit
        Commit SHA for snapshot identity, when applicable.
    rows_written
        Number of rows written when known.
    started_at
        Wall-clock start timestamp (UTC).
    completed_at
        Wall-clock completion timestamp (UTC).
    schema_hash
        Canonical schema hash when available from contracts.
    schema_version
        Optional schema version string from dataset contracts.
    profiling_artifact
        Optional profiling artifact path written during materialization.
    """

    table_key: str
    repo: str | None
    commit: str | None
    rows_written: int | None
    started_at: datetime
    completed_at: datetime
    schema_hash: str | None
    schema_version: str | None
    profiling_artifact: str | None


@dataclass(frozen=True, slots=True)
class UpsertConfig:
    """Conflict handling configuration for upsert mode."""

    conflict_columns: tuple[str, ...]
    update_columns: tuple[str, ...] | None = None
    update_condition: exp.Expression | None = None


@dataclass(frozen=True, slots=True)
class MaterializeOptions:
    """Options for warehouse materialization operations."""

    snapshot: SnapshotRef | None = None
    mode: WriteMode = "replace"
    replace_scope: ReplaceScope = "snapshot"
    owner_target: str | None = None
    input_hash: str | None = None
    asset_type: str = "table"
    upsert: UpsertConfig | None = None
    use_staging: bool = False
    fallback_upsert_on_conflict: bool = False


@dataclass(frozen=True, slots=True)
class RelationWriteState:
    """Row-count metadata for relation writes."""

    row_count: int | None
    skip_row_count: bool


@dataclass(frozen=True, slots=True)
class Warehouse:
    """Warehouse façade over `StorageGateway`.

    Parameters
    ----------
    context
        Storage context providing DuckDB access and snapshot identity.
    """

    context: StorageContext

    @property
    def gateway(self) -> StorageGateway:
        """Return the underlying storage gateway."""
        return self.context.gateway

    def _resolve_scope(self, snapshot: RepoCommitScope | None) -> RepoCommitScope | None:
        return snapshot or self.context.snapshot

    def _resolve_snapshot_ref(self, snapshot: SnapshotRef | None) -> SnapshotRef | None:
        return snapshot or self.context.snapshot

    def _require_snapshot(self, snapshot: SnapshotRef | None) -> SnapshotRef:
        if snapshot is not None:
            return snapshot
        return self.context.require_snapshot()

    def _resolve_options(self, options: MaterializeOptions) -> MaterializeOptions:
        resolved_snapshot = self._resolve_snapshot_ref(options.snapshot)
        if resolved_snapshot is None:
            return options
        if resolved_snapshot is options.snapshot:
            return options
        return replace(options, snapshot=resolved_snapshot)

    def delete_for_snapshot(
        self,
        table_key: str,
        *,
        snapshot: SnapshotRef | None = None,
    ) -> None:
        """Delete rows for a snapshot from a specific table."""
        resolved = self._require_snapshot(snapshot)
        self.gateway.policy.delete_for_snapshot(
            table_key, repo=resolved.repo, commit=resolved.commit
        )

    def read(
        self,
        table_key: str,
        *,
        snapshot: RepoCommitScope | None = None,
    ) -> DuckDBRelation:
        """Return a DuckDB relation, optionally snapshot-filtered.

        Returns
        -------
        DuckDBRelation
            Relation for the requested table, optionally filtered by snapshot.
        """
        relation = self.gateway.relation_from_table_key(table_key)
        resolved_snapshot = self._resolve_scope(snapshot)
        if resolved_snapshot is None:
            return relation
        if not _relation_has_repo_commit_columns(relation):
            return relation
        predicate = _snapshot_filter_expression(
            repo=resolved_snapshot.repo,
            commit=resolved_snapshot.commit,
        )
        return relation.filter(predicate)

    def exists(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> bool:
        """Return True if the table/view exists.

        When `snapshot` is provided, this also checks for the presence of at
        least one row matching `repo` and `commit`, but only when those columns
        exist on the relation.

        Returns
        -------
        bool
            True when the object exists (and has snapshot rows when requested).
        """
        try:
            relation = self.gateway.relation_from_table_key(table_key)
        except (DuckDBError, FileNotFoundError, RuntimeError, ValueError):
            return False

        resolved_snapshot = self._resolve_scope(snapshot)
        if resolved_snapshot is None:
            return True

        if not _relation_has_repo_commit_columns(relation):
            return True

        return _relation_has_snapshot_rows(
            relation,
            repo=resolved_snapshot.repo,
            commit=resolved_snapshot.commit,
        )

    def count(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> int:
        """Count rows in a table, optionally snapshot-filtered.

        Returns
        -------
        int
            Row count for the requested object.
        """
        relation = self.gateway.relation_from_table_key(table_key)
        resolved_snapshot = self._resolve_scope(snapshot)
        if resolved_snapshot is not None and _relation_has_repo_commit_columns(relation):
            predicate = _snapshot_filter_expression(
                repo=resolved_snapshot.repo,
                commit=resolved_snapshot.commit,
            )
            relation = relation.filter(predicate)
        row = relation.count("*").fetchone()
        return int(row[0]) if row is not None else 0

    def materialize_table(
        self,
        table_key: str,
        relation: TabularInput,
        *,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize a tabular input to a DuckDB table.

        Returns
        -------
        MaterializationResult
            Result metadata for the materialization.
        """
        active = self._resolve_options(options or MaterializeOptions())
        _validate_materialize_options(
            active,
            supports_upsert=True,
            upsert_unsupported_message="mode='upsert' requires options.upsert to be provided",
        )

        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)
        schema_version, computed_schema_hash = _contract_schema_metadata(
            self.gateway,
            table_key=table_key,
        )

        def _write() -> int | None:
            coerced = _coerce_tabular_input(
                relation,
                batch_size=DEFAULT_ARROW_BATCH_SIZE,
            )
            return _write_tabular(
                gateway=self.gateway,
                table_key=table_key,
                relation=coerced,
                options=active,
            )

        ctx = _MaterializeWriterContext(
            gateway=self.gateway,
            table_key=table_key,
            started_at=started_at,
            schema_version=schema_version,
            schema_hash=computed_schema_hash,
        )
        return _materialize_with_writer(ctx, options=active, writer=_write)

    def materialize_mappings(
        self,
        table_key: str,
        rows: Iterable[Mapping[str, object]],
        *,
        columns: Sequence[str] | None = None,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize mapping-shaped rows to DuckDB.

        Parameters
        ----------
        table_key
            Destination table key (schema.table).
        rows
            Iterable of mapping rows keyed by column name (e.g., TypedDict models).
        columns
            Optional column order override. When omitted, columns are derived from
            the configured schema provider.
        options
            Materialization options. Defaults to append semantics (no snapshot required).

        Returns
        -------
        MaterializationResult
            Structured result describing the write.
        """
        active = self._resolve_options(options or MaterializeOptions(mode="append"))
        _validate_materialize_options(
            active,
            supports_upsert=False,
            upsert_unsupported_message=(
                "materialize_mappings does not support mode='upsert'; use materialize_table"
            ),
        )

        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)
        schema_version, computed_schema_hash = _contract_schema_metadata(
            self.gateway, table_key=table_key
        )

        def _write() -> int:
            return self.gateway.policy.bulk_insert_mappings(
                table_key,
                rows,
                columns=columns,
            )

        ctx = _MaterializeWriterContext(
            gateway=self.gateway,
            table_key=table_key,
            started_at=started_at,
            schema_version=schema_version,
            schema_hash=computed_schema_hash,
        )
        return _materialize_with_writer(ctx, options=active, writer=_write)

    def ensure_all_views(
        self,
        *,
        overwrite: bool = True,
        strict: bool = False,
        tag_query: TagQuery | None = None,
    ) -> None:
        """Reject SQLGlot view materialization."""
        self.gateway.policy.ensure_all_views(
            overwrite=overwrite,
            strict=strict,
            tag_query=tag_query,
        )

    def explain_table(
        self,
        table_key: str,
        *,
        analyze: bool = False,
        limit: int = 50,
    ) -> str:
        """Return EXPLAIN (or EXPLAIN ANALYZE) output for a table/view query.

        Parameters
        ----------
        table_key
            Fully qualified table or view key (schema.table).
        analyze
            When True, use ``EXPLAIN ANALYZE`` instead of ``EXPLAIN``.
        limit
            LIMIT applied to the SELECT being explained.

        Returns
        -------
        str
            Plan text emitted by DuckDB.

        Raises
        ------
        DuckDBCatalogException
            If the requested table/view is missing and cannot be materialized.
        """
        limited = max(0, limit)
        try:
            relation = self.gateway.relation_from_table_key(table_key)
        except DuckDBCatalogException:
            if self._maybe_materialize_view(table_key):
                relation = self.gateway.relation_from_table_key(table_key)
            else:
                raise
        relation = relation.limit(limited)
        if not analyze:
            return normalize_explain_output(relation.explain()) or ""
        self.gateway.policy.execute_sql("PRAGMA enable_profiling")
        try:
            plan = relation.explain(ExplainType.ANALYZE)
            return normalize_explain_output(plan) or ""
        finally:
            self.gateway.policy.execute_sql("PRAGMA disable_profiling")

    def profile_views(
        self,
        *,
        views: Sequence[str],
        output_dir: Path,
        analyze: bool = False,
        limit: int = 50,
        db_path: Path | None = None,
    ) -> None:
        """Write EXPLAIN/EXPLAIN ANALYZE artifacts for a set of views.

        Parameters
        ----------
        views
            Fully qualified table/view keys to profile.
        output_dir
            Directory to write artifacts into.
        analyze
            When True, run ``EXPLAIN ANALYZE`` instead of ``EXPLAIN``.
        limit
            LIMIT applied to the SELECT being explained.
        db_path
            Optional database path included in the metadata artifact.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "db_path": str(db_path) if db_path is not None else None,
            "analyze": analyze,
            "limit": limit,
            "views": list(views),
        }
        (output_dir / "profile_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        suffix = "analyze" if analyze else "explain"
        for view in views:
            plan = self.explain_table(view, analyze=analyze, limit=limit)
            artifact = output_dir / f"{view.replace('.', '_')}.{suffix}.txt"
            artifact.write_text(plan, encoding="utf-8")

    def _maybe_materialize_view(self, table_key: str) -> bool:
        if not self._is_writable_gateway():
            return False
        schema, _ = split_table_key(table_key)
        contract = self.gateway.datasets.by_table_key.get(table_key)
        _ = (schema, contract)
        return False

    def _is_writable_gateway(self) -> bool:
        config = getattr(self.gateway, "config", None)
        return getattr(config, "read_only", False) is False

    def delete_snapshot(
        self,
        snapshot: SnapshotRef | None = None,
        *,
        include_views: bool = False,
    ) -> int:
        """Delete snapshot-scoped rows for a repo/commit across all datasets.

        Parameters
        ----------
        snapshot
            Snapshot identity whose rows should be removed.
        include_views
            When True, attempt deletion for contracts marked as views. Defaults
            to False because views are read-only.

        Returns
        -------
        int
            Number of dataset tables considered for deletion.
        """
        resolved = self._require_snapshot(snapshot)
        targets: list[str] = []
        for contract in self.gateway.datasets.by_table_key.values():
            if not include_views and contract.is_view:
                continue
            if contract.schema is None:
                continue
            if not _is_snapshot_scoped(contract):
                continue
            targets.append(contract.table_key)

        for table_key in sorted(set(targets)):
            self.gateway.policy.delete_for_snapshot(
                table_key,
                repo=resolved.repo,
                commit=resolved.commit,
            )

        return len(set(targets))


def _relation_has_snapshot_rows(relation: DuckDBRelation, *, repo: str, commit: str) -> bool:
    try:
        predicate = _snapshot_filter_expression(repo=repo, commit=commit)
        filtered = relation.filter(predicate)
        return filtered.limit(1).fetchone() is not None
    except DuckDBError:
        return False


def _snapshot_filter_expression(*, repo: str, commit: str) -> Expression:
    filters = (
        FilterSpecInput(column="repo", op="eq", value=repo),
        FilterSpecInput(column="commit", op="eq", value=commit),
    )
    predicates = compile_filter_predicates(
        filters,
        allowed_columns=_SNAPSHOT_FILTER_COLUMNS,
    )
    expression = duckdb_filter_expression(predicates)
    if expression is None:
        msg = "Snapshot filter compilation returned empty predicate"
        raise FilterCompilerError(msg)
    return expression


def _relation_has_repo_commit_columns(relation: DuckDBRelation) -> bool:
    try:
        names = set(relation.columns)
    except AttributeError:
        return False
    return "repo" in names and "commit" in names


def _dataset_contract(gateway: StorageGateway, *, table_key: str) -> DatasetContract | None:
    contract = gateway.datasets.by_table_key.get(table_key)
    return contract if contract is not None else None


def _is_snapshot_scoped(contract: DatasetContract) -> bool:
    schema = contract.schema
    if schema is None:
        return False
    names = {c.name for c in schema.columns}
    return "repo" in names and "commit" in names


def _profiling_dir_from_env() -> Path | None:
    raw = os.environ.get(_PROFILE_DIR_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _sanitize_table_key_for_path(table_key: str) -> str:
    return table_key.replace("/", "_").replace(".", "__")


def _profiling_output_path(
    *,
    table_key: str,
    snapshot: SnapshotRef | None,
    owner_target: str | None,
) -> Path | None:
    profiling_dir = _profiling_dir_from_env()
    if profiling_dir is None:
        return None

    safe_table = _sanitize_table_key_for_path(table_key)
    safe_target = owner_target.replace("/", "_") if owner_target else "unknown_target"

    repo = snapshot.repo.replace("/", "_") if snapshot is not None else "unknown_repo"
    commit = snapshot.commit if snapshot is not None else "unknown_commit"

    stamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S%f")
    return profiling_dir / repo / commit / safe_target / f"{safe_table}.{stamp}.json"


def _maybe_enable_profiling(
    *,
    con: DuckDBConnection,
    table_key: str,
    snapshot: SnapshotRef | None,
    owner_target: str | None,
) -> Path | None:
    output = _profiling_output_path(
        table_key=table_key, snapshot=snapshot, owner_target=owner_target
    )
    if output is None:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    con.execute("PRAGMA enable_profiling='json'")
    con.execute("PRAGMA profiling_output=?", [str(output)])
    return output


def _disable_profiling_if_enabled(con: DuckDBConnection, path: Path | None) -> None:
    if path is None:
        return
    con.execute("PRAGMA disable_profiling")


def _validate_materialize_options(
    options: MaterializeOptions,
    *,
    supports_upsert: bool,
    upsert_unsupported_message: str,
) -> None:
    snapshot = options.snapshot
    if options.mode == "replace" and options.replace_scope == "snapshot" and snapshot is None:
        msg = "mode='replace' with replace_scope='snapshot' requires snapshot"
        raise ValueError(msg)

    if options.mode != "upsert":
        return

    if not supports_upsert:
        raise ValueError(upsert_unsupported_message)
    if options.upsert is None:
        raise ValueError(upsert_unsupported_message)


def _resolve_fallback_upsert(
    gateway: StorageGateway,
    *,
    table_key: str,
    resolved_columns: Sequence[str],
) -> UpsertConfig | None:
    provider = gateway.policy.schema_provider
    if provider is None:
        return None
    schema = provider.get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return None
    conflict_columns = tuple(schema.primary_key)
    if not conflict_columns:
        return None
    if not resolved_columns:
        return None
    update_columns = tuple(col for col in resolved_columns if col not in conflict_columns)
    if not update_columns:
        update_columns = None
    return UpsertConfig(
        conflict_columns=conflict_columns,
        update_columns=update_columns,
    )


def _resolve_stable_sort_keys(
    table_schema: TableSchema | None,
) -> tuple[str, ...] | None:
    if table_schema is None:
        return None
    policy = table_schema.write_policy
    if policy is not None and policy.stable_sort_keys is not None:
        return policy.stable_sort_keys
    return table_schema.primary_key or None


def _stable_sort_keys_for_table(
    table: pa.Table,
    *,
    table_schema: TableSchema | None,
) -> list[SortKey]:
    stable_sort_keys = _resolve_stable_sort_keys(table_schema)
    if not stable_sort_keys:
        return []
    available = [key for key in stable_sort_keys if key in table.column_names]
    return [(key, "ascending") for key in available]


def _hash_columns_for_table(
    table: pa.Table,
    *,
    table_schema: TableSchema | None,
) -> list[str]:
    if table_schema is not None and table_schema.primary_key:
        return [name for name in table_schema.primary_key if name in table.column_names]
    return list(table.column_names)


def _temp_column_name(table: pa.Table, *, base: str) -> str:
    existing = set(table.column_names)
    name = base
    suffix = 1
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _stable_order_by_hash(table: pa.Table, *, columns: Sequence[str]) -> pa.Table:
    if table.num_rows <= 1:
        return table
    available = [name for name in columns if name in table.column_names]
    if not available:
        return table
    try:
        ordinal = hash_struct_ordinal(
            table,
            columns=available,
            modulus=_HASH_ORDINAL_MODULUS,
        )
    except (RuntimeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        return table
    temp_name = _temp_column_name(table, base="__stable_ordinal")
    try:
        table_with = table.append_column(temp_name, ordinal)
        indices = stable_sort_indices(table_with, sort_keys=[(temp_name, "ascending")])
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
        return table
    return table.take(indices)


def _apply_table_ordering(
    table: pa.Table,
    *,
    table_schema: TableSchema | None,
) -> pa.Table:
    if table.num_rows <= 1:
        return table
    sort_keys = _stable_sort_keys_for_table(table, table_schema=table_schema)
    if sort_keys:
        try:
            indices = stable_sort_indices(table, sort_keys=sort_keys)
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            TypeError,
            ValueError,
        ):
            indices = None
        if indices is not None:
            return table.take(indices)
    hash_columns = _hash_columns_for_table(table, table_schema=table_schema)
    return _stable_order_by_hash(table, columns=hash_columns)


def _contract_schema_metadata(
    gateway: StorageGateway,
    *,
    table_key: str,
) -> tuple[str | None, str | None]:
    contract = _dataset_contract(gateway, table_key=table_key)
    schema_version = contract.schema_version if contract is not None else None
    inferred_schema = table_schema_for_table_key(con=gateway.con, table_key=table_key)
    if inferred_schema is not None:
        return schema_version, schema_hash(inferred_schema)
    return schema_version, None


@dataclass(frozen=True, slots=True)
class _MaterializeWriterContext:
    gateway: StorageGateway
    table_key: str
    started_at: datetime
    schema_version: str | None
    schema_hash: str | None


def _materialize_with_writer(
    ctx: _MaterializeWriterContext,
    *,
    options: MaterializeOptions,
    writer: Callable[[], int | None],
) -> MaterializationResult:
    snapshot = options.snapshot
    profiling_path = _maybe_enable_profiling(
        con=ctx.gateway.con,
        table_key=ctx.table_key,
        snapshot=snapshot,
        owner_target=options.owner_target,
    )

    rows_written: int | None = None
    try:
        with ctx.gateway.policy.transaction():
            if options.mode == "replace":
                if options.replace_scope == "table":
                    ctx.gateway.policy.delete(ctx.table_key)
                elif snapshot is not None:
                    ctx.gateway.policy.delete_for_snapshot(
                        ctx.table_key,
                        repo=snapshot.repo,
                        commit=snapshot.commit,
                    )

            rows_written = writer()
    finally:
        _disable_profiling_if_enabled(ctx.gateway.con, profiling_path)

    completed_at = datetime.now(tz=UTC)

    return MaterializationResult(
        table_key=ctx.table_key,
        repo=snapshot.repo if snapshot is not None else None,
        commit=snapshot.commit if snapshot is not None else None,
        rows_written=rows_written,
        started_at=ctx.started_at,
        completed_at=completed_at,
        schema_hash=ctx.schema_hash,
        schema_version=ctx.schema_version,
        profiling_artifact=str(profiling_path) if profiling_path is not None else None,
    )


def _relation_columns(relation: DuckDBRelation) -> list[str]:
    columns = getattr(relation, "columns", None)
    if columns is None:
        msg = "DuckDB relation does not expose columns"
        raise TypeError(msg)
    return [str(col) for col in columns]


def _relation_row_count(relation: DuckDBRelation, *, table_key: str) -> int:
    row = relation.count("*").fetchone()
    return coerce_int(row[0], ctx=f"{table_key}.count()") if row is not None else 0


def _relation_select_expr(
    relation: DuckDBRelation,
    *,
    columns: Sequence[str],
) -> exp.Select:
    alias = "ci_src"
    subquery = exp.Subquery(
        this=parse_one(relation.sql_query(), dialect=DUCKDB_DIALECT),
        alias=exp.TableAlias(this=exp.to_identifier(alias)),
    )
    return exp.Select(
        expressions=[
            exp.Column(
                this=exp.to_identifier(column),
                table=exp.to_identifier(alias),
            )
            for column in columns
        ],
    ).from_(subquery)


def _write_relation(
    *,
    gateway: StorageGateway,
    table_key: str,
    relation: DuckDBRelation,
    options: MaterializeOptions,
) -> int | None:
    return _write_relation_inner(
        gateway=gateway,
        table_key=table_key,
        relation=relation,
        options=options,
        write_state=RelationWriteState(row_count=None, skip_row_count=False),
    )


def _write_relation_inner(
    *,
    gateway: StorageGateway,
    table_key: str,
    relation: DuckDBRelation,
    options: MaterializeOptions,
    write_state: RelationWriteState,
) -> int | None:
    columns = _relation_columns(relation)
    if not columns:
        return 0
    if write_state.skip_row_count:
        if write_state.row_count == 0:
            return 0
        resolved_count = write_state.row_count
    else:
        resolved_count = _relation_row_count(relation, table_key=table_key)
        if resolved_count == 0:
            return 0

    def _apply_select(select_expr: exp.Expression) -> None:
        if options.mode == "upsert" and options.upsert is not None:
            gateway.policy.upsert_select(
                table_key,
                columns=columns,
                select_sql=select_expr,
                upsert=UpsertSpec(
                    conflict_columns=options.upsert.conflict_columns,
                    update_columns=options.upsert.update_columns,
                    update_condition=options.upsert.update_condition,
                ),
            )
        else:
            gateway.policy.insert_select(
                table_key,
                columns=columns,
                select_sql=select_expr,
            )

    try:
        select_expr = _relation_select_expr(relation, columns=columns)
        _apply_select(select_expr)
    except ParseError:
        reader = relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        with registered_temp_relation(gateway.con, reader, prefix="ci_rel_") as name:
            select_expr = exp.Select(
                expressions=[exp.Column(this=exp.to_identifier(column)) for column in columns],
            ).from_(exp.Table(this=exp.to_identifier(name)))
            _apply_select(select_expr)
    return resolved_count


def _table_row_count(
    relation: DuckDBRelation | pa.Table | pa.RecordBatchReader,
) -> int | None:
    if isinstance(relation, pa.Table):
        return cast("int", relation.num_rows)
    return None


def _materialize_relation_for_validation(
    relation: DuckDBRelation | pa.Table | pa.RecordBatchReader,
    *,
    contract_schema: pa.Schema | None,
    validation_mode: ValidationMode,
) -> DuckDBRelation | pa.Table | pa.RecordBatchReader:
    if isinstance(relation, DuckDBRelation) and (
        contract_schema is not None or validation_mode != "skip"
    ):
        return relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    return relation


def _validate_tabular_relation(
    relation: DuckDBRelation | pa.Table | pa.RecordBatchReader,
    *,
    gateway: StorageGateway,
    table_key: str,
    validation_mode: ValidationMode,
) -> DuckDBRelation | pa.Table | pa.RecordBatchReader:
    if validation_mode == "skip":
        return relation
    context = _validation_context_for_table(gateway, table_key=table_key)
    if isinstance(relation, pa.Table):
        return validate_table(
            table_key,
            relation,
            context=context,
            mode=validation_mode,
        )
    if isinstance(relation, pa.RecordBatchReader):
        return validate_record_batch_reader(
            table_key,
            relation,
            context=context,
            mode=validation_mode,
        )
    return relation


def _ensure_schema_metadata(table_key: str, *, schema: pa.Schema) -> None:
    errors = schema_metadata_errors(schema)
    if not errors:
        return
    raise TableValidationError(table_key, errors)


def _write_tabular(
    *,
    gateway: StorageGateway,
    table_key: str,
    relation: DuckDBRelation | pa.Table | pa.RecordBatchReader,
    options: MaterializeOptions,
) -> int | None:
    table_row_count = _table_row_count(relation)
    contract = _dataset_contract(gateway, table_key=table_key)
    table_schema = contract.schema if contract is not None else None
    contract_schema = _contract_schema_for_table(gateway, table_key=table_key)
    validation_mode = _validation_mode(gateway)
    relation = _materialize_relation_for_validation(
        relation,
        contract_schema=contract_schema,
        validation_mode=validation_mode,
    )
    if contract_schema is not None:
        relation = _align_tabular_input(
            relation,
            contract_schema=contract_schema,
        )
        _ensure_schema_metadata(table_key, schema=relation.schema)
    relation = _validate_tabular_relation(
        relation,
        gateway=gateway,
        table_key=table_key,
        validation_mode=validation_mode,
    )
    if contract_schema is not None and not isinstance(relation, DuckDBRelation):
        if isinstance(relation, pa.RecordBatchReader):
            table = reader_to_table(relation)
        else:
            table = relation
        table = combine_table_chunks(table)
        if contract_schema is not None:
            with contextlib.suppress(
                TypeError,
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
            ):
                table = deep_cast_table_to_contract(table, contract_schema)
        finalized = finalize_table(
            table,
            spec=FinalizeSpec(
                table_key=table_key,
                mode=_finalize_mode(validation_mode),
            ),
        )
        relation = _apply_table_ordering(finalized.good, table_schema=table_schema)
    if isinstance(relation, DuckDBRelation):
        return _write_relation(
            gateway=gateway,
            table_key=table_key,
            relation=relation,
            options=options,
        )
    if isinstance(relation, pa.Table):
        if relation.num_rows == 0:
            return 0
        with registered_temp_relation(gateway.con, relation, prefix="ci_tab_") as name:
            rel = gateway.con.table(name)
            return _write_relation_inner(
                gateway=gateway,
                table_key=table_key,
                relation=rel,
                options=options,
                write_state=RelationWriteState(
                    row_count=relation.num_rows,
                    skip_row_count=True,
                ),
            )
    if isinstance(relation, pa.RecordBatchReader):
        with registered_temp_relation(gateway.con, relation, prefix="ci_rb_") as name:
            rel = gateway.con.table(name)
            return _write_relation_inner(
                gateway=gateway,
                table_key=table_key,
                relation=rel,
                options=options,
                write_state=RelationWriteState(row_count=table_row_count, skip_row_count=True),
            )
    msg = f"Unsupported tabular input for {table_key}: {type(relation)!r}"
    raise TypeError(msg)


def _contract_schema_for_table(
    gateway: StorageGateway,
    *,
    table_key: str,
) -> pa.Schema | None:
    contract = gateway.datasets.by_table_key.get(table_key)
    if contract is not None and contract.schema is not None:
        return arrow_contract_for_table_schema(table_schema=contract.schema)
    try:
        dataset_root_dir = getattr(gateway.config, "dataset_root_dir", None)
        snapshot_id = getattr(gateway.config, "commit", None)
        options = ContractSchemaOptions(
            dataset_root_dir=dataset_root_dir,
            snapshot_id=snapshot_id,
        )
        return contract_schema_for_table_key(
            con=gateway.con,
            table_key=table_key,
            options=options,
        )
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return None


def _validation_context_for_table(
    gateway: StorageGateway,
    *,
    table_key: str,
) -> ColumnarValidationContext:
    schema_provider = getattr(gateway.policy, "schema_provider", None)
    observation_provider = getattr(gateway, "schemas", None)
    resolution = resolve_table_schema(
        table_key,
        observation_provider=observation_provider,
        schema_provider=schema_provider,
    )
    dataset = gateway.datasets.by_table_key.get(table_key)
    validation_profile = dataset.validation_profile if dataset is not None else None
    return ColumnarValidationContext(
        table_schema=resolution.table_schema,
        schema_observation=resolution.observation,
        validation_profile=validation_profile,
        list_alignments=list_alignment_specs_for_table_key(table_key),
    )


def _validation_mode(gateway: StorageGateway) -> ValidationMode:
    mode = getattr(gateway.config, "validation_mode", ContractValidationMode.LENIENT)
    if mode == ContractValidationMode.OFF:
        return "skip"
    if mode == ContractValidationMode.LENIENT:
        return "warn"
    return "strict"


def _finalize_mode(validation_mode: ValidationMode) -> Literal["strict", "tolerant"]:
    if validation_mode == "strict":
        return "strict"
    return "tolerant"


def _align_tabular_input(
    relation: DuckDBRelation | pa.Table | pa.RecordBatchReader,
    *,
    contract_schema: pa.Schema,
) -> DuckDBRelation | pa.RecordBatchReader:
    if isinstance(relation, DuckDBRelation):
        return relation
    reader: pa.RecordBatchReader
    if isinstance(relation, pa.Table):
        reader = table_to_reader(relation, batch_size=None)
    else:
        reader = relation
    return align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=extras_policy_from_schema(contract_schema),
    )


def _coerce_tabular_input(
    value: TabularInput,
    *,
    batch_size: int,
) -> DuckDBRelation | pa.Table | pa.RecordBatchReader:
    if isinstance(value, DuckDBRelation):
        return value
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatchReader):
        return value
    if isinstance(value, ColumnarStream):
        return value.to_reader(batch_size=batch_size)
    reader = coerce_arrow_reader(value, batch_size=batch_size)
    if reader is not None:
        return reader
    table = coerce_arrow_table(value)
    if table is not None:
        return table
    msg = f"Unsupported tabular input: {type(value)!r}"
    raise TypeError(msg)


__all__ = [
    "MaterializationResult",
    "MaterializeOptions",
    "ReplaceScope",
    "TabularInput",
    "UpsertConfig",
    "Warehouse",
    "WriteMode",
]
