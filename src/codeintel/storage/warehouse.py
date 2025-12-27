"""Typed warehouse API for storage.

The warehouse is the intended single I/O boundary for build + serving. It owns:
- read/exists/count primitives (snapshot-aware where applicable)
- table materialization (snapshot-scoped replace) and view creation
- contract-aware metadata capture (schema hash/version) and optional profiling artifacts

Implementation intentionally composes existing primitives (`StorageGateway`,
`DuckDBPolicyBackend`) so callers can adopt incrementally.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import sqlglot.expressions as exp
from duckdb import ColumnExpression, ConstantExpression, ExplainType

from codeintel.core.schemas.hashing import schema_hash
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.query_results import coerce_int
from codeintel.storage.snapshot_scoping import RepoCommitScope
from codeintel.storage.staging import registered_temp_relation
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.duckdb_types import DuckDBConnection, DuckDBRelation
    from codeintel.storage.gateway import StorageGateway

from codeintel.storage.duckdb_types import DuckDBCatalogException, DuckDBError

WriteMode = Literal["append", "replace", "upsert"]
ReplaceScope = Literal["snapshot", "table"]

_PROFILE_DIR_ENV = "CODEINTEL_WAREHOUSE_PROFILING_DIR"

log = logging.getLogger(__name__)


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
class Warehouse:
    """Warehouse façade over `StorageGateway`.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    """

    gateway: StorageGateway

    def delete_for_snapshot(self, table_key: str, *, snapshot: SnapshotRef) -> None:
        """Delete rows for a snapshot from a specific table."""
        self.gateway.policy.delete_for_snapshot(
            table_key, repo=snapshot.repo, commit=snapshot.commit
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
        if snapshot is None:
            return relation
        if not _relation_has_repo_commit_columns(relation):
            return relation
        return relation.filter(
            (ColumnExpression("repo") == ConstantExpression(snapshot.repo))
            & (ColumnExpression("commit") == ConstantExpression(snapshot.commit))
        )

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
        schema, name = split_table_key(table_key)
        try:
            relation = self.gateway.con.table(f"{schema}.{name}")
        except DuckDBError:
            return False

        if snapshot is None:
            return True

        if not _relation_has_repo_commit_columns(relation):
            return True

        return _relation_has_snapshot_rows(
            relation,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )

    def count(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> int:
        """Count rows in a table, optionally snapshot-filtered.

        Returns
        -------
        int
            Row count for the requested object.
        """
        schema, name = split_table_key(table_key)
        relation = self.gateway.con.table(f"{schema}.{name}")
        if snapshot is not None and _relation_has_repo_commit_columns(relation):
            relation = relation.filter(
                (ColumnExpression("repo") == ConstantExpression(snapshot.repo))
                & (ColumnExpression("commit") == ConstantExpression(snapshot.commit))
            )
        row = relation.count("*").fetchone()
        return int(row[0]) if row is not None else 0

    def materialize_table(
        self,
        table_key: str,
        relation: DuckDBRelation,
        *,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize a DuckDB relation to a table.

        Returns
        -------
        MaterializationResult
            Result metadata for the materialization.
        """
        active = options or MaterializeOptions()
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
            return _write_relation(
                gateway=self.gateway,
                table_key=table_key,
                relation=relation,
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

    def materialize_rows(
        self,
        table_key: str,
        rows: Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str] | None,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize row tuples to DuckDB.

        Parameters
        ----------
        table_key
            Destination table key (schema.table).
        rows
            Row tuples matching the provided columns.
        columns
            Column names matching the row tuple positions. When omitted, column order is derived
            from the configured schema provider.
        options
            Materialization options, including snapshot identity and write mode.

        Returns
        -------
        MaterializationResult
            Structured result describing the write.
        """
        active = options or MaterializeOptions()
        _validate_materialize_options(
            active,
            supports_upsert=True,
            upsert_unsupported_message="mode='upsert' requires options.upsert to be provided",
        )

        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)
        schema_version, computed_schema_hash = _contract_schema_metadata(
            self.gateway, table_key=table_key
        )
        resolved_columns = (
            list(columns) if columns is not None else _require_columns(self.gateway, table_key)
        )

        def _write() -> int | None:
            return _write_rows(
                gateway=self.gateway,
                table_key=table_key,
                rows=rows,
                resolved_columns=resolved_columns,
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
        active = options or MaterializeOptions(mode="append")
        _validate_materialize_options(
            active,
            supports_upsert=False,
            upsert_unsupported_message=(
                "materialize_mappings does not support mode='upsert'; "
                "use materialize_rows/dataframe"
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
        """Ensure all registered views are materialized."""
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
        schema, name = split_table_key(table_key)
        select_expr = (
            exp.select("*")
            .from_(exp.Table(this=exp.to_identifier(name), db=exp.to_identifier(schema)))
            .limit(limited)
        )
        sql = select_expr.sql(dialect=DUCKDB_DIALECT)
        try:
            relation = self.gateway.con.sql(sql)
        except DuckDBCatalogException:
            if self._maybe_materialize_view(table_key):
                relation = self.gateway.con.sql(sql)
            else:
                raise
        if not analyze:
            return relation.explain()
        self.gateway.policy.execute_sql("PRAGMA enable_profiling")
        try:
            return relation.explain(ExplainType.ANALYZE)
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
        if self._is_writable_gateway() and any(view.startswith("docs.") for view in views):
            self.ensure_all_views(overwrite=True, strict=False)
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
        if schema == "docs" or (contract is not None and contract.is_view):
            self.ensure_all_views(overwrite=True, strict=False)
            return True
        return False

    def _is_writable_gateway(self) -> bool:
        config = getattr(self.gateway, "config", None)
        return getattr(config, "read_only", False) is False

    def delete_snapshot(self, snapshot: SnapshotRef, *, include_views: bool = False) -> int:
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
                table_key, repo=snapshot.repo, commit=snapshot.commit
            )

        return len(set(targets))


def _relation_has_snapshot_rows(relation: DuckDBRelation, *, repo: str, commit: str) -> bool:
    try:
        filtered = relation.filter(
            (ColumnExpression("repo") == ConstantExpression(repo))
            & (ColumnExpression("commit") == ConstantExpression(commit))
        )
        return filtered.limit(1).fetchone() is not None
    except DuckDBError:
        return False


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


def _contract_schema_metadata(
    gateway: StorageGateway,
    *,
    table_key: str,
) -> tuple[str | None, str | None]:
    contract = _dataset_contract(gateway, table_key=table_key)
    schema_version = contract.schema_version if contract is not None else None
    computed_schema_hash = None
    if contract is not None and contract.schema is not None:
        computed_schema_hash = schema_hash(contract.schema)
    return schema_version, computed_schema_hash


def _require_columns(gateway: StorageGateway, table_key: str) -> list[str]:
    provider = gateway.policy.schema_provider
    if provider is None:
        msg = f"Schema provider is required to infer columns for {table_key!r}"
        raise RuntimeError(msg)
    schema = provider.require_table_schema(table_key)
    return list(schema.column_names())


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


_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")


def _temp_relation_name(prefix: str) -> str:
    cleaned = _NAME_SANITIZER.sub("_", prefix.strip()) or "ci_rel_"
    return f"{cleaned}{uuid.uuid4().hex}"


def _relation_columns(relation: DuckDBRelation) -> list[str]:
    columns = getattr(relation, "columns", None)
    if columns is None:
        msg = "DuckDB relation does not expose columns"
        raise TypeError(msg)
    return [str(col) for col in columns]


def _relation_row_count(relation: DuckDBRelation, *, table_key: str) -> int:
    row = relation.count("*").fetchone()
    return coerce_int(row[0], ctx=f"{table_key}.count()") if row is not None else 0


def _register_relation_view(
    relation: DuckDBRelation,
    *,
    prefix: str = "ci_rel_",
) -> str:
    name = _temp_relation_name(prefix)
    create_view = getattr(relation, "create_view", None)
    if not callable(create_view):
        msg = "DuckDB relation does not support create_view"
        raise TypeError(msg)
    create_view(name)
    return name


def _drop_relation_view(gateway: StorageGateway, view_name: str) -> None:
    gateway.con.execute(f"DROP VIEW IF EXISTS {view_name}")


def _write_relation(
    *,
    gateway: StorageGateway,
    table_key: str,
    relation: DuckDBRelation,
    options: MaterializeOptions,
) -> int | None:
    columns = _relation_columns(relation)
    if not columns:
        return 0
    row_count = _relation_row_count(relation, table_key=table_key)
    if row_count == 0:
        return 0
    view_name = _register_relation_view(relation)
    try:
        select_expr = exp.Select(
            expressions=[exp.Column(this=exp.to_identifier(col)) for col in columns],
        ).from_(exp.Table(this=exp.to_identifier(view_name)))
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
    finally:
        _drop_relation_view(gateway, view_name)
    return row_count


def _rows_to_arrow_table(
    rows: Sequence[tuple[object, ...]],
    *,
    resolved_columns: Sequence[str],
) -> pa.Table:
    if not rows:
        return pa.table({name: [] for name in resolved_columns})
    columns = list(zip(*rows, strict=False))
    data = {
        name: list(values)
        for name, values in zip(resolved_columns, columns, strict=False)
    }
    return pa.table(data)


def _write_rows(
    *,
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[tuple[object, ...]],
    resolved_columns: Sequence[str],
    options: MaterializeOptions,
) -> int | None:
    if not rows:
        return 0
    if options.mode == "upsert" and options.upsert is not None:
        return _write_rows_upsert(
            gateway=gateway,
            table_key=table_key,
            rows=rows,
            resolved_columns=resolved_columns,
            upsert=options.upsert,
        )
    if options.fallback_upsert_on_conflict:
        upsert = _resolve_fallback_upsert(
            gateway,
            table_key=table_key,
            resolved_columns=resolved_columns,
        )
        if upsert is not None:
            log.warning("Using upsert fallback for %s row write", table_key)
            rows_affected = _write_rows_upsert(
                gateway=gateway,
                table_key=table_key,
                rows=rows,
                resolved_columns=resolved_columns,
                upsert=upsert,
            )
            return (
                coerce_int(
                    rows_affected,
                    ctx=f"{table_key} upsert rows_affected",
                )
                or 0
            )
        log.warning("Upsert fallback requested for %s, but no primary key found", table_key)
    return _write_rows_with_staging(
        gateway=gateway,
        table_key=table_key,
        rows=rows,
        resolved_columns=resolved_columns,
        options=options,
    )


def _write_rows_with_staging(
    *,
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[tuple[object, ...]],
    resolved_columns: Sequence[str],
    options: MaterializeOptions,
) -> int:
    table = _rows_to_arrow_table(rows, resolved_columns=resolved_columns)
    if table.num_rows == 0:
        return 0
    with registered_temp_relation(gateway.con, table, prefix="ci_rows_") as temp_name:
        select_expr = exp.Select(
            expressions=[exp.Column(this=exp.to_identifier(col)) for col in resolved_columns],
        ).from_(exp.Table(this=exp.to_identifier(temp_name)))
        if options.mode == "upsert" and options.upsert is not None:
            gateway.policy.upsert_select(
                table_key,
                columns=resolved_columns,
                select_sql=select_expr,
                upsert=UpsertSpec(
                    conflict_columns=options.upsert.conflict_columns,
                    update_columns=options.upsert.update_columns,
                    update_condition=options.upsert.update_condition,
                ),
            )
            return table.num_rows
        gateway.policy.insert_select(
            table_key,
            columns=resolved_columns,
            select_sql=select_expr,
        )
        return table.num_rows


def _write_rows_upsert(
    *,
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[tuple[object, ...]],
    resolved_columns: Sequence[str],
    upsert: UpsertConfig,
) -> int | None:
    table = _rows_to_arrow_table(rows, resolved_columns=resolved_columns)
    if table.num_rows == 0:
        return 0
    with registered_temp_relation(gateway.con, table, prefix="ci_rows_") as temp_name:
        select_expr = exp.Select(
            expressions=[exp.Column(this=exp.to_identifier(col)) for col in resolved_columns],
        ).from_(exp.Table(this=exp.to_identifier(temp_name)))
        gateway.policy.upsert_select(
            table_key,
            columns=resolved_columns,
            select_sql=select_expr,
            upsert=UpsertSpec(
                conflict_columns=upsert.conflict_columns,
                update_columns=upsert.update_columns,
                update_condition=upsert.update_condition,
            ),
        )
    return table.num_rows


__all__ = [
    "MaterializationResult",
    "MaterializeOptions",
    "ReplaceScope",
    "UpsertConfig",
    "Warehouse",
    "WriteMode",
]
