"""Typed warehouse API for storage.

The warehouse is the intended single I/O boundary for build + serving. It owns:
- read/exists/count primitives (snapshot-aware where applicable)
- table materialization (snapshot-scoped replace) and view creation
- contract-aware metadata capture (schema hash/version) and optional profiling artifacts

Implementation intentionally composes existing primitives (`StorageGateway`,
`IbisGateway`, `DuckDBPolicyBackend`) so callers can adopt incrementally.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import sqlglot.expressions as exp
from duckdb import ColumnExpression, ConstantExpression, ExplainType

from codeintel.core.schemas.hashing import schema_hash
from codeintel.storage.constants import DUCKDB_DIALECT
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.ibis_adapter import OnConflict
from codeintel.storage.ibis_types import filter_by
from codeintel.storage.tracking.asset_tracking import AssetRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import ibis.expr.types as ir
    import pandas as pd

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBConnection, DuckDBRelation

from codeintel.storage.gateway.protocol import DuckDBCatalogException, DuckDBError

WriteMode = Literal["append", "replace", "upsert"]

_PROFILE_DIR_ENV = "CODEINTEL_WAREHOUSE_PROFILING_DIR"


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


@dataclass(frozen=True, slots=True)
class MaterializeOptions:
    """Options for warehouse materialization operations."""

    snapshot: SnapshotRef | None = None
    mode: WriteMode = "replace"
    owner_target: str | None = None
    input_hash: str | None = None
    asset_type: str = "table"
    upsert: UpsertConfig | None = None


@dataclass(frozen=True, slots=True)
class Warehouse:
    """Warehouse façade over `StorageGateway`.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB + Ibis access.
    """

    gateway: StorageGateway

    def delete_for_snapshot(self, table_key: str, *, snapshot: SnapshotRef) -> None:
        """Delete rows for a snapshot from a specific table."""
        self.gateway.policy.delete_for_snapshot(
            table_key, repo=snapshot.repo, commit=snapshot.commit
        )

    def read(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> ir.Table:
        """Return an Ibis table expression, optionally snapshot-filtered.

        Snapshot filtering is applied only when both `repo` and `commit` columns
        exist on the table and a snapshot is provided.

        Returns
        -------
        ir.Table
            Ibis expression for the requested table, optionally filtered.
        """
        expr = self.gateway.ibis.table(table_key)
        if snapshot is None:
            return expr

        schema = expr.schema()
        names = set(schema.keys())
        if "repo" in names and "commit" in names:
            return filter_by(expr, expr["repo"] == snapshot.repo, expr["commit"] == snapshot.commit)
        return expr

    def exists(self, table_key: str, *, snapshot: SnapshotRef | None = None) -> bool:
        """Return True if the table/view exists.

        When `snapshot` is provided, this also checks for the presence of at
        least one row matching `repo` and `commit`.

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
        if snapshot is not None:
            relation = relation.filter(
                (ColumnExpression("repo") == ConstantExpression(snapshot.repo))
                & (ColumnExpression("commit") == ConstantExpression(snapshot.commit))
            )
        row = relation.count("*").fetchone()
        return int(row[0]) if row is not None else 0

    def materialize_table(
        self,
        table_key: str,
        expr: ir.Table,
        *,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize an Ibis table expression to DuckDB.

        Parameters
        ----------
        table_key
            Destination table key (schema.table).
        expr
            Ibis table expression to write.
        options
            Materialization options, including snapshot identity and write mode.

        Returns
        -------
        MaterializationResult
            Structured result describing the write.

        Raises
        ------
        ValueError
            If ``mode="replace"`` is requested without a snapshot.
        """
        active = options or MaterializeOptions()
        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)

        contract = _dataset_contract(self.gateway, table_key=table_key)
        schema_version = contract.schema_version if contract is not None else None
        computed_schema_hash = None
        if contract is not None and contract.schema is not None:
            computed_schema_hash = schema_hash(contract.schema)

        snapshot = active.snapshot
        if active.mode == "replace" and snapshot is None:
            msg = "mode='replace' requires snapshot for safe snapshot-scoped semantics"
            raise ValueError(msg)
        if active.mode == "upsert" and active.upsert is None:
            msg = "mode='upsert' requires options.upsert to be provided"
            raise ValueError(msg)

        profiling_path = _maybe_enable_profiling(
            con=self.gateway.con,
            table_key=table_key,
            snapshot=snapshot,
            owner_target=active.owner_target,
        )

        rows_written: int | None = None
        try:
            with self.gateway.policy.transaction():
                if active.mode == "replace" and snapshot is not None:
                    self.gateway.policy.delete_for_snapshot(
                        table_key, repo=snapshot.repo, commit=snapshot.commit
                    )

                raw_count = self.gateway.ibis.execute_scalar(expr.count())
                rows_written = _coerce_int(raw_count, ctx=f"{table_key}.count()")
                on_conflict = None
                if active.mode == "upsert" and active.upsert is not None:
                    on_conflict = OnConflict(
                        conflict_columns=active.upsert.conflict_columns,
                        update_columns=active.upsert.update_columns,
                    )
                self.gateway.ibis.write(table_key, expr, on_conflict=on_conflict)

                record = _asset_record_from_options(
                    table_key=table_key,
                    schema_version=schema_version,
                    rows_written=rows_written,
                    options=active,
                    profiling_path=profiling_path,
                )
                if record is not None:
                    self.gateway.assets.record_asset(record)
        finally:
            _disable_profiling_if_enabled(self.gateway.con, profiling_path)

        completed_at = datetime.now(tz=UTC)

        return MaterializationResult(
            table_key=table_key,
            repo=snapshot.repo if snapshot is not None else None,
            commit=snapshot.commit if snapshot is not None else None,
            rows_written=rows_written,
            started_at=started_at,
            completed_at=completed_at,
            schema_hash=computed_schema_hash,
            schema_version=schema_version,
            profiling_artifact=str(profiling_path) if profiling_path is not None else None,
        )

    def materialize_dataframe(
        self,
        table_key: str,
        df: pd.DataFrame,
        *,
        options: MaterializeOptions | None = None,
    ) -> MaterializationResult:
        """Materialize a DataFrame to DuckDB using schema-aware inserts.

        Parameters
        ----------
        table_key
            Destination table key (schema.table).
        df
            DataFrame containing rows to write.
        options
            Materialization options, including snapshot identity and write mode.

        Returns
        -------
        MaterializationResult
            Structured result describing the write.

        Raises
        ------
        ValueError
            If ``mode="replace"`` is requested without a snapshot.
        """
        active = options or MaterializeOptions()
        snapshot = active.snapshot
        if active.mode == "replace" and snapshot is None:
            msg = "mode='replace' requires snapshot for safe snapshot-scoped semantics"
            raise ValueError(msg)
        if active.mode == "upsert" and active.upsert is None:
            msg = "mode='upsert' requires options.upsert to be provided"
            raise ValueError(msg)

        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)

        contract = _dataset_contract(self.gateway, table_key=table_key)
        schema_version = contract.schema_version if contract is not None else None
        computed_schema_hash = None
        if contract is not None and contract.schema is not None:
            computed_schema_hash = schema_hash(contract.schema)

        profiling_path = _maybe_enable_profiling(
            con=self.gateway.con,
            table_key=table_key,
            snapshot=snapshot,
            owner_target=active.owner_target,
        )

        rows_written: int | None = None
        try:
            with self.gateway.policy.transaction():
                if active.mode == "replace" and snapshot is not None:
                    self.gateway.policy.delete_for_snapshot(
                        table_key, repo=snapshot.repo, commit=snapshot.commit
                    )

                if active.mode == "upsert" and active.upsert is not None:
                    on_conflict = OnConflict(
                        conflict_columns=active.upsert.conflict_columns,
                        update_columns=active.upsert.update_columns,
                    )
                    result = self.gateway.ibis.write(table_key, df, on_conflict=on_conflict)
                    rows_written = result.rows_affected
                else:
                    rows_written = len(df)
                    if rows_written:
                        self.gateway.ibis.write(table_key, df)

                record = _asset_record_from_options(
                    table_key=table_key,
                    schema_version=schema_version,
                    rows_written=rows_written,
                    options=active,
                    profiling_path=profiling_path,
                )
                if record is not None:
                    self.gateway.assets.record_asset(record)
        finally:
            _disable_profiling_if_enabled(self.gateway.con, profiling_path)

        completed_at = datetime.now(tz=UTC)

        return MaterializationResult(
            table_key=table_key,
            repo=snapshot.repo if snapshot is not None else None,
            commit=snapshot.commit if snapshot is not None else None,
            rows_written=rows_written,
            started_at=started_at,
            completed_at=completed_at,
            schema_hash=computed_schema_hash,
            schema_version=schema_version,
            profiling_artifact=str(profiling_path) if profiling_path is not None else None,
        )

    def materialize_rows(
        self,
        table_key: str,
        rows: Sequence[tuple[object, ...]],
        *,
        columns: Sequence[str],
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
            Column names matching the row tuple positions.
        options
            Materialization options, including snapshot identity and write mode.

        Returns
        -------
        MaterializationResult
            Structured result describing the write.

        Raises
        ------
        ValueError
            If ``mode="replace"`` is requested without a snapshot.
        """
        active = options or MaterializeOptions()
        snapshot = active.snapshot
        if active.mode == "replace" and snapshot is None:
            msg = "mode='replace' requires snapshot for safe snapshot-scoped semantics"
            raise ValueError(msg)
        if active.mode == "upsert" and active.upsert is None:
            msg = "mode='upsert' requires options.upsert to be provided"
            raise ValueError(msg)

        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)

        contract = _dataset_contract(self.gateway, table_key=table_key)
        schema_version = contract.schema_version if contract is not None else None
        computed_schema_hash = None
        if contract is not None and contract.schema is not None:
            computed_schema_hash = schema_hash(contract.schema)

        profiling_path = _maybe_enable_profiling(
            con=self.gateway.con,
            table_key=table_key,
            snapshot=snapshot,
            owner_target=active.owner_target,
        )

        rows_written: int | None = None
        try:
            with self.gateway.policy.transaction():
                if active.mode == "replace" and snapshot is not None:
                    self.gateway.policy.delete_for_snapshot(
                        table_key, repo=snapshot.repo, commit=snapshot.commit
                    )

                if active.mode == "upsert" and active.upsert is not None:
                    on_conflict = OnConflict(
                        conflict_columns=active.upsert.conflict_columns,
                        update_columns=active.upsert.update_columns,
                    )
                    result = self.gateway.ibis.write(
                        table_key,
                        rows,
                        columns=list(columns),
                        on_conflict=on_conflict,
                    )
                    rows_written = result.rows_affected
                else:
                    rows_written = len(rows)
                    if rows_written:
                        self.gateway.ibis.write(table_key, rows, columns=list(columns))

                record = _asset_record_from_options(
                    table_key=table_key,
                    schema_version=schema_version,
                    rows_written=rows_written,
                    options=active,
                    profiling_path=profiling_path,
                )
                if record is not None:
                    self.gateway.assets.record_asset(record)
        finally:
            _disable_profiling_if_enabled(self.gateway.con, profiling_path)

        completed_at = datetime.now(tz=UTC)

        return MaterializationResult(
            table_key=table_key,
            repo=snapshot.repo if snapshot is not None else None,
            commit=snapshot.commit if snapshot is not None else None,
            rows_written=rows_written,
            started_at=started_at,
            completed_at=completed_at,
            schema_hash=computed_schema_hash,
            schema_version=schema_version,
            profiling_artifact=str(profiling_path) if profiling_path is not None else None,
        )

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

        Raises
        ------
        ValueError
            If ``mode="replace"`` is requested without a snapshot.
        """
        active = options or MaterializeOptions(mode="append")
        snapshot = active.snapshot
        if active.mode == "replace" and snapshot is None:
            msg = "mode='replace' requires snapshot for safe snapshot-scoped semantics"
            raise ValueError(msg)
        if active.mode == "upsert":
            msg = "materialize_mappings does not support mode='upsert'; use materialize_rows/dataframe"
            raise ValueError(msg)

        started_at = datetime.now(tz=UTC)
        self.gateway.policy.ensure_table(table_key, create_if_missing=True)

        contract = _dataset_contract(self.gateway, table_key=table_key)
        schema_version = contract.schema_version if contract is not None else None
        computed_schema_hash = None
        if contract is not None and contract.schema is not None:
            computed_schema_hash = schema_hash(contract.schema)

        replace_snapshot = snapshot if active.mode == "replace" and snapshot is not None else None

        profiling_path = _maybe_enable_profiling(
            con=self.gateway.con,
            table_key=table_key,
            snapshot=snapshot,
            owner_target=active.owner_target,
        )

        rows_written: int | None = None
        try:
            with self.gateway.policy.transaction():
                if replace_snapshot is not None:
                    self.gateway.policy.delete_for_snapshot(
                        table_key, repo=replace_snapshot.repo, commit=replace_snapshot.commit
                    )

                rows_written = self.gateway.policy.bulk_insert_mappings(
                    table_key,
                    rows,
                    columns=columns,
                )

                record = _asset_record_from_options(
                    table_key=table_key,
                    schema_version=schema_version,
                    rows_written=rows_written,
                    options=active,
                    profiling_path=profiling_path,
                )
                if record is not None:
                    self.gateway.assets.record_asset(record)
        finally:
            _disable_profiling_if_enabled(self.gateway.con, profiling_path)

        completed_at = datetime.now(tz=UTC)

        return MaterializationResult(
            table_key=table_key,
            repo=snapshot.repo if snapshot is not None else None,
            commit=snapshot.commit if snapshot is not None else None,
            rows_written=rows_written,
            started_at=started_at,
            completed_at=completed_at,
            schema_hash=computed_schema_hash,
            schema_version=schema_version,
            profiling_artifact=str(profiling_path) if profiling_path is not None else None,
        )

    def create_or_replace_view(
        self, table_key: str, expr: ir.Table, *, overwrite: bool = True
    ) -> None:
        """Create or replace a view from an Ibis expression."""
        schema, name = split_table_key(table_key)
        self.gateway.policy.create_schema_if_not_exists(schema)
        self.gateway.ibis.con.create_view(name, expr, database=schema, overwrite=overwrite)

    def ensure_all_views(self, *, overwrite: bool = True, strict: bool = False) -> None:
        """Ensure all registered views are materialized."""
        self.gateway.policy.ensure_all_views(overwrite=overwrite, strict=strict)

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
    escaped = str(output).replace("'", "''")
    con.execute(f"PRAGMA profiling_output='{escaped}'")
    return output


def _disable_profiling_if_enabled(con: DuckDBConnection, path: Path | None) -> None:
    if path is None:
        return
    con.execute("PRAGMA disable_profiling")


def _asset_record_from_options(
    *,
    table_key: str,
    schema_version: str | None,
    rows_written: int | None,
    options: MaterializeOptions,
    profiling_path: Path | None,
) -> AssetRecord | None:
    snapshot = options.snapshot
    if snapshot is None or options.owner_target is None:
        return None

    metadata: dict[str, object] = {}
    if profiling_path is not None:
        metadata["profiling_artifact"] = str(profiling_path)

    return AssetRecord(
        asset_key=table_key,
        asset_type=options.asset_type,
        repo=snapshot.repo,
        commit=snapshot.commit,
        owner_target=options.owner_target,
        schema_version=schema_version,
        row_count=rows_written,
        input_hash=options.input_hash,
        metadata=metadata or None,
    )


def _coerce_int(value: object, *, ctx: str) -> int:
    if isinstance(value, bool):
        msg = f"Expected int-like value for {ctx}, got bool"
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            msg = f"Expected int-like string for {ctx}, got {value!r}"
            raise TypeError(msg) from exc

    msg = f"Expected int-like value for {ctx}, got {type(value).__name__}"
    raise TypeError(msg)


__all__ = [
    "MaterializationResult",
    "MaterializeOptions",
    "UpsertConfig",
    "Warehouse",
    "WriteMode",
]
