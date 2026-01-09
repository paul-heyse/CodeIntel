"""Build manifest and run tracking persistence for DuckDB.

This module provides persistent tracking of build output manifests and
build runs, enabling cache invalidation and observability of the build
system.

All DuckDB access is encapsulated here, following the storage layer pattern.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
from sqlglot import exp

from codeintel.core.build_manifest import BuildRunRecord, OutputManifest
from codeintel.core.columnar.arrowdsl import ExecutionContext, ExecutionPlan, run_pipeline
from codeintel.core.columnar.compute_helpers import call_compute, require_array
from codeintel.core.columnar.conversion import reader_to_table, table_to_reader
from codeintel.core.columnar.finalize_ops import FinalizeDedupe, FinalizeSpec
from codeintel.core.columnar.iter import iter_array_values
from codeintel.core.columnar.kernels import stable_sort_indices
from codeintel.core.columnar.masks import and_mask, fill_null_false, invert_mask
from codeintel.core.columnar.normalization import normalize_array
from codeintel.core.columnar.plan_ops import ScanPlanOptions, build_scan_plan
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.columnar.streaming import DatasetScanOptions
from codeintel.core.constants import DEFAULT_ARROW_PROVENANCE_COLUMNS
from codeintel.core.datasets.arrow_store import (
    ArrowDatasetWriteOptions,
    scan_dataset,
    write_dataset,
)
from codeintel.core.datasets.paths import dataset_snapshot_dir
from codeintel.core.gateway import ScipRunRecordProtocol
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.primitives import resolve_stable_sort_keys
from codeintel.core.schemas.service import get_schema_service
from codeintel.core.serialization.json import (
    decode_json_dict,
    deserialize_str_tuple,
    serialize_str_sequence,
)
from codeintel.core.serialization.payload import encode_payload
from codeintel.core.sqlglot_tools import render_sql_duckdb, table_expr_from_ref
from codeintel.core.time import utc_now
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.helpers.table_key import split_table_key
from codeintel.storage.query_results import iter_tuples_from_arrow_reader
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from codeintel.core.build_manifest import BuildStatus
    from codeintel.core.hamilton.records import NodeExecutionRecord, TargetRunRecord
    from codeintel.storage.gateway.protocol import StorageGateway

log = logging.getLogger(__name__)
_MANIFEST_TABLE_KEY = "build.output_manifests"
_MANIFEST_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ScipRunRecord:
    """Structured record for build.scip_runs telemetry rows."""

    run_id: str
    repo: str
    commit: str
    mode: str
    options_hash: str | None
    project_version: str | None
    project_namespace: str | None
    tool_version: str | None
    total_modules: int
    changed_modules: int
    deleted_modules: int
    changed_ratio: float | None
    batch_size: int | None
    batch_count: int
    decision: str | None
    ratio_gate_applied: bool | None
    ratio_gate_min_modules: int | None
    ratio_gate_min_changed: int | None
    hash_source: str | None
    hash_source_breakdown: str | None
    hash_reused: int
    hash_computed: int
    plan_ms: float | None
    hash_ms: float | None
    tool_ms: float | None
    parse_ms: float | None
    merge_ms: float | None
    write_ms: float | None
    total_ms: float | None
    status: str
    error_summary: str | None
    output_scip: str | None
    recorded_at: datetime


def _parse_manifest_row(row: tuple[Any, ...]) -> OutputManifest:
    """Parse a DuckDB row into an OutputManifest.

    Centralizes type coercion from DuckDB result tuples to typed dataclass.

    Parameters
    ----------
    row
        DuckDB row tuple from output_manifests table.
        Expected column order: target, repo, commit, impl_kind, computed_at,
        duration_ms, input_hash, output_hash, row_count, options_hash,
        change_delta

    Returns
    -------
    OutputManifest
        Typed manifest dataclass.
    """
    return OutputManifest(
        target=str(row[0]),
        repo=str(row[1]),
        commit=str(row[2]),
        impl_kind=str(row[3]),
        computed_at=cast("datetime", row[4]),
        duration_ms=float(row[5]),
        input_hash=str(row[6]),
        output_hash=str(row[7]) if row[7] is not None else None,
        row_count=int(row[8]) if row[8] is not None else None,
        options_hash=str(row[9]) if row[9] is not None else None,
        change_delta=decode_json_dict(row[10]) if row[10] is not None else None,
    )


def _parse_run_row(row: tuple[Any, ...]) -> BuildRunRecord:
    """Parse a DuckDB row into a BuildRunRecord.

    Centralizes type coercion from DuckDB result tuples to typed dataclass.

    Parameters
    ----------
    row
        DuckDB row tuple from build.runs table.
        Expected column order: run_id, repo, commit, requested_targets,
        computed_targets, skipped_targets, started_at, completed_at,
        status, error_summary, duration_ms

    Returns
    -------
    BuildRunRecord
        Typed run record dataclass.
    """
    return BuildRunRecord(
        run_id=str(row[0]),
        repo=str(row[1]),
        commit=str(row[2]),
        requested_targets=deserialize_str_tuple(cast("str | None", row[3])),
        computed_targets=deserialize_str_tuple(cast("str | None", row[4])),
        skipped_targets=deserialize_str_tuple(cast("str | None", row[5])),
        started_at=cast("datetime", row[6]),
        completed_at=cast("datetime | None", row[7]),
        status=cast("BuildStatus", row[8]),
        error_summary=str(row[9]) if row[9] is not None else None,
        duration_ms=float(row[10]) if row[10] is not None else None,
    )


def _combine_conditions(conditions: Sequence[exp.Expression]) -> exp.Expression | None:
    if not conditions:
        return None
    combined = conditions[0]
    for condition in conditions[1:]:
        combined = exp.and_(combined, condition)
    return combined


def _manifest_select_exprs(impl_column: str) -> list[exp.Expression]:
    return [
        exp.Column(this=exp.to_identifier("target")),
        exp.Column(this=exp.to_identifier("repo")),
        exp.Column(this=exp.to_identifier("commit")),
        exp.alias_(exp.Column(this=exp.to_identifier(impl_column)), "impl_kind"),
        exp.Column(this=exp.to_identifier("computed_at")),
        exp.Column(this=exp.to_identifier("duration_ms")),
        exp.Column(this=exp.to_identifier("input_hash")),
        exp.Column(this=exp.to_identifier("output_hash")),
        exp.Column(this=exp.to_identifier("row_count")),
        exp.Column(this=exp.to_identifier("options_hash")),
        exp.Column(this=exp.to_identifier("change_delta")),
    ]


def _manifest_row_payload(
    manifest: OutputManifest,
    *,
    impl_column: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": manifest.target,
        "repo": manifest.repo,
        "commit": manifest.commit,
        "impl_kind": manifest.impl_kind,
        "computed_at": manifest.computed_at,
        "duration_ms": manifest.duration_ms,
        "input_hash": manifest.input_hash,
        "output_hash": manifest.output_hash,
        "row_count": manifest.row_count,
        "options_hash": manifest.options_hash,
        "change_delta": manifest.change_delta,
    }
    if impl_column != "impl_kind":
        payload[impl_column] = payload.pop("impl_kind")
    return payload


def _run_select_exprs() -> list[exp.Expression]:
    return [
        exp.Column(this=exp.to_identifier("run_id")),
        exp.Column(this=exp.to_identifier("repo")),
        exp.Column(this=exp.to_identifier("commit")),
        exp.Column(this=exp.to_identifier("requested_targets")),
        exp.Column(this=exp.to_identifier("computed_targets")),
        exp.Column(this=exp.to_identifier("skipped_targets")),
        exp.Column(this=exp.to_identifier("started_at")),
        exp.Column(this=exp.to_identifier("completed_at")),
        exp.Column(this=exp.to_identifier("status")),
        exp.Column(this=exp.to_identifier("error_summary")),
        exp.Column(this=exp.to_identifier("duration_ms")),
    ]


class BuildTracking:
    """Accessor for build manifest and run tracking tables.

    This class provides CRUD operations for:
    - ``build.output_manifests``: Records of computed targets
    - ``build.runs``: Records of build system runs

    All operations are performed directly on the DuckDB connection
    without caching, following the storage accessor pattern.

    Parameters
    ----------
    con
        DuckDB connection to use for queries.

    Examples
    --------
    >>> tracking = BuildTracking(gateway)
    >>> tracking.save_manifest(manifest)
    >>> loaded = tracking.load_manifest("function_types", "org/repo", "abc123")
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize build tracking accessor.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        """
        self._gateway = gateway
        self._con = gateway.con
        self._backend = gateway.policy
        self._impl_kind_columns: dict[str, str] = {}
        self._table_columns_cache: dict[str, set[str]] = {}

    def _table_columns(self, table_key: str) -> set[str]:
        cached = self._table_columns_cache.get(table_key)
        if cached is not None:
            return cached
        schema, table = split_table_key(table_key)
        query = (
            exp.select(exp.Column(this=exp.to_identifier("column_name")))
            .from_(table_expr_from_ref("information_schema.columns"))
            .where(
                _combine_conditions(
                    [
                        exp.EQ(
                            this=exp.Column(this=exp.to_identifier("table_schema")),
                            expression=exp.Placeholder(),
                        ),
                        exp.EQ(
                            this=exp.Column(this=exp.to_identifier("table_name")),
                            expression=exp.Placeholder(),
                        ),
                    ]
                )
            )
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            [schema, table],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
        columns = {str(row[0]) for row in iter_tuples_from_arrow_reader(reader)}
        self._table_columns_cache[table_key] = columns
        return columns

    def _impl_kind_column(self, table_key: str) -> str:
        cached = self._impl_kind_columns.get(table_key)
        if cached is not None:
            return cached
        columns = self._table_columns(table_key)
        if "impl_kind" in columns:
            column = "impl_kind"
        elif "plugin" in columns:
            column = "plugin"
        else:
            column = "impl_kind"
        self._impl_kind_columns[table_key] = column
        return column

    def _manifest_impl_column(self, table_key: str) -> str:
        try:
            table_schema = get_schema_service().require_table_schema(table_key)
        except (KeyError, RuntimeError):
            return self._impl_kind_column(table_key)
        columns = {col.name for col in table_schema.columns}
        if "impl_kind" in columns:
            return "impl_kind"
        if "plugin" in columns:
            return "plugin"
        return "impl_kind"

    @staticmethod
    def _manifest_arrow_schema(table_key: str) -> pa.Schema:
        schema_service = get_schema_service()
        table_schema = schema_service.require_table_schema(table_key)
        return arrow_contract_for_table_schema(table_schema=table_schema)

    def _manifest_dataset_context(self, *, commit: str | None) -> tuple[Path, str]:
        dataset_root_dir = self._gateway.config.dataset_root_dir
        snapshot_id = self._gateway.config.commit or commit
        if dataset_root_dir is None or snapshot_id is None:
            msg = f"Dataset root and snapshot id required for {_MANIFEST_TABLE_KEY}"
            raise RuntimeError(msg)
        return dataset_root_dir, snapshot_id

    def _is_parquet_backed(self, table_key: str) -> bool:
        dataset = self._gateway.datasets.by_table_key.get(table_key)
        return dataset is not None and not dataset.is_view

    @staticmethod
    def _load_manifest_table(
        *,
        table_key: str,
        dataset_root_dir: Path,
        snapshot_id: str,
        arrow_schema: pa.Schema,
    ) -> pa.Table | None:
        snapshot_dir = dataset_snapshot_dir(
            dataset_root_dir,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        if not snapshot_dir.is_dir():
            return None
        try:
            dataset = scan_dataset(
                dataset_root=dataset_root_dir,
                table_key=table_key,
                snapshot_id=snapshot_id,
            )
        except FileNotFoundError:
            return None
        scan_options = DatasetScanOptions(
            columns=list(arrow_schema.names),
            provenance_columns=DEFAULT_ARROW_PROVENANCE_COLUMNS,
            implicit_ordering=True,
            require_sequenced_output=True,
        )
        plan = build_scan_plan(
            dataset,
            options=ScanPlanOptions(
                columns=scan_options.projection_columns(),
                filter_expr=None,
                implicit_ordering=scan_options.implicit_ordering,
                require_sequenced_output=scan_options.require_sequenced_output,
            ),
        )
        resolved_threads = (
            scan_options.use_threads if scan_options.use_threads is not None else True
        )
        execution_ctx = ExecutionContext(
            use_threads=resolved_threads,
            determinism="canonical",
            combine_chunks=True,
        )

        def _read_table() -> pa.Table:
            return reader_to_table(plan.to_reader(use_threads=resolved_threads))

        finalize_spec = FinalizeSpec(
            table_key=_MANIFEST_TABLE_KEY,
            mode="tolerant",
            context_fields=DEFAULT_ARROW_PROVENANCE_COLUMNS,
            dedupe=FinalizeDedupe(
                prefer_columns=("computed_at",),
                keys=(),
                tie_breakers=(
                    ("input_hash", "ascending"),
                    ("output_hash", "ascending"),
                    ("options_hash", "ascending"),
                ),
                tier="canonical",
                strategy="first",
            ),
        )
        result = run_pipeline(
            plan=ExecutionPlan(table_thunk=_read_table),
            finalize=finalize_spec,
            ctx=execution_ctx,
        )
        return result.good if result.good.num_rows else None

    @staticmethod
    def _manifest_match_mask(
        table: pa.Table,
        *,
        target: str,
        repo: str,
        commit: str,
    ) -> pa.Array:
        targets = table.column("target")
        repos = table.column("repo")
        commits = table.column("commit")
        try:
            target_mask = require_array(
                call_compute("equal", [targets, pa.scalar(target)]),
                name="equal",
            )
            repo_mask = require_array(
                call_compute("equal", [repos, pa.scalar(repo)]),
                name="equal",
            )
            commit_mask = require_array(
                call_compute("equal", [commits, pa.scalar(commit)]),
                name="equal",
            )
            combined = and_mask(target_mask, and_mask(repo_mask, commit_mask))
            return normalize_array(fill_null_false(combined))
        except (
            TypeError,
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            ValueError,
        ):
            pass
        mask = [
            row_target == target and row_repo == repo and row_commit == commit
            for row_target, row_repo, row_commit in zip(
                iter_array_values(targets),
                iter_array_values(repos),
                iter_array_values(commits),
                strict=True,
            )
        ]
        return pa.array(mask, type=pa.bool_())

    @staticmethod
    def _manifest_repo_commit_mask(
        table: pa.Table,
        *,
        repo: str,
        commit: str,
    ) -> pa.Array:
        repos = table.column("repo")
        commits = table.column("commit")
        try:
            repo_mask = require_array(
                call_compute("equal", [repos, pa.scalar(repo)]),
                name="equal",
            )
            commit_mask = require_array(
                call_compute("equal", [commits, pa.scalar(commit)]),
                name="equal",
            )
            combined = and_mask(repo_mask, commit_mask)
            return normalize_array(fill_null_false(combined))
        except (
            TypeError,
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            ValueError,
        ):
            pass
        mask = [
            row_repo == repo and row_commit == commit
            for row_repo, row_commit in zip(
                iter_array_values(repos),
                iter_array_values(commits),
                strict=True,
            )
        ]
        return pa.array(mask, type=pa.bool_())

    @staticmethod
    def _invert_mask(mask: pa.Array) -> pa.Array:
        try:
            return normalize_array(invert_mask(fill_null_false(mask)))
        except (
            TypeError,
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            ValueError,
        ):
            pass
        values = [not value for value in iter_array_values(mask)]
        return pa.array(values, type=pa.bool_())

    @staticmethod
    def _manifest_rows_from_table(
        table: pa.Table,
        *,
        impl_column: str,
    ) -> tuple[OutputManifest, ...]:
        columns = [
            "target",
            "repo",
            "commit",
            impl_column,
            "computed_at",
            "duration_ms",
            "input_hash",
            "output_hash",
            "row_count",
            "options_hash",
            "change_delta",
        ]
        selected = table.select(columns)
        reader = table_to_reader(selected, batch_size=None)
        return tuple(_parse_manifest_row(row) for row in iter_tuples_from_arrow_reader(reader))

    def _save_manifest_parquet(self, manifest: OutputManifest) -> None:
        table_key = _MANIFEST_TABLE_KEY
        dataset_root_dir, snapshot_id = self._manifest_dataset_context(commit=manifest.commit)
        impl_column = self._manifest_impl_column(table_key)
        payload = _manifest_row_payload(manifest, impl_column=impl_column)
        reader, _ = table_for_rows(table_key, [payload])
        new_table = reader.read_all()

        with _MANIFEST_WRITE_LOCK:
            existing_table = self._load_manifest_table(
                table_key=table_key,
                dataset_root_dir=dataset_root_dir,
                snapshot_id=snapshot_id,
                arrow_schema=new_table.schema,
            )
            if existing_table is not None:
                match_mask = self._manifest_match_mask(
                    existing_table,
                    target=manifest.target,
                    repo=manifest.repo,
                    commit=manifest.commit,
                )
                filtered = existing_table.filter(self._invert_mask(match_mask))
                if filtered.num_rows > 0:
                    new_table = pa.concat_tables([filtered, new_table], promote=True)
            try:
                table_schema = get_schema_service().get_table_schema(table_key)
            except RuntimeError:
                table_schema = None
            manifest_entry = write_dataset(
                dataset_root=dataset_root_dir,
                table_key=table_key,
                snapshot_id=snapshot_id,
                data=new_table,
                options=ArrowDatasetWriteOptions(
                    existing_data_behavior="delete_matching",
                    stable_sort_keys=resolve_stable_sort_keys(table_schema),
                ),
            )
            manifests = dict(self._gateway.datasets.dataset_manifests)
            manifests[table_key] = manifest_entry
            self._gateway.datasets = self._gateway.datasets.with_dataset_manifests(manifests)

        log.debug(
            "build.manifest.parquet.saved target=%s input_hash=%s",
            manifest.target,
            manifest.input_hash,
        )

    def _load_manifest_parquet(
        self,
        *,
        target: str,
        repo: str,
        commit: str,
    ) -> OutputManifest | None:
        table_key = _MANIFEST_TABLE_KEY
        dataset_root_dir, snapshot_id = self._manifest_dataset_context(commit=commit)
        arrow_schema = self._manifest_arrow_schema(table_key)
        table = self._load_manifest_table(
            table_key=table_key,
            dataset_root_dir=dataset_root_dir,
            snapshot_id=snapshot_id,
            arrow_schema=arrow_schema,
        )
        if table is None:
            return None
        match_mask = self._manifest_match_mask(
            table,
            target=target,
            repo=repo,
            commit=commit,
        )
        matched = table.filter(match_mask)
        if matched.num_rows == 0:
            return None
        if matched.num_rows > 1 and "computed_at" in matched.column_names:
            try:
                indices = stable_sort_indices(
                    matched,
                    sort_keys=[("computed_at", "ascending")],
                )
                matched = matched.take(indices)
            except (TypeError, pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
                matched = matched.sort_by([("computed_at", "ascending")])
            matched = matched.slice(matched.num_rows - 1, 1)
        impl_column = self._manifest_impl_column(table_key)
        manifests = self._manifest_rows_from_table(matched, impl_column=impl_column)
        return manifests[-1] if manifests else None

    def _list_manifests_parquet(
        self,
        *,
        repo: str,
        commit: str,
    ) -> tuple[OutputManifest, ...]:
        table_key = _MANIFEST_TABLE_KEY
        dataset_root_dir, snapshot_id = self._manifest_dataset_context(commit=commit)
        arrow_schema = self._manifest_arrow_schema(table_key)
        table = self._load_manifest_table(
            table_key=table_key,
            dataset_root_dir=dataset_root_dir,
            snapshot_id=snapshot_id,
            arrow_schema=arrow_schema,
        )
        if table is None:
            return ()
        filter_mask = self._manifest_repo_commit_mask(table, repo=repo, commit=commit)
        filtered = table.filter(filter_mask)
        if filtered.num_rows == 0:
            return ()
        if "target" in filtered.column_names:
            try:
                indices = stable_sort_indices(
                    filtered,
                    sort_keys=[("target", "ascending")],
                )
                filtered = filtered.take(indices)
            except (TypeError, pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
                filtered = filtered.sort_by([("target", "ascending")])
        impl_column = self._manifest_impl_column(table_key)
        return self._manifest_rows_from_table(filtered, impl_column=impl_column)

    def save_manifest(self, manifest: OutputManifest) -> None:
        """Save or update an output manifest.

        Uses upsert to insert or update the manifest record.

        Parameters
        ----------
        manifest
            The manifest to save.
        """
        if self._is_parquet_backed(_MANIFEST_TABLE_KEY):
            self._save_manifest_parquet(manifest)
            return
        change_delta = dict(manifest.change_delta) if manifest.change_delta is not None else None
        impl_column = self._impl_kind_column(_MANIFEST_TABLE_KEY)
        self._backend.upsert(
            _MANIFEST_TABLE_KEY,
            [
                (
                    manifest.target,
                    manifest.repo,
                    manifest.commit,
                    manifest.impl_kind,
                    manifest.computed_at,
                    manifest.duration_ms,
                    manifest.input_hash,
                    manifest.output_hash,
                    manifest.row_count,
                    manifest.options_hash,
                    change_delta,
                )
            ],
            columns=(
                "target",
                "repo",
                "commit",
                impl_column,
                "computed_at",
                "duration_ms",
                "input_hash",
                "output_hash",
                "row_count",
                "options_hash",
                "change_delta",
            ),
            upsert=UpsertSpec(
                conflict_columns=("target", "repo", "commit"),
                update_columns=(
                    impl_column,
                    "computed_at",
                    "duration_ms",
                    "input_hash",
                    "output_hash",
                    "row_count",
                    "options_hash",
                    "change_delta",
                ),
            ),
        )

    def load_manifest(self, target: str, repo: str, commit: str) -> OutputManifest | None:
        """Load an output manifest by primary key.

        Parameters
        ----------
        target
            Target name.
        repo
            Repository slug.
        commit
            Commit SHA.

        Returns
        -------
        OutputManifest | None
            The manifest if found, None otherwise.
        """
        if self._is_parquet_backed(_MANIFEST_TABLE_KEY):
            return self._load_manifest_parquet(target=target, repo=repo, commit=commit)
        impl_column = self._impl_kind_column(_MANIFEST_TABLE_KEY)
        where_expr = _combine_conditions(
            [
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("target")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("repo")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("commit")),
                    expression=exp.Placeholder(),
                ),
            ]
        )
        query = (
            exp.select(*_manifest_select_exprs(impl_column))
            .from_(table_expr_from_ref("build.output_manifests"))
            .where(where_expr)
        )
        result = self._con.execute(
            render_sql_duckdb(query),
            [target, repo, commit],
        ).fetchone()

        if result is None:
            return None

        return _parse_manifest_row(result)

    def list_manifests(self, repo: str, commit: str) -> tuple[OutputManifest, ...]:
        """List all manifests for a repo/commit.

        Parameters
        ----------
        repo
            Repository slug.
        commit
            Commit SHA.

        Returns
        -------
        tuple[OutputManifest, ...]
            All manifests for the given repo/commit.
        """
        if self._is_parquet_backed(_MANIFEST_TABLE_KEY):
            return self._list_manifests_parquet(repo=repo, commit=commit)
        impl_column = self._impl_kind_column(_MANIFEST_TABLE_KEY)
        where_expr = _combine_conditions(
            [
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("repo")),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("commit")),
                    expression=exp.Placeholder(),
                ),
            ]
        )
        query = (
            exp.select(*_manifest_select_exprs(impl_column))
            .from_(table_expr_from_ref("build.output_manifests"))
            .where(where_expr)
            .order_by(exp.Ordered(this=exp.Column(this=exp.to_identifier("target"))))
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            [repo, commit],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

        return tuple(_parse_manifest_row(row) for row in iter_tuples_from_arrow_reader(reader))

    def delete_manifests(self, repo: str, commit: str) -> None:
        """Delete all manifests for a repo/commit.

        Parameters
        ----------
        repo
            Repository slug.
        commit
            Commit SHA.
        """
        delete_expr = exp.Delete(
            this=table_expr_from_ref(_MANIFEST_TABLE_KEY),
            where=exp.Where(
                this=_combine_conditions(
                    [
                        exp.EQ(
                            this=exp.Column(this=exp.to_identifier("repo")),
                            expression=exp.Placeholder(),
                        ),
                        exp.EQ(
                            this=exp.Column(this=exp.to_identifier("commit")),
                            expression=exp.Placeholder(),
                        ),
                    ]
                )
            ),
        )
        self._con.execute(render_sql_duckdb(delete_expr), [repo, commit])

    def start_run(self, record: BuildRunRecord) -> None:
        """Record the start of a build run.

        Parameters
        ----------
        record
            The run record to save.
        """
        inserted = self._backend.upsert(
            "build.runs",
            [
                (
                    record.run_id,
                    record.repo,
                    record.commit,
                    serialize_str_sequence(record.requested_targets),
                    serialize_str_sequence(record.computed_targets),
                    serialize_str_sequence(record.skipped_targets),
                    record.started_at,
                    record.completed_at,
                    record.status,
                    record.error_summary,
                    record.duration_ms,
                )
            ],
            columns=(
                "run_id",
                "repo",
                "commit",
                "requested_targets",
                "computed_targets",
                "skipped_targets",
                "started_at",
                "completed_at",
                "status",
                "error_summary",
                "duration_ms",
            ),
            upsert=UpsertSpec(
                conflict_columns=("run_id",),
                update_columns=(),
            ),
        )
        if inserted == 0:
            log.warning("build.run start ignored due to duplicate run_id: %s", record.run_id)

    def complete_run(
        self,
        run_id: str,
        status: BuildStatus,
        computed_targets: tuple[str, ...],
        skipped_targets: tuple[str, ...],
        error_summary: str | None = None,
    ) -> None:
        """Update a run record upon completion.

        Parameters
        ----------
        run_id
            Run identifier.
        status
            Final status (succeeded or failed).
        computed_targets
            Targets that were computed.
        skipped_targets
            Targets that were skipped.
        error_summary
            Error summary if failed.
        """
        completed_at = utc_now()

        select_started = (
            exp.select(exp.Column(this=exp.to_identifier("started_at")))
            .from_(table_expr_from_ref("build.runs"))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
        )
        result = self._con.execute(
            render_sql_duckdb(select_started),
            [run_id],
        ).fetchone()

        duration_ms: float | None = None
        if result is not None and result[0] is not None:
            started_at: datetime = cast("datetime", result[0])
            duration_ms = (completed_at - started_at).total_seconds() * 1000

        update_expr = exp.Update(
            this=table_expr_from_ref("build.runs"),
            expressions=[
                exp.EQ(
                    this=exp.to_identifier("completed_at"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.to_identifier("status"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.to_identifier("computed_targets"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.to_identifier("skipped_targets"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.to_identifier("error_summary"),
                    expression=exp.Placeholder(),
                ),
                exp.EQ(
                    this=exp.to_identifier("duration_ms"),
                    expression=exp.Placeholder(),
                ),
            ],
            where=exp.Where(
                this=exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            ),
        )
        self._con.execute(
            render_sql_duckdb(update_expr),
            [
                completed_at,
                status,
                serialize_str_sequence(computed_targets),
                serialize_str_sequence(skipped_targets),
                error_summary,
                duration_ms,
                run_id,
            ],
        )

    def fetch_run(self, run_id: str) -> BuildRunRecord | None:
        """Fetch a run record by ID.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        BuildRunRecord | None
            The run record if found, None otherwise.
        """
        query = (
            exp.select(*_run_select_exprs())
            .from_(table_expr_from_ref("build.runs"))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
        )
        result = self._con.execute(render_sql_duckdb(query), [run_id]).fetchone()

        if result is None:
            return None

        return _parse_run_row(result)

    def list_runs(self, repo: str, limit: int = 100) -> tuple[BuildRunRecord, ...]:
        """List recent runs for a repository.

        Parameters
        ----------
        repo
            Repository slug.
        limit
            Maximum number of runs to return.

        Returns
        -------
        tuple[BuildRunRecord, ...]
            Recent runs, newest first.
        """
        query = (
            exp.select(*_run_select_exprs())
            .from_(table_expr_from_ref("build.runs"))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("repo")),
                    expression=exp.Placeholder(),
                )
            )
            .order_by(exp.Ordered(this=exp.Column(this=exp.to_identifier("started_at")), desc=True))
            .limit(exp.Placeholder())
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            [repo, limit],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

        return tuple(_parse_run_row(row) for row in iter_tuples_from_arrow_reader(reader))

    def save_run_targets(
        self,
        run_id: str,
        repo: str,
        commit: str,
        records: Sequence[TargetRunRecord],
    ) -> int:
        """Save per-target execution records for a build run.

        Parameters
        ----------
        run_id
            Parent run identifier.
        repo
            Repository slug.
        commit
            Commit SHA.
        records
            Sequence of TargetRunRecord objects from execution.

        Returns
        -------
        int
            Number of records inserted.
        """
        if not records:
            return 0

        self._backend.ensure_table("build.run_targets", create_if_missing=True)
        recorded_at = utc_now()
        impl_column = self._impl_kind_column("build.run_targets")
        available_columns = self._table_columns("build.run_targets")
        include_drift = "drift_summaries" in available_columns
        include_dep_hashes = "dep_hashes" in available_columns
        columns = [
            "run_id",
            "repo",
            "commit",
            "target",
            impl_column,
            "status",
            "input_hash",
            "options_hash",
            "duration_ms",
            "row_counts",
        ]
        if include_drift:
            columns.append("drift_summaries")
        columns.append("error")
        if include_dep_hashes:
            columns.append("dep_hashes")
        columns.append("recorded_at")
        rows: list[tuple[object, ...]] = []

        for rec in records:
            row_counts_payload = encode_payload(dict(rec.row_counts) if rec.row_counts else {})
            row_values: list[object] = [
                run_id,
                repo,
                commit,
                rec.target,
                rec.impl_kind,
                rec.status,
                rec.input_hash,
                rec.options_hash,
                rec.duration_ms,
                row_counts_payload,
            ]
            if include_drift:
                drift_summaries_payload = encode_payload(
                    {table_key: dict(summary) for table_key, summary in rec.drift_summaries.items()}
                )
                row_values.append(drift_summaries_payload)
            row_values.append(rec.error)
            if include_dep_hashes:
                row_values.append(None)
            row_values.append(recorded_at)
            rows.append(tuple(row_values))

        return self._backend.bulk_insert(
            "build.run_targets",
            rows,
            columns=tuple(columns),
        )

    def list_run_targets(self, run_id: str) -> list[dict[str, Any]]:
        """List per-target records for a specific run.

        Parameters
        ----------
        run_id
            Run identifier to fetch targets for.

        Returns
        -------
        list[dict[str, Any]]
            List of target record dictionaries.
        """
        impl_column = self._impl_kind_column("build.run_targets")
        query = (
            exp.select(
                exp.Column(this=exp.to_identifier("target")),
                exp.alias_(exp.Column(this=exp.to_identifier(impl_column)), "impl_kind"),
                exp.Column(this=exp.to_identifier("status")),
                exp.Column(this=exp.to_identifier("input_hash")),
                exp.Column(this=exp.to_identifier("options_hash")),
                exp.Column(this=exp.to_identifier("duration_ms")),
                exp.Column(this=exp.to_identifier("row_counts")),
                exp.Column(this=exp.to_identifier("drift_summaries")),
                exp.Column(this=exp.to_identifier("error")),
                exp.Column(this=exp.to_identifier("recorded_at")),
            )
            .from_(table_expr_from_ref("build.run_targets"))
            .where(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("run_id")),
                    expression=exp.Placeholder(),
                )
            )
            .order_by(exp.Ordered(this=exp.Column(this=exp.to_identifier("target"))))
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            [run_id],
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

        return [
            {
                "target": row[0],
                "impl_kind": row[1],
                "status": row[2],
                "input_hash": row[3],
                "options_hash": row[4],
                "duration_ms": row[5],
                "row_counts": decode_json_dict(row[6]) if row[6] else {},
                "drift_summaries": decode_json_dict(row[7]) if row[7] else {},
                "error": row[8],
                "recorded_at": row[9],
            }
            for row in iter_tuples_from_arrow_reader(reader)
        ]

    def save_run_nodes(
        self,
        run_id: str,
        records: Sequence[NodeExecutionRecord],
    ) -> int:
        """Save node-level execution records for a build run.

        Parameters
        ----------
        run_id
            Parent run identifier.
        records
            Sequence of NodeExecutionRecord objects.

        Returns
        -------
        int
            Number of records inserted.
        """
        if not records:
            return 0

        deduped_records = self._dedupe_run_nodes(records)
        rows = [
            (
                run_id,
                r.node_name,
                r.target,
                r.node_type,
                r.status,
                r.started_at,
                r.completed_at,
                r.duration_ms,
                r.error,
                encode_payload(r.tags or {}),
            )
            for r in deduped_records
        ]

        return self._backend.upsert(
            "build.run_nodes",
            rows,
            columns=(
                "run_id",
                "node_name",
                "target",
                "node_type",
                "status",
                "started_at",
                "completed_at",
                "duration_ms",
                "error",
                "tags",
            ),
            upsert=UpsertSpec(
                conflict_columns=("run_id", "node_name"),
                update_columns=(
                    "target",
                    "node_type",
                    "status",
                    "started_at",
                    "completed_at",
                    "duration_ms",
                    "error",
                    "tags",
                ),
            ),
        )

    @staticmethod
    def _pick_node_record(
        existing: NodeExecutionRecord,
        candidate: NodeExecutionRecord,
    ) -> NodeExecutionRecord:
        if existing.status != candidate.status:
            if candidate.status == "failed":
                return candidate
            if existing.status == "failed":
                return existing
        existing_completed = existing.completed_at or existing.started_at
        candidate_completed = candidate.completed_at or candidate.started_at
        if candidate_completed >= existing_completed:
            return candidate
        return existing

    @staticmethod
    def _dedupe_run_nodes(
        records: Sequence[NodeExecutionRecord],
    ) -> list[NodeExecutionRecord]:
        deduped: dict[str, NodeExecutionRecord] = {}
        for record in records:
            current = deduped.get(record.node_name)
            if current is None:
                deduped[record.node_name] = record
                continue
            deduped[record.node_name] = BuildTracking._pick_node_record(current, record)
        return [deduped[name] for name in sorted(deduped)]

    def record_scip_run(self, record: ScipRunRecordProtocol) -> None:
        """Upsert a SCIP telemetry record into build.scip_runs."""
        self._backend.ensure_table("build.scip_runs", create_if_missing=True)
        self._backend.upsert(
            "build.scip_runs",
            [
                (
                    record.run_id,
                    record.repo,
                    record.commit,
                    record.mode,
                    record.options_hash,
                    record.project_version,
                    record.project_namespace,
                    record.tool_version,
                    record.total_modules,
                    record.changed_modules,
                    record.deleted_modules,
                    record.changed_ratio,
                    record.batch_size,
                    record.batch_count,
                    record.decision,
                    record.ratio_gate_applied,
                    record.ratio_gate_min_modules,
                    record.ratio_gate_min_changed,
                    record.hash_source,
                    record.hash_source_breakdown,
                    record.hash_reused,
                    record.hash_computed,
                    record.plan_ms,
                    record.hash_ms,
                    record.tool_ms,
                    record.parse_ms,
                    record.merge_ms,
                    record.write_ms,
                    record.total_ms,
                    record.status,
                    record.error_summary,
                    record.output_scip,
                    record.recorded_at,
                )
            ],
            columns=(
                "run_id",
                "repo",
                "commit",
                "mode",
                "options_hash",
                "project_version",
                "project_namespace",
                "tool_version",
                "total_modules",
                "changed_modules",
                "deleted_modules",
                "changed_ratio",
                "batch_size",
                "batch_count",
                "decision",
                "ratio_gate_applied",
                "ratio_gate_min_modules",
                "ratio_gate_min_changed",
                "hash_source",
                "hash_source_breakdown",
                "hash_reused",
                "hash_computed",
                "plan_ms",
                "hash_ms",
                "tool_ms",
                "parse_ms",
                "merge_ms",
                "write_ms",
                "total_ms",
                "status",
                "error_summary",
                "output_scip",
                "recorded_at",
            ),
            upsert=UpsertSpec(
                conflict_columns=("run_id",),
                update_columns=(
                    "mode",
                    "options_hash",
                    "project_version",
                    "project_namespace",
                    "tool_version",
                    "total_modules",
                    "changed_modules",
                    "deleted_modules",
                    "changed_ratio",
                    "batch_size",
                    "batch_count",
                    "decision",
                    "ratio_gate_applied",
                    "ratio_gate_min_modules",
                    "ratio_gate_min_changed",
                    "hash_source",
                    "hash_source_breakdown",
                    "hash_reused",
                    "hash_computed",
                    "plan_ms",
                    "hash_ms",
                    "tool_ms",
                    "parse_ms",
                    "merge_ms",
                    "write_ms",
                    "total_ms",
                    "status",
                    "error_summary",
                    "output_scip",
                    "recorded_at",
                ),
            ),
        )

    def list_run_nodes(
        self,
        run_id: str,
        *,
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        """List node records for a specific run.

        Parameters
        ----------
        run_id
            Run identifier to fetch nodes for.
        target
            Optional target filter.

        Returns
        -------
        list[dict[str, Any]]
            List of node record dictionaries.
        """
        conditions: list[exp.Expression] = [
            exp.EQ(
                this=exp.Column(this=exp.to_identifier("run_id")),
                expression=exp.Placeholder(),
            )
        ]
        params: list[Any] = [run_id]
        if target:
            conditions.append(
                exp.EQ(
                    this=exp.Column(this=exp.to_identifier("target")),
                    expression=exp.Placeholder(),
                )
            )
            params.append(target)
        query = (
            exp.select(
                exp.Column(this=exp.to_identifier("node_name")),
                exp.Column(this=exp.to_identifier("target")),
                exp.Column(this=exp.to_identifier("node_type")),
                exp.Column(this=exp.to_identifier("status")),
                exp.Column(this=exp.to_identifier("started_at")),
                exp.Column(this=exp.to_identifier("completed_at")),
                exp.Column(this=exp.to_identifier("duration_ms")),
                exp.Column(this=exp.to_identifier("error")),
                exp.Column(this=exp.to_identifier("tags")),
            )
            .from_(table_expr_from_ref("build.run_nodes"))
            .where(_combine_conditions(conditions))
            .order_by(exp.Ordered(this=exp.Column(this=exp.to_identifier("started_at"))))
        )
        reader = self._con.execute(
            render_sql_duckdb(query),
            params,
        ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

        return [
            {
                "node_name": row[0],
                "target": row[1],
                "node_type": row[2],
                "status": row[3],
                "started_at": row[4],
                "completed_at": row[5],
                "duration_ms": row[6],
                "error": row[7],
                "tags": decode_json_dict(row[8]) if row[8] else {},
            }
            for row in iter_tuples_from_arrow_reader(reader)
        ]


__all__ = [
    "BuildTracking",
    "ScipRunRecord",
]
