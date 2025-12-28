"""File profile recipe helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.profiles.types import FileProfileInputs
from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
)
from codeintel.analytics.profiles.writer_guard import (
    PolicyWriterConfig,
    write_rows_via_policy_backend,
)
from codeintel.analytics.utilities.type_coercion import (
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsFileProfileRow as FileProfileRowModel,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.query_results import records_from_relation
from codeintel.storage.snapshot_scoping import maybe_scope_by_repo_commit

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.duckdb_types import DuckDBRelation
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def compute_file_profile_inputs(
    gateway: StorageGateway, snapshot: SnapshotRef
) -> FileProfileInputs:
    """
    Construct snapshot inputs for file profile generation.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    FileProfileInputs
        Snapshot handle for file profile helpers.
    """
    return FileProfileInputs(
        gateway=gateway,
        con=gateway.con,
        repo=snapshot.repo,
        commit=snapshot.commit,
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=0.0,
    )


def _load_file_profile_tables(
    inputs: FileProfileInputs, module_table: str
) -> tuple[
    DuckDBRelation,
    DuckDBRelation,
    DuckDBRelation,
    DuckDBRelation,
    DuckDBRelation,
    DuckDBRelation,
] | None:
    """
    Load filtered profile source tables.

    Returns
    -------
    tuple[DuckDBRelation, ...] | None
        Filtered source tables or None on access failure.
    """
    gw = inputs.gateway
    try:
        fp = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.function_profile"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        ast_metrics = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("core.ast_metrics"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        hotspots = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.hotspots"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        typedness = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.typedness"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        static_diag = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.static_diagnostics"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        modules = maybe_scope_by_repo_commit(
            gw.relation_from_table_key(module_table),
            repo=inputs.repo,
            commit=inputs.commit,
        )
    except DuckDBError as exc:
        log.warning("file_profile: failed to access tables: %s", exc)
        return None
    else:
        return fp, ast_metrics, hotspots, typedness, static_diag, modules


def build_file_profile_rows(
    inputs: FileProfileInputs,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> Iterable[FileProfileRowModel]:
    """
    Compute file_profile rows by aggregating function_profile data.

    Yields
    ------
    FileProfileRowModel
        Row models ready for insertion into ``analytics.file_profile``.

    Raises
    ------
    ValueError
        If an unexpected module table name is provided.
    """
    if module_table not in {DEFAULT_MODULE_TABLE, CATALOG_MODULE_TABLE}:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    tables = _load_file_profile_tables(inputs, module_table)
    if tables is None:
        return

    fp, ast_metrics, hotspots, typedness, static_diag, modules = tables

    fm = fp.group_by("repo", "commit", "rel_path").aggregate(
        [
            "count(rel_path) as total_functions",
            "sum(case when call_is_public then 1 else 0 end) as public_functions",
            "avg(loc) as avg_loc",
            "max(loc) as max_loc",
            "avg(cyclomatic_complexity) as avg_cyclomatic_complexity",
            "max(cyclomatic_complexity) as max_cyclomatic_complexity",
            "sum(case when risk_level = 'high' then 1 else 0 end) as high_risk_function_count",
            "sum(case when risk_level = 'medium' then 1 else 0 end) as medium_risk_function_count",
            "max(risk_score) as max_risk_score",
            "sum(covered_lines) as sum_covered_lines",
            "sum(executable_lines) as sum_exec_lines",
            "sum(case when tested then 1 else 0 end) as tested_function_count",
            "sum(case when tested then 0 else 1 end) as untested_function_count",
            "sum(tests_touching) as tests_touching",
        ]
    )
    typedness_rel = typedness.select(
        "repo",
        "commit",
        "path as rel_path",
        "annotation_ratio",
        "untyped_defs",
        "overlay_needed",
        "type_error_count",
    )
    modules_rel = modules.select(
        "repo",
        "commit",
        "path as rel_path",
        "module",
        "language",
        "tags",
        "owners",
    )

    joined = (
        fm.join(ast_metrics, ["rel_path"], how="left")
        .join(hotspots, ["rel_path"], how="left")
        .join(typedness_rel, ["repo", "commit", "rel_path"], how="left")
        .join(static_diag, ["repo", "commit", "rel_path"], how="left")
        .join(modules_rel, ["repo", "commit", "rel_path"], how="left")
    )

    try:
        selected = joined.select(
            "repo",
            "commit",
            "rel_path",
            "module",
            "language",
            "node_count",
            "function_count",
            "class_count",
            "avg_depth",
            "max_depth",
            "complexity as ast_complexity",
            "score as hotspot_score",
            "commit_count",
            "author_count",
            "lines_added",
            "lines_deleted",
            "cast(json_extract(annotation_ratio, '$.params') as double) as annotation_ratio",
            "untyped_defs",
            "overlay_needed",
            "type_error_count",
            "total_errors as static_error_count",
            "has_errors as has_static_errors",
            "total_functions",
            "public_functions",
            "avg_loc",
            "max_loc",
            "avg_cyclomatic_complexity",
            "max_cyclomatic_complexity",
            "high_risk_function_count",
            "medium_risk_function_count",
            "max_risk_score",
            "cast(sum_covered_lines as double) / nullif(sum_exec_lines, 0) as file_coverage_ratio",
            "tested_function_count",
            "untested_function_count",
            "tests_touching",
            "tags",
            "owners",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("file_profile: failed to execute aggregation: %s", exc)
        return

    for record in records:
        record["created_at"] = inputs.created_at
        yield _row_to_file_profile_model(record, inputs)


def _row_to_file_profile_model(
    record: dict[str, object], inputs: FileProfileInputs
) -> FileProfileRowModel:
    """
    Convert a DuckDB row mapping into a FileProfileRowModel.

    Returns
    -------
    FileProfileRowModel
        Row model derived from the provided record.
    """
    return FileProfileRowModel(
        repo=str(record["repo"]),
        commit=str(record["commit"]),
        rel_path=str(record["rel_path"]),
        module=optional_str(record["module"]),
        language=optional_str(record["language"]),
        node_count=optional_int(record["node_count"]),
        function_count=optional_int(record["function_count"]),
        class_count=optional_int(record["class_count"]),
        avg_depth=optional_float(record["avg_depth"]),
        max_depth=optional_int(record["max_depth"]),
        ast_complexity=optional_float(record["ast_complexity"]),
        hotspot_score=optional_float(record["hotspot_score"]),
        commit_count=optional_int(record["commit_count"]),
        author_count=optional_int(record["author_count"]),
        lines_added=optional_int(record["lines_added"]),
        lines_deleted=optional_int(record["lines_deleted"]),
        annotation_ratio=optional_float(record["annotation_ratio"]),
        untyped_defs=optional_int(record["untyped_defs"]),
        overlay_needed=bool(record["overlay_needed"])
        if record["overlay_needed"] is not None
        else None,
        type_error_count=optional_int(record["type_error_count"]),
        static_error_count=optional_int(record["static_error_count"]),
        has_static_errors=(
            bool(record["has_static_errors"]) if record["has_static_errors"] is not None else None
        ),
        total_functions=optional_int(record["total_functions"]),
        public_functions=optional_int(record["public_functions"]),
        avg_loc=optional_float(record["avg_loc"]),
        max_loc=optional_int(record["max_loc"]),
        avg_cyclomatic_complexity=optional_float(record["avg_cyclomatic_complexity"]),
        max_cyclomatic_complexity=optional_int(record["max_cyclomatic_complexity"]),
        high_risk_function_count=optional_int(record["high_risk_function_count"]),
        medium_risk_function_count=optional_int(record["medium_risk_function_count"]),
        max_risk_score=optional_float(record["max_risk_score"]),
        file_coverage_ratio=optional_float(record["file_coverage_ratio"]),
        tested_function_count=optional_int(record["tested_function_count"]),
        untested_function_count=optional_int(record["untested_function_count"]),
        tests_touching=optional_int(record["tests_touching"]),
        tags=record["tags"] if record["tags"] is not None else "[]",
        owners=record["owners"] if record["owners"] is not None else "[]",
        created_at=(
            record["created_at"]
            if isinstance(record["created_at"], datetime)
            else inputs.created_at
        ),
    )


def build_file_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """Compute and persist analytics.file_profile rows.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.
    module_table
        Name of the module table to use.

    Returns
    -------
    int
        Number of rows inserted.
    """
    inputs = compute_file_profile_inputs(gateway, snapshot)
    rows = list(build_file_profile_rows(inputs, module_table=module_table))
    if not rows:
        return 0

    config = PolicyWriterConfig(
        table_key="analytics.file_profile",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return write_rows_via_policy_backend(gateway, rows=rows, config=config)
