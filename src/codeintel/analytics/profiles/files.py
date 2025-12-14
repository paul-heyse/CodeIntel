"""File profile recipe helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import ibis

from codeintel.analytics.profiles.types import FileProfileInputs
from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
)
from codeintel.analytics.profiles.writer_guard import create_profile_writer
from codeintel.analytics.utilities.type_coercion import (
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.config.datasets import (
    FILE_PROFILE_COLUMNS,
    FileProfileRowModel,
    file_profile_row_to_tuple,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import (
    bool_not,
    col_count,
    col_max,
    col_mean,
    col_sum,
    filter_by,
    ibis_bool,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.config.primitives import SnapshotRef
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
) -> tuple[ibis.Table, ibis.Table, ibis.Table, ibis.Table, ibis.Table, ibis.Table] | None:
    """
    Load filtered profile source tables.

    Returns
    -------
    tuple[ibis.Table, ...] | None
        Filtered source tables or None on access failure.
    """
    gw = inputs.gateway
    try:
        fp_table = gw.ibis.table("analytics.function_profile")
        fp = filter_by(fp_table, fp_table.repo == inputs.repo, fp_table.commit == inputs.commit)
        ast_table = gw.ibis.table("core.ast_metrics")
        ast_metrics = ast_table
        hotspots_table = gw.ibis.table("analytics.hotspots")
        hotspots = hotspots_table
        typedness_table = gw.ibis.table("analytics.typedness")
        typedness = filter_by(
            typedness_table,
            typedness_table.repo == inputs.repo,
            typedness_table.commit == inputs.commit,
        )
        static_diag_table = gw.ibis.table("analytics.static_diagnostics")
        static_diag = filter_by(
            static_diag_table,
            static_diag_table.repo == inputs.repo,
            static_diag_table.commit == inputs.commit,
        )
        module_table_expr = gw.ibis.table(module_table)
        modules = filter_by(
            module_table_expr,
            module_table_expr.repo == inputs.repo,
            module_table_expr.commit == inputs.commit,
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

    fm = fp.group_by(fp.repo, fp.commit, fp.rel_path).aggregate(
        total_functions=col_count(fp.rel_path),
        public_functions=col_sum(fp.call_is_public),
        avg_loc=col_mean(fp.loc),
        max_loc=col_max(fp.loc),
        avg_cyclomatic_complexity=col_mean(fp.cyclomatic_complexity),
        max_cyclomatic_complexity=col_max(fp.cyclomatic_complexity),
        high_risk_function_count=col_sum(ibis_bool(fp.risk_level == "high")),
        medium_risk_function_count=col_sum(ibis_bool(fp.risk_level == "medium")),
        max_risk_score=col_max(fp.risk_score),
        sum_covered_lines=col_sum(fp.covered_lines),
        sum_exec_lines=col_sum(fp.executable_lines),
        tested_function_count=col_sum(fp.tested),
        untested_function_count=col_sum(bool_not(fp.tested)),
        tests_touching=col_sum(fp.tests_touching),
    )

    joined = (
        fm.left_join(ast_metrics, [fm.rel_path == ast_metrics.rel_path])
        .left_join(hotspots, [fm.rel_path == hotspots.rel_path])
        .left_join(typedness, [fm.rel_path == typedness.path])
        .left_join(static_diag, [fm.rel_path == static_diag.rel_path])
        .left_join(
            modules,
            [
                (fm.repo == modules.repo)
                & (fm.commit == modules.commit)
                & (fm.rel_path == modules.path)
            ],
        )
    )

    try:
        df = joined.select(
            fm.repo,
            fm.commit,
            fm.rel_path,
            modules.module,
            modules.language,
            ast_metrics.node_count,
            ast_metrics.function_count,
            ast_metrics.class_count,
            ast_metrics.avg_depth,
            ast_metrics.max_depth,
            ast_metrics.complexity.name("ast_complexity"),
            hotspots.score.name("hotspot_score"),
            hotspots.commit_count,
            hotspots.author_count,
            hotspots.lines_added,
            hotspots.lines_deleted,
            typedness.annotation_ratio["params"].cast("float64").name("annotation_ratio"),
            typedness.untyped_defs,
            typedness.overlay_needed,
            typedness.type_error_count,
            static_diag.total_errors.name("static_error_count"),
            static_diag.has_errors.name("has_static_errors"),
            fm.total_functions,
            fm.public_functions,
            fm.avg_loc,
            fm.max_loc,
            fm.avg_cyclomatic_complexity,
            fm.max_cyclomatic_complexity,
            fm.high_risk_function_count,
            fm.medium_risk_function_count,
            fm.max_risk_score,
            (fm.sum_covered_lines.cast("float64") / fm.sum_exec_lines.nullif(ibis.literal(0))).name(
                "file_coverage_ratio"
            ),
            fm.tested_function_count,
            fm.untested_function_count,
            fm.tests_touching,
            modules.tags,
            modules.owners,
            ibis.literal(inputs.created_at).name("created_at"),
        ).execute()
    except DuckDBError as exc:
        log.warning("file_profile: failed to execute aggregation: %s", exc)
        return
    columns = [
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
        "ast_complexity",
        "hotspot_score",
        "commit_count",
        "author_count",
        "lines_added",
        "lines_deleted",
        "annotation_ratio",
        "untyped_defs",
        "overlay_needed",
        "type_error_count",
        "static_error_count",
        "has_static_errors",
        "total_functions",
        "public_functions",
        "avg_loc",
        "max_loc",
        "avg_cyclomatic_complexity",
        "max_cyclomatic_complexity",
        "high_risk_function_count",
        "medium_risk_function_count",
        "max_risk_score",
        "file_coverage_ratio",
        "tested_function_count",
        "untested_function_count",
        "tests_touching",
        "tags",
        "owners",
        "created_at",
    ]

    for row in df.itertuples(index=False, name=None):
        record = dict(zip(columns, row, strict=False))
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


# Factory-created writer for file profiles
write_file_profile_rows: Callable[[StorageGateway, Iterable[FileProfileRowModel]], int] = (
    create_profile_writer(
        "analytics.file_profile",
        FILE_PROFILE_COLUMNS,
        cast("Callable[[Mapping[str, object]], tuple[object, ...]]", file_profile_row_to_tuple),
    )
)


def build_file_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """
    Compute and persist analytics.file_profile rows.

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
    rows = build_file_profile_rows(inputs, module_table=module_table)
    return write_file_profile_rows(gateway, rows)
