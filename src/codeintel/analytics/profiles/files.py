"""File profile recipe helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from codeintel.analytics.profiles.types import FileProfileInputs
from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.config import ProfilesAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import FileProfileRowModel, file_profile_row_to_tuple
from codeintel.storage.sql_helpers import ensure_schema


def compute_file_profile_inputs(
    gateway: StorageGateway, cfg: ProfilesAnalyticsStepConfig
) -> FileProfileInputs:
    """
    Construct snapshot inputs for file profile generation.

    Returns
    -------
    FileProfileInputs
        Snapshot handle for file profile helpers.
    """
    return FileProfileInputs(
        con=gateway.con,
        repo=cfg.repo,
        commit=cfg.commit,
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=0.0,
    )


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
    con = inputs.con
    sql_core = """
        WITH fm AS (
            SELECT
                repo,
                commit,
                rel_path,
                COUNT(*) AS total_functions,
                COUNT(*) FILTER (WHERE call_is_public) AS public_functions,
                AVG(loc) AS avg_loc,
                MAX(loc) AS max_loc,
                AVG(cyclomatic_complexity) AS avg_cyclomatic_complexity,
                MAX(cyclomatic_complexity) AS max_cyclomatic_complexity,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END)
                    AS high_risk_function_count,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END)
                    AS medium_risk_function_count,
                MAX(risk_score) AS max_risk_score,
                SUM(covered_lines) AS sum_covered_lines,
                SUM(executable_lines) AS sum_exec_lines,
                SUM(CASE WHEN tested THEN 1 ELSE 0 END) AS tested_function_count,
                SUM(CASE WHEN NOT tested THEN 1 ELSE 0 END) AS untested_function_count,
                SUM(tests_touching) AS tests_touching
            FROM analytics.function_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, rel_path
        ),
        ast AS (
            SELECT * FROM core.ast_metrics
        ),
        hs AS (
            SELECT * FROM analytics.hotspots
        ),
        ty AS (
            SELECT * FROM analytics.typedness
        ),
        sd AS (
            SELECT
                rel_path,
                total_errors AS static_error_count,
                has_errors AS has_static_errors
            FROM analytics.static_diagnostics
        ),
        mod AS (
            SELECT repo, commit, path, module, language, tags, owners
            FROM core.modules
        )
        SELECT
            fm.repo,
            fm.commit,
            fm.rel_path,
            mod.module,
            mod.language,
            ast.node_count,
            ast.function_count,
            ast.class_count,
            ast.avg_depth,
            ast.max_depth,
            ast.complexity AS ast_complexity,
            hs.score AS hotspot_score,
            hs.commit_count,
            hs.author_count,
            hs.lines_added,
            hs.lines_deleted,
            CAST(ty.annotation_ratio->>'params' AS DOUBLE) AS annotation_ratio,
            ty.untyped_defs,
            ty.overlay_needed,
            ty.type_error_count,
            sd.static_error_count,
            sd.has_static_errors,
            fm.total_functions,
            fm.public_functions,
            fm.avg_loc,
            fm.max_loc,
            fm.avg_cyclomatic_complexity,
            fm.max_cyclomatic_complexity,
            fm.high_risk_function_count,
            fm.medium_risk_function_count,
            fm.max_risk_score,
            CASE
                WHEN fm.sum_exec_lines > 0 THEN fm.sum_covered_lines * 1.0 / fm.sum_exec_lines
                ELSE NULL
            END AS file_coverage_ratio,
            fm.tested_function_count,
            fm.untested_function_count,
            fm.tests_touching,
            mod.tags,
            mod.owners,
            ?
        FROM fm
        LEFT JOIN ast
          ON fm.rel_path = ast.rel_path
        LEFT JOIN hs
          ON fm.rel_path = hs.rel_path
        LEFT JOIN ty
          ON fm.rel_path = ty.path
        LEFT JOIN sd
          ON fm.rel_path = sd.rel_path
        LEFT JOIN mod
          ON fm.repo = mod.repo
         AND fm.commit = mod.commit
         AND fm.rel_path = mod.path;
        """
    sql_catalog = sql_core.replace("core.modules", CATALOG_MODULE_TABLE)
    if module_table == DEFAULT_MODULE_TABLE:
        sql = sql_core
    elif module_table == CATALOG_MODULE_TABLE:
        sql = sql_catalog
    else:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    rows = con.execute(sql, [inputs.repo, inputs.commit, inputs.created_at]).fetchall()
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

    for row in rows:
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


def write_file_profile_rows(gateway: StorageGateway, rows: Iterable[FileProfileRowModel]) -> int:
    """
    Insert rows into analytics.file_profile.

    Returns
    -------
    int
        Number of rows inserted.
    """
    rows = list(rows)
    if not rows:
        return 0
    repo = rows[0]["repo"]
    commit = rows[0]["commit"]
    con = gateway.con
    ensure_schema(con, "analytics.file_profile")
    con.execute(
        "DELETE FROM analytics.file_profile WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    tuples = [file_profile_row_to_tuple(row) for row in rows]
    con.executemany(
        """
        INSERT INTO analytics.file_profile (
            repo,
            commit,
            rel_path,
            module,
            language,
            node_count,
            function_count,
            class_count,
            avg_depth,
            max_depth,
            ast_complexity,
            hotspot_score,
            commit_count,
            author_count,
            lines_added,
            lines_deleted,
            annotation_ratio,
            untyped_defs,
            overlay_needed,
            type_error_count,
            static_error_count,
            has_static_errors,
            total_functions,
            public_functions,
            avg_loc,
            max_loc,
            avg_cyclomatic_complexity,
            max_cyclomatic_complexity,
            high_risk_function_count,
            medium_risk_function_count,
            max_risk_score,
            file_coverage_ratio,
            tested_function_count,
            untested_function_count,
            tests_touching,
            tags,
            owners,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        tuples,
    )
    return len(tuples)


def build_file_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """
    Compute and persist analytics.file_profile rows.

    Returns
    -------
    int
        Number of rows inserted.
    """
    inputs = compute_file_profile_inputs(gateway, cfg)
    rows = build_file_profile_rows(inputs, module_table=module_table)
    return write_file_profile_rows(gateway, rows)
