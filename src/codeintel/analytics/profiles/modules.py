"""Module profile recipe helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from codeintel.analytics.profiles.types import ModuleProfileInputs
from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.config import ProfilesAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import ModuleProfileRowModel, module_profile_row_to_tuple
from codeintel.storage.sql_helpers import ensure_schema


def compute_module_profile_inputs(
    gateway: StorageGateway, cfg: ProfilesAnalyticsStepConfig
) -> ModuleProfileInputs:
    """
    Construct snapshot inputs for module profile generation.

    Returns
    -------
    ModuleProfileInputs
        Snapshot handle for module profile helpers.
    """
    return ModuleProfileInputs(
        con=gateway.con,
        repo=cfg.repo,
        commit=cfg.commit,
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=0.0,
    )


def build_module_profile_rows(
    inputs: ModuleProfileInputs,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> Iterable[ModuleProfileRowModel]:
    """
    Compute module_profile rows by aggregating file and function profiles.

    Yields
    ------
    ModuleProfileRowModel
        Row models ready for insertion into ``analytics.module_profile``.

    Raises
    ------
    ValueError
        If an unexpected module table name is provided.
    """
    con = inputs.con
    sql_core = """
        WITH func_stats AS (
            SELECT
                repo,
                commit,
                module,
                COUNT(*) AS function_count,
                SUM(COALESCE(loc, 0)) AS total_loc,
                SUM(COALESCE(logical_loc, 0)) AS total_logical_loc,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_function_count,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END)
                    AS medium_risk_function_count,
                SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END) AS low_risk_function_count,
                MAX(risk_score) AS max_risk_score,
                AVG(risk_score) AS avg_risk_score,
                SUM(CASE WHEN tested THEN 1 ELSE 0 END) AS tested_function_count,
                SUM(CASE WHEN NOT tested THEN 1 ELSE 0 END) AS untested_function_count
            FROM analytics.function_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, module
        ),
        files AS (
            SELECT
                repo,
                commit,
                module,
                COUNT(*) AS file_count,
                SUM(class_count) AS class_count,
                AVG(ast_complexity) AS avg_file_complexity,
                MAX(ast_complexity) AS max_file_complexity
            FROM analytics.file_profile
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, module
        ),
        mod AS (
            SELECT repo, commit, module, path, language, tags, owners
            FROM core.modules
        ),
        imports AS (
            SELECT
                repo,
                commit,
                src_module AS module,
                MAX(src_fan_out) AS import_fan_out,
                MAX(dst_fan_in) FILTER (WHERE dst_module = src_module) AS import_fan_in,
                MAX(cycle_group) AS cycle_group,
                MAX(CASE WHEN cycle_group IS NOT NULL THEN 1 ELSE 0 END) AS in_cycle_flag
            FROM graph.import_graph_edges
            WHERE repo = ? AND commit = ?
            GROUP BY repo, commit, src_module
        ),
        roles AS (
            SELECT repo, commit, module, role, role_confidence, role_sources_json
            FROM analytics.semantic_roles_modules
            WHERE repo = ? AND commit = ?
        )
        SELECT
            mod.repo,
            mod.commit,
            mod.module,
            mod.path,
            mod.language,
            COALESCE(files.file_count, 0),
            COALESCE(func_stats.total_loc, 0),
            COALESCE(func_stats.total_logical_loc, 0),
            COALESCE(func_stats.function_count, 0),
            COALESCE(files.class_count, 0),
            files.avg_file_complexity,
            files.max_file_complexity,
            COALESCE(func_stats.high_risk_function_count, 0),
            COALESCE(func_stats.medium_risk_function_count, 0),
            COALESCE(func_stats.low_risk_function_count, 0),
            func_stats.max_risk_score,
            func_stats.avg_risk_score,
            CASE
                WHEN COALESCE(func_stats.tested_function_count, 0)
                     + COALESCE(func_stats.untested_function_count, 0) > 0
                THEN
                    CAST(func_stats.tested_function_count AS DOUBLE)
                    / (func_stats.tested_function_count + func_stats.untested_function_count)
                ELSE NULL
            END AS module_coverage_ratio,
            func_stats.tested_function_count,
            func_stats.untested_function_count,
            COALESCE(imports.import_fan_in, 0),
            COALESCE(imports.import_fan_out, 0),
            imports.cycle_group,
            imports.in_cycle_flag > 0 AS in_cycle,
            roles.role,
            roles.role_confidence,
            roles.role_sources_json,
            mod.tags,
            mod.owners,
            ?
        FROM mod
        LEFT JOIN func_stats
          ON func_stats.module = mod.module
         AND func_stats.repo = mod.repo
         AND func_stats.commit = mod.commit
        LEFT JOIN files
          ON files.module = mod.module
         AND files.repo = mod.repo
         AND files.commit = mod.commit
        LEFT JOIN imports
          ON imports.module = mod.module
         AND imports.repo = mod.repo
         AND imports.commit = mod.commit
        LEFT JOIN roles
          ON roles.module = mod.module
         AND roles.repo = mod.repo
         AND roles.commit = mod.commit
        WHERE mod.repo = ?
          AND mod.commit = ?;
        """
    sql_catalog = sql_core.replace("core.modules", CATALOG_MODULE_TABLE)
    if module_table == DEFAULT_MODULE_TABLE:
        sql = sql_core
    elif module_table == CATALOG_MODULE_TABLE:
        sql = sql_catalog
    else:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    rows = con.execute(
        sql,
        [
            inputs.repo,
            inputs.commit,
            inputs.repo,
            inputs.commit,
            inputs.repo,
            inputs.commit,
            inputs.repo,
            inputs.commit,
            inputs.created_at,
            inputs.repo,
            inputs.commit,
        ],
    ).fetchall()

    columns = [
        "repo",
        "commit",
        "module",
        "path",
        "language",
        "file_count",
        "total_loc",
        "total_logical_loc",
        "function_count",
        "class_count",
        "avg_file_complexity",
        "max_file_complexity",
        "high_risk_function_count",
        "medium_risk_function_count",
        "low_risk_function_count",
        "max_risk_score",
        "avg_risk_score",
        "module_coverage_ratio",
        "tested_function_count",
        "untested_function_count",
        "import_fan_in",
        "import_fan_out",
        "cycle_group",
        "in_cycle",
        "role",
        "role_confidence",
        "role_sources_json",
        "tags",
        "owners",
        "created_at",
    ]

    for row in rows:
        record = dict(zip(columns, row, strict=False))
        yield _row_to_module_profile_model(record, inputs)


def _row_to_module_profile_model(
    record: dict[str, object], inputs: ModuleProfileInputs
) -> ModuleProfileRowModel:
    """
    Convert a DuckDB row mapping into a ModuleProfileRowModel.

    Returns
    -------
    ModuleProfileRowModel
        Row model derived from the provided record.
    """
    return ModuleProfileRowModel(
        repo=str(record["repo"]),
        commit=str(record["commit"]),
        module=str(record["module"]),
        path=optional_str(record["path"]),
        language=optional_str(record["language"]),
        file_count=optional_int(record["file_count"]),
        total_loc=optional_int(record["total_loc"]),
        total_logical_loc=optional_int(record["total_logical_loc"]),
        function_count=optional_int(record["function_count"]),
        class_count=optional_int(record["class_count"]),
        avg_file_complexity=optional_float(record["avg_file_complexity"]),
        max_file_complexity=optional_float(record["max_file_complexity"]),
        high_risk_function_count=optional_int(record["high_risk_function_count"]),
        medium_risk_function_count=optional_int(record["medium_risk_function_count"]),
        low_risk_function_count=optional_int(record["low_risk_function_count"]),
        max_risk_score=optional_float(record["max_risk_score"]),
        avg_risk_score=optional_float(record["avg_risk_score"]),
        module_coverage_ratio=optional_float(record["module_coverage_ratio"]),
        tested_function_count=optional_int(record["tested_function_count"]),
        untested_function_count=optional_int(record["untested_function_count"]),
        import_fan_in=optional_int(record["import_fan_in"]),
        import_fan_out=optional_int(record["import_fan_out"]),
        cycle_group=optional_int(record["cycle_group"]),
        in_cycle=bool(record["in_cycle"]) if record["in_cycle"] is not None else None,
        role=optional_str(record["role"]),
        role_confidence=optional_float(record["role_confidence"]),
        role_sources_json=record["role_sources_json"]
        if record["role_sources_json"] is not None
        else "[]",
        tags=record["tags"] if record["tags"] is not None else "[]",
        owners=record["owners"] if record["owners"] is not None else "[]",
        created_at=(
            record["created_at"]
            if isinstance(record["created_at"], datetime)
            else inputs.created_at
        ),
    )


def write_module_profile_rows(
    gateway: StorageGateway, rows: Iterable[ModuleProfileRowModel]
) -> int:
    """
    Insert rows into analytics.module_profile.

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
    ensure_schema(con, "analytics.module_profile")
    con.execute(
        "DELETE FROM analytics.module_profile WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    tuples = [module_profile_row_to_tuple(row) for row in rows]
    con.executemany(
        """
        INSERT INTO analytics.module_profile (
            repo,
            commit,
            module,
            path,
            language,
            file_count,
            total_loc,
            total_logical_loc,
            function_count,
            class_count,
            avg_file_complexity,
            max_file_complexity,
            high_risk_function_count,
            medium_risk_function_count,
            low_risk_function_count,
            max_risk_score,
            avg_risk_score,
            module_coverage_ratio,
            tested_function_count,
            untested_function_count,
            import_fan_in,
            import_fan_out,
            cycle_group,
            in_cycle,
            role,
            role_confidence,
            role_sources_json,
            tags,
            owners,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        tuples,
    )
    return len(tuples)


def build_module_profile(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """
    Compute and persist analytics.module_profile rows.

    Returns
    -------
    int
        Number of rows inserted.
    """
    inputs = compute_module_profile_inputs(gateway, cfg)
    rows = build_module_profile_rows(inputs, module_table=module_table)
    return write_module_profile_rows(gateway, rows)
