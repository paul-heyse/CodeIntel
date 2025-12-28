"""Module profile recipe helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.profiles.types import ModuleProfileInputs
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
    AnalyticsModuleProfileRow as ModuleProfileRowModel,
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


def compute_module_profile_inputs(
    gateway: StorageGateway, snapshot: SnapshotRef
) -> ModuleProfileInputs:
    """
    Construct snapshot inputs for module profile generation.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.

    Returns
    -------
    ModuleProfileInputs
        Snapshot handle for module profile helpers.
    """
    return ModuleProfileInputs(
        gateway=gateway,
        con=gateway.con,
        repo=snapshot.repo,
        commit=snapshot.commit,
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
    if module_table not in {DEFAULT_MODULE_TABLE, CATALOG_MODULE_TABLE}:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    try:
        modules_scoped, func_stats, files, imports, roles = _load_module_aggregates(
            inputs.gateway,
            module_table,
            inputs.repo,
            inputs.commit,
        )
    except DuckDBError as exc:
        log.warning("module_profile: failed to access tables: %s", exc)
        return

    base = modules_scoped.set_alias("base")
    func_rel = func_stats.set_alias("func_stats")
    files_rel = files.set_alias("files")
    imports_rel = imports.set_alias("imports")
    roles_rel = roles.set_alias("roles")

    joined = base.join(
        func_rel,
        (
            "base.repo = func_stats.repo "
            "AND base.commit = func_stats.commit "
            "AND base.module = func_stats.module"
        ),
        how="left",
    ).set_alias("base")
    joined = joined.join(
        files_rel,
        "base.repo = files.repo AND base.commit = files.commit AND base.module = files.module",
        how="left",
    ).set_alias("base")
    joined = joined.join(
        imports_rel,
        (
            "base.repo = imports.repo "
            "AND base.commit = imports.commit "
            "AND base.module = imports.module"
        ),
        how="left",
    ).set_alias("base")
    joined = joined.join(
        roles_rel,
        "base.repo = roles.repo AND base.commit = roles.commit AND base.module = roles.module",
        how="left",
    ).set_alias("base")

    try:
        selected = joined.select(
            "base.repo as repo",
            "base.commit as commit",
            "base.module as module",
            "base.path as path",
            "base.language as language",
            "coalesce(base.file_count, 0) as file_count",
            "coalesce(base.total_loc, 0) as total_loc",
            "coalesce(base.total_logical_loc, 0) as total_logical_loc",
            "coalesce(base.function_count, 0) as function_count",
            "coalesce(base.class_count, 0) as class_count",
            "base.avg_file_complexity as avg_file_complexity",
            "base.max_file_complexity as max_file_complexity",
            "coalesce(base.high_risk_function_count, 0) as high_risk_function_count",
            "coalesce(base.medium_risk_function_count, 0) as medium_risk_function_count",
            "coalesce(base.low_risk_function_count, 0) as low_risk_function_count",
            "base.max_risk_score as max_risk_score",
            "base.avg_risk_score as avg_risk_score",
            (
                "case when "
                "(coalesce(base.tested_function_count, 0) + "
                "coalesce(base.untested_function_count, 0)) = 0 "
                "then NULL "
                "else cast(coalesce(base.tested_function_count, 0) as double) / "
                "(coalesce(base.tested_function_count, 0) + "
                "coalesce(base.untested_function_count, 0)) "
                "end as module_coverage_ratio"
            ),
            "coalesce(base.tested_function_count, 0) as tested_function_count",
            "coalesce(base.untested_function_count, 0) as untested_function_count",
            "coalesce(base.import_fan_in, 0) as import_fan_in",
            "coalesce(base.import_fan_out, 0) as import_fan_out",
            "base.cycle_group as cycle_group",
            "coalesce(base.in_cycle_flag, 0) > 0 as in_cycle",
            "base.role as role",
            "base.role_confidence as role_confidence",
            "base.role_sources_json as role_sources_json",
            "base.tags as tags",
            "base.owners as owners",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("module_profile: failed to execute aggregation: %s", exc)
        return

    for record in records:
        record["created_at"] = inputs.created_at
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


def _load_module_aggregates(
    gateway: StorageGateway, module_table: str, repo: str, commit: str
) -> tuple[DuckDBRelation, DuckDBRelation, DuckDBRelation, DuckDBRelation, DuckDBRelation]:
    """
    Load scoped module and aggregate tables for module profiles.

    Returns
    -------
    tuple[DuckDBRelation, DuckDBRelation, DuckDBRelation, DuckDBRelation, DuckDBRelation]
        Modules table plus function, file, import, and roles aggregates.
    """
    modules_table = gateway.relation_from_table_key(module_table)
    modules_scoped = maybe_scope_by_repo_commit(modules_table, repo=repo, commit=commit)

    func_profile = maybe_scope_by_repo_commit(
        gateway.relation_from_table_key("analytics.function_profile"),
        repo=repo,
        commit=commit,
    )
    func_stats = func_profile.aggregate(
        ", ".join(
            [
                "count(function_goid_h128) as function_count",
                "sum(loc) as total_loc",
                "sum(logical_loc) as total_logical_loc",
                (
                    "sum(case when risk_level = 'high' then 1 else 0 end) "
                    "as high_risk_function_count"
                ),
                (
                    "sum(case when risk_level = 'medium' then 1 else 0 end) "
                    "as medium_risk_function_count"
                ),
                ("sum(case when risk_level = 'low' then 1 else 0 end) as low_risk_function_count"),
                "max(risk_score) as max_risk_score",
                "avg(risk_score) as avg_risk_score",
                "sum(case when tested then 1 else 0 end) as tested_function_count",
                "sum(case when tested then 0 else 1 end) as untested_function_count",
            ]
        ),
        "repo, commit, module",
    )

    file_profile = maybe_scope_by_repo_commit(
        gateway.relation_from_table_key("analytics.file_profile"),
        repo=repo,
        commit=commit,
    )
    files = file_profile.aggregate(
        ", ".join(
            [
                "count(rel_path) as file_count",
                "sum(class_count) as class_count",
                "avg(ast_complexity) as avg_file_complexity",
                "max(ast_complexity) as max_file_complexity",
            ]
        ),
        "repo, commit, module",
    )

    imports_table = maybe_scope_by_repo_commit(
        gateway.relation_from_table_key("graph.import_graph_edges"),
        repo=repo,
        commit=commit,
    )
    imports = imports_table.aggregate(
        ", ".join(
            [
                "max(src_fan_out) as import_fan_out",
                "max(dst_fan_in) as import_fan_in",
                "max(cycle_group) as cycle_group",
                "sum(case when cycle_group is null then 0 else 1 end) as in_cycle_flag",
            ]
        ),
        "repo, commit, src_module",
    ).select(
        "repo",
        "commit",
        "src_module as module",
        "import_fan_out",
        "import_fan_in",
        "cycle_group",
        "in_cycle_flag",
    )

    roles_table = maybe_scope_by_repo_commit(
        gateway.relation_from_table_key("analytics.semantic_roles_modules"),
        repo=repo,
        commit=commit,
    )
    roles = roles_table.select(
        "repo",
        "commit",
        "module",
        "role",
        "role_confidence",
        "role_sources_json",
    )

    return modules_scoped, func_stats, files, imports, roles


def build_module_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """Compute and persist analytics.module_profile rows.

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
    inputs = compute_module_profile_inputs(gateway, snapshot)
    rows = list(build_module_profile_rows(inputs, module_table=module_table))
    if not rows:
        return 0

    config = PolicyWriterConfig(
        table_key="analytics.module_profile",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return write_rows_via_policy_backend(gateway, rows=rows, config=config)
