"""Module profile recipe helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import ibis

from codeintel.analytics.compute.ibis_utils import safe_ratio, zero_if_null
from codeintel.analytics.profiles.types import ModuleProfileInputs
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
    MODULE_PROFILE_COLUMNS,
    ModuleProfileRowModel,
    module_profile_row_to_tuple,
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

    import ibis.expr.types as it
    from ibis import BaseBackend

    from codeintel.config.primitives import SnapshotRef
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
    gateway = inputs.gateway
    repo = inputs.repo
    commit = inputs.commit

    if module_table not in {DEFAULT_MODULE_TABLE, CATALOG_MODULE_TABLE}:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    try:
        modules_scoped, func_stats, files, imports, roles = _load_module_aggregates(
            cast("BaseBackend", gateway.ibis), module_table, repo, commit
        )
    except DuckDBError as exc:
        log.warning("module_profile: failed to access tables: %s", exc)
        return

    joined = (
        modules_scoped.left_join(
            func_stats,
            predicates=[
                (modules_scoped.repo, func_stats.repo),
                (modules_scoped.commit, func_stats.commit),
                (modules_scoped.module, func_stats.module),
            ],
        )
        .left_join(
            files,
            predicates=[
                (modules_scoped.repo, files.repo),
                (modules_scoped.commit, files.commit),
                (modules_scoped.module, files.module),
            ],
        )
        .left_join(
            imports,
            predicates=[
                (modules_scoped.repo, imports.repo),
                (modules_scoped.commit, imports.commit),
                (modules_scoped.module, imports.module),
            ],
        )
        .left_join(
            roles,
            predicates=[
                (modules_scoped.repo, roles.repo),
                (modules_scoped.commit, roles.commit),
                (modules_scoped.module, roles.module),
            ],
        )
    )

    tested_expr = zero_if_null(func_stats.tested_function_count)
    untested_expr = zero_if_null(func_stats.untested_function_count)

    try:
        df = joined.select(
            modules_scoped.repo.name("repo"),
            modules_scoped.commit.name("commit"),
            modules_scoped.module.name("module"),
            modules_scoped.path.name("path"),
            modules_scoped.language.name("language"),
            zero_if_null(files.file_count).name("file_count"),
            zero_if_null(func_stats.total_loc).name("total_loc"),
            zero_if_null(func_stats.total_logical_loc).name("total_logical_loc"),
            zero_if_null(func_stats.function_count).name("function_count"),
            zero_if_null(files.class_count).name("class_count"),
            files.avg_file_complexity.name("avg_file_complexity"),
            files.max_file_complexity.name("max_file_complexity"),
            zero_if_null(func_stats.high_risk_function_count).name("high_risk_function_count"),
            zero_if_null(func_stats.medium_risk_function_count).name("medium_risk_function_count"),
            zero_if_null(func_stats.low_risk_function_count).name("low_risk_function_count"),
            func_stats.max_risk_score.name("max_risk_score"),
            func_stats.avg_risk_score.name("avg_risk_score"),
            safe_ratio(
                tested_expr,
                tested_expr + untested_expr,
            ).name("module_coverage_ratio"),
            tested_expr.name("tested_function_count"),
            untested_expr.name("untested_function_count"),
            zero_if_null(imports.import_fan_in).name("import_fan_in"),
            zero_if_null(imports.import_fan_out).name("import_fan_out"),
            imports.cycle_group.name("cycle_group"),
            zero_if_null(imports.in_cycle_flag).cast("boolean").name("in_cycle"),
            roles.role.name("role"),
            roles.role_confidence.name("role_confidence"),
            roles.role_sources_json.name("role_sources_json"),
            modules_scoped.tags.name("tags"),
            modules_scoped.owners.name("owners"),
            ibis.literal(inputs.created_at).name("created_at"),
        ).execute()
    except DuckDBError as exc:
        log.warning("module_profile: failed to execute aggregation: %s", exc)
        return

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

    for row in df.itertuples(index=False, name=None):
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


def _load_module_aggregates(
    ibis_api: BaseBackend, module_table: str, repo: str, commit: str
) -> tuple[it.Table, it.Table, it.Table, it.Table, it.Table]:
    """
    Load scoped module and aggregate tables for module profiles.

    Returns
    -------
    tuple[Table, Table, Table, Table, Table]
        Modules table plus function, file, import, and roles aggregates.
    """
    modules_table = ibis_api.table(module_table)
    modules_scoped = filter_by(
        modules_table,
        modules_table.repo == repo,
        modules_table.commit == commit,
    )

    func_profile = ibis_api.table("analytics.function_profile")
    func_scoped = filter_by(func_profile, func_profile.repo == repo, func_profile.commit == commit)
    func_stats = func_scoped.group_by(
        func_profile.repo,
        func_profile.commit,
        func_profile.module,
    ).aggregate(
        function_count=col_count(func_profile.function_goid_h128),
        total_loc=col_sum(func_profile.loc),
        total_logical_loc=col_sum(func_profile.logical_loc),
        high_risk_function_count=col_sum(
            ibis_bool(func_profile.risk_level == "high").cast("int64")
        ),
        medium_risk_function_count=col_sum(
            ibis_bool(func_profile.risk_level == "medium").cast("int64")
        ),
        low_risk_function_count=col_sum(ibis_bool(func_profile.risk_level == "low").cast("int64")),
        max_risk_score=col_max(func_profile.risk_score),
        avg_risk_score=col_mean(func_profile.risk_score),
        tested_function_count=col_sum(ibis_bool(func_profile.tested).cast("int64")),
        untested_function_count=col_sum(bool_not(func_profile.tested).cast("int64")),
    )

    file_profile = ibis_api.table("analytics.file_profile")
    file_scoped = filter_by(file_profile, file_profile.repo == repo, file_profile.commit == commit)
    files = file_scoped.group_by(
        file_profile.repo,
        file_profile.commit,
        file_profile.module,
    ).aggregate(
        file_count=file_profile.rel_path.count(),
        class_count=col_sum(file_profile.class_count),
        avg_file_complexity=col_mean(file_profile.ast_complexity),
        max_file_complexity=col_max(file_profile.ast_complexity),
    )

    imports_table = ibis_api.table("graph.import_graph_edges")
    imports_scoped = filter_by(
        imports_table,
        imports_table.repo == repo,
        imports_table.commit == commit,
    )
    imports = imports_scoped.group_by(
        imports_table.repo,
        imports_table.commit,
        imports_table.src_module.name("module"),
    ).aggregate(
        import_fan_out=col_max(imports_table.src_fan_out),
        import_fan_in=col_max(imports_table.dst_fan_in),
        cycle_group=col_max(imports_table.cycle_group),
        in_cycle_flag=col_sum(ibis_bool(imports_table.cycle_group.notnull()).cast("int64")),
    )

    roles_table = ibis_api.table("analytics.semantic_roles_modules")
    roles = filter_by(
        roles_table,
        roles_table.repo == repo,
        roles_table.commit == commit,
    ).select(
        roles_table.repo,
        roles_table.commit,
        roles_table.module,
        roles_table.role,
        roles_table.role_confidence,
        roles_table.role_sources_json,
    )

    return modules_scoped, func_stats, files, imports, roles


# Factory-created writer for module profiles
write_module_profile_rows: Callable[[StorageGateway, Iterable[ModuleProfileRowModel]], int] = (
    create_profile_writer(
        "analytics.module_profile",
        MODULE_PROFILE_COLUMNS,
        cast("Callable[[Mapping[str, object]], tuple[object, ...]]", module_profile_row_to_tuple),
    )
)


def build_module_profile(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """
    Compute and persist analytics.module_profile rows.

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
    rows = build_module_profile_rows(inputs, module_table=module_table)
    return write_module_profile_rows(gateway, rows)
