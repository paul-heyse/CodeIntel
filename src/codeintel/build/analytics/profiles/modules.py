"""Module profile recipe helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.profiles.types import ModuleProfileFrames, ModuleProfileInputs
from codeintel.build.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
)
from codeintel.build.analytics.utilities.type_coercion import (
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsModuleProfileRow as ModuleProfileRowModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)


def _scope_frame(frame: pl.DataFrame, repo: str, commit: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "repo" in frame.columns and "commit" in frame.columns:
        return frame.filter((pl.col("repo") == repo) & (pl.col("commit") == commit))
    return frame


def compute_module_profile_inputs(
    snapshot: SnapshotRef,
    frames: ModuleProfileFrames,
) -> ModuleProfileInputs:
    """
    Construct snapshot inputs for module profile generation.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    frames
        Frame bundle for module profile inputs.

    Returns
    -------
    ModuleProfileInputs
        Snapshot handle for module profile helpers.
    """
    return ModuleProfileInputs(
        repo=snapshot.repo,
        commit=snapshot.commit,
        created_at=datetime.now(tz=UTC),
        modules=frames.modules,
        function_profile=frames.function_profile,
        file_profile=frames.file_profile,
        import_graph_edges=frames.import_graph_edges,
        semantic_roles_modules=frames.semantic_roles_modules,
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

    modules = _scope_frame(inputs.modules, inputs.repo, inputs.commit)
    if modules.is_empty():
        return
    for col in ["repo", "commit", "module", "path", "language", "tags", "owners"]:
        if col not in modules.columns:
            modules = modules.with_columns(pl.lit(None).alias(col))

    function_profile = _scope_frame(inputs.function_profile, inputs.repo, inputs.commit)
    func_stats = (
        function_profile.group_by(["repo", "commit", "module"]).agg(
            [
                pl.len().alias("function_count"),
                pl.col("loc").fill_null(0).sum().alias("total_loc"),
                pl.col("logical_loc").fill_null(0).sum().alias("total_logical_loc"),
                (pl.col("risk_level") == "high")
                .cast(pl.Int64)
                .sum()
                .alias("high_risk_function_count"),
                (pl.col("risk_level") == "medium")
                .cast(pl.Int64)
                .sum()
                .alias("medium_risk_function_count"),
                (pl.col("risk_level") == "low")
                .cast(pl.Int64)
                .sum()
                .alias("low_risk_function_count"),
                pl.col("risk_score").max().alias("max_risk_score"),
                pl.col("risk_score").mean().alias("avg_risk_score"),
                pl.col("tested").cast(pl.Int64).sum().alias("tested_function_count"),
                pl.col("tested").not_().cast(pl.Int64).sum().alias("untested_function_count"),
            ]
        )
        if not function_profile.is_empty()
        else pl.DataFrame(
            schema={
                "repo": pl.Utf8,
                "commit": pl.Utf8,
                "module": pl.Utf8,
                "function_count": pl.Int64,
                "total_loc": pl.Int64,
                "total_logical_loc": pl.Int64,
                "high_risk_function_count": pl.Int64,
                "medium_risk_function_count": pl.Int64,
                "low_risk_function_count": pl.Int64,
                "max_risk_score": pl.Float64,
                "avg_risk_score": pl.Float64,
                "tested_function_count": pl.Int64,
                "untested_function_count": pl.Int64,
            }
        )
    )

    file_profile = _scope_frame(inputs.file_profile, inputs.repo, inputs.commit)
    files = (
        file_profile.group_by(["repo", "commit", "module"]).agg(
            [
                pl.len().alias("file_count"),
                pl.col("class_count").fill_null(0).sum().alias("class_count"),
                pl.col("ast_complexity").mean().alias("avg_file_complexity"),
                pl.col("ast_complexity").max().alias("max_file_complexity"),
            ]
        )
        if not file_profile.is_empty()
        else pl.DataFrame(
            schema={
                "repo": pl.Utf8,
                "commit": pl.Utf8,
                "module": pl.Utf8,
                "file_count": pl.Int64,
                "class_count": pl.Int64,
                "avg_file_complexity": pl.Float64,
                "max_file_complexity": pl.Float64,
            }
        )
    )

    import_edges = _scope_frame(inputs.import_graph_edges, inputs.repo, inputs.commit)
    imports = (
        import_edges.group_by(["repo", "commit", "src_module"]).agg(
            [
                pl.col("src_fan_out").max().alias("import_fan_out"),
                pl.col("dst_fan_in").max().alias("import_fan_in"),
                pl.col("cycle_group").max().alias("cycle_group"),
                pl.col("cycle_group").is_not_null().cast(pl.Int64).sum().alias("in_cycle_flag"),
            ]
        )
        if not import_edges.is_empty()
        else pl.DataFrame(
            schema={
                "repo": pl.Utf8,
                "commit": pl.Utf8,
                "src_module": pl.Utf8,
                "import_fan_out": pl.Int64,
                "import_fan_in": pl.Int64,
                "cycle_group": pl.Int64,
                "in_cycle_flag": pl.Int64,
            }
        )
    )
    if not imports.is_empty():
        imports = imports.rename({"src_module": "module"})

    roles = _scope_frame(inputs.semantic_roles_modules, inputs.repo, inputs.commit)
    for col in ["repo", "commit", "module", "role", "role_confidence", "role_sources_json"]:
        if col not in roles.columns:
            roles = roles.with_columns(pl.lit(None).alias(col))

    base = modules.join(func_stats, on=["repo", "commit", "module"], how="left")
    base = base.join(files, on=["repo", "commit", "module"], how="left")
    base = base.join(imports, on=["repo", "commit", "module"], how="left")
    base = base.join(roles, on=["repo", "commit", "module"], how="left")
    tested_count = pl.col("tested_function_count").fill_null(0)
    untested_count = pl.col("untested_function_count").fill_null(0)
    base = base.with_columns(
        pl.when((tested_count + untested_count) > 0)
        .then(tested_count / (tested_count + untested_count))
        .otherwise(None)
        .alias("module_coverage_ratio")
    )
    base = base.with_columns((pl.col("in_cycle_flag").fill_null(0) > 0).alias("in_cycle"))

    for record in base.iter_rows(named=True):
        record["created_at"] = inputs.created_at
        yield _row_to_module_profile_model(record, inputs)


def _row_to_module_profile_model(
    record: dict[str, object], inputs: ModuleProfileInputs
) -> ModuleProfileRowModel:
    """
    Convert a tabular row mapping into a ModuleProfileRowModel.

    Returns
    -------
    ModuleProfileRowModel
        Row model derived from the provided record.
    """
    created_at_value = record.get("created_at")
    created_at = created_at_value if isinstance(created_at_value, datetime) else inputs.created_at
    return ModuleProfileRowModel(
        repo=str(record.get("repo")),
        commit=str(record.get("commit")),
        module=str(record.get("module")),
        path=optional_str(record.get("path")),
        language=optional_str(record.get("language")),
        file_count=optional_int(record.get("file_count")),
        total_loc=optional_int(record.get("total_loc")),
        total_logical_loc=optional_int(record.get("total_logical_loc")),
        function_count=optional_int(record.get("function_count")),
        class_count=optional_int(record.get("class_count")),
        avg_file_complexity=optional_float(record.get("avg_file_complexity")),
        max_file_complexity=optional_float(record.get("max_file_complexity")),
        high_risk_function_count=optional_int(record.get("high_risk_function_count")),
        medium_risk_function_count=optional_int(record.get("medium_risk_function_count")),
        low_risk_function_count=optional_int(record.get("low_risk_function_count")),
        max_risk_score=optional_float(record.get("max_risk_score")),
        avg_risk_score=optional_float(record.get("avg_risk_score")),
        module_coverage_ratio=optional_float(record.get("module_coverage_ratio")),
        tested_function_count=optional_int(record.get("tested_function_count")),
        untested_function_count=optional_int(record.get("untested_function_count")),
        import_fan_in=optional_int(record.get("import_fan_in")),
        import_fan_out=optional_int(record.get("import_fan_out")),
        cycle_group=optional_int(record.get("cycle_group")),
        in_cycle=bool(record.get("in_cycle")) if record.get("in_cycle") is not None else None,
        role=optional_str(record.get("role")),
        role_confidence=optional_float(record.get("role_confidence")),
        role_sources_json=record.get("role_sources_json")
        if record.get("role_sources_json") is not None
        else "[]",
        tags=record.get("tags") if record.get("tags") is not None else "[]",
        owners=record.get("owners") if record.get("owners") is not None else "[]",
        created_at=created_at,
    )
