"""File profile recipe helpers."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.profiles.types import FileProfileFrames, FileProfileInputs
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
    AnalyticsFileProfileRow as FileProfileRowModel,
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


def _extract_annotation_ratio(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        ratio = value.get("params")
        return float(ratio) if isinstance(ratio, (int, float)) else None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            ratio = decoded.get("params")
            return float(ratio) if isinstance(ratio, (int, float)) else None
    return None


def _ensure_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for col in columns:
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(col))
    return frame


def _normalize_ast_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    if "complexity" in frame.columns:
        frame = frame.rename({"complexity": "ast_complexity"})
    return _ensure_columns(
        frame,
        [
            "node_count",
            "function_count",
            "class_count",
            "avg_depth",
            "max_depth",
            "ast_complexity",
        ],
    )


def _normalize_hotspots(frame: pl.DataFrame) -> pl.DataFrame:
    if "score" in frame.columns:
        frame = frame.rename({"score": "hotspot_score"})
    return _ensure_columns(
        frame,
        ["commit_count", "author_count", "lines_added", "lines_deleted", "hotspot_score"],
    )


def _normalize_typedness(frame: pl.DataFrame, *, repo: str, commit: str) -> pl.DataFrame:
    frame = _scope_frame(frame, repo, commit)
    frame = _ensure_columns(frame, ["repo", "commit", "path"])
    return _ensure_columns(
        frame,
        ["annotation_ratio", "untyped_defs", "overlay_needed", "type_error_count"],
    )


def _normalize_static_diagnostics(frame: pl.DataFrame, *, repo: str, commit: str) -> pl.DataFrame:
    frame = _scope_frame(frame, repo, commit)
    frame = _ensure_columns(frame, ["repo", "commit", "rel_path"])
    if "total_errors" in frame.columns:
        frame = frame.rename(
            {"total_errors": "static_error_count", "has_errors": "has_static_errors"}
        )
    return _ensure_columns(frame, ["static_error_count", "has_static_errors"])


def _normalize_modules(frame: pl.DataFrame, *, repo: str, commit: str) -> pl.DataFrame:
    frame = _scope_frame(frame, repo, commit)
    frame = _ensure_columns(frame, ["repo", "commit", "path"])
    return _ensure_columns(frame, ["module", "language", "tags", "owners"])


_PROFILE_GROUP_BY = ["repo", "commit", "rel_path"]


def compute_file_profile_inputs(
    snapshot: SnapshotRef,
    frames: FileProfileFrames,
) -> FileProfileInputs:
    """
    Construct snapshot inputs for file profile generation.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    frames
        Frame bundle for file profile inputs.

    Returns
    -------
    FileProfileInputs
        Snapshot handle for file profile helpers.
    """
    return FileProfileInputs(
        repo=snapshot.repo,
        commit=snapshot.commit,
        created_at=datetime.now(tz=UTC),
        function_profile=frames.function_profile,
        ast_metrics=frames.ast_metrics,
        hotspots=frames.hotspots,
        typedness=frames.typedness,
        static_diagnostics=frames.static_diagnostics,
        modules=frames.modules,
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
    if module_table not in {DEFAULT_MODULE_TABLE, CATALOG_MODULE_TABLE}:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    function_profile = _scope_frame(inputs.function_profile, inputs.repo, inputs.commit)
    if function_profile.is_empty():
        return

    agg = function_profile.group_by(_PROFILE_GROUP_BY).agg(
        [
            pl.len().alias("total_functions"),
            pl.col("call_is_public").cast(pl.Int64).sum().alias("public_functions"),
            pl.col("loc").mean().alias("avg_loc"),
            pl.col("loc").max().alias("max_loc"),
            pl.col("cyclomatic_complexity").mean().alias("avg_cyclomatic_complexity"),
            pl.col("cyclomatic_complexity").max().alias("max_cyclomatic_complexity"),
            (pl.col("risk_level") == "high").cast(pl.Int64).sum().alias("high_risk_function_count"),
            (pl.col("risk_level") == "medium")
            .cast(pl.Int64)
            .sum()
            .alias("medium_risk_function_count"),
            pl.col("risk_score").max().alias("max_risk_score"),
            pl.col("covered_lines").fill_null(0).sum().alias("sum_covered_lines"),
            pl.col("executable_lines").fill_null(0).sum().alias("sum_exec_lines"),
            pl.col("tested").cast(pl.Int64).sum().alias("tested_function_count"),
            pl.col("tested").not_().cast(pl.Int64).sum().alias("untested_function_count"),
            pl.col("tests_touching").fill_null(0).sum().alias("tests_touching"),
        ]
    )
    agg = agg.with_columns(
        pl.when(pl.col("sum_exec_lines") > 0)
        .then(pl.col("sum_covered_lines") / pl.col("sum_exec_lines"))
        .otherwise(None)
        .alias("file_coverage_ratio")
    )

    ast_metrics = _normalize_ast_metrics(inputs.ast_metrics)
    hotspots = _normalize_hotspots(inputs.hotspots)
    typedness = _normalize_typedness(inputs.typedness, repo=inputs.repo, commit=inputs.commit)
    static_diag = _normalize_static_diagnostics(
        inputs.static_diagnostics,
        repo=inputs.repo,
        commit=inputs.commit,
    )
    modules = _normalize_modules(inputs.modules, repo=inputs.repo, commit=inputs.commit)

    base = agg.join(ast_metrics, on="rel_path", how="left")
    base = base.join(hotspots, on="rel_path", how="left")
    base = base.join(
        typedness,
        left_on=["repo", "commit", "rel_path"],
        right_on=["repo", "commit", "path"],
        how="left",
    ).drop("path")
    base = base.join(
        static_diag,
        on=["repo", "commit", "rel_path"],
        how="left",
    )
    base = base.join(
        modules,
        left_on=["repo", "commit", "rel_path"],
        right_on=["repo", "commit", "path"],
        how="left",
    ).drop("path")

    for record in base.iter_rows(named=True):
        record["created_at"] = inputs.created_at
        yield _row_to_file_profile_model(record, inputs)


def _row_to_file_profile_model(
    record: dict[str, object], inputs: FileProfileInputs
) -> FileProfileRowModel:
    """
    Convert a tabular row mapping into a FileProfileRowModel.

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
        annotation_ratio=_extract_annotation_ratio(record.get("annotation_ratio")),
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
