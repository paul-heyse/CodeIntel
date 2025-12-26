"""Native Hamilton implementation for hotspots target.

This module computes file hotspot metrics from AST complexity and git churn
statistics, materializing rows via Hamilton-native DuckDB row savers.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import ibis.expr.types as ir
from hamilton.function_modifiers import source

from codeintel.analytics.hotspots import ChurnSummary, compute_hotspot_rows, parse_git_log_lines
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import DataAccessSpec, load_table_spec
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    SaverContext,
    TableSaveSpec,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.nodes.module_attach import tagged_attach_node
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
)
from codeintel.build.hamilton.tag_spec import TagSpec
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.ingestion.engine.infrastructure import ToolRunner, ToolRunOptions
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Iterable

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

HOTSPOTS_TARGET_NAME = "hotspots"
HOTSPOTS_TABLE_KEY = "analytics.hotspots"
HOTSPOTS_TABLE_KEYS = (HOTSPOTS_TABLE_KEY,)
HOTSPOTS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=HOTSPOTS_TARGET_NAME,
    hash_options_node="hotspots__hash_options",
)
AST_METRICS_TABLE_KEY = "core.ast_metrics"
MODULES_TABLE_KEY = "core.modules"
MAX_STDERR_CHARS = 500

_MODULE: ModuleType = sys.modules[__name__]

hotspots__ast_metrics_table = load_table_spec(
    DataAccessSpec(
        domain="analytics",
        target=HOTSPOTS_TARGET_NAME,
        table_key=AST_METRICS_TABLE_KEY,
        node_name="hotspots__ast_metrics_table",
    )
)
tagged_attach_node(
    _MODULE,
    node_name=hotspots__ast_metrics_table.__name__,
    fn=hotspots__ast_metrics_table,
    tag_spec=TagSpec.for_loader_query(
        domain="analytics",
        target=HOTSPOTS_TARGET_NAME,
        table_key=AST_METRICS_TABLE_KEY,
    ),
)
hotspots__modules_table = load_table_spec(
    DataAccessSpec(
        domain="analytics",
        target=HOTSPOTS_TARGET_NAME,
        table_key=MODULES_TABLE_KEY,
        node_name="hotspots__modules_table",
    )
)
tagged_attach_node(
    _MODULE,
    node_name=hotspots__modules_table.__name__,
    fn=hotspots__modules_table,
    tag_spec=TagSpec.for_loader_query(
        domain="analytics",
        target=HOTSPOTS_TARGET_NAME,
        table_key=MODULES_TABLE_KEY,
    ),
)


@tag_helper(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def hotspots__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for hotspots execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, HOTSPOTS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def hotspots__skip(
    env: BuildEnv,
    graph: TargetGraph,
    hotspots__hash_options: InputHashOptions,
) -> bool:
    """Return True when hotspots should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        HOTSPOTS_TARGET_NAME,
        hash_options=hotspots__hash_options,
    )
    return executor.should_skip()


@tag_helper(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def hotspots__options(env: BuildEnv) -> HotspotsOptions:
    """Load hotspots options from the build environment.

    Returns
    -------
    HotspotsOptions
        Loaded options for hotspots computation.
    """
    return load_target_options(
        env,
        target_name=HOTSPOTS_TARGET_NAME,
        options_type=HotspotsOptions,
    )


@tag_helper(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def hotspots__repo_root(env: BuildEnv) -> Path:
    """Expose the repository root for hotspots computation.

    Returns
    -------
    Path
        Repository root path.
    """
    return env.snapshot.repo_root


@tag_helper(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def hotspots__tool_runner(env: BuildEnv) -> ToolRunner | None:
    """Expose the tool runner for git log collection.

    Returns
    -------
    ToolRunner | None
        Tool runner when configured, otherwise None.
    """
    return env.providers.tool_runner


@dataclass(frozen=True)
class HotspotsOptions:
    """Options for computing hotspots."""

    max_commits: int = 2000


@dataclass(frozen=True)
class HotspotsResult:
    """Result from hotspots computation."""

    rows: tuple[dict[str, object], ...] | None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True when computation succeeded."""
        return self.error is None


def _run_git_log(
    repo_root: Path,
    max_commits: int,
    *,
    runner: ToolRunner,
) -> list[str] | None:
    resolved_root = repo_root.resolve()
    args = [
        "git",
        "log",
        f"--max-count={max_commits}",
        "--numstat",
        "--date=short",
        "--pretty=format:COMMIT\t%H\t%an",
        "--no-renames",
    ]
    result = runner.run(
        "git",
        args,
        options=ToolRunOptions(cwd=resolved_root),
    )
    if result.returncode not in {0, 1}:
        LOG.warning(
            "git log exited with code %s; stdout=%s stderr=%s",
            result.returncode,
            result.stdout[:MAX_STDERR_CHARS],
            result.stderr[:MAX_STDERR_CHARS],
        )
    if result.returncode not in {0, 1}:
        return None
    return result.stdout.splitlines()


@tag_helper(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def collect_git_file_stats(
    repo_root: Path,
    *,
    max_commits: int,
    runner: ToolRunner | None,
) -> dict[str, ChurnSummary]:
    if max_commits <= 0:
        return {}
    if runner is None:
        LOG.warning("ToolRunner not provided for git churn; skipping git log.")
        return {}

    git_lines = _run_git_log(repo_root, max_commits, runner=runner)
    if git_lines is None:
        return {}
    return parse_git_log_lines(git_lines)


def _load_module_paths(modules_table: ir.Table) -> set[str] | None:
    try:
        df = modules_table.select("path").execute()
    except DuckDBError as exc:
        LOG.warning("Failed to read core.modules for hotspots: %s", exc)
        return None
    if getattr(df, "empty", True):
        LOG.info("No rows in core.modules; skipping hotspots.")
        return None

    paths: set[str] = set()
    for _, row in df.iterrows():
        path = row["path"]
        if path is not None:
            paths.add(str(path))
    return paths


def _load_ast_metrics(
    ast_metrics_table: ir.Table,
    module_paths: set[str],
) -> list[tuple[str, float]] | None:
    try:
        df = ast_metrics_table.select("rel_path", "complexity").execute()
    except DuckDBError as exc:
        LOG.warning("Failed to read core.ast_metrics for hotspots: %s", exc)
        return None
    if getattr(df, "empty", True):
        LOG.info("No rows in core.ast_metrics; skipping hotspots.")
        return None

    rows: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        rel_path = str(row["rel_path"])
        if module_paths and rel_path not in module_paths:
            continue
        complexity = float(row["complexity"]) if row["complexity"] is not None else 0.0
        rows.append((rel_path, complexity))
    return rows


def _rows_to_tuples(
    rows: Iterable[dict[str, object]],
) -> tuple[tuple[object, ...], ...]:
    return tuple(row_to_tuple(HOTSPOTS_TABLE_KEY, row) for row in rows)


@tag_compute(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def t__hotspots__compute(
    hotspots__ast_metrics_table: ir.Table,
    hotspots__modules_table: ir.Table,
    hotspots__options: HotspotsOptions,
    hotspots__repo_root: Path,
    hotspots__tool_runner: ToolRunner | None,
    *,
    hotspots__skip: bool,
) -> HotspotsResult:
    """Compute file hotspot metrics.

    Parameters
    ----------
    hotspots__ast_metrics_table
        Loader node for core.ast_metrics.
    hotspots__modules_table
        Loader node for core.modules.

    Returns
    -------
    HotspotsResult
        Computed hotspot rows or an error message.
    """
    if hotspots__skip:
        return HotspotsResult(rows=None)

    module_paths = _load_module_paths(hotspots__modules_table)
    if not module_paths:
        return HotspotsResult(rows=None)

    ast_metrics = _load_ast_metrics(hotspots__ast_metrics_table, module_paths)
    if not ast_metrics:
        return HotspotsResult(rows=None)

    churn_stats = collect_git_file_stats(
        hotspots__repo_root,
        max_commits=hotspots__options.max_commits,
        runner=hotspots__tool_runner,
    )

    try:
        rows = compute_hotspot_rows(ast_metrics, churn_stats)
    except (ValueError, TypeError) as exc:
        return HotspotsResult(rows=None, error=str(exc))
    return HotspotsResult(rows=rows)


@save_rows(
    context=HOTSPOTS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=HOTSPOTS_TABLE_KEY),
)
@tag_compute(domain="analytics", target=HOTSPOTS_TARGET_NAME, target_="hotspots__rows")
def hotspots__rows(
    t__hotspots__compute: HotspotsResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Convert hotspot row mappings into tuple rows.

    Parameters
    ----------
    t__hotspots__compute
        Computed hotspot rows.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Tuple rows for DuckDB materialization, or None when skipped.
    """
    if t__hotspots__compute.rows is None:
        return None
    if not t__hotspots__compute.success:
        LOG.warning("Hotspots computation failed: %s", t__hotspots__compute.error)
        return None
    return _rows_to_tuples(t__hotspots__compute.rows)


hotspots__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=HOTSPOTS_TARGET_NAME,
    table_keys=HOTSPOTS_TABLE_KEYS,
)


@codeintel_target(domain="analytics", target=HOTSPOTS_TARGET_NAME)
def t__hotspots(
    env: BuildEnv,
    graph: TargetGraph,
    hotspots__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """File hotspot analysis based on churn.

    Returns
    -------
    TargetRunRecord
        Run record for the hotspots target execution.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=HOTSPOTS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=hotspots__table_materializations,
    )


__all__ = [
    "hotspots__ast_metrics_table",
    "hotspots__hash_options",
    "hotspots__modules_table",
    "hotspots__options",
    "hotspots__repo_root",
    "hotspots__rows",
    "hotspots__skip",
    "hotspots__tool_runner",
    "t__hotspots",
    "t__hotspots__compute",
]
