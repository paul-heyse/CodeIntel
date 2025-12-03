"""GOID builder plugin.

This module provides the GOID builder as a graph plugin, implementing
the full orchestration for building Global Object Identifiers.

Architecture
------------
This plugin follows the hexagonal architecture pattern:
- Uses resources (StorageResource) for I/O
- Delegates pure computation to compute/goid.py
- Orchestrates the full build pipeline

The orchestration logic includes:
1. Loading AST nodes from storage
2. Building GOID entries via compute layer
3. Persisting results back to storage
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.config.datasets import GoidCrosswalkRow as DatasetGoidCrosswalkRow
from codeintel.config.datasets import GoidRow as DatasetGoidRow
from codeintel.config.datasets import goid_crosswalk_to_tuple, goid_to_tuple
from codeintel.config.steps_graphs import GoidBuilderStepConfig
from codeintel.graphs.catalog import load_function_catalog
from codeintel.graphs.compute.goid import (
    GoidDescriptor,
    build_crosswalk_row,
    build_goid_row,
    build_urn,
    compute_goid,
    determine_kind,
)
from codeintel.graphs.core import (
    ComputationResult,
    GraphExecutionContext,
    GraphPluginProtocol,
    make_builder_plugin,
)
from codeintel.graphs.resources import StorageResource
from codeintel.ingestion.infrastructure_utilities.paths import relpath_to_module
from codeintel.ingestion.services.storage import IngestStorageService

log = logging.getLogger(__name__)


def _relpath_to_module(path: str | Path) -> str:
    """Convert a repository-relative path to a dotted module path.

    Parameters
    ----------
    path
        Repository-relative path to a Python file.

    Returns
    -------
    str
        Dotted module path derived from the relative path.
    """
    return relpath_to_module(path)


def _safe_int(value: object, default: int | None = None) -> int | None:
    """Convert optional values to int, treating pandas nulls as missing.

    Parameters
    ----------
    value
        Value to convert.
    default
        Default value when conversion fails.

    Returns
    -------
    int | None
        Integer value when conversion succeeds; otherwise the provided default.
    """
    if value is None or isinstance(value, (pd.Series, pd.DataFrame)):
        return default

    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        return default

    if not isinstance(value, (int, float, str, bool)):
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compute_start_line(row: pd.Series) -> int:
    """Compute the effective start line, considering decorator spans.

    Returns
    -------
    int
        Start line number, adjusted for decorators if present.
    """
    start_line = _safe_int(row["lineno"], default=1) or 1
    decorator_start = _safe_int(row.get("decorator_start_line"))
    if decorator_start is not None and decorator_start > 0:
        return min(start_line, decorator_start)
    return start_line


def _build_goid_entries(
    row: pd.Series,
    cfg: GoidBuilderStepConfig,
    now: datetime,
    module_by_path: Mapping[str, str],
) -> tuple[DatasetGoidRow, DatasetGoidCrosswalkRow]:
    """Build GOID and crosswalk entries from an AST node row.

    Delegates pure computation to the compute layer and converts
    results to dataset row types for persistence.

    Parameters
    ----------
    row
        Pandas row containing AST node data.
    cfg
        GOID builder configuration.
    now
        Current timestamp for created_at/updated_at.
    module_by_path
        Mapping of file paths to module names.

    Returns
    -------
    tuple[DatasetGoidRow, DatasetGoidCrosswalkRow]
        GOID row and crosswalk row for persistence.
    """
    rel_path = str(row["path"]).replace("\\", "/")
    module_name = module_by_path.get(rel_path, _relpath_to_module(Path(rel_path)))
    parent_raw = row["parent_qualname"]
    parent_qualname = str(parent_raw) if parent_raw is not None else None

    # Build descriptor and delegate computation
    descriptor = GoidDescriptor(
        repo=cfg.repo,
        commit=cfg.commit,
        language=cfg.language,
        rel_path=rel_path,
        kind=determine_kind(str(row["node_type"]), parent_qualname, rel_path, module_name),
        qualname=str(row["qualname"]),
        start_line=_compute_start_line(row),
        end_line=_safe_int(row["end_lineno"]),
    )

    goid_h128 = compute_goid(descriptor)
    urn = build_urn(descriptor)
    goid_data = build_goid_row(descriptor, goid_h128, urn, now)
    xwalk_data = build_crosswalk_row(descriptor, urn, module_name, now)

    return (
        DatasetGoidRow(
            goid_h128=goid_data.goid_h128,
            urn=goid_data.urn,
            repo=goid_data.repo,
            commit=goid_data.commit,
            rel_path=goid_data.rel_path,
            language=goid_data.language,
            kind=goid_data.kind,
            qualname=goid_data.qualname,
            start_line=goid_data.start_line,
            end_line=goid_data.end_line,
            created_at=goid_data.created_at,
        ),
        DatasetGoidCrosswalkRow(
            repo=xwalk_data.repo,
            commit=xwalk_data.commit,
            goid=xwalk_data.goid,
            lang=xwalk_data.lang,
            module_path=xwalk_data.module_path,
            file_path=xwalk_data.file_path,
            start_line=xwalk_data.start_line,
            end_line=xwalk_data.end_line,
            scip_symbol=xwalk_data.scip_symbol,
            ast_qualname=xwalk_data.ast_qualname,
            cst_node_id=xwalk_data.cst_node_id,
            chunk_id=xwalk_data.chunk_id,
            symbol_id=xwalk_data.symbol_id,
            updated_at=xwalk_data.updated_at,
        ),
    )


def _build_goids(ctx: GraphExecutionContext) -> ComputationResult:
    """Build GOIDs and crosswalk entries from AST nodes.

    Orchestrates the full GOID build pipeline:
    1. Loads AST nodes from storage
    2. Builds GOID entries via compute layer
    3. Persists results back to storage

    Returns
    -------
    ComputationResult
        Success result with row counts after populating GOID tables.
    """
    # Get storage via resource injection
    storage = ctx.require(StorageResource)
    gateway = storage.gateway

    cfg = GoidBuilderStepConfig(snapshot=ctx.snapshot)
    con = gateway.con

    # Load AST nodes from storage
    df = con.execute(
        """
        SELECT
            an.path,
            an.node_type,
            an.name,
            an.qualname,
            an.lineno,
            an.end_lineno,
            an.decorator_start_line,
            an.decorator_end_line,
            an.parent_qualname
        FROM core.ast_nodes an
        JOIN core.modules m
          ON m.path = an.path
        WHERE m.repo = ? AND m.commit = ?
          AND an.node_type IN ('Module', 'ClassDef', 'FunctionDef', 'AsyncFunctionDef')
        """,
        [cfg.repo, cfg.commit],
    ).fetch_df()

    if df.empty:
        log.warning("No AST nodes found in core.ast_nodes; cannot build GOIDs.")
        return ComputationResult.ok(row_counts={"core.goids": 0, "core.goid_crosswalk": 0})

    goid_rows: list[DatasetGoidRow] = []
    xwalk_rows: list[DatasetGoidCrosswalkRow] = []

    now = datetime.now(UTC)

    module_by_path = load_function_catalog(gateway, repo=cfg.repo, commit=cfg.commit).module_by_path

    for _, row in df.iterrows():
        goid_row, xwalk_row = _build_goid_entries(row, cfg, now, module_by_path)
        goid_rows.append(goid_row)
        xwalk_rows.append(xwalk_row)

    # Persist results
    storage_service = IngestStorageService.from_gateway(gateway)
    storage_service.run_batch(
        "core.goids",
        [goid_to_tuple(row) for row in goid_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    storage_service.run_batch(
        "core.goid_crosswalk",
        [goid_crosswalk_to_tuple(row) for row in xwalk_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "GOID build complete for repo=%s commit=%s: %d entities",
        cfg.repo,
        cfg.commit,
        len(goid_rows),
    )

    return ComputationResult.ok(
        row_counts={"core.goids": len(goid_rows), "core.goid_crosswalk": len(xwalk_rows)}
    )


goid_builder_plugin = make_builder_plugin(
    name="goid_builder",
    computation=_build_goids,
    stage="goid",
    produces_graphs=(),
    depends_on=(),
    provides=("goids",),
    produces_tables=("core.goids", "core.goid_crosswalk"),
)


def get_goid_builder_plugin() -> GraphPluginProtocol:
    """Return the GOID builder plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured GOID builder plugin.
    """
    return goid_builder_plugin


def build_goid_entries_for_testing(
    row: pd.Series,
    cfg: GoidBuilderStepConfig,
    now: datetime,
    module_by_path: Mapping[str, str],
) -> tuple[DatasetGoidRow, DatasetGoidCrosswalkRow]:
    """Exercise GOID entry construction in tests.

    Parameters
    ----------
    row
        Pandas row containing AST node data.
    cfg
        GOID builder configuration.
    now
        Current timestamp.
    module_by_path
        Mapping of file paths to module names.

    Returns
    -------
    tuple[DatasetGoidRow, DatasetGoidCrosswalkRow]
        GOID row and crosswalk row dataclasses.
    """
    return _build_goid_entries(row, cfg, now, module_by_path)


def build_goids(
    gateway: StorageGateway,
    cfg: GoidBuilderStepConfig,
) -> None:
    """Build GOIDs using the plugin orchestration.

    Convenience function for pipeline steps to invoke the goid_builder plugin
    with a specific configuration. Creates the execution context and resources
    internally.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    cfg
        GOID builder configuration.
    """
    from codeintel.graphs.resources.container import ResourceContainer  # noqa: PLC0415

    # Create context with resources
    container = ResourceContainer()
    container.register(StorageResource(gateway, cfg.snapshot.repo_root))

    ctx = GraphExecutionContext(
        snapshot=cfg.snapshot,
        resources=container,
    )

    result = _build_goids(ctx)
    if not result.success:
        log.warning("GOID build failed: %s", result.message)


# Type import for gateway parameter
if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


__all__ = [
    "build_goid_entries_for_testing",
    "build_goids",
    "get_goid_builder_plugin",
    "goid_builder_plugin",
]
