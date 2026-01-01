"""Pure compute functions for entrypoint detection.

This module provides pure compute functions that return row data without
performing any database writes. The materialization is handled by the
Hamilton native module in `build/hamilton/native/analytics/entrypoints.py`.

The functions detect HTTP, CLI, and job entrypoints from source code,
returning structured result containers that can be materialized to DuckDB
tables by the build system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.entrypoints.detection import DetectorSettings
from codeintel.build.analytics.entrypoints.core import (
    ENTRYPOINT_TESTS_COLS,
    ENTRYPOINTS_COLS,
    EntrypointContextInputs,
    _build_entrypoint_context,
    _collect_entrypoint_rows,
)

if TYPE_CHECKING:
    from codeintel.build.analytics.entrypoints.core import EntrypointBuildInputs
    from codeintel.config.primitives import SnapshotRef


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntrypointsResult:
    """Result container for entrypoints computation.

    Contains row data for both entrypoint tables without performing writes.
    The rows are tuples matching the column specifications in the schema.

    Attributes
    ----------
    entrypoint_rows
        Rows for analytics.entrypoints table.
    test_rows
        Rows for analytics.entrypoint_tests table.
    """

    entrypoint_rows: tuple[tuple[object, ...], ...]
    test_rows: tuple[tuple[object, ...], ...]


def compute_entrypoints_pure(
    snapshot: SnapshotRef,
    inputs: EntrypointBuildInputs,
    context_inputs: EntrypointContextInputs,
) -> EntrypointsResult:
    """Compute entrypoints without writing to database.

    Detect HTTP, CLI, and job entrypoints from source code in the snapshot,
    returning row data that can be materialized separately.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    inputs
        Bundled inputs containing catalog, module map, features, and settings.
    context_inputs
        Optional frames and overrides used to build the entrypoint context.

    Returns
    -------
    EntrypointsResult
        Container with rows for entrypoints and entrypoint_tests tables.

    Notes
    -----
    This function is a pure transformation that reads from the database but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.

    The detection identifies:
    - HTTP endpoints (FastAPI, Flask, Django, etc.)
    - CLI commands (Click, argparse, Typer)
    - Scheduled jobs and background tasks
    - Event handlers and message consumers
    """
    resolved_context_inputs = EntrypointContextInputs(
        module_map_override=inputs.module_map,
        features=inputs.features_map,
        modules_frame=context_inputs.modules_frame,
        coverage_functions_frame=context_inputs.coverage_functions_frame,
        test_coverage_edges_frame=context_inputs.test_coverage_edges_frame,
        test_catalog_frame=context_inputs.test_catalog_frame,
        subsystem_modules_frame=context_inputs.subsystem_modules_frame,
        subsystems_frame=context_inputs.subsystems_frame,
    )
    entrypoint_context = _build_entrypoint_context(
        snapshot,
        inputs.catalog_provider,
        resolved_context_inputs,
    )

    if entrypoint_context is None:
        log.warning(
            "No modules available to scan for entrypoints in %s@%s",
            snapshot.repo,
            snapshot.commit,
        )
        return EntrypointsResult(entrypoint_rows=(), test_rows=())

    effective_settings = inputs.settings or DetectorSettings()
    entrypoint_rows, test_rows = _collect_entrypoint_rows(
        context=entrypoint_context,
        repo_root=snapshot.repo_root,
        settings=effective_settings,
        scan_profile=inputs.scan_profile,
    )

    log.info(
        "entrypoints computed: %d entrypoints, %d entrypoint_test edges for %s@%s",
        len(entrypoint_rows),
        len(test_rows),
        snapshot.repo,
        snapshot.commit,
    )

    return EntrypointsResult(
        entrypoint_rows=tuple(entrypoint_rows),
        test_rows=tuple(test_rows),
    )


__all__ = [
    "ENTRYPOINTS_COLS",
    "ENTRYPOINT_TESTS_COLS",
    "EntrypointsResult",
    "compute_entrypoints_pure",
]
