"""Native Hamilton implementation for function_effects target.

This module provides the Hamilton native nodes for function effects classification
with DAG-visible I/O via SaveToDecorator/DuckDBRowsSaver:

- `t__function_effects__compute`: Pure compute node returning effect rows
- `function_effects__rows`: Extract rows for materialization
- `t__function_effects`: Materialize node combining table writes

The compute node calls `build_function_effects_rows` which returns pure rows
without persistence. Persistence is handled by DuckDBRowsSaver via SaveToDecorator,
making I/O visible in the Hamilton DAG for caching and observability.

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
    build_function_effects_rows,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService
from codeintel.graphs.runtime import GraphRuntimeOptions, resolve_graph_runtime

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

# Column definitions for function_effects table
FUNCTION_EFFECTS_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "function_goid_h128",
    "is_pure",
    "uses_io",
    "touches_db",
    "uses_time",
    "uses_randomness",
    "modifies_globals",
    "modifies_closure",
    "spawns_threads_or_tasks",
    "has_transitive_effects",
    "purity_confidence",
    "effects_json",
    "created_at",
)


@dataclass(frozen=True)
class FunctionEffectsResult:
    """Result from function effects computation.

    Attributes
    ----------
    rows
        Effect rows ready for persistence, or None if skipped.
    error
        Error message if computation failed.
    """

    rows: list[dict[str, object]] | None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if computation succeeded.

        Returns
        -------
        bool
            True if rows is not None and no error occurred.
        """
        return self.error is None


def _row_to_tuple(row: Mapping[str, object], cols: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a dict row to a tuple in column order.

    Parameters
    ----------
    row
        Row mapping from column name to value.
    cols
        Column names in the desired order.

    Returns
    -------
    tuple[object, ...]
        Values in column order.
    """
    return tuple(row.get(col) for col in cols)


@tag(domain="analytics", target="function_effects", node_type="compute")
def t__function_effects__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__call_graph: TargetRunRecord,
) -> FunctionEffectsResult:
    """Compute side effects classification for functions.

    This is a pure compute node that returns rows without persistence.
    The actual DB writes are handled by downstream SaveToDecorator nodes.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for skip detection.
    t__call_graph
        Upstream call_graph target result (for dependency).

    Returns
    -------
    FunctionEffectsResult
        Result containing effect rows, or None if skipped.

    Notes
    -----
    The effects classified include:
    - Pure functions vs impure
    - Side effect types (I/O, state mutation, etc.)
    - Effect evidence and reasoning
    """
    if t__call_graph.status != "succeeded":
        return FunctionEffectsResult(
            rows=None,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    target = graph.get("function_effects")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return FunctionEffectsResult(rows=None)

    try:
        # Load catalog and graph runtime
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        try:
            graph_runtime = resolve_graph_runtime(
                env.gateway,
                env.snapshot,
                GraphRuntimeOptions(),
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to resolve graph runtime: %s", exc)
            graph_runtime = None

        # Build options
        opts = FunctionEffectsOptions(
            max_call_depth=3,
            require_all_callees_pure=True,
        )

        # Build inputs
        inputs = FunctionEffectsInputs(
            catalog_provider=catalog,
            runtime=graph_runtime,
            ast_map=None,
            missing_goids=None,
        )

        # Compute effects (pure compute - no persistence)
        rows = build_function_effects_rows(
            env.gateway,
            env.snapshot,
            options=opts,
            inputs=inputs,
        )

        return FunctionEffectsResult(rows=rows)

    except Exception as exc:
        log.exception("Function effects computation failed")
        return FunctionEffectsResult(
            rows=None,
            error=str(exc),
        )


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.function_effects"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("function_effects"),
    table_key=value("analytics.function_effects"),
    columns=value(FUNCTION_EFFECTS_COLS),
)
@tag(
    domain="analytics",
    target="function_effects",
    node_type="compute",
    target_="function_effects__rows",
)
def function_effects__rows(
    t__function_effects__compute: FunctionEffectsResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.function_effects table.

    Parameters
    ----------
    t__function_effects__compute
        Computed function effects result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.function_effects table, or None if compute
        result is None or failed.
    """
    if t__function_effects__compute.rows is None:
        return None
    return tuple(
        _row_to_tuple(row, FUNCTION_EFFECTS_COLS)
        for row in t__function_effects__compute.rows
    )


@tag(domain="analytics", target="function_effects", node_type="materialize")
def t__function_effects(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_effects: dict[str, Any],
) -> TargetRunRecord:
    """Materialize function effects target.

    Converts materialization metadata into a TargetRunRecord for the
    function_effects target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__function_effects
        Materialization metadata for function_effects table.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="function_effects",
        expected_table_key="analytics.function_effects",
        materialization=m__analytics__function_effects,
    )


__all__ = [
    "FUNCTION_EFFECTS_COLS",
    "FunctionEffectsResult",
    "function_effects__rows",
    "t__function_effects",
    "t__function_effects__compute",
]
