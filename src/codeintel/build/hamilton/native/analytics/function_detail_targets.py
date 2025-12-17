"""Native Hamilton implementations for function-level detail targets.

This module consolidates function-detail analytics targets that produce row
tables in DuckDB via DAG-visible I/O (SaveToDecorator/DuckDBRowsSaver):

- ``function_contracts``: inferred pre/postconditions and nullability contracts
- ``function_effects``: purity and side-effect classification

Each target follows the same pattern:

1. A pure compute node returning row dicts (no persistence)
2. A SaveToDecorator node that materializes rows to DuckDB
3. A materialize node that converts materialization metadata to TargetRunRecord

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_contracts import build_function_contracts_rows
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
    build_function_effects_rows,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
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

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

# Column definitions for function_contracts table
FUNCTION_CONTRACTS_COLS: tuple[str, ...] = (
    "repo",
    "commit",
    "function_goid_h128",
    "preconditions_json",
    "postconditions_json",
    "raises_json",
    "param_nullability_json",
    "return_nullability",
    "contract_confidence",
    "created_at",
)


@dataclass(frozen=True)
class FunctionContractsResult:
    """Result from function contracts computation.

    Attributes
    ----------
    rows
        Contract rows ready for persistence, or None if skipped.
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


@tag(domain="analytics", target="function_contracts", node_type="compute")
def t__function_contracts__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__goids: TargetRunRecord,
) -> FunctionContractsResult:
    """Compute pre/postconditions and nullability contracts for functions.

    This is a pure compute node that returns rows without persistence.
    The actual DB writes are handled by downstream SaveToDecorator nodes.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for skip detection.
    t__goids
        Upstream goids target result (for dependency).

    Returns
    -------
    FunctionContractsResult
        Result containing contract rows, or None if skipped.

    Notes
    -----
    The contracts inferred include:
    - Preconditions (required input states)
    - Postconditions (guaranteed output states)
    - Nullability contracts
    """
    if t__goids.status != "succeeded":
        return FunctionContractsResult(
            rows=None,
            error=f"Upstream goids target failed: {t__goids.error}",
        )

    target = graph.get("function_contracts")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return FunctionContractsResult(rows=None)

    try:
        # Load catalog
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            return FunctionContractsResult(
                rows=None,
                error=f"CatalogProvider is required: {exc}",
            )

        # Load function ASTs
        try:
            function_ast_map, _missing = load_function_asts(
                env.gateway,
                FunctionAstLoadRequest(
                    repo=env.snapshot.repo,
                    commit=env.snapshot.commit,
                    repo_root=env.snapshot.repo_root,
                    catalog_provider=catalog,
                ),
            )
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Failed to load function ASTs: %s", exc)
            function_ast_map = {}

        # Compute contracts (pure compute - no persistence)
        rows = build_function_contracts_rows(
            env.gateway,
            env.snapshot,
            function_ast_map=function_ast_map,
            catalog=catalog,
        )

        return FunctionContractsResult(rows=rows)

    except Exception as exc:
        log.exception("Function contracts computation failed")
        return FunctionContractsResult(
            rows=None,
            error=str(exc),
        )


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.function_contracts"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("function_contracts"),
    table_key=value("analytics.function_contracts"),
    columns=value(FUNCTION_CONTRACTS_COLS),
)
@tag(
    domain="analytics",
    target="function_contracts",
    node_type="compute",
    target_="function_contracts__rows",
)
def function_contracts__rows(
    t__function_contracts__compute: FunctionContractsResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.function_contracts table.

    Parameters
    ----------
    t__function_contracts__compute
        Computed function contracts result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.function_contracts table, or None if compute
        result is None or failed.
    """
    if t__function_contracts__compute.rows is None:
        return None
    return tuple(
        _row_to_tuple(row, FUNCTION_CONTRACTS_COLS)
        for row in t__function_contracts__compute.rows
    )


@tag(domain="analytics", target="function_contracts", node_type="materialize")
def t__function_contracts(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_contracts: dict[str, Any],
) -> TargetRunRecord:
    """Materialize function contracts target.

    Converts materialization metadata into a TargetRunRecord for the
    function_contracts target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__function_contracts
        Materialization metadata for function_contracts table.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="function_contracts",
        expected_table_key="analytics.function_contracts",
        materialization=m__analytics__function_contracts,
    )


# ---------------------------------------------------------------------------
# function_effects target
# ---------------------------------------------------------------------------


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
        """Check if computation succeeded."""
        return self.error is None


@tag(domain="analytics", target="function_effects", node_type="compute")
def t__function_effects__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__call_graph: TargetRunRecord,
) -> FunctionEffectsResult:
    """Compute side effects classification for functions.

    Returns
    -------
    FunctionEffectsResult
        Computed rows and optional error message.
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

        opts = FunctionEffectsOptions(
            max_call_depth=3,
            require_all_callees_pure=True,
        )
        inputs = FunctionEffectsInputs(
            catalog_provider=catalog,
            runtime=graph_runtime,
            ast_map=None,
            missing_goids=None,
        )

        rows = build_function_effects_rows(
            env.gateway,
            env.snapshot,
            options=opts,
            inputs=inputs,
        )
        return FunctionEffectsResult(rows=rows)

    except Exception as exc:
        log.exception("Function effects computation failed")
        return FunctionEffectsResult(rows=None, error=str(exc))


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
    """Extract rows for analytics.function_effects.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when computation produced no rows.
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

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="function_effects",
        expected_table_key="analytics.function_effects",
        materialization=m__analytics__function_effects,
    )


__all__ = [
    "FUNCTION_CONTRACTS_COLS",
    "FUNCTION_EFFECTS_COLS",
    "FunctionContractsResult",
    "FunctionEffectsResult",
    "function_contracts__rows",
    "function_effects__rows",
    "t__function_contracts",
    "t__function_contracts__compute",
    "t__function_effects",
    "t__function_effects__compute",
]
