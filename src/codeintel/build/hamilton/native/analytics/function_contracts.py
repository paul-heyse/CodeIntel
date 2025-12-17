"""Native Hamilton implementation for function_contracts target.

This module provides the Hamilton native nodes for function contracts inference
with DAG-visible I/O via SaveToDecorator/DuckDBRowsSaver:

- `t__function_contracts__compute`: Pure compute node returning contract rows
- `function_contracts__rows`: Extract rows for materialization
- `t__function_contracts`: Materialize node combining table writes

The compute node calls `build_function_contracts_rows` which returns pure rows
without persistence. Persistence is handled by DuckDBRowsSaver via SaveToDecorator,
making I/O visible in the Hamilton DAG for caching and observability.

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.analytics.functions.function_contracts import build_function_contracts_rows
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


__all__ = [
    "FUNCTION_CONTRACTS_COLS",
    "FunctionContractsResult",
    "function_contracts__rows",
    "t__function_contracts",
    "t__function_contracts__compute",
]
