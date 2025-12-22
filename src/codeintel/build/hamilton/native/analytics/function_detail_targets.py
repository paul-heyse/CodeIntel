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

from hamilton.function_modifiers import source, value

from codeintel.analytics.functions.function_contracts import build_function_contracts_rows
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
    build_function_effects_rows,
)
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.native.target_override_tables import (
    FUNCTION_CONTRACTS_OVERRIDE_TABLES,
    FUNCTION_EFFECTS_OVERRIDE_TABLES,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.resources import ResourceNotFoundError
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.graphs.runtime import resolve_graph_runtime

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

FUNCTION_CONTRACTS_TARGET_NAME = "function_contracts"
FUNCTION_EFFECTS_TARGET_NAME = "function_effects"

FUNCTION_CONTRACTS_TABLE_KEY = "analytics.function_contracts"
FUNCTION_EFFECTS_TABLE_KEY = "analytics.function_effects"

register_output_targets(
    make_output_target(
        name=FUNCTION_EFFECTS_TARGET_NAME,
        module="analytics",
        description="Function purity and side-effect analysis.",
        options=TargetSpecOptions(
            table_keys=(FUNCTION_EFFECTS_TABLE_KEY,),
            override_tables=FUNCTION_EFFECTS_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=FUNCTION_CONTRACTS_TARGET_NAME,
        module="analytics",
        description="Inferred function pre/postconditions.",
        options=TargetSpecOptions(
            table_keys=(FUNCTION_CONTRACTS_TABLE_KEY,),
            override_tables=FUNCTION_CONTRACTS_OVERRIDE_TABLES,
        ),
    ),
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


@tag_compute(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
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

    target = graph.get(FUNCTION_CONTRACTS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, FUNCTION_CONTRACTS_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return FunctionContractsResult(rows=None)

    try:
        registry = build_registry(
            gateway=env.gateway,
            snapshot=env.snapshot,
            registry_options=ProviderRegistryOptions(
                include_graphs=False,
                include_asts=True,
            ),
        )

        # Load catalog
        try:
            catalog = registry.require(CatalogProvider).get()
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            return FunctionContractsResult(
                rows=None,
                error=f"CatalogProvider is required: {exc}",
            )

        # Load function ASTs
        try:
            ast_data = registry.require(AstProvider).get()
            function_ast_map = ast_data.function_ast_map
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_CONTRACTS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_CONTRACTS_TARGET_NAME),
    table_key=value(FUNCTION_CONTRACTS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FUNCTION_CONTRACTS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_CONTRACTS_TARGET_NAME,
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
        row_to_tuple(FUNCTION_CONTRACTS_TABLE_KEY, row)
        for row in t__function_contracts__compute.rows
    )


@tag_materialize(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
def t__function_contracts(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_contracts: MaterializationMetadata,
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
        target_name=FUNCTION_CONTRACTS_TARGET_NAME,
        expected_table_key=FUNCTION_CONTRACTS_TABLE_KEY,
        materialization=m__analytics__function_contracts,
    )


# ---------------------------------------------------------------------------
# function_effects target
# ---------------------------------------------------------------------------


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


@tag_compute(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
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

    target = graph.get(FUNCTION_EFFECTS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, FUNCTION_EFFECTS_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return FunctionEffectsResult(rows=None)

    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        try:
            catalog = registry.require(CatalogProvider).get()
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        try:
            graph_runtime = resolve_graph_runtime(
                env.gateway,
                env.snapshot,
                load_graph_runtime_options(env, target_name=FUNCTION_EFFECTS_TARGET_NAME),
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to resolve graph runtime: %s", exc)
            graph_runtime = None

        opts = load_target_options(
            env,
            target_name=FUNCTION_EFFECTS_TARGET_NAME,
            options_type=FunctionEffectsOptions,
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_EFFECTS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_EFFECTS_TARGET_NAME),
    table_key=value(FUNCTION_EFFECTS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FUNCTION_EFFECTS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_EFFECTS_TARGET_NAME,
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
        row_to_tuple(FUNCTION_EFFECTS_TABLE_KEY, row) for row in t__function_effects__compute.rows
    )


@tag_materialize(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
def t__function_effects(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_effects: MaterializationMetadata,
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
        target_name=FUNCTION_EFFECTS_TARGET_NAME,
        expected_table_key=FUNCTION_EFFECTS_TABLE_KEY,
        materialization=m__analytics__function_effects,
    )


__all__ = [
    "FunctionContractsResult",
    "FunctionEffectsResult",
    "function_contracts__rows",
    "function_effects__rows",
    "t__function_contracts",
    "t__function_contracts__compute",
    "t__function_effects",
    "t__function_effects__compute",
]
