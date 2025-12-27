"""Native Hamilton implementations for function-level detail targets.

This module consolidates function-detail analytics targets that produce row
tables in DuckDB via DAG-visible I/O (SaveToDecorator/DuckDBRowsSaver):

- ``function_contracts``: inferred pre/postconditions and nullability contracts
- ``function_effects``: purity and side-effect classification

Each target follows the same pattern:

1. A pure compute node returning row dicts (no persistence)
2. A SaveToDecorator node that materializes rows to DuckDB
3. A materialize node that converts materialization results to TargetRunRecord

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codeintel.analytics.functions.function_contracts import build_function_contracts_rows
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    FunctionEffectsOptions,
    build_function_effects_rows,
)
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    SaverContext,
    TableSaveSpec,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
)
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hashing import InputHashOptions
from codeintel.core.resources import ResourceNotFoundError
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.graphs.runtime import resolve_graph_runtime
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord)

FUNCTION_CONTRACTS_TARGET_NAME = "function_contracts"
FUNCTION_EFFECTS_TARGET_NAME = "function_effects"

FUNCTION_CONTRACTS_TABLE_KEY = "analytics.function_contracts"
FUNCTION_EFFECTS_TABLE_KEY = "analytics.function_effects"
FUNCTION_CONTRACTS_TABLE_KEYS = (FUNCTION_CONTRACTS_TABLE_KEY,)
FUNCTION_EFFECTS_TABLE_KEYS = (FUNCTION_EFFECTS_TABLE_KEY,)
FUNCTION_CONTRACTS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_CONTRACTS_TARGET_NAME,
    hash_options_node="function_contracts__hash_options",
)
FUNCTION_EFFECTS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_EFFECTS_TARGET_NAME,
    hash_options_node="function_effects__hash_options",
)


@tag_helper(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
def function_contracts__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for function_contracts execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, FUNCTION_CONTRACTS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
def function_contracts__skip(
    env: BuildEnv,
    catalog: DagCatalog,
    function_contracts__hash_options: InputHashOptions,
) -> bool:
    """Return True when function_contracts should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        catalog,
        FUNCTION_CONTRACTS_TARGET_NAME,
        hash_options=function_contracts__hash_options,
    )
    return executor.should_skip()


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for function detail nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


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
    gateway: StorageGateway,
    t__goids: TargetRunRecord,
    *,
    function_contracts__skip: bool,
) -> FunctionContractsResult:
    """Compute pre/postconditions and nullability contracts for functions.

    This is a pure compute node that returns rows without persistence.
    The actual DB writes are handled by downstream SaveToDecorator nodes.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__goids
        Upstream goids target result (for dependency).
    function_contracts__skip
        Skip flag derived from manifest-based input hash evaluation.

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

    if function_contracts__skip:
        return FunctionContractsResult(rows=None)

    try:
        registry = build_registry(
            gateway=gateway,
            snapshot=env.snapshot,
            registry_options=ProviderRegistryOptions(
                include_graphs=False,
                include_asts=True,
            ),
        )

        # Load catalog
        try:
            catalog = registry.require(CatalogProvider)
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            return FunctionContractsResult(
                rows=None,
                error=f"CatalogProvider is required: {exc}",
            )

        # Load function ASTs
        try:
            ast_data = registry.require(AstProvider)
            function_ast_map = ast_data.function_ast_map
        except (RuntimeError, ValueError, OSError) as exc:
            log.warning("Failed to load function ASTs: %s", exc)
            function_ast_map = {}

        # Compute contracts (pure compute - no persistence)
        rows = build_function_contracts_rows(
            gateway,
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


@save_rows(
    context=FUNCTION_CONTRACTS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FUNCTION_CONTRACTS_TABLE_KEY),
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


@codeintel_target(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
def t__function_contracts(
    env: BuildEnv,
    catalog: DagCatalog,
    function_contracts__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Infer function pre/postconditions.

    Converts materialization results into a TargetRunRecord for the
    function_contracts target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    function_contracts__table_materializations
        Materialization results keyed by table name.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=FUNCTION_CONTRACTS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=function_contracts__table_materializations,
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
    gateway: StorageGateway,
    t__call_graph: TargetRunRecord,
    *,
    function_effects__skip: bool,
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

    if function_effects__skip:
        return FunctionEffectsResult(rows=None)

    registry = build_registry(
        gateway=gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        try:
            catalog = registry.require(CatalogProvider)
        except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        try:
            graph_runtime = resolve_graph_runtime(
                gateway,
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
            gateway,
            env.snapshot,
            options=opts,
            inputs=inputs,
        )
        return FunctionEffectsResult(rows=rows)

    except Exception as exc:
        log.exception("Function effects computation failed")
        return FunctionEffectsResult(rows=None, error=str(exc))


@save_rows(
    context=FUNCTION_EFFECTS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FUNCTION_EFFECTS_TABLE_KEY),
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


@codeintel_target(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
def t__function_effects(
    env: BuildEnv,
    catalog: DagCatalog,
    function_effects__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Analyze function purity and side effects.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=FUNCTION_EFFECTS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=function_effects__table_materializations,
    )


function_effects__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=FUNCTION_EFFECTS_TARGET_NAME,
    table_keys=FUNCTION_EFFECTS_TABLE_KEYS,
)


__all__ = [
    "FunctionContractsResult",
    "FunctionEffectsResult",
    "function_contracts__hash_options",
    "function_contracts__rows",
    "function_contracts__skip",
    "function_contracts__table_materializations",
    "function_effects__hash_options",
    "function_effects__rows",
    "function_effects__skip",
    "function_effects__table_materializations",
    "t__function_contracts",
    "t__function_contracts__compute",
    "t__function_effects",
    "t__function_effects__compute",
]
function_contracts__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=FUNCTION_CONTRACTS_TARGET_NAME,
    table_keys=FUNCTION_CONTRACTS_TABLE_KEYS,
)


# ---------------------------------------------------------------------------
# function_effects target
# ---------------------------------------------------------------------------


@tag_helper(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
def function_effects__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for function_effects execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, FUNCTION_EFFECTS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
def function_effects__skip(
    env: BuildEnv,
    catalog: DagCatalog,
    function_effects__hash_options: InputHashOptions,
) -> bool:
    """Return True when function_effects should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        catalog,
        FUNCTION_EFFECTS_TARGET_NAME,
        hash_options=function_effects__hash_options,
    )
    return executor.should_skip()
