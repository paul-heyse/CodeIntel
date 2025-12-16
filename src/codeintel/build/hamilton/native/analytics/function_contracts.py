"""Native Hamilton implementation for function_contracts target.

This module provides the Hamilton native nodes for function contracts inference:
- `t__function_contracts__compute`: Pure compute node for contracts inference
- `t__function_contracts`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.functions import compute_function_contracts
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class FunctionContractsResult:
    """Result from function contracts computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    error
        Error message if computation failed.
    """

    success: bool
    error: str | None = None


@tag(domain="analytics", target="function_contracts", node_type="compute")
def t__function_contracts__compute(
    env: BuildEnv,
    t__goids: TargetRunRecord,
) -> FunctionContractsResult:
    """Compute pre/postconditions and nullability contracts for functions.

    This is a compute node that calls the function contracts computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__goids
        Upstream goids target result (for dependency).

    Returns
    -------
    FunctionContractsResult
        Result indicating success or failure.

    Notes
    -----
    The contracts inferred include:
    - Preconditions (required input states)
    - Postconditions (guaranteed output states)
    - Nullability contracts
    """
    if t__goids.status != "succeeded":
        return FunctionContractsResult(
            success=False,
            error=f"Upstream goids target failed: {t__goids.error}",
        )

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
                success=False,
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

        # Compute contracts (handles persistence internally)
        compute_function_contracts(
            env.gateway,
            env.snapshot,
            function_ast_map=function_ast_map,
            catalog=catalog,
        )

        return FunctionContractsResult(success=True)

    except Exception as exc:
        log.exception("Function contracts computation failed")
        return FunctionContractsResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="function_contracts", node_type="materialize")
def t__function_contracts(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_contracts__compute: FunctionContractsResult,
) -> TargetRunRecord:
    """Materialize function contracts target.

    This is the entry point for the function_contracts target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__function_contracts__compute
        Computed function contracts result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.function_contracts
    """
    executor = NativeTargetExecutor.for_target(env, graph, "function_contracts")

    if executor.should_skip():
        return executor.skip()

    if not t__function_contracts__compute.success:
        return executor.fail(
            RuntimeError(t__function_contracts__compute.error or "Function contracts failed")
        )

    def compute() -> dict[str, int]:
        # Contracts are persisted during compute - return empty count
        return {"analytics.function_contracts": 0}

    return executor.execute(compute)


__all__ = [
    "FunctionContractsResult",
    "t__function_contracts",
    "t__function_contracts__compute",
]
