"""Native Hamilton implementation for semantic_roles target.

This module provides the Hamilton native nodes for semantic roles computation:
- `t__semantic_roles__compute`: Pure compute node for semantic roles
- `t__semantic_roles`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class SemanticRolesResult:
    """Result from semantic roles computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    error
        Error message if computation failed.
    """

    success: bool
    error: str | None = None


@tag(domain="analytics", target="semantic_roles", node_type="compute")
def t__semantic_roles__compute(
    env: BuildEnv,
    t__modules: TargetRunRecord,
    t__function_ast_features: TargetRunRecord,
) -> SemanticRolesResult:
    """Compute semantic roles for functions and modules.

    This is a compute node that calls the semantic roles computation
    which handles both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__modules
        Upstream modules target result (for dependency).
    t__function_ast_features
        Upstream function_ast_features target result (for dependency).

    Returns
    -------
    SemanticRolesResult
        Result indicating success or failure.

    Notes
    -----
    The classifications include:
    - Semantic role (producer, consumer, transformer, etc.)
    - Data flow patterns
    - Behavioral classification
    """
    if t__modules.status != "succeeded":
        return SemanticRolesResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    if t__function_ast_features.status != "succeeded":
        return SemanticRolesResult(
            success=False,
            error=f"Upstream function_ast_features target failed: {t__function_ast_features.error}",
        )

    try:
        # Load catalog for module info
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
            module_by_path = dict(catalog.catalog().module_by_path)
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            module_by_path = {}

        # AST and features maps (currently not loaded from upstream)
        ast_map: dict[int, FunctionAst] = {}
        features_map: dict[int, FunctionAstFeatures] = {}

        # Compute semantic roles (handles persistence internally)
        compute_semantic_roles(
            env.gateway,
            env.snapshot,
            module_by_path=module_by_path,
            ast_map=ast_map,
            features_map=features_map,
        )

        return SemanticRolesResult(success=True)

    except Exception as exc:
        log.exception("Semantic roles computation failed")
        return SemanticRolesResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="semantic_roles", node_type="materialize")
def t__semantic_roles(
    env: BuildEnv,
    graph: TargetGraph,
    t__semantic_roles__compute: SemanticRolesResult,
) -> TargetRunRecord:
    """Materialize semantic roles target.

    This is the entry point for the semantic_roles target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__semantic_roles__compute
        Computed semantic roles result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.semantic_roles_modules
    - analytics.semantic_roles_functions
    """
    executor = NativeTargetExecutor.for_target(env, graph, "semantic_roles")

    if executor.should_skip():
        return executor.skip()

    if not t__semantic_roles__compute.success:
        return executor.fail(
            RuntimeError(t__semantic_roles__compute.error or "Semantic roles failed")
        )

    def compute() -> dict[str, int]:
        # Roles are persisted during compute - return empty counts
        return {
            "analytics.semantic_roles_modules": 0,
            "analytics.semantic_roles_functions": 0,
        }

    return executor.execute(compute)


__all__ = [
    "SemanticRolesResult",
    "t__semantic_roles",
    "t__semantic_roles__compute",
]
