"""Native Hamilton implementation for function_ast_features target.

This module provides the Hamilton native nodes for function AST features:
- `t__function_ast_features__compute`: Pure compute node for AST features
- `t__function_ast_features`: Materialize node that writes the table

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.utilities.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.analytics.utilities.persistence import DeleteScope
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class AstFeaturesResult:
    """Result from AST features computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    features_map
        Mapping of GOID to AST features.
    error
        Error message if computation failed.
    """

    success: bool
    features_map: dict[int, FunctionAstFeatures] = field(default_factory=dict)
    error: str | None = None


@tag(domain="analytics", target="function_ast_features", node_type="compute")
def t__function_ast_features__compute(env: BuildEnv) -> AstFeaturesResult:
    """Compute AST-derived semantic features for functions.

    This is a pure compute node that loads function ASTs and extracts
    semantic features from them.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    AstFeaturesResult
        Result containing features map keyed by GOID.

    Notes
    -----
    The features extracted include:
    - Control flow patterns
    - Statement types and distribution
    - Expression complexity
    """
    try:
        catalog = CatalogService.from_db(
            env.gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
    except (RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return AstFeaturesResult(
            success=False,
            error=f"CatalogProvider is required: {exc}",
        )

    try:
        provider = FeaturesProvider(
            gateway=env.gateway,
            snapshot=env.snapshot,
            catalog_provider=catalog,
        )
        features_map = provider.get()
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("Failed to compute function features: %s", exc)
        return AstFeaturesResult(
            success=True,
            features_map={},
        )

    return AstFeaturesResult(
        success=True,
        features_map=features_map,
    )


@tag(domain="analytics", target="function_ast_features", node_type="materialize")
def t__function_ast_features(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_ast_features__compute: AstFeaturesResult,
) -> TargetRunRecord:
    """Materialize function AST features to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed AST features to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__function_ast_features__compute
        Computed AST features from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following table:
    - analytics.function_ast_features
    """
    executor = NativeTargetExecutor.for_target(env, graph, "function_ast_features")

    if executor.should_skip():
        return executor.skip()

    if not t__function_ast_features__compute.success:
        return executor.fail(
            RuntimeError(t__function_ast_features__compute.error or "AST features failed")
        )

    def compute() -> dict[str, int]:
        features_map = t__function_ast_features__compute.features_map
        if not features_map:
            log.info(
                "No function features computed for %s@%s",
                env.snapshot.repo,
                env.snapshot.commit,
            )
            return {"analytics.function_ast_features": 0}

        rows = [
            features_to_row(
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                features=features,
            )
            for features in features_map.values()
        ]

        contract = get_function_ast_features_contract(env.gateway)
        delete_scope = DeleteScope(repo=env.snapshot.repo, commit=env.snapshot.commit)
        insert_analytics_rows(
            env.gateway,
            contract,
            rows,
            delete_scope=delete_scope,
            scope=f"{env.snapshot.repo}@{env.snapshot.commit}",
        )

        return {"analytics.function_ast_features": len(rows)}

    return executor.execute(compute)


__all__ = [
    "AstFeaturesResult",
    "t__function_ast_features",
    "t__function_ast_features__compute",
]
