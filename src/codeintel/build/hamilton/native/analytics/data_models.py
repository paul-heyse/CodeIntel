"""Native Hamilton implementation for data_models target.

This module provides the Hamilton native nodes for data model extraction:
- `t__data_models__compute`: Pure compute node for data model extraction
- `t__data_models`: Materialize node that writes all 3 tables

The compute node calls pure functions from `codeintel.analytics.data_models.compute`
which return structured result containers. The materialize node uses
`materialize_rows` to persist the data to DuckDB with proper asset tracking.
"""

from __future__ import annotations

import logging

from hamilton.function_modifiers import tag

from codeintel.analytics.compute.data_models import (
    DATA_MODEL_USAGE_COLS,
    build_data_model_usage_rows,
)
from codeintel.analytics.data_models.compute import DataModelsResult, compute_data_models_pure
from codeintel.analytics.data_models.core import (
    DATA_MODEL_FIELDS_COLS,
    DATA_MODEL_RELATIONSHIPS_COLS,
    DATA_MODELS_COLS,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materializer import (
    MaterializationContext,
    materialize_rows,
)
from codeintel.build.targets import TargetGraph
from codeintel.storage.helpers.module_index import load_module_map

_HAMILTON_TYPE_HINTS = (BuildEnv, DataModelsResult, TargetGraph, TargetRunRecord)

LOG = logging.getLogger(__name__)


@tag(domain="analytics", target="data_models", node_type="compute")
def t__data_models__compute(env: BuildEnv) -> DataModelsResult:
    """Compute data models for all classes in the snapshot.

    This is a pure compute node with no side effects. It reads class
    metadata and docstrings from the database and extracts structured
    data model definitions for each class.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    DataModelsResult
        Container with rows for data_models, data_model_fields,
        and data_model_relationships tables.

    Notes
    -----
    The extraction identifies:
    - Dataclasses, Pydantic models, TypedDicts, Protocols
    - Django and SQLAlchemy ORM models
    - Field types, constraints, and defaults
    - Relationships between models
    """
    return compute_data_models_pure(env.gateway, env.snapshot)


@tag(domain="analytics", target="data_models", node_type="materialize")
def t__data_models(
    env: BuildEnv,
    graph: TargetGraph,
    t__data_models__compute: DataModelsResult,
) -> TargetRunRecord:
    """Materialize all 3 data model tables to DuckDB.

    This is the only side-effect boundary for this target. It writes
    the computed data models to DuckDB and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__data_models__compute
        Computed data models from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.data_models
    - analytics.data_model_fields
    - analytics.data_model_relationships
    """
    executor = NativeTargetExecutor.for_target(env, graph, "data_models")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        # Ensure tables exist
        backend = env.gateway.policy
        backend.ensure_table("analytics.data_models")
        backend.ensure_table("analytics.data_model_fields")
        backend.ensure_table("analytics.data_model_relationships")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="data_models",
            input_hash=executor.input_hash,
        )

        row_counts: dict[str, int] = {}

        # Materialize all 3 tables
        result = t__data_models__compute
        tables = [
            ("analytics.data_models", result.model_rows, DATA_MODELS_COLS),
            ("analytics.data_model_fields", result.field_rows, DATA_MODEL_FIELDS_COLS),
            (
                "analytics.data_model_relationships",
                result.relationship_rows,
                DATA_MODEL_RELATIONSHIPS_COLS,
            ),
        ]
        for table_key, rows, cols in tables:
            ref = materialize_rows(ctx, table_key, rows, cols)
            row_counts[table_key] = ref.row_count or 0

        return row_counts

    return executor.execute(compute)


@tag(domain="analytics", target="data_model_usage", node_type="compute")
def t__data_model_usage__compute(env: BuildEnv) -> tuple[tuple[object, ...], ...]:
    """Compute rows for analytics.data_model_usage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Row tuples for analytics.data_model_usage in DATA_MODEL_USAGE_COLS order.
    """
    module_map = load_module_map(
        env.gateway,
        repo=env.snapshot.repo,
        commit=env.snapshot.commit,
        logger=LOG,
    )

    ast_by_goid, missing_goids = load_function_asts(
        env.gateway,
        FunctionAstLoadRequest(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            repo_root=env.snapshot.repo_root,
        ),
    )

    return build_data_model_usage_rows(
        env.gateway,
        env.snapshot,
        module_map=module_map,
        ast_by_goid=ast_by_goid,
        missing_goids=missing_goids,
    )


@tag(domain="analytics", target="data_model_usage", node_type="materialize")
def t__data_model_usage(
    env: BuildEnv,
    graph: TargetGraph,
    t__data_model_usage__compute: tuple[tuple[object, ...], ...],
) -> TargetRunRecord:
    """Materialize analytics.data_model_usage rows to DuckDB.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__data_model_usage__compute
        Computed row tuples for analytics.data_model_usage.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "data_model_usage")

    if executor.should_skip():
        return executor.skip()

    def compute() -> dict[str, int]:
        backend = env.gateway.policy
        backend.ensure_table("analytics.data_model_usage")

        ctx = MaterializationContext(
            gateway=env.gateway,
            snapshot=env.snapshot,
            validate=env.validate_outputs,
            owner_target="data_model_usage",
            input_hash=executor.input_hash,
        )

        ref = materialize_rows(
            ctx,
            "analytics.data_model_usage",
            t__data_model_usage__compute,
            DATA_MODEL_USAGE_COLS,
        )
        return {ref.table_key: ref.row_count or 0}

    return executor.execute(compute)


# Export node names for Hamilton discovery
__all__ = [
    "t__data_model_usage",
    "t__data_model_usage__compute",
    "t__data_models",
    "t__data_models__compute",
]
