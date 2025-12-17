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
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

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
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.runner import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.storage.helpers.module_index import load_module_map

_HAMILTON_TYPE_HINTS = (BuildEnv, DataModelsResult, TargetGraph, TargetRunRecord)

LOG = logging.getLogger(__name__)


@tag(domain="analytics", target="data_models", node_type="compute")
def t__data_models__compute(env: BuildEnv, graph: TargetGraph) -> DataModelsResult | None:
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
    DataModelsResult | None
        Container with rows for data_models, data_model_fields,
        and data_model_relationships tables.
        Returns None when manifest-skip indicates the target is current.

    Notes
    -----
    The extraction identifies:
    - Dataclasses, Pydantic models, TypedDicts, Protocols
    - Django and SQLAlchemy ORM models
    - Field types, constraints, and defaults
    - Relationships between models
    """
    target = graph.get("data_models")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return compute_data_models_pure(env.gateway, env.snapshot)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.data_models"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("data_models"),
    table_key=value("analytics.data_models"),
    columns=value(tuple(DATA_MODELS_COLS)),
)
@tag(domain="analytics", target="data_models", node_type="compute", target_="data_models__model_rows")
def data_models__model_rows(
    t__data_models__compute: DataModelsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.data_models."""
    if t__data_models__compute is None:
        return None
    return tuple(t__data_models__compute.model_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.data_model_fields"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("data_models"),
    table_key=value("analytics.data_model_fields"),
    columns=value(tuple(DATA_MODEL_FIELDS_COLS)),
)
@tag(domain="analytics", target="data_models", node_type="compute", target_="data_models__field_rows")
def data_models__field_rows(
    t__data_models__compute: DataModelsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.data_model_fields."""
    if t__data_models__compute is None:
        return None
    return tuple(t__data_models__compute.field_rows)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.data_model_relationships"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("data_models"),
    table_key=value("analytics.data_model_relationships"),
    columns=value(tuple(DATA_MODEL_RELATIONSHIPS_COLS)),
)
@tag(
    domain="analytics",
    target="data_models",
    node_type="compute",
    target_="data_models__relationship_rows",
)
def data_models__relationship_rows(
    t__data_models__compute: DataModelsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.data_model_relationships."""
    if t__data_models__compute is None:
        return None
    return tuple(t__data_models__compute.relationship_rows)


@tag(domain="analytics", target="data_models", node_type="materialize")
def t__data_models(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__data_models: dict[str, Any],
    m__analytics__data_model_fields: dict[str, Any],
    m__analytics__data_model_relationships: dict[str, Any],
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
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name="data_models",
        materializations={
            "analytics.data_models": m__analytics__data_models,
            "analytics.data_model_fields": m__analytics__data_model_fields,
            "analytics.data_model_relationships": m__analytics__data_model_relationships,
        },
    )


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node("analytics.data_model_usage"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("data_model_usage"),
    table_key=value("analytics.data_model_usage"),
    columns=value(tuple(DATA_MODEL_USAGE_COLS)),
)
@tag(
    domain="analytics",
    target="data_model_usage",
    node_type="compute",
    target_="t__data_model_usage__compute",
)
def t__data_model_usage__compute(
    env: BuildEnv,
    graph: TargetGraph,
) -> tuple[tuple[object, ...], ...] | None:
    """Compute rows for analytics.data_model_usage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for analytics.data_model_usage in DATA_MODEL_USAGE_COLS order.
        Returns None when manifest-skip indicates the target is current.
    """
    target = graph.get("data_model_usage")
    if target is not None:
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=None,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

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
    m__analytics__data_model_usage: dict[str, Any],
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
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="data_model_usage",
        expected_table_key="analytics.data_model_usage",
        materialization=m__analytics__data_model_usage,
    )


# Export node names for Hamilton discovery
__all__ = [
    "t__data_model_usage",
    "t__data_model_usage__compute",
    "t__data_models",
    "t__data_models__compute",
]
