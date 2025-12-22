"""Native Hamilton implementations for metadata-oriented analytics targets.

This module consolidates targets that extract metadata about code structure:

- ``data_models`` / ``data_model_usage``: Data model extraction + usage analytics.
- ``function_ast_features``: AST-derived function features.
- ``profiles``: Aggregated profiles for functions/files/modules.

The compute node calls pure functions from `codeintel.analytics.data_models.compute`
which return structured result containers. The materialize node uses
`materialize_rows` to persist the data to DuckDB with proper asset tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.compute.data_models import build_data_model_usage_rows
from codeintel.analytics.data_models.compute import DataModelsResult, compute_data_models_pure
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.analytics.utilities.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.analytics.utilities.persistence import DeleteScope
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_override_tables import (
    DATA_MODEL_USAGE_OVERRIDE_TABLES,
    DATA_MODELS_OVERRIDE_TABLES,
    FUNCTION_AST_FEATURES_OVERRIDE_TABLES,
    PROFILES_OVERRIDE_TABLES,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
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

_HAMILTON_TYPE_HINTS = (BuildEnv, DataModelsResult, TargetGraph, TargetRunRecord)

DATA_MODELS_TARGET_NAME = "data_models"
DATA_MODEL_USAGE_TARGET_NAME = "data_model_usage"
FUNCTION_AST_FEATURES_TARGET_NAME = "function_ast_features"
PROFILES_TARGET_NAME = "profiles"

DATA_MODELS_TABLE_KEY = "analytics.data_models"
DATA_MODEL_FIELDS_TABLE_KEY = "analytics.data_model_fields"
DATA_MODEL_RELATIONSHIPS_TABLE_KEY = "analytics.data_model_relationships"
DATA_MODELS_TABLE_KEYS = (
    DATA_MODELS_TABLE_KEY,
    DATA_MODEL_FIELDS_TABLE_KEY,
    DATA_MODEL_RELATIONSHIPS_TABLE_KEY,
)

DATA_MODEL_USAGE_TABLE_KEY = "analytics.data_model_usage"
FUNCTION_AST_FEATURES_TABLE_KEY = "analytics.function_ast_features"

FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
FILE_PROFILE_TABLE_KEY = "analytics.file_profile"
MODULE_PROFILE_TABLE_KEY = "analytics.module_profile"
PROFILES_TABLE_KEYS = (
    FUNCTION_PROFILE_TABLE_KEY,
    FILE_PROFILE_TABLE_KEY,
    MODULE_PROFILE_TABLE_KEY,
)

register_output_targets(
    make_output_target(
        name=DATA_MODELS_TARGET_NAME,
        module="analytics",
        description="Data model extraction (dataclasses, Pydantic, etc.).",
        options=TargetSpecOptions(
            table_keys=DATA_MODELS_TABLE_KEYS,
            override_tables=DATA_MODELS_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=DATA_MODEL_USAGE_TARGET_NAME,
        module="analytics",
        description="Function-level data model usage tracking.",
        options=TargetSpecOptions(
            table_keys=(DATA_MODEL_USAGE_TABLE_KEY,),
            override_tables=DATA_MODEL_USAGE_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=FUNCTION_AST_FEATURES_TARGET_NAME,
        module="analytics",
        description="AST-derived semantic features for functions.",
        options=TargetSpecOptions(
            table_keys=(FUNCTION_AST_FEATURES_TABLE_KEY,),
            override_tables=FUNCTION_AST_FEATURES_OVERRIDE_TABLES,
        ),
    ),
    make_output_target(
        name=PROFILES_TARGET_NAME,
        module="analytics",
        description="Denormalized profile tables for querying.",
        options=TargetSpecOptions(
            table_keys=PROFILES_TABLE_KEYS,
            override_tables=PROFILES_OVERRIDE_TABLES,
        ),
    ),
)

LOG = logging.getLogger(__name__)
log = LOG

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures


@tag_compute(domain="analytics", target=DATA_MODELS_TARGET_NAME)
def t__data_models__compute(env: BuildEnv, graph: TargetGraph) -> DataModelsResult | None:
    """Compute data models for all classes in the snapshot.

    This is a pure compute node with no side effects. It reads class
    metadata and docstrings from the database and extracts structured
    data model definitions for each class.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for manifest-driven skip checks.

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
    target = graph.get(DATA_MODELS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, DATA_MODELS_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return compute_data_models_pure(env.gateway, env.snapshot)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(DATA_MODELS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DATA_MODELS_TARGET_NAME),
    table_key=value(DATA_MODELS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(DATA_MODELS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    target_="data_models__model_rows",
)
def data_models__model_rows(
    t__data_models__compute: DataModelsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.data_models.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when skipped.
    """
    if t__data_models__compute is None:
        return None
    return tuple(t__data_models__compute.model_rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(DATA_MODEL_FIELDS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DATA_MODELS_TARGET_NAME),
    table_key=value(DATA_MODEL_FIELDS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(DATA_MODEL_FIELDS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    target_="data_models__field_rows",
)
def data_models__field_rows(
    t__data_models__compute: DataModelsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.data_model_fields.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when skipped.
    """
    if t__data_models__compute is None:
        return None
    return tuple(t__data_models__compute.field_rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(DATA_MODEL_RELATIONSHIPS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DATA_MODELS_TARGET_NAME),
    table_key=value(DATA_MODEL_RELATIONSHIPS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(DATA_MODEL_RELATIONSHIPS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=DATA_MODELS_TARGET_NAME,
    target_="data_models__relationship_rows",
)
def data_models__relationship_rows(
    t__data_models__compute: DataModelsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.data_model_relationships.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples to materialize, or None when skipped.
    """
    if t__data_models__compute is None:
        return None
    return tuple(t__data_models__compute.relationship_rows)


@tag_materialize(domain="analytics", target=DATA_MODELS_TARGET_NAME)
def t__data_models(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__data_models: MaterializationMetadata,
    m__analytics__data_model_fields: MaterializationMetadata,
    m__analytics__data_model_relationships: MaterializationMetadata,
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
    m__analytics__data_models
        Materialization metadata for analytics.data_models.
    m__analytics__data_model_fields
        Materialization metadata for analytics.data_model_fields.
    m__analytics__data_model_relationships
        Materialization metadata for analytics.data_model_relationships.

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
        target_name=DATA_MODELS_TARGET_NAME,
        materializations={
            DATA_MODELS_TABLE_KEY: m__analytics__data_models,
            DATA_MODEL_FIELDS_TABLE_KEY: m__analytics__data_model_fields,
            DATA_MODEL_RELATIONSHIPS_TABLE_KEY: m__analytics__data_model_relationships,
        },
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(DATA_MODEL_USAGE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DATA_MODEL_USAGE_TARGET_NAME),
    table_key=value(DATA_MODEL_USAGE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(DATA_MODEL_USAGE_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=DATA_MODEL_USAGE_TARGET_NAME,
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
    graph
        Target graph for manifest-driven skip checks.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for analytics.data_model_usage in schema order.
        Returns None when manifest-skip indicates the target is current.
    """
    target = graph.get(DATA_MODEL_USAGE_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, DATA_MODEL_USAGE_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(
            include_graphs=False,
            include_asts=True,
            include_module_map=True,
        ),
    )
    module_map_provider = registry.require(ModuleMapProvider)
    ast_data = registry.require(AstProvider).get()

    return build_data_model_usage_rows(
        env.gateway,
        env.snapshot,
        module_map=module_map_provider.module_map,
        ast_by_goid=ast_data.function_ast_map,
        missing_goids=ast_data.missing_function_goids,
    )


@tag_materialize(domain="analytics", target=DATA_MODEL_USAGE_TARGET_NAME)
def t__data_model_usage(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__data_model_usage: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize analytics.data_model_usage rows to DuckDB.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__data_model_usage
        Materialization metadata for analytics.data_model_usage.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=DATA_MODEL_USAGE_TARGET_NAME,
        expected_table_key=DATA_MODEL_USAGE_TABLE_KEY,
        materialization=m__analytics__data_model_usage,
    )


# ---------------------------------------------------------------------------
# function_ast_features target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AstFeaturesResult:
    """Result from AST features computation."""

    success: bool
    features_map: dict[int, FunctionAstFeatures] = field(default_factory=dict)
    error: str | None = None


@tag_compute(domain="analytics", target=FUNCTION_AST_FEATURES_TARGET_NAME)
def t__function_ast_features__compute(env: BuildEnv) -> AstFeaturesResult:
    """Compute AST-derived semantic features for functions.

    Returns
    -------
    AstFeaturesResult
        Feature map and optional error message.
    """
    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(
            include_graphs=False,
            include_features=True,
        ),
    )

    try:
        _ = registry.require(CatalogProvider).get()
    except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return AstFeaturesResult(success=False, error=f"CatalogProvider is required: {exc}")

    try:
        features_map = registry.require(FeaturesProvider).get()
    except (RuntimeError, ValueError, OSError) as exc:
        log.warning("Failed to compute function features: %s", exc)
        return AstFeaturesResult(success=True, features_map={})

    return AstFeaturesResult(success=True, features_map=features_map)


@tag_materialize(domain="analytics", target=FUNCTION_AST_FEATURES_TARGET_NAME)
def t__function_ast_features(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_ast_features__compute: AstFeaturesResult,
) -> TargetRunRecord:
    """Materialize function AST features to DuckDB.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, FUNCTION_AST_FEATURES_TARGET_NAME)

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
            return {FUNCTION_AST_FEATURES_TABLE_KEY: 0}

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

        return {FUNCTION_AST_FEATURES_TABLE_KEY: len(rows)}

    return executor.execute(compute)


# ---------------------------------------------------------------------------
# profiles target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfilesResult:
    """Result from profiles computation."""

    success: bool
    error: str | None = None


@tag_compute(domain="analytics", target=PROFILES_TARGET_NAME)
def t__profiles__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__symbol_uses: TargetRunRecord,
) -> ProfilesResult:
    """Build aggregated profiles for functions, files, and modules.

    Returns
    -------
    ProfilesResult
        Status indicator and optional error message.
    """
    if t__call_graph.status != "succeeded":
        return ProfilesResult(
            success=False,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    if t__symbol_uses.status != "succeeded":
        return ProfilesResult(
            success=False,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )

    registry = build_registry(
        gateway=env.gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        catalog = registry.require(CatalogProvider).get()
    except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return ProfilesResult(success=False, error=f"CatalogProvider is required: {exc}")

    try:
        build_function_profile(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
            module_map=None,
        )
        build_file_profile(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
        )
        build_module_profile(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
        )

        return ProfilesResult(success=True)
    except Exception as exc:
        log.exception("Profiles computation failed")
        return ProfilesResult(success=False, error=str(exc))


@tag_materialize(domain="analytics", target=PROFILES_TARGET_NAME)
def t__profiles(
    env: BuildEnv,
    graph: TargetGraph,
    t__profiles__compute: ProfilesResult,
) -> TargetRunRecord:
    """Materialize profiles target.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, PROFILES_TARGET_NAME)

    if executor.should_skip():
        return executor.skip()

    if not t__profiles__compute.success:
        return executor.fail(RuntimeError(t__profiles__compute.error or "Profiles failed"))

    def compute() -> dict[str, int]:
        return {
            FUNCTION_PROFILE_TABLE_KEY: 0,
            FILE_PROFILE_TABLE_KEY: 0,
            MODULE_PROFILE_TABLE_KEY: 0,
        }

    return executor.execute(compute)


__all__ = [
    "AstFeaturesResult",
    "ProfilesResult",
    "t__data_model_usage",
    "t__data_model_usage__compute",
    "t__data_models",
    "t__data_models__compute",
    "t__function_ast_features",
    "t__function_ast_features__compute",
    "t__profiles",
    "t__profiles__compute",
]
