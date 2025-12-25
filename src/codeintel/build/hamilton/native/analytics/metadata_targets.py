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
from codeintel.analytics.profiles.files import build_file_profile_rows, compute_file_profile_inputs
from codeintel.analytics.profiles.functions import (
    FunctionProfileViews,
    build_function_profile_rows,
    compute_function_profile_inputs,
    join_function_contracts,
    join_function_coverage,
    join_function_docs,
    join_function_effects,
    join_function_history,
    join_function_risk,
    join_function_roles,
    load_function_base_info,
)
from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.analytics.profiles.modules import (
    build_module_profile_rows,
    compute_module_profile_inputs,
)
from codeintel.analytics.profiles.utils import DEFAULT_MODULE_TABLE, seed_catalog_modules
from codeintel.analytics.resources import ProviderRegistryOptions, build_registry
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.catalog import CatalogProvider
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.resources import ResourceNotFoundError
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.storage.gateway import StorageGateway

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

LOG = logging.getLogger(__name__)
log = LOG

if TYPE_CHECKING:
    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.profiles.types import FunctionProfileInputs


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for analytics metadata nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


@tag_compute(domain="analytics", target=DATA_MODELS_TARGET_NAME)
def t__data_models__compute(
    env: BuildEnv,
    graph: TargetGraph,
    gateway: StorageGateway,
) -> DataModelsResult | None:
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
    gateway
        Storage gateway for analytics queries.

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
            gateway=gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None
    return compute_data_models_pure(gateway, env.snapshot)


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


@codeintel_target(domain="analytics", target=DATA_MODELS_TARGET_NAME)
def t__data_models(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__data_models: MaterializationMetadata,
    m__analytics__data_model_fields: MaterializationMetadata,
    m__analytics__data_model_relationships: MaterializationMetadata,
) -> TargetRunRecord:
    """Extract data models (dataclasses, Pydantic, etc.).

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
    gateway: StorageGateway,
) -> tuple[tuple[object, ...], ...] | None:
    """Compute rows for analytics.data_model_usage.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for manifest-driven skip checks.
    gateway
        Storage gateway for analytics queries.

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
            gateway=gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    registry = build_registry(
        gateway=gateway,
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
        gateway,
        env.snapshot,
        module_map=module_map_provider.module_map,
        ast_by_goid=ast_data.function_ast_map,
        missing_goids=ast_data.missing_function_goids,
    )


@codeintel_target(domain="analytics", target=DATA_MODEL_USAGE_TARGET_NAME)
def t__data_model_usage(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__data_model_usage: MaterializationMetadata,
) -> TargetRunRecord:
    """Track function-level data model usage.

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
def t__function_ast_features__compute(
    env: BuildEnv,
    gateway: StorageGateway,
) -> AstFeaturesResult:
    """Compute AST-derived semantic features for functions.

    Returns
    -------
    AstFeaturesResult
        Feature map and optional error message.
    """
    registry = build_registry(
        gateway=gateway,
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


@codeintel_target(domain="analytics", target=FUNCTION_AST_FEATURES_TARGET_NAME)
def t__function_ast_features(
    env: BuildEnv,
    graph: TargetGraph,
    t__function_ast_features__compute: AstFeaturesResult,
    m__analytics__function_ast_features: MaterializationMetadata,
) -> TargetRunRecord:
    """AST-derived semantic features for functions.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if not t__function_ast_features__compute.success:
        options_hash = options_hash_for_target(env, FUNCTION_AST_FEATURES_TARGET_NAME)
        return TargetRunRecord(
            target=FUNCTION_AST_FEATURES_TARGET_NAME,
            plugin_name=f"native:{FUNCTION_AST_FEATURES_TARGET_NAME}",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__function_ast_features__compute.error or "AST features failed",
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=FUNCTION_AST_FEATURES_TARGET_NAME,
        expected_table_key=FUNCTION_AST_FEATURES_TABLE_KEY,
        materialization=m__analytics__function_ast_features,
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_AST_FEATURES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_AST_FEATURES_TARGET_NAME),
    table_key=value(FUNCTION_AST_FEATURES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FUNCTION_AST_FEATURES_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_AST_FEATURES_TARGET_NAME,
    target_="function_ast_features__rows",
)
def function_ast_features__rows(
    env: BuildEnv,
    t__function_ast_features__compute: AstFeaturesResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.function_ast_features table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if not t__function_ast_features__compute.success:
        return None

    features_map = t__function_ast_features__compute.features_map
    if not features_map:
        log.info(
            "No function features computed for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        return None

    rows = [
        features_to_row(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            features=features,
        )
        for features in features_map.values()
    ]
    return tuple(row_to_tuple(FUNCTION_AST_FEATURES_TABLE_KEY, row) for row in rows)


# ---------------------------------------------------------------------------
# profiles target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfilesComputeResult:
    """Result from profiles computation."""

    module_table: str | None
    error: str | None = None


@dataclass(frozen=True)
class ProfilesMaterializations:
    """Materialization metadata bundle for profile tables."""

    function_profile: MaterializationMetadata
    file_profile: MaterializationMetadata
    module_profile: MaterializationMetadata


def _build_function_profile_views(
    inputs: FunctionProfileInputs,
    module_table: str,
) -> FunctionProfileViews:
    return FunctionProfileViews(
        base_by_func=load_function_base_info(inputs, module_table=module_table),
        risk_by_func=join_function_risk(inputs),
        coverage_by_func=join_function_coverage(inputs),
        graph_by_func=summarize_graph_for_function_profile(inputs),
        effects_by_func=join_function_effects(inputs),
        contracts_by_func=join_function_contracts(inputs),
        roles_by_func=join_function_roles(inputs),
        docs_by_func=join_function_docs(inputs),
        history_by_func=join_function_history(inputs),
    )


@tag_compute(domain="analytics", target=PROFILES_TARGET_NAME)
def t__profiles__compute(
    env: BuildEnv,
    graph: TargetGraph,
    gateway: StorageGateway,
    t__call_graph: TargetRunRecord,
    t__symbol_uses: TargetRunRecord,
) -> ProfilesComputeResult | None:
    """Build aggregated profiles for functions, files, and modules.

    Returns
    -------
    ProfilesComputeResult | None
        Computed module table name or None when skipped.
    """
    if t__call_graph.status != "succeeded":
        return ProfilesComputeResult(
            module_table=None,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    if t__symbol_uses.status != "succeeded":
        return ProfilesComputeResult(
            module_table=None,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )

    target = graph.get(PROFILES_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, PROFILES_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    registry = build_registry(
        gateway=gateway,
        snapshot=env.snapshot,
        registry_options=ProviderRegistryOptions(include_graphs=False),
    )

    try:
        catalog = registry.require(CatalogProvider).get()
    except (ResourceNotFoundError, RuntimeError, ValueError) as exc:
        log.warning("Failed to load catalog: %s", exc)
        return ProfilesComputeResult(
            module_table=None,
            error=f"CatalogProvider is required: {exc}",
        )

    try:
        module_table = seed_catalog_modules(
            gateway,
            catalog,
            env.snapshot.repo,
            env.snapshot.commit,
        )
        return ProfilesComputeResult(module_table=module_table)
    except Exception as exc:
        log.exception("Profiles computation failed")
        return ProfilesComputeResult(module_table=None, error=str(exc))


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_PROFILE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(PROFILES_TARGET_NAME),
    table_key=value(FUNCTION_PROFILE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FUNCTION_PROFILE_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=PROFILES_TARGET_NAME,
    target_="function_profile__rows",
)
def function_profile__rows(
    env: BuildEnv,
    gateway: StorageGateway,
    t__profiles__compute: ProfilesComputeResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.function_profile table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__profiles__compute is None or t__profiles__compute.error:
        return None

    module_table = t__profiles__compute.module_table or DEFAULT_MODULE_TABLE
    inputs = compute_function_profile_inputs(gateway, env.snapshot)
    views = _build_function_profile_views(inputs, module_table)
    rows = build_function_profile_rows(inputs, views=views)
    return tuple(row_to_tuple(FUNCTION_PROFILE_TABLE_KEY, row) for row in rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FILE_PROFILE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(PROFILES_TARGET_NAME),
    table_key=value(FILE_PROFILE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(FILE_PROFILE_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=PROFILES_TARGET_NAME,
    target_="file_profile__rows",
)
def file_profile__rows(
    env: BuildEnv,
    gateway: StorageGateway,
    t__profiles__compute: ProfilesComputeResult | None,
    m__analytics__function_profile: MaterializationMetadata,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.file_profile table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__profiles__compute is None or t__profiles__compute.error:
        return None

    meta = DuckDBMaterializationMetadata.from_mapping(
        m__analytics__function_profile,
        default_table_key=FUNCTION_PROFILE_TABLE_KEY,
    )
    if meta.status != "succeeded":
        return None

    module_table = t__profiles__compute.module_table or DEFAULT_MODULE_TABLE
    inputs = compute_file_profile_inputs(gateway, env.snapshot)
    rows = build_file_profile_rows(inputs, module_table=module_table)
    if rows is None:
        return None
    return tuple(row_to_tuple(FILE_PROFILE_TABLE_KEY, row) for row in rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(MODULE_PROFILE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(PROFILES_TARGET_NAME),
    table_key=value(MODULE_PROFILE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(MODULE_PROFILE_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=PROFILES_TARGET_NAME,
    target_="module_profile__rows",
)
def module_profile__rows(
    env: BuildEnv,
    gateway: StorageGateway,
    t__profiles__compute: ProfilesComputeResult | None,
    m__analytics__file_profile: MaterializationMetadata,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.module_profile table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__profiles__compute is None or t__profiles__compute.error:
        return None

    meta = DuckDBMaterializationMetadata.from_mapping(
        m__analytics__file_profile,
        default_table_key=FILE_PROFILE_TABLE_KEY,
    )
    if meta.status != "succeeded":
        return None

    module_table = t__profiles__compute.module_table or DEFAULT_MODULE_TABLE
    inputs = compute_module_profile_inputs(gateway, env.snapshot)
    rows = build_module_profile_rows(inputs, module_table=module_table)
    if rows is None:
        return None
    return tuple(row_to_tuple(MODULE_PROFILE_TABLE_KEY, row) for row in rows)


@codeintel_target(domain="analytics", target=PROFILES_TARGET_NAME)
def t__profiles(
    env: BuildEnv,
    graph: TargetGraph,
    t__profiles__compute: ProfilesComputeResult | None,
    profiles__materializations: ProfilesMaterializations,
) -> TargetRunRecord:
    """Denormalized profile tables for querying.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if t__profiles__compute is not None and t__profiles__compute.error:
        options_hash = options_hash_for_target(env, PROFILES_TARGET_NAME)
        return TargetRunRecord(
            target=PROFILES_TARGET_NAME,
            plugin_name=f"native:{PROFILES_TARGET_NAME}",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__profiles__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=PROFILES_TARGET_NAME,
        materializations={
            FUNCTION_PROFILE_TABLE_KEY: profiles__materializations.function_profile,
            FILE_PROFILE_TABLE_KEY: profiles__materializations.file_profile,
            MODULE_PROFILE_TABLE_KEY: profiles__materializations.module_profile,
        },
    )


@tag_compute(domain="analytics", target=PROFILES_TARGET_NAME, target_="profiles__materializations")
def profiles__materializations(
    m__analytics__function_profile: MaterializationMetadata,
    m__analytics__file_profile: MaterializationMetadata,
    m__analytics__module_profile: MaterializationMetadata,
) -> ProfilesMaterializations:
    """Bundle profile materialization metadata for the profiles target.

    Returns
    -------
    ProfilesMaterializations
        Grouped metadata for profile table materializations.
    """
    return ProfilesMaterializations(
        function_profile=m__analytics__function_profile,
        file_profile=m__analytics__file_profile,
        module_profile=m__analytics__module_profile,
    )


__all__ = [
    "AstFeaturesResult",
    "ProfilesComputeResult",
    "t__data_model_usage",
    "t__data_model_usage__compute",
    "t__data_models",
    "t__data_models__compute",
    "t__function_ast_features",
    "t__function_ast_features__compute",
    "t__profiles",
    "t__profiles__compute",
]
