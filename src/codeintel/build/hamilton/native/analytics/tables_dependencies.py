"""Dependency analytics tables built with relation-first nodes."""

from __future__ import annotations

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_relation_for_table
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_relation_table,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.storage.gateway import DuckDBRelation

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, DuckDBRelation)

EXTERNAL_DEPS_TARGET_NAME = "external_deps"
EXTERNAL_DEPENDENCIES_TABLE_KEY = "analytics.external_dependencies"
EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY = "analytics.external_dependency_calls"
EXTERNAL_DEPS_TABLE_KEYS = (
    EXTERNAL_DEPENDENCIES_TABLE_KEY,
    EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
)
EXTERNAL_DEPS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
)


@save_relation_table(
    context=EXTERNAL_DEPS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=EXTERNAL_DEPENDENCIES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
    table_key=EXTERNAL_DEPENDENCIES_TABLE_KEY,
)
def external_deps__table(env: BuildEnv) -> DuckDBRelation:
    """Return an empty external dependencies relation.

    Returns
    -------
    DuckDBRelation
        Empty relation with the external dependencies schema.
    """
    return empty_relation_for_table(env.gateway.con, EXTERNAL_DEPENDENCIES_TABLE_KEY)


@save_relation_table(
    context=EXTERNAL_DEPS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
    table_key=EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
)
def external_deps__calls_table(env: BuildEnv) -> DuckDBRelation:
    """Return an empty external dependency calls relation.

    Returns
    -------
    DuckDBRelation
        Empty relation with the external dependency calls schema.
    """
    return empty_relation_for_table(env.gateway.con, EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY)


external_deps__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
    table_keys=EXTERNAL_DEPS_TABLE_KEYS,
    node_name="external_deps__table_materializations",
)


@codeintel_target(domain="analytics", target=EXTERNAL_DEPS_TARGET_NAME)
def t__external_deps(
    env: BuildEnv,
    catalog: DagCatalog,
    external_deps__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize external_deps target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the external_deps target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=EXTERNAL_DEPS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=external_deps__table_materializations,
    )


__all__ = [
    "external_deps__calls_table",
    "external_deps__table",
    "external_deps__table_materializations",
    "t__external_deps",
]
