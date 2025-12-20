"""Target-based table relationship extraction for dataset introspection.

This module originally extracted constraints from plugin metadata. The build
system has moved to a Hamilton-first architecture where targets and their
contracts are the source of truth.

We preserve the public API (`get_producer_plugins`, `get_consumer_plugins`, ...)
but implement it in terms of build targets:

- Producers are targets whose contracts produce the table.
- Consumers are targets whose (Hamilton-derived) dependencies include at least
  one target that produces the table.

Architecture Reference: Section 5.4.1 - Implement ConstraintSet aggregation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.engine_version import get_build_engine_version
from codeintel.build.hamilton.contracts.schemas.constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
)
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    ...

__all__ = [
    "PluginTableRelation",
    "extract_constraints_from_plugins",
    "get_consumer_plugins",
    "get_producer_plugins",
    "get_table_plugin_relations",
    "merge_constraint_sets",
]

log = logging.getLogger(__name__)

_DOMAIN_BY_MODULE: dict[str, PluginDomain] = {
    "ingestion": PluginDomain.INGEST,
    "graphs": PluginDomain.GRAPH,
    "analytics": PluginDomain.ANALYTICS,
    "export": PluginDomain.EXPORT,
}


@lru_cache(maxsize=1)
def _get_all_plugins_metadata() -> dict[str, CorePluginMetadata]:
    """Build target-derived "plugin metadata" for dataset relationships.

    Returns
    -------
    dict[str, CorePluginMetadata]
        Mapping of target name to core metadata.
    """
    result: dict[str, CorePluginMetadata] = {}
    graph = get_target_metadata_service().system.graph
    build_version = get_build_engine_version()

    for target in graph.all_targets:
        consumed: set[str] = set()
        for dep_name in graph.dependencies_of(target.name):
            try:
                dep_target = graph.get(dep_name)
            except KeyError:
                continue
            consumed.update(dep_target.contract.table_keys)

        domain = _DOMAIN_BY_MODULE.get(target.module, PluginDomain.CLI)
        result[target.name] = CorePluginMetadata(
            name=f"{target.module}.{target.name}",
            version=build_version,
            description=target.description or "",
            domain=domain,
            kind="build_target",
            stage=target.module,
            provides=(),
            requires=(),
            produces_tables=tuple(sorted(target.contract.table_keys)),
            consumes_tables=tuple(sorted(consumed)),
            supports_incremental=True,
            scope_aware=False,
        )

    return result


@dataclass(frozen=True)
class PluginTableRelation:
    """A relationship between a plugin and a table.

    Parameters
    ----------
    plugin_name
        Name of the plugin.
    plugin_version
        Version of the plugin.
    table_key
        Fully qualified table name.
    relation_type
        Either "produces" or "consumes".
    domain
        Plugin domain (analytics, graph, ingest, etc.).

    Examples
    --------
    >>> rel = PluginTableRelation(
    ...     plugin_name="analytics.function_metrics",
    ...     plugin_version="3.0.0",
    ...     table_key="analytics.function_metrics",
    ...     relation_type="produces",
    ...     domain="analytics",
    ... )
    >>> rel.is_producer
    True
    """

    plugin_name: str
    plugin_version: str
    table_key: str
    relation_type: str
    domain: str

    @property
    def is_producer(self) -> bool:
        """Check if this is a producer relationship.

        Returns
        -------
        bool
            True if the plugin produces this table.
        """
        return self.relation_type == "produces"

    @property
    def is_consumer(self) -> bool:
        """Check if this is a consumer relationship.

        Returns
        -------
        bool
            True if the plugin consumes this table.
        """
        return self.relation_type == "consumes"


def get_producer_plugins(table_key: str) -> list[CorePluginMetadata]:
    """Find plugins that produce the given dataset.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    list[CorePluginMetadata]
        Plugin metadata for all producers of this table.
    """
    result: list[CorePluginMetadata] = []
    for meta in _get_all_plugins_metadata().values():
        produces = getattr(meta, "produces_tables", None)
        if produces and table_key in produces:
            result.append(meta)
    return result


def get_consumer_plugins(table_key: str) -> list[CorePluginMetadata]:
    """Find plugins that consume the given dataset.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    list[CorePluginMetadata]
        Plugin metadata for all consumers of this table.
    """
    result: list[CorePluginMetadata] = []
    for meta in _get_all_plugins_metadata().values():
        consumes = getattr(meta, "consumes_tables", None)
        if consumes and table_key in consumes:
            result.append(meta)
    return result


def _get_domain_str(meta: CorePluginMetadata) -> str:
    """Get domain string from plugin metadata.

    Parameters
    ----------
    meta
        Plugin metadata.

    Returns
    -------
    str
        Domain as string.
    """
    return str(meta.domain.value) if hasattr(meta.domain, "value") else str(meta.domain)


def get_table_plugin_relations(table_key: str) -> list[PluginTableRelation]:
    """Get all plugin relationships for a table.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    list[PluginTableRelation]
        All producer and consumer relationships for this table.
    """
    producer_relations = [
        PluginTableRelation(
            plugin_name=meta.name,
            plugin_version=meta.version,
            table_key=table_key,
            relation_type="produces",
            domain=_get_domain_str(meta),
        )
        for meta in get_producer_plugins(table_key)
    ]

    consumer_relations = [
        PluginTableRelation(
            plugin_name=meta.name,
            plugin_version=meta.version,
            table_key=table_key,
            relation_type="consumes",
            domain=_get_domain_str(meta),
        )
        for meta in get_consumer_plugins(table_key)
    ]

    return producer_relations + consumer_relations


def extract_constraints_from_plugins(table_key: str) -> ConstraintSet:
    """Extract constraints from plugin metadata for a dataset.

    This function examines all plugins that produce or consume the given
    table and generates COMPUTATION constraints representing these
    dependencies. These constraints enable tracing data flow through
    the system.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    ConstraintSet
        Constraints derived from plugin producer/consumer relationships.

    Examples
    --------
    >>> cs = extract_constraints_from_plugins("analytics.function_metrics")
    >>> isinstance(cs, ConstraintSet)
    True
    """
    cs = ConstraintSet(table_key=table_key)

    producers = get_producer_plugins(table_key)
    for meta in producers:
        cs.add(
            Constraint(
                kind=ConstraintKind.COMPUTATION,
                column=None,
                expression=f"produced_by({meta.name})",
                source="plugin.produces_tables",
                description=f"Table is produced by plugin {meta.name} v{meta.version}",
            )
        )

    consumers = get_consumer_plugins(table_key)
    for meta in consumers:
        cs.add(
            Constraint(
                kind=ConstraintKind.COMPUTATION,
                column=None,
                expression=f"consumed_by({meta.name})",
                source="plugin.consumes_tables",
                description=f"Table is consumed by plugin {meta.name} v{meta.version}",
            )
        )

    for meta in producers:
        if meta.consumes_tables:
            for consumed_table in meta.consumes_tables:
                cs.add(
                    Constraint(
                        kind=ConstraintKind.FOREIGN_KEY,
                        column=None,
                        expression=f"depends_on({consumed_table})",
                        source="plugin.consumes_tables",
                        description=f"Producer {meta.name} requires {consumed_table}",
                    )
                )

    return cs


def merge_constraint_sets(*sets: ConstraintSet) -> ConstraintSet:
    """Merge multiple ConstraintSets into one.

    Parameters
    ----------
    *sets
        ConstraintSets to merge.

    Returns
    -------
    ConstraintSet
        Merged constraint set with all constraints from inputs.

    Raises
    ------
    ValueError
        If constraint sets have different table keys.
    """
    if not sets:
        msg = "At least one ConstraintSet required"
        raise ValueError(msg)

    first = sets[0]
    merged = ConstraintSet(table_key=first.table_key)

    for cs in sets:
        if cs.table_key != first.table_key:
            msg = f"Cannot merge ConstraintSets with different table keys: {first.table_key} vs {cs.table_key}"
            raise ValueError(msg)

        for constraint in cs.constraints:
            merged.add(constraint)

    return merged
