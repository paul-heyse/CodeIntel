"""Plugin-based constraint extraction for dataset schema introspection.

This module extracts constraints from plugin metadata, specifically from
produces_tables and consumes_tables declarations. These constraints
represent data dependencies and producer/consumer relationships that
are implicit in the plugin DAG.

Architecture Reference: Section 5.4.1 - Implement ConstraintSet aggregation
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.datasets.constraints import Constraint, ConstraintKind, ConstraintSet

if TYPE_CHECKING:
    from codeintel.core.plugins.types.metadata import CorePluginMetadata

__all__ = [
    "PluginTableRelation",
    "extract_constraints_from_plugins",
    "get_consumer_plugins",
    "get_producer_plugins",
    "get_table_plugin_relations",
]

log = logging.getLogger(__name__)

# Lazy reference to avoid circular imports
_PLUGIN_CATALOG: object | None = None


def _get_plugin_catalog() -> object | None:
    """Lazily load the plugin catalog.

    Returns
    -------
    object | None
        The plugin catalog if available.
    """
    global _PLUGIN_CATALOG  # noqa: PLW0603

    if _PLUGIN_CATALOG is not None:
        return _PLUGIN_CATALOG

    try:
        # Use importlib to avoid circular dependency at module load time
        analytics_registry = importlib.import_module("codeintel.analytics.core.registry")
        _PLUGIN_CATALOG = analytics_registry.ANALYTICS_REGISTRY
    except ImportError:
        log.debug("Analytics registry not available for plugin constraint extraction")
        return None
    else:
        return _PLUGIN_CATALOG


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

    Notes
    -----
    NOTE(logic-framework): Full functionality requires complete plugin catalog
    Functional Intent: Return all plugins whose produces_tables includes table_key
    Architecture Reference: Section 5.4.1 - ConstraintSet aggregation
    Activation Steps:
      1. Ensure ANALYTICS_REGISTRY is fully populated at startup
      2. Add graph and ingest plugin registries to the search
    """
    catalog = _get_plugin_catalog()
    if catalog is None:
        return []

    catalog_all = getattr(catalog, "all", None)
    if catalog_all is None:
        return []

    result: list[CorePluginMetadata] = []
    for plugin in catalog_all():
        core_meta = getattr(plugin, "core_metadata", None)
        if core_meta is None:
            continue

        produces = getattr(core_meta, "produces_tables", None)
        if produces and table_key in produces:
            result.append(core_meta)

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

    Notes
    -----
    NOTE(logic-framework): Full functionality requires complete plugin catalog
    Functional Intent: Return all plugins whose consumes_tables includes table_key
    Architecture Reference: Section 5.4.1 - ConstraintSet aggregation
    Activation Steps:
      1. Ensure ANALYTICS_REGISTRY is fully populated at startup
      2. Add graph and ingest plugin registries to the search
    """
    catalog = _get_plugin_catalog()
    if catalog is None:
        return []

    catalog_all = getattr(catalog, "all", None)
    if catalog_all is None:
        return []

    result: list[CorePluginMetadata] = []
    for plugin in catalog_all():
        core_meta = getattr(plugin, "core_metadata", None)
        if core_meta is None:
            continue

        consumes = getattr(core_meta, "consumes_tables", None)
        if consumes and table_key in consumes:
            result.append(core_meta)

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

    Notes
    -----
    NOTE(logic-framework): Full constraint extraction requires complete plugin DAG
    Functional Intent: Generate COMPUTATION constraints from plugin dependencies
    Architecture Reference: Section 5.4.1 - ConstraintSet aggregation
    Activation Steps:
      1. Complete plugin catalog population
      2. Add column-level dependency tracking in plugins
      3. Wire to schema_builder for automatic constraint aggregation

    Examples
    --------
    >>> cs = extract_constraints_from_plugins("analytics.function_metrics")
    >>> # When fully activated, this returns computation constraints
    >>> # representing which plugins produce/consume this table
    """
    cs = ConstraintSet(table_key=table_key)

    # Add producer constraints
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

    # Add consumer constraints
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

    # Add foreign key constraints for consumed tables
    # These represent implicit dependencies
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
