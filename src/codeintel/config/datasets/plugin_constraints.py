"""Plugin-based constraint extraction for dataset schema introspection.

This module extracts constraints from plugin metadata using the build
registry as the single source of truth. It specifically extracts
produces_tables and consumes_tables declarations to represent data
dependencies and producer/consumer relationships in the plugin DAG.

Architecture Reference: Section 5.4.1 - Implement ConstraintSet aggregation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
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
    "merge_constraint_sets",
]

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_all_plugins_metadata() -> dict[str, CorePluginMetadata]:
    """Load all plugin metadata from the build registry.

    Returns
    -------
    dict[str, CorePluginMetadata]
        Mapping of target name to core metadata.
    """
    try:
        # Lazy import to avoid circular dependency at module load time
        from codeintel.build.plugin_registry import get_all_plugins  # noqa: PLC0415
    except ImportError:
        log.debug("Build plugin registry not available")
        return {}

    result: dict[str, CorePluginMetadata] = {}
    for target_name, plugin_class in get_all_plugins().items():
        try:
            plugin = plugin_class()
        except (TypeError, ValueError, AttributeError):
            log.debug("Failed to instantiate plugin %s", target_name)
            continue

        core_meta = getattr(plugin, "core_metadata", None)
        if core_meta is not None:
            result[target_name] = core_meta

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
