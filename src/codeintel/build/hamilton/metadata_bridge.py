"""Unified plugin metadata access for Hamilton integration.

This module provides a bridge between existing target/plugin metadata and
the canonical format needed for Hamilton node generation. Phase 0 falls back
to deriving metadata from OutputTarget when plugins don't expose it directly.

Design Principles
-----------------
1. CanonicalPluginMeta is the single source of truth for node metadata.
2. from_plugin_or_target() checks plugin.metadata first, then falls back.
3. This allows gradual migration: plugins can opt-in to canonical metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codeintel.build.targets import OutputTarget


class PluginWithMetadata(Protocol):
    """Protocol for plugins that may have metadata."""

    @property
    def metadata(self) -> object | None:
        """Return plugin metadata if available."""
        ...


@dataclass(frozen=True)
class CanonicalPluginMeta:
    """Canonical metadata for a plugin/target unit of work.

    This dataclass provides a unified view of plugin identity, dependencies,
    and I/O contracts regardless of whether the information comes from a
    plugin's explicit metadata or is derived from the target graph.

    Attributes
    ----------
    name
        Stable identifier for policy/config/lineage (e.g., "analytics.function_metrics").
    version
        Plugin version string for cache invalidation.
    domain
        Domain classification: "ingestion", "graphs", "analytics", or "export".
    description
        Human-readable description of what this plugin does.
    requires
        Tuple of capability or target names this plugin depends on.
    provides
        Tuple of capability or target names this plugin produces.
    produces_tables
        Tuple of table keys this plugin writes to.
    consumes_tables
        Tuple of table keys this plugin reads from.
    options_type
        Optional type for plugin configuration options.

    Examples
    --------
    >>> meta = CanonicalPluginMeta(
    ...     name="analytics.risk_factors",
    ...     version="1.0.0",
    ...     domain="analytics",
    ...     description="Compute risk factors from function metrics",
    ...     requires=("analytics.function_metrics", "graphs.call_graph"),
    ...     provides=("analytics.risk_factors",),
    ...     produces_tables=("analytics.risk_factors",),
    ... )
    """

    name: str
    version: str
    domain: str
    description: str
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    consumes_tables: tuple[str, ...] = ()
    options_type: type[object] | None = field(default=None, compare=False)


def from_target(target: OutputTarget) -> CanonicalPluginMeta:
    """Derive canonical metadata from an OutputTarget.

    This is the Phase 0 fallback when a plugin doesn't expose explicit
    metadata. The information is extracted from the target graph's
    knowledge of the target.

    Parameters
    ----------
    target
        The OutputTarget to extract metadata from.

    Returns
    -------
    CanonicalPluginMeta
        Metadata derived from target properties.

    Examples
    --------
    >>> from codeintel.build.targets import OutputTarget
    >>> target = OutputTarget(
    ...     name="function_metrics",
    ...     module="analytics",
    ...     dependencies=("goids", "ast"),
    ... )
    >>> meta = from_target(target)
    >>> meta.name
    'analytics.function_metrics'
    >>> meta.domain
    'analytics'
    """
    name = f"{target.module}.{target.name}"

    requires = tuple(f"{target.module}.{dep}" for dep in target.dependencies)

    produces_tables = target.contract.table_keys if target.contract else ()

    return CanonicalPluginMeta(
        name=name,
        version="0.0.0",
        domain=target.module,
        description=target.description or f"Target {target.name} ({target.module})",
        requires=requires,
        provides=(name,),
        produces_tables=produces_tables,
        consumes_tables=(),
        options_type=None,
    )


def from_plugin_or_target(
    *,
    plugin: object,
    target: OutputTarget,
) -> CanonicalPluginMeta:
    """Extract metadata from plugin if available, otherwise from target.

    This function checks if the plugin exposes a ``metadata`` attribute
    with the canonical format. If not, it falls back to deriving metadata
    from the OutputTarget.

    Parameters
    ----------
    plugin
        Plugin instance that may have a ``metadata`` attribute.
    target
        OutputTarget to use as fallback.

    Returns
    -------
    CanonicalPluginMeta
        Metadata from plugin or derived from target.

    Examples
    --------
    >>> meta = from_plugin_or_target(plugin=my_plugin, target=target)
    >>> meta.name
    'analytics.function_metrics'
    """
    meta = getattr(plugin, "metadata", None)
    if meta is None:
        return from_target(target)

    if isinstance(meta, dict):
        return CanonicalPluginMeta(
            name=str(meta.get("name", f"{target.module}.{target.name}")),
            version=str(meta.get("version", "0.0.0")),
            domain=str(meta.get("domain", target.module)),
            description=str(meta.get("description", target.description or f"Plugin {target.name}")),
            requires=tuple(meta.get("requires", ())),
            provides=tuple(meta.get("provides", (f"{target.module}.{target.name}",))),
            produces_tables=tuple(meta.get("produces_tables", ())),
            consumes_tables=tuple(meta.get("consumes_tables", ())),
            options_type=meta.get("options_type"),
        )

    return CanonicalPluginMeta(
        name=getattr(meta, "name", f"{target.module}.{target.name}"),
        version=getattr(meta, "version", "0.0.0"),
        domain=getattr(meta, "domain", target.module),
        description=getattr(meta, "description", target.description or f"Plugin {target.name}"),
        requires=tuple(getattr(meta, "requires", ())),
        provides=tuple(getattr(meta, "provides", (f"{target.module}.{target.name}",))),
        produces_tables=tuple(getattr(meta, "produces_tables", ())),
        consumes_tables=tuple(getattr(meta, "consumes_tables", ())),
        options_type=getattr(meta, "options_type", None),
    )


__all__ = [
    "CanonicalPluginMeta",
    "from_plugin_or_target",
    "from_target",
]
