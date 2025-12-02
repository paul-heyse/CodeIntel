"""Configuration registry for analytics step configs.

This module provides a central registry mapping config types to plugins,
enabling automatic plugin resolution based on provided configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

T = TypeVar("T", bound="AnalyticsStepConfigBase")


@runtime_checkable
class AnalyticsStepConfigBase(Protocol):
    """Protocol for all analytics step configurations.

    All step configs must provide access to the snapshot reference
    and derived properties (repo, commit, repo_root).
    """

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        ...

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        ...

    @property
    def commit(self) -> str:
        """Return the commit identifier."""
        ...

    @property
    def repo_root(self) -> Path:
        """Return the repository root path."""
        ...


@dataclass(frozen=True)
class ConfigPluginMapping:
    """Mapping between a config type and its associated plugins.

    Attributes
    ----------
    config_type
        The configuration type.
    plugins
        Plugin names that use this config.
    primary
        The primary plugin for this config (used for auto-resolution).
    """

    config_type: type[AnalyticsStepConfigBase]
    plugins: tuple[str, ...]
    primary: str | None = None


class ConfigRegistry:
    """Central registry mapping config types to plugins.

    Enables automatic plugin resolution based on provided configurations
    and validation of config requirements.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._mappings: dict[type[AnalyticsStepConfigBase], ConfigPluginMapping] = {}
        self._by_plugin: dict[str, set[type[AnalyticsStepConfigBase]]] = {}

    def register(
        self,
        config_type: type[AnalyticsStepConfigBase],
        plugins: tuple[str, ...],
        *,
        primary: str | None = None,
    ) -> None:
        """Register a config type with its associated plugins.

        Parameters
        ----------
        config_type
            The configuration type.
        plugins
            Plugin names that use this config.
        primary
            The primary plugin (defaults to first in list).

        Raises
        ------
        ValueError
            If the config type is already registered.
        """
        if config_type in self._mappings:
            message = f"Config type {config_type.__name__} already registered"
            raise ValueError(message)

        resolved_primary = primary or (plugins[0] if plugins else None)
        mapping = ConfigPluginMapping(
            config_type=config_type,
            plugins=plugins,
            primary=resolved_primary,
        )
        self._mappings[config_type] = mapping

        for plugin_name in plugins:
            self._by_plugin.setdefault(plugin_name, set()).add(config_type)

        log.debug(
            "Registered config %s for plugins %s",
            config_type.__name__,
            plugins,
        )

    def get_plugins_for_config(
        self,
        config_type: type[AnalyticsStepConfigBase],
    ) -> tuple[str, ...]:
        """Return plugin names associated with a config type.

        Parameters
        ----------
        config_type
            The configuration type.

        Returns
        -------
        tuple[str, ...]
            Plugin names that use this config.
        """
        mapping = self._mappings.get(config_type)
        return mapping.plugins if mapping else ()

    def get_primary_plugin(
        self,
        config_type: type[AnalyticsStepConfigBase],
    ) -> str | None:
        """Return the primary plugin for a config type.

        Parameters
        ----------
        config_type
            The configuration type.

        Returns
        -------
        str | None
            Primary plugin name or None.
        """
        mapping = self._mappings.get(config_type)
        return mapping.primary if mapping else None

    def get_required_configs(
        self,
        plugin_name: str,
    ) -> tuple[type[AnalyticsStepConfigBase], ...]:
        """Return config types required by a plugin.

        Parameters
        ----------
        plugin_name
            Plugin name to look up.

        Returns
        -------
        tuple[type[AnalyticsStepConfigBase], ...]
            Config types required by the plugin.
        """
        config_types = self._by_plugin.get(plugin_name, set())
        return tuple(config_types)

    def resolve_plugins_from_configs(
        self,
        configs: dict[type[AnalyticsStepConfigBase], object],
    ) -> tuple[str, ...]:
        """Resolve plugin names from provided configs.

        Parameters
        ----------
        configs
            Mapping of config types to instances.

        Returns
        -------
        tuple[str, ...]
            Unique plugin names that can be run with the provided configs.
        """
        plugins: set[str] = set()
        for config_type in configs:
            mapping = self._mappings.get(config_type)
            if mapping:
                plugins.update(mapping.plugins)
        return tuple(sorted(plugins))

    def list_all(self) -> tuple[ConfigPluginMapping, ...]:
        """Return all registered mappings.

        Returns
        -------
        tuple[ConfigPluginMapping, ...]
            All config-plugin mappings.
        """
        return tuple(self._mappings.values())


# Global registry instance
_CONFIG_REGISTRY: ConfigRegistry | None = None


def get_config_registry() -> ConfigRegistry:
    """Return the global config registry.

    Returns
    -------
    ConfigRegistry
        The singleton registry instance.
    """
    global _CONFIG_REGISTRY  # noqa: PLW0603
    if _CONFIG_REGISTRY is None:
        _CONFIG_REGISTRY = ConfigRegistry()
    return _CONFIG_REGISTRY


def register_config(
    config_type: type[AnalyticsStepConfigBase],
    plugins: tuple[str, ...],
    *,
    primary: str | None = None,
) -> None:
    """Register a config type with the global registry.

    Parameters
    ----------
    config_type
        The configuration type.
    plugins
        Plugin names that use this config.
    primary
        The primary plugin.
    """
    get_config_registry().register(config_type, plugins, primary=primary)


@dataclass(frozen=True)
class BaseStepConfig:
    """Concrete base class for analytics step configurations.

    Provides common snapshot-derived properties to reduce duplication
    across step config classes.

    Attributes
    ----------
    snapshot
        Snapshot reference containing repo, commit, and repo_root.
    """

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return the commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Return the repository root path."""
        return self.snapshot.repo_root


__all__ = [
    "AnalyticsStepConfigBase",
    "BaseStepConfig",
    "ConfigPluginMapping",
    "ConfigRegistry",
    "get_config_registry",
    "register_config",
]
