"""Shared options infrastructure for plugin configuration."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.plugins.types.metadata import CorePluginMetadata

T = TypeVar("T")


@runtime_checkable
class ConfigSource(Protocol):
    """Protocol for loading plugin configuration."""

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return raw option values for a plugin, or None if not configured.

        Returns
        -------
        Mapping[str, Any] | None
            Option key-value pairs, or None when no configuration exists.
        """
        _ = self
        _ = plugin_name
        return None


class EmptyConfigSource:
    """ConfigSource that always returns no options."""

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return None for any plugin.

        Returns
        -------
        None
            Always returns None.
        """
        _ = self
        _ = plugin_name
        return None


@runtime_checkable
class _PydanticV2Model(Protocol):
    """Subset of the Pydantic v2 API we rely on."""

    def model_copy(self: Self, *, update: Mapping[str, Any] | None = None) -> Self: ...


@runtime_checkable
class _PydanticV1Model(Protocol):
    """Subset of the Pydantic v1 API we rely on."""

    def copy(self: Self, *, update: Mapping[str, Any] | None = None) -> Self: ...


class PluginOptionsResolver:
    """Construct typed options objects for plugins."""

    def __init__(self, config_source: ConfigSource | None = None) -> None:
        self._config_source = config_source or EmptyConfigSource()

    @property
    def config_source(self) -> ConfigSource:
        """Return the ConfigSource used by this resolver.

        Returns
        -------
        ConfigSource
            The configuration source backing this resolver.
        """
        return self._config_source

    def get_options(
        self,
        plugin_metadata: CorePluginMetadata,
        model: type[T],
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> T:
        """Construct an options instance for a plugin.

        Returns
        -------
        T
            Options model instance populated from config and overrides.
        """
        raw = self._config_source.get_plugin_options(plugin_metadata.name) or {}
        base = model(**raw)

        if not dynamic_overrides:
            return base

        overrides = dict(dynamic_overrides)

        if is_dataclass(base) and not isinstance(base, type):
            return cast("T", replace(base, **overrides))

        if isinstance(base, _PydanticV2Model):
            return cast("T", base.model_copy(update=overrides))

        if isinstance(base, _PydanticV1Model):
            return cast("T", base.copy(update=overrides))

        for key, value in overrides.items():
            setattr(base, key, value)
        return base


def _merge_dicts(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Shallow merge two option dictionaries.

    Returns
    -------
    dict[str, Any]
        Combined mapping with overrides applied.
    """
    merged: dict[str, Any] = {}
    if base:
        merged.update(base)
    if override:
        merged.update(override)
    return merged


@dataclass(frozen=True)
class PluginConfigBundle:
    """Configuration data for all plugins for a single layer."""

    plugin_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Normalize plugin options to a mutable dict."""
        normalized = dict(self.plugin_options or {})
        object.__setattr__(self, "plugin_options", normalized)

    def get(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return raw options for a plugin in this bundle.

        Returns
        -------
        Mapping[str, Any] | None
            Options mapping for the plugin, or None if not configured.
        """
        options = cast("Mapping[str, Mapping[str, Any]]", self.plugin_options)
        return options.get(plugin_name)


class ProfiledConfigSource(ConfigSource):
    """ConfigSource that merges base, profile, and CLI overrides."""

    def __init__(
        self,
        *,
        base: PluginConfigBundle | None = None,
        profile: PluginConfigBundle | None = None,
        cli: PluginConfigBundle | None = None,
        active_profile_name: str | None = None,
    ) -> None:
        self._base: PluginConfigBundle = base or PluginConfigBundle(plugin_options={})
        self._profile: PluginConfigBundle = profile or PluginConfigBundle(plugin_options={})
        self._cli: PluginConfigBundle = cli or PluginConfigBundle(plugin_options={})
        self._active_profile_name: str | None = active_profile_name

    @property
    def active_profile_name(self) -> str | None:
        """Return the active profile name.

        Returns
        -------
        str | None
            Active profile identifier, or None when no profile is active.
        """
        return self._active_profile_name

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return merged options for a plugin.

        Returns
        -------
        Mapping[str, Any] | None
            Merged configuration for the plugin, or None when unset.
        """
        base_raw = self._base.get(plugin_name)
        profile_raw = self._profile.get(plugin_name) if self._active_profile_name else None
        cli_raw = self._cli.get(plugin_name)

        merged = _merge_dicts(base_raw, profile_raw)
        merged = _merge_dicts(merged, cli_raw)

        return merged or None


__all__ = [
    "ConfigSource",
    "EmptyConfigSource",
    "PluginConfigBundle",
    "PluginOptionsResolver",
    "ProfiledConfigSource",
]
