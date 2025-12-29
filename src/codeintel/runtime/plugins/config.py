"""Plugin configuration models for runtime composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from codeintel.build.config import BuildConfig


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """Configuration for target pack discovery and workspace modules."""

    enabled: tuple[str, ...] | None = None
    disabled: tuple[str, ...] = ()
    strict: bool = True
    namespace_enforcement: bool = True
    allow_workspace_modules: bool = True
    hamilton_config: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation of the config.

        Returns
        -------
        dict[str, object]
            Serialized plugin configuration payload.
        """
        return {
            "enabled": list(self.enabled) if self.enabled is not None else None,
            "disabled": list(self.disabled),
            "strict": self.strict,
            "namespace_enforcement": self.namespace_enforcement,
            "allow_workspace_modules": self.allow_workspace_modules,
            "hamilton_config": dict(self.hamilton_config),
        }


def plugin_config_from_build_config(config: BuildConfig) -> PluginConfig:
    """Parse plugin config from a BuildConfig instance.

    Parameters
    ----------
    config
        Build config to inspect for plugin settings.

    Returns
    -------
    PluginConfig
        Parsed plugin configuration.
    """
    raw_plugins = _coerce_optional_mapping(config.get("plugins"), name="plugins")
    raw_runtime_plugins = _coerce_optional_mapping(
        config.get("runtime.plugins"),
        name="runtime.plugins",
    )
    if raw_plugins and raw_runtime_plugins:
        merged = {**raw_plugins, **raw_runtime_plugins}
        return plugin_config_from_mapping(merged)
    if raw_runtime_plugins:
        return plugin_config_from_mapping(raw_runtime_plugins)
    if raw_plugins:
        return plugin_config_from_mapping(raw_plugins)
    return PluginConfig()


def plugin_config_from_mapping(mapping: Mapping[str, object]) -> PluginConfig:
    """Create PluginConfig from a mapping.

    Parameters
    ----------
    mapping
        Mapping containing plugin configuration values.

    Returns
    -------
    PluginConfig
        Parsed plugin configuration.
    """
    enabled = _coerce_optional_str_tuple(mapping.get("enabled"), name="enabled")
    disabled = _coerce_str_tuple(mapping.get("disabled"), name="disabled", default=())
    strict = _coerce_bool(mapping.get("strict"), name="strict", default=True)
    namespace_enforcement = _coerce_bool(
        mapping.get("namespace_enforcement"),
        name="namespace_enforcement",
        default=True,
    )
    allow_workspace_modules = _coerce_bool(
        mapping.get("allow_workspace_modules"),
        name="allow_workspace_modules",
        default=True,
    )
    hamilton_config = _coerce_mapping(
        mapping.get("hamilton_config"),
        name="hamilton_config",
        default={},
    )
    return PluginConfig(
        enabled=enabled,
        disabled=disabled,
        strict=strict,
        namespace_enforcement=namespace_enforcement,
        allow_workspace_modules=allow_workspace_modules,
        hamilton_config=hamilton_config,
    )


def _coerce_optional_mapping(value: object, *, name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _coerce_mapping(value, name=name, default=None)


def _coerce_mapping(
    value: object,
    *,
    name: str,
    default: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if value is None:
        if default is None:
            msg = f"{name} must be a mapping"
            raise TypeError(msg)
        return default
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping"
        raise TypeError(msg)
    return dict(value)


def _coerce_bool(value: object, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    msg = f"{name} must be a boolean"
    raise TypeError(msg)


def _coerce_optional_str_tuple(value: object, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _coerce_str_tuple(value, name=name, default=())


def _coerce_str_tuple(
    value: object,
    *,
    name: str,
    default: Sequence[str],
) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        values = []
        for item in value:
            if not isinstance(item, str):
                msg = f"{name} must contain only strings"
                raise TypeError(msg)
            values.append(item)
        return tuple(values)
    msg = f"{name} must be a string or sequence of strings"
    raise TypeError(msg)


__all__ = [
    "PluginConfig",
    "plugin_config_from_build_config",
    "plugin_config_from_mapping",
]
