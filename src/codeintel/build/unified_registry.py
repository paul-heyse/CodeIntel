"""Unified registry for targets and their implementations.

This module provides a single registry that atomically associates targets
with their plugin implementations, preventing the mismatch issues that can
occur when targets and plugins are registered separately.

The UnifiedRegistry ensures:
1. Every target with a plugin field has a corresponding implementation
2. No plugin can be registered without a target
3. Consistency validation at registry build time

Example
-------
>>> registry = UnifiedRegistry()
>>> registry.register(my_target, plugin=MyPlugin)
>>> registry.get_target("my_target")
OutputTarget(name='my_target', ...)
>>> registry.get_plugin("my_target")
<class 'MyPlugin'>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.errors import RegistryValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.build.plugin import TargetPlugin, TargetPluginProtocol
    from codeintel.build.targets import OutputTarget

__all__ = [
    "TargetRegistration",
    "UnifiedRegistry",
    "get_unified_registry",
]


@dataclass(frozen=True)
class TargetRegistration:
    """Complete registration for a target.

    Associates a target with its implementation details.

    Attributes
    ----------
    target
        The OutputTarget definition.
    plugin_class
        Plugin class that implements this target (if any).
    native_module
        Module path for native Hamilton implementation (if any).
    """

    target: OutputTarget
    plugin_class: type[TargetPlugin] | None = None
    native_module: str | None = None

    @property
    def has_implementation(self) -> bool:
        """Check if target has an implementation.

        Returns
        -------
        bool
            True if plugin_class or native_module is set.
        """
        return self.plugin_class is not None or self.native_module is not None


@dataclass
class UnifiedRegistry:
    """Single registry for targets, plugins, and native implementations.

    This registry ensures atomic registration of targets with their
    implementations, preventing the mismatch issues that occur when
    targets and plugins are registered separately.

    Attributes
    ----------
    _registrations
        Internal mapping of target names to registrations.

    Examples
    --------
    >>> registry = UnifiedRegistry()
    >>> registry.register(my_target, plugin=MyPlugin)
    >>> registry.validate()
    []
    """

    _registrations: dict[str, TargetRegistration] = field(default_factory=dict)

    def register(
        self,
        target: OutputTarget,
        *,
        plugin: type[TargetPlugin] | None = None,
        native_module: str | None = None,
    ) -> None:
        """Register a target with its implementation atomically.

        Parameters
        ----------
        target
            The OutputTarget to register.
        plugin
            Plugin class implementing this target.
        native_module
            Module path for native Hamilton implementation.

        Raises
        ------
        RegistryValidationError
            If target declares a plugin but no implementation is provided.
        """
        if target.plugin and plugin is None and native_module is None:
            msg = f"declares plugin='{target.plugin}' but no implementation provided"
            raise RegistryValidationError(target.name, [msg])

        self._registrations[target.name] = TargetRegistration(
            target=target,
            plugin_class=plugin,
            native_module=native_module,
        )

    def register_target_only(self, target: OutputTarget) -> None:
        """Register a target without an implementation.

        Use this for targets that don't require plugins (e.g., native-only
        targets computed entirely within Hamilton DAGs).

        Parameters
        ----------
        target
            The OutputTarget to register.
        """
        if target.name in self._registrations:
            return  # Already registered

        self._registrations[target.name] = TargetRegistration(
            target=target,
            plugin_class=None,
            native_module=None,
        )

    def get_target(self, name: str) -> OutputTarget:
        """Get a target by name.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        OutputTarget
            The target definition.

        Raises
        ------
        KeyError
            If target is not registered.
        """
        if name not in self._registrations:
            msg = f"Target '{name}' not found in registry"
            raise KeyError(msg)
        return self._registrations[name].target

    def get_plugin(self, name: str) -> type[TargetPlugin] | None:
        """Get the plugin class for a target.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        type[TargetPlugin] | None
            Plugin class if registered, None otherwise.
        """
        reg = self._registrations.get(name)
        return reg.plugin_class if reg else None

    def get_all_plugins(self) -> dict[str, type[TargetPlugin]]:
        """Return all plugin implementations keyed by target name.

        Returns
        -------
        dict[str, type[TargetPlugin]]
            Mapping of target name to plugin class.
        """
        plugins: dict[str, type[TargetPlugin]] = {}
        for name, reg in self._registrations.items():
            if reg.plugin_class is not None:
                plugins[name] = reg.plugin_class
        return plugins

    def instantiate_plugin(self, target_name: str) -> TargetPluginProtocol:
        """Instantiate a plugin for a target.

        Parameters
        ----------
        target_name
            Target name to instantiate a plugin for.

        Returns
        -------
        TargetPluginProtocol
            Instantiated plugin.

        Raises
        ------
        KeyError
            If the target is missing or has no registered plugin implementation.
        """
        if target_name not in self._registrations:
            msg = f"Target '{target_name}' not found in registry"
            raise KeyError(msg)
        plugin_class = self.get_plugin(target_name)
        if plugin_class is None:
            msg = f"Target '{target_name}' has no registered plugin implementation"
            raise KeyError(msg)
        return plugin_class()

    def get_native_module(self, name: str) -> str | None:
        """Get the native module path for a target.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        str | None
            Module path if registered, None otherwise.
        """
        reg = self._registrations.get(name)
        return reg.native_module if reg else None

    def get_registration(self, name: str) -> TargetRegistration | None:
        """Get the full registration for a target.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        TargetRegistration | None
            Full registration if found, None otherwise.
        """
        return self._registrations.get(name)

    def get_all_targets(self) -> tuple[OutputTarget, ...]:
        """Get all registered targets.

        Returns
        -------
        tuple[OutputTarget, ...]
            All targets in registration order.
        """
        return tuple(reg.target for reg in self._registrations.values())

    def get_all_registrations(self) -> tuple[TargetRegistration, ...]:
        """Get all registrations.

        Returns
        -------
        tuple[TargetRegistration, ...]
            All registrations.
        """
        return tuple(self._registrations.values())

    def __contains__(self, name: str) -> bool:
        """Check if a target is registered.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        bool
            True if target is registered.
        """
        return name in self._registrations

    def __iter__(self) -> Iterator[str]:
        """Iterate over target names.

        Returns
        -------
        Iterator[str]
            Iterator of target names.
        """
        return iter(self._registrations)

    def __len__(self) -> int:
        """Return number of registered targets.

        Returns
        -------
        int
            Number of targets.
        """
        return len(self._registrations)

    def native_target_names(self) -> frozenset[str]:
        """Return names of targets with native implementations.

        Returns
        -------
        frozenset[str]
            Set of target names that have native module implementations.

        Examples
        --------
        >>> registry = get_unified_registry()
        >>> natives = registry.native_target_names()
        >>> "risk_factors" in natives
        True
        """
        return frozenset(
            name for name, reg in self._registrations.items() if reg.native_module is not None
        )

    def is_native_target(self, name: str) -> bool:
        """Check if a target has a native implementation.

        Parameters
        ----------
        name
            Target name to check.

        Returns
        -------
        bool
            True if the target has a native module implementation.

        Examples
        --------
        >>> registry = get_unified_registry()
        >>> registry.is_native_target("risk_factors")
        True
        >>> registry.is_native_target("modules")
        False
        """
        reg = self._registrations.get(name)
        return reg is not None and reg.native_module is not None

    def has_implementation(self, name: str) -> bool:
        """Check if a target has any implementation (plugin or native).

        Parameters
        ----------
        name
            Target name to check.

        Returns
        -------
        bool
            True if target has a plugin or native module implementation.

        Examples
        --------
        >>> registry = get_unified_registry()
        >>> registry.has_implementation("function_metrics")
        True
        >>> registry.has_implementation("some_future_target")
        False
        """
        reg = self._registrations.get(name)
        return reg is not None and reg.has_implementation

    def validate(self) -> list[str]:
        """Validate all registrations are complete and consistent.

        Returns
        -------
        list[str]
            List of error messages (empty if valid).
        """
        errors: list[str] = []

        for name, reg in self._registrations.items():
            # Check if target with plugin field has implementation
            if reg.target.plugin and not reg.has_implementation:
                errors.append(
                    f"Target '{name}' declares plugin='{reg.target.plugin}' "
                    f"but has no registered implementation"
                )

        return errors


# Singleton registry instance
_UNIFIED_REGISTRY: UnifiedRegistry | None = None


def get_unified_registry() -> UnifiedRegistry:
    """Get the singleton unified registry.

    Lazily builds the registry on first access.

    Returns
    -------
    UnifiedRegistry
        The global registry instance.
    """
    global _UNIFIED_REGISTRY  # noqa: PLW0603
    if _UNIFIED_REGISTRY is None:
        _UNIFIED_REGISTRY = _build_unified_registry()
    return _UNIFIED_REGISTRY


def _build_unified_registry() -> UnifiedRegistry:
    """Build the unified registry from existing registrations.

    This function uses the registrations module to populate the registry,
    providing a single source of truth for target/plugin associations.

    Returns
    -------
    UnifiedRegistry
        Populated registry.
    """
    from codeintel.build.registrations import register_all_targets  # noqa: PLC0415

    registry = UnifiedRegistry()
    register_all_targets(registry)

    return registry


def reset_unified_registry() -> None:
    """Reset the singleton registry (for testing).

    Clears the cached registry so it will be rebuilt on next access.
    """
    global _UNIFIED_REGISTRY  # noqa: PLW0603
    _UNIFIED_REGISTRY = None
