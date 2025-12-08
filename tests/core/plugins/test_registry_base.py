"""Tests for registry hooks and provider indexing."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
from typing import TypeGuard

import pytest

from codeintel.core.plugins.registry.base import BasePluginRegistry, RegistryHooks
from codeintel.core.plugins.types.protocol import PluginMetadata


class DummyPlugin:
    """Minimal plugin implementation for registry tests."""

    def __init__(
        self, name: str, provides: Sequence[str] = (), requires: Sequence[str] = ()
    ) -> None:
        self._metadata = PluginMetadata(
            name=name,
            description="test plugin",
            kind="builder",
            stage="core",
            provides=tuple(provides),
            requires=tuple(requires),
        )

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return self._metadata


class DummyHooks(RegistryHooks[DummyPlugin]):
    """Hooks that record lifecycle calls for assertions."""

    def __init__(self, plugin: DummyPlugin | None = None) -> None:
        self._plugin = plugin
        self.builtins_loaded = False
        self._entrypoint_group = "tests.plugins"

    @property
    def entrypoint_group(self) -> str:
        """Entrypoint group for discovery."""
        return self._entrypoint_group

    def load_builtins(self) -> None:
        """Mark builtins as loaded."""
        self.builtins_loaded = True

    def resolve_entrypoint(self, loaded: object) -> DummyPlugin | None:
        """Return plugin for sentinel entrypoint payload.

        Returns
        -------
        DummyPlugin | None
            Plugin when the payload matches the sentinel.
        """
        if loaded == "dummy" and self._plugin is not None:
            return self._plugin
        return None

    def is_valid_plugin(self, obj: object) -> TypeGuard[DummyPlugin]:
        """Validate plugin candidates.

        Returns
        -------
        TypeGuard[DummyPlugin]
            True when the object is a DummyPlugin.
        """
        _ = self.entrypoint_group
        return isinstance(obj, DummyPlugin)


class DummyRegistry(BasePluginRegistry[DummyPlugin]):
    """Concrete registry for testing."""

    @staticmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return default plugin names.

        Returns
        -------
        Sequence[str]
            Default plugin identifiers.
        """
        return ()


class DummyEntryPoint:
    """Simple entrypoint stub."""

    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        """Return stored entry point value.

        Returns
        -------
        object
            Entrypoint payload.
        """
        return self._value


class DummyEntryPoints:
    """Container emulating importlib.metadata entry_points()."""

    def __init__(self, entries: list[DummyEntryPoint]) -> None:
        self._entries = entries

    def select(self, group: str) -> list[DummyEntryPoint]:
        """Return all entry points for the requested group.

        Returns
        -------
        list[DummyEntryPoint]
            Entrypoints matching the group.
        """
        _ = group
        return self._entries


def test_hooks_load_builtins_and_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hooks should load builtins and resolve entrypoints."""
    plugin = DummyPlugin("dummy_plugin")
    hooks = DummyHooks(plugin=plugin)
    registry = DummyRegistry(hooks=hooks)

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: DummyEntryPoints([DummyEntryPoint("ep", "dummy")]),
    )

    discovered = registry.load_from_entrypoints(force=True)
    if discovered != (plugin,):
        pytest.fail("Entry point resolution did not return expected plugin")
    if not registry.contains("dummy_plugin"):
        pytest.fail("Registered plugin not found after discovery")

    names = registry.list_names()
    if not hooks.builtins_loaded:
        pytest.fail("Builtins did not load when registry accessed")
    if names != ("dummy_plugin",):
        pytest.fail(f"Unexpected registry names: {names}")


def test_dependency_resolution_uses_requires_provides() -> None:
    """Dependencies should wire requires to providers."""
    provider = DummyPlugin("provider", provides=("cap",))
    consumer = DummyPlugin("consumer", requires=("cap",))
    selected: dict[str, DummyPlugin] = {
        provider.metadata.name: provider,
        consumer.metadata.name: consumer,
    }

    dependencies = DummyRegistry.resolve_dependencies_debug(selected)

    if dependencies["consumer"] != {"provider"}:
        pytest.fail(f"Expected provider dependency, got {dependencies['consumer']}")
    if dependencies["provider"] != set():
        pytest.fail(f"Expected no dependencies for provider, got {dependencies['provider']}")
