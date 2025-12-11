"""Tests for registry hooks and provider indexing."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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


ENTRYPOINT_SENTINEL = "dummy"


def _make_entry_point() -> importlib.metadata.EntryPoint:
    """
    Build a typed entry point pointing to the sentinel payload.

    Returns
    -------
    importlib.metadata.EntryPoint
        Entry point referencing the sentinel constant.
    """
    return importlib.metadata.EntryPoint(
        name="ep",
        value=f"{__name__}:ENTRYPOINT_SENTINEL",
        group="tests.plugins",
    )


@contextmanager
def override_entry_points(
    entries: list[importlib.metadata.EntryPoint],
) -> Iterator[None]:
    """
    Temporarily override importlib metadata entry points.

    Parameters
    ----------
    entries
        Entry points to return from importlib.metadata.entry_points().

    Yields
    ------
    Iterator[None]
        Context with entry points patched to the provided entries.
    """
    original_entry_points = importlib.metadata.entry_points

    def _entry_points(**kwargs: object) -> importlib.metadata.EntryPoints:
        _ = kwargs
        return importlib.metadata.EntryPoints(entries)

    importlib.metadata.entry_points = _entry_points
    try:
        yield
    finally:
        importlib.metadata.entry_points = original_entry_points


def test_hooks_load_builtins_and_entrypoints() -> None:
    """Hooks should load builtins and resolve entrypoints."""
    plugin = DummyPlugin("dummy_plugin")
    hooks = DummyHooks(plugin=plugin)
    registry = DummyRegistry(hooks=hooks)

    with override_entry_points([_make_entry_point()]):
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
