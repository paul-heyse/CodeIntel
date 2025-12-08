"""Tests for plugin registry and decorator-based plugin discovery."""

from __future__ import annotations

from typing import ClassVar

import pytest

from codeintel.build import plugin_registry
from codeintel.build.plugin_registry import (
    PluginRegistryStore,
    get_all_plugins,
    get_plugin_for_target,
    register_plugin,
)
from codeintel.build.plugins import (
    PluginCatalog,
    TargetPlugin,
    all_plugins,
    get_plugin,
)
from codeintel.build.plugins import (
    register_plugin as decorator_register_plugin,
)
from tests._helpers.assertions import expect_equal, expect_in, expect_is_instance, expect_true
from tests._helpers.build import RecordingPlugin, make_plugin_registry_store


@pytest.fixture
def registry_store() -> PluginRegistryStore:
    """Fresh plugin registry store for isolation.

    Returns
    -------
    PluginRegistryStore
        New registry store without built-in loader.
    """
    return make_plugin_registry_store(loader=None)


def test_register_and_get_plugin(registry_store: PluginRegistryStore) -> None:
    """Manual registration returns instantiated plugin."""
    register_plugin("record", RecordingPlugin, registry=registry_store)

    plugin = get_plugin_for_target("record", registry=registry_store)

    expect_is_instance(plugin, RecordingPlugin)
    expect_equal(plugin.plugin_name, "recording_plugin")


def test_get_all_plugins_returns_copy(registry_store: PluginRegistryStore) -> None:
    """get_all_plugins should not expose internal registry for mutation."""
    register_plugin("record", RecordingPlugin, registry=registry_store)

    plugins_before = get_all_plugins(registry=registry_store)
    plugins_copy = dict(plugins_before)
    plugins_copy["new"] = RecordingPlugin

    expect_equal(get_all_plugins(registry=registry_store), plugins_before)


def test_missing_plugin_error_lists_available(registry_store: PluginRegistryStore) -> None:
    """KeyError message includes available plugin names."""
    register_plugin("present", RecordingPlugin, registry=registry_store)

    with pytest.raises(KeyError) as excinfo:
        get_plugin_for_target("absent", registry=registry_store)

    expect_in("available", str(excinfo.value).lower())
    expect_in("present", str(excinfo.value))


def test_duplicate_registration_logs_warning(
    registry_store: PluginRegistryStore, caplog: pytest.LogCaptureFixture
) -> None:
    """Overwriting an existing plugin emits a warning."""
    caplog.set_level("WARNING")

    register_plugin("record", RecordingPlugin, registry=registry_store)
    register_plugin("record", RecordingPlugin, registry=registry_store)

    expect_true(any("Overwriting plugin" in rec.message for rec in caplog.records))


def test_lazy_registration_handles_import_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Import errors during lazy registration are logged and do not raise."""
    caplog.set_level("WARNING")

    def loader(_registry: PluginRegistryStore) -> None:
        exc = ImportError("missing.module")
        plugin_registry.log.warning(
            "Failed to register plugin %s.%s: %s", "missing.module", "MissingClass", exc
        )

    registry_store = make_plugin_registry_store(loader=loader)

    plugins = get_all_plugins(registry=registry_store)

    expect_equal(plugins, {})
    expect_true(any("Failed to register plugin" in rec.message for rec in caplog.records))


def test_lazy_registration_handles_attribute_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing attributes during registration are logged."""
    caplog.set_level("WARNING")

    def loader(_registry: PluginRegistryStore) -> None:
        module_path = "dummy.module"
        class_name = "MissingClass"
        exc = AttributeError("missing attribute")
        plugin_registry.log.warning(
            "Failed to register plugin %s.%s: %s", module_path, class_name, exc
        )

    registry_store = make_plugin_registry_store(loader=loader)

    plugins = get_all_plugins(registry=registry_store)

    expect_equal(plugins, {})
    expect_true(any("Failed to register plugin" in rec.message for rec in caplog.records))


def test_register_plugin_decorator() -> None:
    """Decorator registers plugin class and returns it."""
    catalog = PluginCatalog()

    class DecoratedPlugin(RecordingPlugin, TargetPlugin):
        plugin_name: ClassVar[str] = "decorated"

    decorator_register_plugin(plugin_class=DecoratedPlugin, catalog=catalog)

    expect_true(get_plugin("decorated", catalog=catalog) is DecoratedPlugin)
    expect_in("decorated", all_plugins(catalog=catalog))


def test_all_plugins_is_copy() -> None:
    """all_plugins returns a copy of the registry."""
    catalog = PluginCatalog()

    class AnotherPlugin(RecordingPlugin, TargetPlugin):
        plugin_name: ClassVar[str] = "another"

    decorator_register_plugin(plugin_class=AnotherPlugin, catalog=catalog)

    plugins = all_plugins(catalog=catalog)
    plugins.pop("another")

    expect_in("another", all_plugins(catalog=catalog))


def test_get_plugin_missing() -> None:
    """get_plugin returns None when plugin is absent."""
    catalog = PluginCatalog()
    expect_true(get_plugin("absent", catalog=catalog) is None)
