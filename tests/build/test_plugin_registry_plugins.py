"""Tests for plugin registry and decorator-based plugin discovery."""

from __future__ import annotations

import types
from typing import ClassVar

import pytest

from codeintel.build import plugin_registry
from codeintel.build.plugin_registry import get_all_plugins, get_plugin_for_target
from codeintel.build.plugins import TargetPlugin, all_plugins, get_plugin, register_plugin
from tests._helpers.build import RecordingPlugin


def _reset_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset plugin registry globals for isolated tests."""
    monkeypatch.setattr(plugin_registry, "_PLUGINS", {})
    monkeypatch.setattr(plugin_registry, "_REGISTERED", True)


def test_register_and_get_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual registration returns instantiated plugin."""
    _reset_registry(monkeypatch)
    plugin_registry.register_plugin("record", RecordingPlugin)

    plugin = get_plugin_for_target("record")

    assert isinstance(plugin, RecordingPlugin)
    assert plugin.plugin_name == "recording_plugin"


def test_get_all_plugins_returns_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_all_plugins should not expose internal registry for mutation."""
    _reset_registry(monkeypatch)
    plugin_registry.register_plugin("record", RecordingPlugin)

    plugins_before = get_all_plugins()
    plugins_copy = dict(plugins_before)
    plugins_copy["new"] = RecordingPlugin

    assert get_all_plugins() == plugins_before


def test_missing_plugin_error_lists_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """KeyError message includes available plugin names."""
    _reset_registry(monkeypatch)
    plugin_registry.register_plugin("present", RecordingPlugin)

    with pytest.raises(KeyError) as excinfo:
        get_plugin_for_target("absent")

    assert "available" in str(excinfo.value).lower()
    assert "present" in str(excinfo.value)


def test_duplicate_registration_logs_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Overwriting an existing plugin emits a warning."""
    _reset_registry(monkeypatch)
    caplog.set_level("WARNING")

    plugin_registry.register_plugin("record", RecordingPlugin)
    plugin_registry.register_plugin("record", RecordingPlugin)

    assert any("Overwriting plugin" in rec.message for rec in caplog.records)


def test_lazy_registration_handles_import_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Import errors during lazy registration are logged and do not raise."""
    monkeypatch.setattr(plugin_registry, "_PLUGINS", {})
    monkeypatch.setattr(plugin_registry, "_REGISTERED", False)
    monkeypatch.setattr(
        plugin_registry,
        "_PLUGIN_DEFINITIONS",
        (("missing.module", "MissingClass", ("missing",)),),
    )
    caplog.set_level("WARNING")

    monkeypatch.setattr(
        plugin_registry.importlib,
        "import_module",
        lambda module_path: (_ for _ in ()).throw(ImportError(module_path)),
    )

    plugins = get_all_plugins()

    assert plugins == {}
    assert any("Failed to register plugin" in rec.message for rec in caplog.records)


def test_lazy_registration_handles_attribute_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing attributes during registration are logged."""
    monkeypatch.setattr(plugin_registry, "_PLUGINS", {})
    monkeypatch.setattr(plugin_registry, "_REGISTERED", False)
    monkeypatch.setattr(
        plugin_registry,
        "_PLUGIN_DEFINITIONS",
        (("dummy.module", "MissingClass", ("dummy",)),),
    )
    caplog.set_level("WARNING")

    module = types.SimpleNamespace()
    monkeypatch.setattr(plugin_registry.importlib, "import_module", lambda _path: module)

    plugins = get_all_plugins()

    assert plugins == {}
    assert any("Failed to register plugin" in rec.message for rec in caplog.records)


def test_register_plugin_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decorator registers plugin class and returns it."""
    monkeypatch.setattr("codeintel.build.plugins._PLUGIN_REGISTRY", {})

    @register_plugin
    class DecoratedPlugin(RecordingPlugin, TargetPlugin):
        plugin_name: ClassVar[str] = "decorated"

    assert get_plugin("decorated") is DecoratedPlugin
    assert "decorated" in all_plugins()


def test_all_plugins_is_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """all_plugins returns a copy of the registry."""
    monkeypatch.setattr("codeintel.build.plugins._PLUGIN_REGISTRY", {})

    @register_plugin
    class AnotherPlugin(RecordingPlugin, TargetPlugin):
        plugin_name: ClassVar[str] = "another"

    plugins = all_plugins()
    plugins.pop("another")

    assert "another" in all_plugins()


def test_get_plugin_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_plugin returns None when plugin is absent."""
    monkeypatch.setattr("codeintel.build.plugins._PLUGIN_REGISTRY", {})
    assert get_plugin("absent") is None
