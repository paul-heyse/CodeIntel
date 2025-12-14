"""Tests for UnifiedRegistry plugin resolution."""

from __future__ import annotations

from codeintel.build.unified_registry import get_unified_registry
from tests._helpers.assertions import expect_true


def test_unified_registry_instantiates_plugins() -> None:
    """UnifiedRegistry should instantiate plugin instances for plugin targets."""
    reg = get_unified_registry()
    plugin = reg.instantiate_plugin("modules")
    expect_true(hasattr(plugin, "execute"), message="Plugin should expose execute()")
