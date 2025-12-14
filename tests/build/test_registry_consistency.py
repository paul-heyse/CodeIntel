"""Test plugin and target registry consistency.

This module ensures that the plugin registry and target registry remain
in sync, preventing orphaned plugins or targets without implementations.
"""

from __future__ import annotations

from codeintel.build.plugin_registry import _PLUGIN_DEFINITIONS
from codeintel.build.registry import ALL_TARGETS
from tests._helpers.assertions import expect_true


class TestRegistryConsistency:
    """Ensure plugin registry and target registry are aligned."""

    @staticmethod
    def test_all_plugin_targets_have_output_targets() -> None:
        """Every plugin target must have a corresponding OutputTarget."""
        target_names = {t.name for t in ALL_TARGETS}
        plugin_targets = {target for _, _, targets in _PLUGIN_DEFINITIONS for target in targets}

        missing = plugin_targets - target_names
        expect_true(
            len(missing) == 0,
            message=f"Plugin targets without OutputTarget: {missing}",
        )

    @staticmethod
    def test_plugin_backed_targets_have_plugins() -> None:
        """Targets with non-empty plugin field must have plugin registration."""
        plugin_targets = {target for _, _, targets in _PLUGIN_DEFINITIONS for target in targets}

        for target in ALL_TARGETS:
            if target.plugin and target.plugin != "":
                expect_true(
                    target.name in plugin_targets,
                    message=(
                        f"Target '{target.name}' has plugin='{target.plugin}' "
                        f"but no plugin registration"
                    ),
                )
