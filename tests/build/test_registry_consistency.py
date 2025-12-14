"""Test target registry and UnifiedRegistry consistency."""

from __future__ import annotations

from codeintel.build.registry import ALL_TARGETS
from codeintel.build.unified_registry import get_unified_registry
from tests._helpers.assertions import expect_true


class TestRegistryConsistency:
    """Ensure UnifiedRegistry and target registry are aligned."""

    @staticmethod
    def test_all_targets_registered_in_unified_registry() -> None:
        """Every OutputTarget should be present in the unified registry."""
        reg = get_unified_registry()
        missing = {t.name for t in ALL_TARGETS if t.name not in reg}
        expect_true(len(missing) == 0, message=f"Targets missing from UnifiedRegistry: {missing}")

    @staticmethod
    def test_targets_with_plugin_field_have_implementations() -> None:
        """Targets with a plugin field must have a plugin or native implementation."""
        reg = get_unified_registry()
        for target in ALL_TARGETS:
            if not target.plugin:
                continue
            expect_true(
                reg.has_implementation(target.name),
                message=(
                    f"Target '{target.name}' declares plugin='{target.plugin}' "
                    "but has no registered implementation"
                ),
            )
