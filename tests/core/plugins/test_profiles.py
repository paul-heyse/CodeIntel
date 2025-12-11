"""Tests for execution profiles."""

from __future__ import annotations

import pytest

from codeintel.core.plugins.execution.profiles import (
    CI_PROFILE,
    FAST_PROFILE,
    FULL_PROFILE,
    ExecutionProfile,
    get_profile,
    list_profiles,
    register_profile,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)


class TestExecutionProfile:
    """Tests for ExecutionProfile dataclass."""

    @staticmethod
    def test_get_plugin_options_existing() -> None:
        """Verify get_plugin_options returns options for configured plugin."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
            plugin_options={
                "plugin.a": {"key": "value"},
            },
        )
        opts = profile.get_plugin_options("plugin.a")
        expect_equal(opts, {"key": "value"})

    @staticmethod
    def test_get_plugin_options_missing() -> None:
        """Verify get_plugin_options returns None for unconfigured plugin."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
        )
        expect_is_none(profile.get_plugin_options("unknown"))

    @staticmethod
    def test_get_effective_options_merges_shared() -> None:
        """Verify get_effective_options merges shared and plugin options."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
            shared_options={"scope_paths": ["src/"]},
            plugin_options={
                "plugin.a": {"enabled": True},
            },
        )
        opts = profile.get_effective_options("plugin.a")
        expect_equal(opts, {"scope_paths": ["src/"], "enabled": True})

    @staticmethod
    def test_plugin_options_override_shared() -> None:
        """Verify plugin options override shared options."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
            shared_options={"key": "shared"},
            plugin_options={
                "plugin.a": {"key": "plugin"},
            },
        )
        opts = profile.get_effective_options("plugin.a")
        expect_equal(opts["key"], "plugin")


class TestBuiltinProfiles:
    """Tests for builtin profiles."""

    @staticmethod
    def test_fast_profile_disables_expensive_features() -> None:
        """Verify fast profile disables expensive computations."""
        opts = FAST_PROFILE.get_plugin_options("analytics.function_metrics")
        expect_is_not_none(opts)
        expect_true(opts is not None and opts.get("include_graph_metrics") is False)

    @staticmethod
    def test_full_profile_enables_all_features() -> None:
        """Verify full profile enables all features."""
        opts = FULL_PROFILE.get_plugin_options("analytics.function_metrics")
        expect_is_not_none(opts)
        expect_true(opts is not None and opts.get("include_graph_metrics") is True)

    @staticmethod
    def test_ci_profile_has_balanced_settings() -> None:
        """Verify CI profile has balanced settings."""
        opts = CI_PROFILE.get_plugin_options("graphs.callgraph")
        expect_is_not_none(opts)
        expect_true(opts is not None and opts.get("include_external_calls") is False)


class TestProfileRegistry:
    """Tests for profile registry functions."""

    @staticmethod
    def test_get_profile_builtin() -> None:
        """Verify get_profile returns builtin profiles."""
        expect_equal(get_profile("fast"), FAST_PROFILE)
        expect_equal(get_profile("full"), FULL_PROFILE)
        expect_equal(get_profile("ci"), CI_PROFILE)

    @staticmethod
    def test_get_profile_unknown_raises() -> None:
        """Verify get_profile raises for unknown profiles."""
        with pytest.raises(KeyError, match="Unknown profile"):
            get_profile("nonexistent")

    @staticmethod
    def test_list_profiles_includes_builtins() -> None:
        """Verify list_profiles includes builtin profiles."""
        profiles = list_profiles()
        expect_in("fast", profiles)
        expect_in("full", profiles)
        expect_in("ci", profiles)

    @staticmethod
    def test_register_profile_custom() -> None:
        """Verify custom profile can be registered."""
        custom = ExecutionProfile(
            name="test_custom_unique",
            description="Custom test profile.",
        )
        register_profile(custom)
        expect_equal(get_profile("test_custom_unique"), custom)

    @staticmethod
    def test_register_profile_duplicate_raises() -> None:
        """Verify registering duplicate profile raises."""
        custom = ExecutionProfile(
            name="test_dup",
            description="Test.",
        )
        register_profile(custom)
        with pytest.raises(ValueError, match="already registered"):
            register_profile(custom)

    @staticmethod
    def test_cannot_override_builtin() -> None:
        """Verify builtin profiles cannot be overridden."""
        fake_fast = ExecutionProfile(
            name="fast",
            description="Fake fast.",
        )
        with pytest.raises(ValueError, match="Cannot override builtin"):
            register_profile(fake_fast)
