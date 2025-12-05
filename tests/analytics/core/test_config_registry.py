"""Tests for config registry.

This module tests:
- ConfigPluginMapping dataclass
- ConfigRegistry class
- Global registry functions
- BaseStepConfig dataclass
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

from codeintel.analytics.core.config_registry import (
    BaseStepConfig,
    ConfigPluginMapping,
    ConfigRegistry,
    get_config_registry,
    register_config,
    reset_config_registry,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.factories import make_snapshot

# Test constants (non-repo/commit)
TEST_PLUGIN_NAME = "test.plugin"
TEST_PLUGIN_NAME_2 = "test.plugin2"
TEST_PLUGIN_NAME_3 = "test.plugin3"
EXPECTED_PLUGIN_COUNT_2 = 2


@dataclass(frozen=True)
class MockStepConfig:
    """Mock step config for testing."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Return repository identifier."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Return repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class MockStepConfig2:
    """Another mock step config for testing."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Return repository identifier."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Return repository root path."""
        return self.snapshot.repo_root


class TestConfigPluginMapping:
    """Tests for ConfigPluginMapping dataclass."""

    @staticmethod
    def test_creates_mapping_with_defaults() -> None:
        """Verify mapping can be created with defaults."""
        mapping = ConfigPluginMapping(
            config_type=MockStepConfig,
            plugins=(TEST_PLUGIN_NAME,),
        )
        assert mapping.config_type is MockStepConfig
        assert mapping.plugins == (TEST_PLUGIN_NAME,)
        assert mapping.primary is None

    @staticmethod
    def test_creates_mapping_with_primary() -> None:
        """Verify mapping can be created with primary plugin."""
        mapping = ConfigPluginMapping(
            config_type=MockStepConfig,
            plugins=(TEST_PLUGIN_NAME, TEST_PLUGIN_NAME_2),
            primary=TEST_PLUGIN_NAME_2,
        )
        assert mapping.primary == TEST_PLUGIN_NAME_2

    @staticmethod
    def test_mapping_is_frozen() -> None:
        """Verify mapping is immutable."""
        mapping = ConfigPluginMapping(
            config_type=MockStepConfig,
            plugins=(TEST_PLUGIN_NAME,),
        )
        with pytest.raises(AttributeError):
            mapping.primary = TEST_PLUGIN_NAME  # type: ignore[misc]


class TestConfigRegistry:
    """Tests for ConfigRegistry class."""

    @staticmethod
    def test_register_config_type() -> None:
        """Verify config type registration."""
        registry = ConfigRegistry()
        registry.register(MockStepConfig, (TEST_PLUGIN_NAME,))

        plugins = registry.get_plugins_for_config(MockStepConfig)
        assert plugins == (TEST_PLUGIN_NAME,)

    @staticmethod
    def test_register_multiple_plugins() -> None:
        """Verify config can have multiple plugins."""
        registry = ConfigRegistry()
        registry.register(
            MockStepConfig,
            (TEST_PLUGIN_NAME, TEST_PLUGIN_NAME_2),
        )

        plugins = registry.get_plugins_for_config(MockStepConfig)
        assert len(plugins) == EXPECTED_PLUGIN_COUNT_2
        assert TEST_PLUGIN_NAME in plugins
        assert TEST_PLUGIN_NAME_2 in plugins

    @staticmethod
    def test_register_with_explicit_primary() -> None:
        """Verify explicit primary plugin is used."""
        registry = ConfigRegistry()
        registry.register(
            MockStepConfig,
            (TEST_PLUGIN_NAME, TEST_PLUGIN_NAME_2),
            primary=TEST_PLUGIN_NAME_2,
        )

        primary = registry.get_primary_plugin(MockStepConfig)
        assert primary == TEST_PLUGIN_NAME_2

    @staticmethod
    def test_register_uses_first_as_default_primary() -> None:
        """Verify first plugin is default primary."""
        registry = ConfigRegistry()
        registry.register(
            MockStepConfig,
            (TEST_PLUGIN_NAME, TEST_PLUGIN_NAME_2),
        )

        primary = registry.get_primary_plugin(MockStepConfig)
        assert primary == TEST_PLUGIN_NAME

    @staticmethod
    def test_register_duplicate_raises() -> None:
        """Verify duplicate registration raises error."""
        registry = ConfigRegistry()
        registry.register(MockStepConfig, (TEST_PLUGIN_NAME,))

        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockStepConfig, (TEST_PLUGIN_NAME_2,))

    @staticmethod
    def test_get_plugins_for_unregistered_returns_empty() -> None:
        """Verify unregistered config returns empty tuple."""
        registry = ConfigRegistry()
        plugins = registry.get_plugins_for_config(MockStepConfig)
        assert plugins == ()

    @staticmethod
    def test_get_primary_for_unregistered_returns_none() -> None:
        """Verify unregistered config returns None for primary."""
        registry = ConfigRegistry()
        primary = registry.get_primary_plugin(MockStepConfig)
        assert primary is None

    @staticmethod
    def test_get_required_configs() -> None:
        """Verify required configs are returned for plugin."""
        registry = ConfigRegistry()
        registry.register(MockStepConfig, (TEST_PLUGIN_NAME,))

        configs = registry.get_required_configs(TEST_PLUGIN_NAME)
        assert MockStepConfig in configs

    @staticmethod
    def test_get_required_configs_unknown_plugin() -> None:
        """Verify unknown plugin returns empty tuple."""
        registry = ConfigRegistry()
        configs = registry.get_required_configs("unknown.plugin")
        assert configs == ()

    @staticmethod
    def test_resolve_plugins_from_configs() -> None:
        """Verify plugins are resolved from configs."""
        registry = ConfigRegistry()
        registry.register(MockStepConfig, (TEST_PLUGIN_NAME,))
        registry.register(MockStepConfig2, (TEST_PLUGIN_NAME_2,))

        snapshot = make_snapshot(repo_root=Path("/test"))
        configs: dict[type, object] = {
            MockStepConfig: MockStepConfig(snapshot=snapshot),
            MockStepConfig2: MockStepConfig2(snapshot=snapshot),
        }

        plugins = registry.resolve_plugins_from_configs(configs)
        assert TEST_PLUGIN_NAME in plugins
        assert TEST_PLUGIN_NAME_2 in plugins

    @staticmethod
    def test_resolve_plugins_skips_unregistered() -> None:
        """Verify unregistered configs are skipped."""
        registry = ConfigRegistry()
        registry.register(MockStepConfig, (TEST_PLUGIN_NAME,))

        snapshot = make_snapshot(repo_root=Path("/test"))
        configs: dict[type, object] = {
            MockStepConfig: MockStepConfig(snapshot=snapshot),
            MockStepConfig2: MockStepConfig2(snapshot=snapshot),  # Not registered
        }

        plugins = registry.resolve_plugins_from_configs(configs)
        assert plugins == (TEST_PLUGIN_NAME,)

    @staticmethod
    def test_list_all_returns_all_mappings() -> None:
        """Verify list_all returns all mappings."""
        registry = ConfigRegistry()
        registry.register(MockStepConfig, (TEST_PLUGIN_NAME,))
        registry.register(MockStepConfig2, (TEST_PLUGIN_NAME_2,))

        mappings = registry.list_all()
        assert len(mappings) == EXPECTED_PLUGIN_COUNT_2


@pytest.fixture(autouse=True)
def _reset_global_registry() -> Generator[None]:
    """Reset registry before and after each test in this module."""
    reset_config_registry()
    yield
    reset_config_registry()


class TestGlobalRegistryFunctions:
    """Tests for global registry functions."""

    @staticmethod
    def test_get_config_registry_returns_singleton() -> None:
        """Verify get_config_registry returns same instance."""
        registry1 = get_config_registry()
        registry2 = get_config_registry()
        assert registry1 is registry2

    @staticmethod
    def test_reset_creates_new_instance() -> None:
        """Verify reset creates a new instance."""
        registry1 = get_config_registry()
        registry1.register(MockStepConfig, (TEST_PLUGIN_NAME,))

        reset_config_registry()

        registry2 = get_config_registry()
        # Should be empty after reset
        plugins = registry2.get_plugins_for_config(MockStepConfig)
        assert plugins == ()

    @staticmethod
    def test_register_config_uses_global_registry() -> None:
        """Verify register_config uses global registry."""
        register_config(MockStepConfig, (TEST_PLUGIN_NAME,))

        registry = get_config_registry()
        plugins = registry.get_plugins_for_config(MockStepConfig)
        assert plugins == (TEST_PLUGIN_NAME,)


class TestBaseStepConfig:
    """Tests for BaseStepConfig dataclass."""

    @staticmethod
    def test_creates_config_with_snapshot() -> None:
        """Verify config stores snapshot."""
        snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=Path("/test/repo"),
        )
        config = BaseStepConfig(snapshot=snapshot)
        assert config.snapshot == snapshot

    @staticmethod
    def test_repo_property() -> None:
        """Verify repo property returns snapshot.repo."""
        snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=Path("/test/repo"),
        )
        config = BaseStepConfig(snapshot=snapshot)
        assert config.repo == DEFAULT_REPO

    @staticmethod
    def test_commit_property() -> None:
        """Verify commit property returns snapshot.commit."""
        snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=Path("/test/repo"),
        )
        config = BaseStepConfig(snapshot=snapshot)
        assert config.commit == DEFAULT_COMMIT

    @staticmethod
    def test_repo_root_property() -> None:
        """Verify repo_root property returns snapshot.repo_root."""
        repo_root = Path("/test/repo")
        snapshot = SnapshotRef(
            repo=DEFAULT_REPO,
            commit=DEFAULT_COMMIT,
            repo_root=repo_root,
        )
        config = BaseStepConfig(snapshot=snapshot)
        assert config.repo_root == repo_root

    @staticmethod
    def test_config_is_frozen() -> None:
        """Verify config is immutable."""
        snapshot = make_snapshot(repo_root=Path("/test"))
        config = BaseStepConfig(snapshot=snapshot)
        with pytest.raises(AttributeError):
            config.snapshot = snapshot  # type: ignore[misc]
