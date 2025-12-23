"""Pytest configuration and shared fixtures for core module tests.

This module provides reusable fixtures for testing the core infrastructure,
including configuration providers, resource registries, and plugin metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.core.plugins.execution.context import ConfigProvider, PluginScratch
from codeintel.core.plugins.types.protocol import (
    PluginMetadata,
)
from codeintel.core.resources.registry import ResourceRegistry
from tests._helpers.env import build_test_gateway
from tests._helpers.fakes.configs import create_test_snapshot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass
class SampleDatabaseConfig:
    """Sample database configuration for testing."""

    host: str
    port: int
    database: str


@dataclass
class SampleAppConfig:
    """Sample application configuration for testing."""

    debug: bool
    log_level: str


@dataclass
class SampleCacheConfig:
    """Sample cache configuration for testing."""

    ttl_seconds: int
    max_entries: int


@pytest.fixture
def config_provider() -> ConfigProvider:
    """Create a ConfigProvider with sample configurations.

    Returns
    -------
    ConfigProvider
        Provider with database and app configs.
    """
    return ConfigProvider(
        {
            SampleDatabaseConfig: SampleDatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
            ),
            SampleAppConfig: SampleAppConfig(
                debug=True,
                log_level="DEBUG",
            ),
        }
    )


@pytest.fixture
def core_gateway() -> Iterator[StorageGateway]:
    """Provide standard gateway for core tests.

    Yields
    ------
    StorageGateway
        Gateway with schema and macros applied; automatically closed.
    """
    gateway = build_test_gateway()
    yield gateway
    gateway.close()


@pytest.fixture
def core_snapshot(tmp_path: Path) -> SnapshotRef:
    """Provide standard snapshot for core tests.

    Parameters
    ----------
    tmp_path
        Pytest temporary path fixture.

    Returns
    -------
    SnapshotRef
        Snapshot with standard test defaults.
    """
    return create_test_snapshot(tmp_path)


@pytest.fixture
def resource_registry() -> ResourceRegistry:
    """Create a fresh ResourceRegistry instance for testing.

    Returns
    -------
    ResourceRegistry
        Empty resource registry instance.
    """
    return ResourceRegistry()


@pytest.fixture
def sample_plugin_metadata() -> PluginMetadata:
    """Create a sample PluginMetadata for testing.

    Returns
    -------
    PluginMetadata
        Fully populated metadata instance.
    """
    return PluginMetadata(
        name="test.sample_plugin",
        description="A sample plugin for testing",
        kind="analytics",
        stage="function",
        version="1.0.0",
        enabled_by_default=True,
        provides=("sample.capability",),
        requires=(),
        depends_on=(),
        produces_tables=("analytics.sample_table",),
    )


@pytest.fixture
def minimal_plugin_metadata() -> PluginMetadata:
    """Create minimal PluginMetadata with only required fields.

    Returns
    -------
    PluginMetadata
        Metadata with only required fields set.
    """
    return PluginMetadata(
        name="test.minimal",
        description="Minimal test plugin",
        kind="analytics",
        stage="other",
    )


@pytest.fixture
def builder_plugin_metadata() -> PluginMetadata:
    """Create PluginMetadata for a builder-type plugin.

    Returns
    -------
    PluginMetadata
        Builder plugin metadata that produces graphs.
    """
    return PluginMetadata(
        name="test.builder",
        description="A builder plugin for testing",
        kind="builder",
        stage="goid",
        produces_graphs=("call_graph",),
    )


@pytest.fixture
def metric_plugin_metadata() -> PluginMetadata:
    """Create PluginMetadata for a metric-type plugin.

    Returns
    -------
    PluginMetadata
        Metric plugin metadata that requires graphs.
    """
    return PluginMetadata(
        name="test.metric",
        description="A metric plugin for testing",
        kind="metric",
        stage="core",
        requires_graphs=("call_graph",),
    )


@pytest.fixture
def plugin_scratch() -> PluginScratch:
    """Create a fresh PluginScratch instance for testing.

    Returns
    -------
    PluginScratch
        Empty scratch store.
    """
    return PluginScratch()


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """Create a temporary repository root directory.

    Returns
    -------
    Path
        Path to the temporary repo directory.
    """
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return repo


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    """Create a temporary build directory.

    Returns
    -------
    Path
        Path to the temporary build directory.
    """
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    return build


__all__ = [
    "SampleAppConfig",
    "SampleCacheConfig",
    "SampleDatabaseConfig",
]
