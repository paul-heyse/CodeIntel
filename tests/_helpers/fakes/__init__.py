"""Typed fakes for ingestion and analytics tests.

This package provides fake implementations for testing, organized by domain:
- coverage: Fake coverage data and loaders
- tools: Fake tool runners and services
- storage: Fake storage adapters
- configs: Fake configuration primitives
- utilities: Shared utility functions
"""

from __future__ import annotations

from tests._helpers.fakes.configs import (
    FakeBuildPaths,
    FakePluginContext,
    FakeSnapshotRef,
)
from tests._helpers.fakes.coverage import (
    CoverageLoader,
    FakeCoverage,
    FakeCoverageData,
)
from tests._helpers.fakes.plugins import (
    GraphPluginPack,
    GraphPluginPackCounters,
    GraphPluginPackSettings,
    TestGraphPlugin,
    build_graph_plugin_pack,
)
from tests._helpers.fakes.serving import (
    ScopeRecordingQuery,
    ServingScopePack,
    build_serving_scope_pack,
)
from tests._helpers.fakes.storage import FakeIngestStorage
from tests._helpers.fakes.tools import (
    FakeScipResult,
    FakeToolRunner,
    FakeToolService,
    FakeToolServiceConfig,
    write_dummy_scip_files,
)
from tests._helpers.fakes.utilities import utcnow

__all__ = [
    "CoverageLoader",
    "FakeBuildPaths",
    "FakeCoverage",
    "FakeCoverageData",
    "FakeIngestStorage",
    "FakePluginContext",
    "FakeScipResult",
    "FakeSnapshotRef",
    "FakeToolRunner",
    "FakeToolService",
    "FakeToolServiceConfig",
    "GraphPluginPack",
    "GraphPluginPackCounters",
    "GraphPluginPackSettings",
    "ScopeRecordingQuery",
    "ServingScopePack",
    "TestGraphPlugin",
    "build_graph_plugin_pack",
    "build_serving_scope_pack",
    "utcnow",
    "write_dummy_scip_files",
]
