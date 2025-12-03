"""Bundled test infrastructure for ingestion tests.

This module provides a unified `TestIngestSetup` dataclass that bundles all
common test components together, reducing local variable sprawl and providing
factory methods for quick setup.

Example
-------
>>> from tests._helpers.ingest_setup import IngestTestSetup
>>> from tests._helpers.gateway import open_ingestion_gateway
>>>
>>> def test_my_plugin(tmp_path):
...     gateway = open_ingestion_gateway()
...     setup = IngestTestSetup.from_repo(tmp_path / "repo", gateway=gateway)
...     ctx = setup.build_context("my_plugin")
...     # Use ctx for plugin execution...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from codeintel.config import BuildPaths, SnapshotRef
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.infrastructure_utilities.source_scanner import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
)
from codeintel.ingestion.plugins import IngestRuntimeScratch
from codeintel.ingestion.resources.modules import ModuleProvider
from codeintel.ingestion.resources.registry import ResourceRegistry
from codeintel.ingestion.resources.tools import ToolsProvider
from codeintel.ingestion.resources.tracker import TrackerConfig, TrackerProvider
from tests._helpers.tooling import make_tools_config

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@dataclass
class IngestTestSetup:
    """Bundled test infrastructure for ingestion tests.

    Centralize all common test components to reduce local variable count
    and provide factory methods for quick setup. This addresses PLR0914
    (too many local variables) by bundling related objects together.

    Attributes
    ----------
    gateway : StorageGateway
        Storage gateway for database access.
    snapshot : SnapshotRef
        Snapshot reference with repo, commit, and root.
    paths : BuildPaths
        Build paths for artifacts and outputs.
    tools : ToolsConfig
        External tool configuration.
    code_profile : ScanProfile
        Code scanning profile.
    config_profile : ScanProfile
        Configuration profile.
    scratch : IngestRuntimeScratch
        Runtime scratch space for inter-plugin communication.
    resources : ResourceRegistry
        Resource registry with registered providers.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    tools: ToolsConfig
    code_profile: ScanProfile
    config_profile: ScanProfile
    scratch: IngestRuntimeScratch = field(default_factory=IngestRuntimeScratch)
    resources: ResourceRegistry = field(default_factory=ResourceRegistry)

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        *,
        gateway: StorageGateway,
        repo: str = "test/repo",
        commit: str = "testcommit",
        build_dir: Path | None = None,
    ) -> Self:
        """Create complete setup from repo root with sensible defaults.

        Factory method that creates all necessary components with reasonable
        defaults, reducing boilerplate in tests.

        Parameters
        ----------
        repo_root
            Path to the repository root directory.
        gateway
            Storage gateway for database access.
        repo
            Repository identifier (default: "test/repo").
        commit
            Commit identifier (default: "testcommit").
        build_dir
            Optional override for build directory.

        Returns
        -------
        Self
            Fully configured TestIngestSetup instance.
        """
        snapshot = SnapshotRef.from_args(repo=repo, commit=commit, repo_root=repo_root)
        paths = BuildPaths.from_repo_root(repo_root, build_dir=build_dir)
        tools = make_tools_config()
        code_profile = default_code_profile(repo_root)
        config_profile = default_config_profile(repo_root)
        scratch = IngestRuntimeScratch()
        resources = ResourceRegistry()

        instance = cls(
            gateway=gateway,
            snapshot=snapshot,
            paths=paths,
            tools=tools,
            code_profile=code_profile,
            config_profile=config_profile,
            scratch=scratch,
            resources=resources,
        )
        instance._register_default_providers()
        return instance

    def _register_default_providers(self) -> None:
        """Register standard resource providers.

        Set up ModuleProvider, ToolsProvider, and TrackerProvider with
        the current configuration.
        """
        tracker_config = TrackerConfig(
            scratch=self.scratch,
            profile=self.code_profile,
        )
        self.resources.register(
            TrackerProvider,
            TrackerProvider(
                self.gateway,
                self.snapshot,
                tracker_config,
            ),
        )
        self.resources.register(
            ToolsProvider,
            ToolsProvider(self.tools, self.paths.tool_cache),
        )
        self.resources.register(
            ModuleProvider,
            ModuleProvider(
                self.gateway,
                self.snapshot,
                profile=self.code_profile,
            ),
        )

    def build_context(self, plugin_name: str = "test") -> IngestExecutionContext:
        """Build execution context from this setup.

        Create an IngestExecutionContext with all components from this setup.

        Parameters
        ----------
        plugin_name
            Name of the plugin for context identification.

        Returns
        -------
        IngestExecutionContext
            Fully configured execution context.
        """
        return IngestExecutionContext(
            gateway=self.gateway,
            snapshot=self.snapshot,
            paths=self.paths,
            tools=self.tools,
            code_profile=self.code_profile,
            config_profile=self.config_profile,
            scratch=self.scratch,
            resources=self.resources,
            plugin_name=plugin_name,
        )

    def with_fresh_scratch(self) -> IngestTestSetup:
        """Return new setup with fresh scratch space.

        Useful when running multiple plugins in sequence where you want
        to reset scratch state between executions.

        Returns
        -------
        IngestTestSetup
            New setup with fresh scratch and re-registered providers.
        """
        new_scratch = IngestRuntimeScratch()
        new_resources = ResourceRegistry()

        new_setup = IngestTestSetup(
            gateway=self.gateway,
            snapshot=self.snapshot,
            paths=self.paths,
            tools=self.tools,
            code_profile=self.code_profile,
            config_profile=self.config_profile,
            scratch=new_scratch,
            resources=new_resources,
        )
        new_setup._register_default_providers()
        return new_setup
