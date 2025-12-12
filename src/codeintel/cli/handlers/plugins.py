"""Plugin management handlers.

Handlers for plugin discovery, listing, and management operations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_invalid_plugin_manifest,
    fail_invalid_plugin_name,
    fail_plugin_no_manifest,
    fail_plugin_not_found,
)
from codeintel.cli.plugins import (
    PluginManifest,
    PluginTestHarness,
    create_plugin_scaffold,
    get_plugin_manager,
)

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginInfo:
    """Information about a single plugin."""

    name: str
    version: str
    description: str
    operations: int
    enabled: bool
    path: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "operations": self.operations,
            "enabled": self.enabled,
        }
        if self.path:
            result["path"] = self.path
        return result


@dataclass(frozen=True)
class PluginsListResult:
    """Result from listing plugins."""

    plugins: list[dict[str, object]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "plugins": self.plugins,
            "count": self.count,
        }


@dataclass(frozen=True)
class PluginsDiscoverResult:
    """Result from discovering plugins."""

    discovered: list[dict[str, str]]
    plugin_dirs: list[str]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "discovered": self.discovered,
            "plugin_dirs": self.plugin_dirs,
            "count": self.count,
        }


@dataclass(frozen=True)
class PluginInfoResult:
    """Result from getting plugin info."""

    name: str
    version: str
    description: str
    path: str
    operations: int
    enabled: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "path": self.path,
            "operations": self.operations,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class PluginPathsResult:
    """Result from listing plugin paths."""

    paths: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "paths": self.paths,
        }


@dataclass(frozen=True)
class PluginNewResult:
    """Result from creating a new plugin."""

    plugin_dir: str
    name: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "plugin_dir": self.plugin_dir,
            "name": self.name,
        }


@dataclass(frozen=True)
class PluginTestResult:
    """Result from testing a plugin."""

    plugin_name: str
    plugin_version: str
    tests_run: int
    passed: int
    results: list[dict[str, object]]
    registered_operations: list[str]
    all_passed: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "tests_run": self.tests_run,
            "passed": self.passed,
            "results": self.results,
            "registered_operations": self.registered_operations,
            "all_passed": self.all_passed,
        }


@dataclass(frozen=True)
class PluginValidateResult:
    """Result from validating a plugin."""

    valid: bool
    name: str
    version: str
    api_version: str
    capabilities: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "valid": self.valid,
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "capabilities": self.capabilities,
            "errors": self.errors,
        }


def plugins_list_handler(ctx: CommandContext) -> CliResult[PluginsListResult]:
    """List installed plugins.

    Parameters
    ----------
    ctx
        Command context (no params required).

    Returns
    -------
    CliResult[PluginsListResult]
        List of installed plugins.
    """
    _ = ctx.params.raw
    LOG.info("Listing installed plugins")

    manager = get_plugin_manager()
    plugins = manager.list_plugins()

    plugin_dicts: list[dict[str, object]] = [p.to_dict() for p in plugins]

    return CliResult.ok(PluginsListResult(plugins=plugin_dicts, count=len(plugins)))


def plugins_discover_handler(
    ctx: CommandContext,
) -> CliResult[PluginsDiscoverResult]:
    """Discover available plugins.

    Parameters
    ----------
    ctx
        Command context (no params required).

    Returns
    -------
    CliResult[PluginsDiscoverResult]
        Discovered plugins and search paths.
    """
    _ = ctx.params.raw
    LOG.info("Discovering available plugins")

    manager = get_plugin_manager()
    paths = manager.discover()
    loaded_names = {p.name for p in manager.loaded_plugins.values()}

    discovered: list[dict[str, str]] = []
    for path in paths:
        is_loaded = any(path.stem in name or name in path.stem for name in loaded_names)
        discovered.append(
            {
                "path": str(path),
                "name": path.name,
                "status": "loaded" if is_loaded else "available",
            }
        )

    plugin_dirs = [str(d) for d in manager.plugin_dirs]

    return CliResult.ok(
        PluginsDiscoverResult(
            discovered=discovered,
            plugin_dirs=plugin_dirs,
            count=len(discovered),
        )
    )


def plugins_info_handler(ctx: CommandContext) -> CliResult[PluginInfoResult]:
    """Get details about a plugin.

    Parameters
    ----------
    ctx
        Command context with params:
        - name: Plugin name

    Returns
    -------
    CliResult[PluginInfoResult]
        Plugin details.
    """
    name = ctx.params.require_str("name")
    LOG.info("Getting info for plugin: %s", name)

    manager = get_plugin_manager()
    plugin = manager.get_plugin(name)

    if plugin is None:
        return fail_plugin_not_found(name)

    return CliResult.ok(
        PluginInfoResult(
            name=plugin.name,
            version=plugin.version,
            description=plugin.description,
            path=str(plugin.path) if plugin.path else "",
            operations=plugin.operations,
            enabled=plugin.enabled,
        )
    )


def plugins_paths_handler(ctx: CommandContext) -> CliResult[PluginPathsResult]:
    """Show plugin search paths.

    Parameters
    ----------
    ctx
        Command context (no params required).

    Returns
    -------
    CliResult[PluginPathsResult]
        Plugin search paths.
    """
    _ = ctx.params.raw
    LOG.info("Listing plugin search paths")

    manager = get_plugin_manager()

    paths: list[dict[str, object]] = [
        {"path": str(plugin_dir), "exists": plugin_dir.exists()}
        for plugin_dir in manager.plugin_dirs
    ]

    return CliResult.ok(PluginPathsResult(paths=paths))


def plugins_new_handler(ctx: CommandContext) -> CliResult[PluginNewResult]:
    """Create a new plugin scaffold.

    Parameters
    ----------
    ctx
        Command context with params:
        - name: Plugin name
        - output: Optional output directory

    Returns
    -------
    CliResult[PluginNewResult]
        Created plugin info.
    """
    name = ctx.params.require_str("name")
    output_dir = ctx.params.get_path("output") or Path.cwd()

    LOG.info("Creating plugin scaffold: %s in %s", name, output_dir)

    pattern = re.compile(r"^[a-z][a-z0-9_-]*$")
    if not pattern.match(name):
        return fail_invalid_plugin_name(
            "Plugin name must be lowercase alphanumeric with hyphens/underscores"
        )

    plugin_dir = create_plugin_scaffold(name, output_dir)

    return CliResult.ok(PluginNewResult(plugin_dir=str(plugin_dir), name=name))


def plugins_test_handler(ctx: CommandContext) -> CliResult[PluginTestResult]:
    """Test a plugin.

    Parameters
    ----------
    ctx
        Command context with params:
        - path: Plugin directory path

    Returns
    -------
    CliResult[PluginTestResult]
        Test results.
    """
    path = ctx.params.get_path("path") or Path.cwd()

    LOG.info("Testing plugin at: %s", path)

    manifest_path = path / "plugin.json"
    if not manifest_path.exists():
        return fail_plugin_no_manifest(str(path))

    manifest = PluginManifest.load(manifest_path)
    harness = PluginTestHarness(manifest)
    results = harness.run_all_tests()

    all_passed = True
    result_dicts: list[dict[str, object]] = []

    for result in results:
        if not result.success:
            all_passed = False
        result_dicts.append(
            {
                "success": result.success,
                "message": result.message,
                "errors": result.errors,
                "warnings": result.warnings,
            }
        )

    summary = harness.get_summary()

    return CliResult.ok(
        PluginTestResult(
            plugin_name=manifest.name,
            plugin_version=manifest.version,
            tests_run=summary["tests_run"],
            passed=summary["passed"],
            results=result_dicts,
            registered_operations=summary["registered_operations"],
            all_passed=all_passed,
        )
    )


def plugins_validate_handler(
    ctx: CommandContext,
) -> CliResult[PluginValidateResult]:
    """Validate a plugin manifest.

    Parameters
    ----------
    ctx
        Command context with params:
        - path: Plugin directory path

    Returns
    -------
    CliResult[PluginValidateResult]
        Validation results.
    """
    path = ctx.params.get_path("path") or Path.cwd()

    LOG.info("Validating plugin at: %s", path)

    manifest_path = path / "plugin.json"
    if not manifest_path.exists():
        return fail_plugin_no_manifest(str(path))

    try:
        manifest = PluginManifest.load(manifest_path)
    except (KeyError, ValueError) as e:
        return fail_invalid_plugin_manifest(str(e))

    errors = manifest.validate()

    capabilities = [cap.value for cap in manifest.capabilities] if manifest.capabilities else []

    return CliResult.ok(
        PluginValidateResult(
            valid=len(errors) == 0,
            name=manifest.name,
            version=manifest.version,
            api_version=manifest.api_version,
            capabilities=capabilities,
            errors=errors,
        )
    )


__all__ = [
    "PluginInfo",
    "PluginInfoResult",
    "PluginNewResult",
    "PluginPathsResult",
    "PluginTestResult",
    "PluginValidateResult",
    "PluginsDiscoverResult",
    "PluginsListResult",
    "plugins_discover_handler",
    "plugins_info_handler",
    "plugins_list_handler",
    "plugins_new_handler",
    "plugins_paths_handler",
    "plugins_test_handler",
    "plugins_validate_handler",
]
