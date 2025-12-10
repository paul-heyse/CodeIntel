"""Plugin management handlers.

Handlers for plugin discovery, listing, and management operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.plugins import (
    PluginManifest,
    PluginTestHarness,
    create_plugin_scaffold,
    get_plugin_manager,
)
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.handlers.protocol import EnhancedHandlerContext

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


def _get_str_param(
    ctx: EnhancedHandlerContext,
    name: str,
    default: str | None = None,
) -> str | None:
    """Extract string parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.
    default
        Default value if not present.

    Returns
    -------
    str | None
        Parameter value or default.
    """
    value = ctx.params.get(name)
    if value is None:
        return default
    return str(value)


def _require_str_param(ctx: EnhancedHandlerContext, name: str) -> str:
    """Extract required string parameter from context.

    Parameters
    ----------
    ctx
        Handler context.
    name
        Parameter name.

    Returns
    -------
    str
        Parameter value.

    Raises
    ------
    ValueError
        If parameter is missing.
    """
    value = ctx.params.get(name)
    if value is None:
        msg = f"{name} parameter is required"
        raise ValueError(msg)
    return str(value)


def plugins_list_handler(ctx: EnhancedHandlerContext) -> CliResult[PluginsListResult]:
    """List installed plugins.

    Parameters
    ----------
    ctx
        Handler context (no params required).

    Returns
    -------
    CliResult[PluginsListResult]
        List of installed plugins.
    """
    _ = ctx.params  # Acknowledge params
    LOG.info("Listing installed plugins")

    manager = get_plugin_manager()
    plugins = manager.list_plugins()

    plugin_dicts: list[dict[str, object]] = [p.to_dict() for p in plugins]

    return CliResult.ok(PluginsListResult(plugins=plugin_dicts, count=len(plugins)))


def plugins_discover_handler(
    ctx: EnhancedHandlerContext,
) -> CliResult[PluginsDiscoverResult]:
    """Discover available plugins.

    Parameters
    ----------
    ctx
        Handler context (no params required).

    Returns
    -------
    CliResult[PluginsDiscoverResult]
        Discovered plugins and search paths.
    """
    _ = ctx.params  # Acknowledge params
    LOG.info("Discovering available plugins")

    manager = get_plugin_manager()
    paths = manager.discover()
    loaded_names = {p.name for p in manager.loaded_plugins.values()}

    discovered: list[dict[str, str]] = []
    for path in paths:
        is_loaded = any(path.stem in name or name in path.stem for name in loaded_names)
        discovered.append({
            "path": str(path),
            "name": path.name,
            "status": "loaded" if is_loaded else "available",
        })

    plugin_dirs = [str(d) for d in manager.plugin_dirs]

    return CliResult.ok(
        PluginsDiscoverResult(
            discovered=discovered,
            plugin_dirs=plugin_dirs,
            count=len(discovered),
        )
    )


def plugins_info_handler(ctx: EnhancedHandlerContext) -> CliResult[PluginInfoResult]:
    """Get details about a plugin.

    Parameters
    ----------
    ctx
        Handler context with params:
        - name: Plugin name

    Returns
    -------
    CliResult[PluginInfoResult]
        Plugin details.
    """
    name = _require_str_param(ctx, "name")
    LOG.info("Getting info for plugin: %s", name)

    manager = get_plugin_manager()
    plugin = manager.get_plugin(name)

    if plugin is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:plugins:not-found",
                title="Plugin Not Found",
                detail=f"Plugin not found: {name}",
                status=404,
            )
        )

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


def plugins_paths_handler(ctx: EnhancedHandlerContext) -> CliResult[PluginPathsResult]:
    """Show plugin search paths.

    Parameters
    ----------
    ctx
        Handler context (no params required).

    Returns
    -------
    CliResult[PluginPathsResult]
        Plugin search paths.
    """
    _ = ctx.params  # Acknowledge params
    LOG.info("Listing plugin search paths")

    manager = get_plugin_manager()

    paths: list[dict[str, object]] = [
        {"path": str(plugin_dir), "exists": plugin_dir.exists()}
        for plugin_dir in manager.plugin_dirs
    ]

    return CliResult.ok(PluginPathsResult(paths=paths))


def plugins_new_handler(ctx: EnhancedHandlerContext) -> CliResult[PluginNewResult]:
    """Create a new plugin scaffold.

    Parameters
    ----------
    ctx
        Handler context with params:
        - name: Plugin name
        - output: Optional output directory

    Returns
    -------
    CliResult[PluginNewResult]
        Created plugin info.
    """
    import re  # noqa: PLC0415

    name = _require_str_param(ctx, "name")
    output_str = _get_str_param(ctx, "output")
    output_dir = Path(output_str) if output_str else Path.cwd()

    LOG.info("Creating plugin scaffold: %s in %s", name, output_dir)

    # Validate name
    pattern = re.compile(r"^[a-z][a-z0-9_-]*$")
    if not pattern.match(name):
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:plugins:invalid-name",
                title="Invalid Plugin Name",
                detail="Plugin name must be lowercase alphanumeric with hyphens/underscores",
                status=400,
            )
        )

    plugin_dir = create_plugin_scaffold(name, output_dir)

    return CliResult.ok(PluginNewResult(plugin_dir=str(plugin_dir), name=name))


def plugins_test_handler(ctx: EnhancedHandlerContext) -> CliResult[PluginTestResult]:
    """Test a plugin.

    Parameters
    ----------
    ctx
        Handler context with params:
        - path: Plugin directory path

    Returns
    -------
    CliResult[PluginTestResult]
        Test results.
    """
    path_str = _require_str_param(ctx, "path")
    path = Path(path_str)

    LOG.info("Testing plugin at: %s", path)

    manifest_path = path / "plugin.json"
    if not manifest_path.exists():
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:plugins:no-manifest",
                title="No Plugin Manifest",
                detail=f"No plugin.json found in {path}",
                status=404,
            )
        )

    manifest = PluginManifest.load(manifest_path)
    harness = PluginTestHarness(manifest)
    results = harness.run_all_tests()

    all_passed = True
    result_dicts: list[dict[str, object]] = []

    for result in results:
        if not result.success:
            all_passed = False
        result_dicts.append({
            "success": result.success,
            "message": result.message,
            "errors": result.errors,
            "warnings": result.warnings,
        })

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
    ctx: EnhancedHandlerContext,
) -> CliResult[PluginValidateResult]:
    """Validate a plugin manifest.

    Parameters
    ----------
    ctx
        Handler context with params:
        - path: Plugin directory path

    Returns
    -------
    CliResult[PluginValidateResult]
        Validation results.
    """
    path_str = _require_str_param(ctx, "path")
    path = Path(path_str)

    LOG.info("Validating plugin at: %s", path)

    manifest_path = path / "plugin.json"
    if not manifest_path.exists():
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:plugins:no-manifest",
                title="No Plugin Manifest",
                detail=f"No plugin.json found in {path}",
                status=404,
            )
        )

    try:
        manifest = PluginManifest.load(manifest_path)
    except (KeyError, ValueError) as e:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:plugins:invalid-manifest",
                title="Invalid Plugin Manifest",
                detail=f"Error loading manifest: {e}",
                status=400,
            )
        )

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
