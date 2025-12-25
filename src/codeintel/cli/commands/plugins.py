"""Plugin management commands.

Provide commands to discover, list, and inspect CLI plugins
using the Command[T] pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import App

from codeintel.cli.commands.decorators import cli_command
from codeintel.cli.core import CliResult
from codeintel.cli.core.command import Command
from codeintel.cli.core.results import result_type
from codeintel.cli.errors.results import (
    fail_invalid_plugin_manifest,
    fail_invalid_plugin_name,
    fail_plugin_no_manifest,
    fail_plugin_not_found,
)
from codeintel.cli.options.registry import PLUGINS_NAME, PLUGINS_OUTPUT_DIR, PLUGINS_PATH
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param
from codeintel.cli.plugins import (
    PluginManifest,
    PluginTestHarness,
    create_plugin_scaffold,
    get_plugin_manager,
)

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext

LOG = logging.getLogger(__name__)

plugins_app = App(name="plugins", help="Manage CLI plugins")

PLUGINS_LIST_PATH: CommandPath = ("plugins", "list")
PLUGINS_DISCOVER_PATH: CommandPath = ("plugins", "discover")
PLUGINS_INFO_PATH: CommandPath = ("plugins", "info")
PLUGINS_PATHS_PATH: CommandPath = ("plugins", "paths")
PLUGINS_NEW_PATH: CommandPath = ("plugins", "new")
PLUGINS_TEST_PATH: CommandPath = ("plugins", "test")
PLUGINS_VALIDATE_PATH: CommandPath = ("plugins", "validate")

_PLUGIN_VALIDATE_CAPABILITIES_FIELD = field(default_factory=list)
_PLUGIN_VALIDATE_ERRORS_FIELD = field(default_factory=list)
_PLUGINS_LIST_FLAGS_FIELD = shared_flags_field(PLUGINS_LIST_PATH)
_PLUGINS_DISCOVER_FLAGS_FIELD = shared_flags_field(PLUGINS_DISCOVER_PATH)
_PLUGINS_INFO_FLAGS_FIELD = shared_flags_field(PLUGINS_INFO_PATH)
_PLUGINS_PATHS_FLAGS_FIELD = shared_flags_field(PLUGINS_PATHS_PATH)
_PLUGINS_NEW_FLAGS_FIELD = shared_flags_field(PLUGINS_NEW_PATH)
_PLUGINS_TEST_FLAGS_FIELD = shared_flags_field(PLUGINS_TEST_PATH)
_PLUGINS_VALIDATE_FLAGS_FIELD = shared_flags_field(PLUGINS_VALIDATE_PATH)


@result_type
@dataclass(frozen=True)
class PluginInfo:
    """Information about a single plugin.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version.
    description
        Plugin description.
    operations
        Number of operations registered.
    enabled
        Whether the plugin is enabled.
    path
        Plugin directory path.
    """

    name: str
    version: str
    description: str
    operations: int
    enabled: bool
    path: str | None = None


@result_type
@dataclass(frozen=True)
class PluginsListResult:
    """Result from listing plugins.

    Parameters
    ----------
    plugins
        List of plugin information dictionaries.
    count
        Total number of plugins.
    """

    plugins: list[dict[str, object]]
    count: int


@result_type
@dataclass(frozen=True)
class PluginsDiscoverResult:
    """Result from discovering plugins.

    Parameters
    ----------
    discovered
        List of discovered plugin paths.
    plugin_dirs
        Plugin search directories.
    count
        Number of plugins discovered.
    """

    discovered: list[dict[str, str]]
    plugin_dirs: list[str]
    count: int


@result_type
@dataclass(frozen=True)
class PluginInfoResult:
    """Result from getting plugin info.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version.
    description
        Plugin description.
    path
        Plugin directory path.
    operations
        Number of operations.
    enabled
        Whether enabled.
    """

    name: str
    version: str
    description: str
    path: str
    operations: int
    enabled: bool


@result_type
@dataclass(frozen=True)
class PluginPathsResult:
    """Result from listing plugin paths.

    Parameters
    ----------
    paths
        Plugin search path information.
    """

    paths: list[dict[str, object]]


@result_type
@dataclass(frozen=True)
class PluginNewResult:
    """Result from creating a new plugin.

    Parameters
    ----------
    plugin_dir
        Created plugin directory.
    name
        Plugin name.
    """

    plugin_dir: str
    name: str


@result_type
@dataclass(frozen=True)
class PluginTestResult:
    """Result from testing a plugin.

    Parameters
    ----------
    plugin_name
        Name of the tested plugin.
    plugin_version
        Version of the tested plugin.
    tests_run
        Number of tests run.
    passed
        Number of tests passed.
    results
        Individual test results.
    registered_operations
        Operations registered by the plugin.
    all_passed
        Whether all tests passed.
    """

    plugin_name: str
    plugin_version: str
    tests_run: int
    passed: int
    results: list[dict[str, object]]
    registered_operations: list[str]
    all_passed: bool


@result_type
@dataclass(frozen=True)
class PluginValidateResult:
    """Result from validating a plugin.

    Parameters
    ----------
    valid
        Whether the manifest is valid.
    name
        Plugin name.
    version
        Plugin version.
    api_version
        API version.
    capabilities
        Plugin capabilities.
    errors
        Validation errors.
    """

    valid: bool
    name: str
    version: str
    api_version: str
    capabilities: list[str] = _PLUGIN_VALIDATE_CAPABILITIES_FIELD
    errors: list[str] = _PLUGIN_VALIDATE_ERRORS_FIELD


@cli_command("plugins.list", require_storage=False)
@plugins_app.command(name="list")
@dataclass(frozen=True)
class PluginsList(Command[PluginsListResult]):
    """List installed plugins.

    Display a table of all loaded plugins with their version,
    operation count, and description.
    """

    __operation_id__ = "plugins.list"

    flags: SharedFlagsProtocol = _PLUGINS_LIST_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginsListResult]:
        """Execute plugin listing.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginsListResult]
            List of installed plugins.
        """
        _ = self.flags
        _ = ctx
        LOG.info("Listing installed plugins")

        manager = get_plugin_manager()
        plugins = manager.list_plugins()

        plugin_dicts: list[dict[str, object]] = [p.to_dict() for p in plugins]

        return CliResult.ok(PluginsListResult(plugins=plugin_dicts, count=len(plugins)))


@cli_command("plugins.discover", require_storage=False)
@plugins_app.command(name="discover")
@dataclass(frozen=True)
class PluginsDiscover(Command[PluginsDiscoverResult]):
    """Discover available plugins.

    Search plugin directories for available plugins and show
    their loading status.
    """

    __operation_id__ = "plugins.discover"

    flags: SharedFlagsProtocol = _PLUGINS_DISCOVER_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginsDiscoverResult]:
        """Execute plugin discovery.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginsDiscoverResult]
            Discovered plugins and search paths.
        """
        _ = self.flags
        _ = ctx
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


@cli_command("plugins.info", require_storage=False)
@plugins_app.command(name="info")
@dataclass(frozen=True)
class PluginsInfo(Command[PluginInfoResult]):
    """Show details about a plugin.

    Display detailed information about a specific loaded plugin
    including its operations and file path.
    """

    __operation_id__ = "plugins.info"

    name: Annotated[str, option_param(PLUGINS_NAME, command_path=PLUGINS_INFO_PATH)]
    flags: SharedFlagsProtocol = _PLUGINS_INFO_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginInfoResult]:
        """Execute plugin info query.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginInfoResult]
            Plugin details.
        """
        _ = ctx
        LOG.info("Getting info for plugin: %s", self.name)

        manager = get_plugin_manager()
        plugin = manager.get_plugin(self.name)

        if plugin is None:
            return fail_plugin_not_found(self.name)

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


@cli_command("plugins.paths", require_storage=False)
@plugins_app.command(name="paths")
@dataclass(frozen=True)
class PluginsPaths(Command[PluginPathsResult]):
    """Show plugin search paths.

    Display all directories where plugins are searched,
    with indicators for which paths exist.
    """

    __operation_id__ = "plugins.paths"

    flags: SharedFlagsProtocol = _PLUGINS_PATHS_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginPathsResult]:
        """Execute paths listing.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginPathsResult]
            Plugin search paths.
        """
        _ = self.flags
        _ = ctx
        LOG.info("Listing plugin search paths")

        manager = get_plugin_manager()

        paths: list[dict[str, object]] = [
            {"path": str(plugin_dir), "exists": plugin_dir.exists()}
            for plugin_dir in manager.plugin_dirs
        ]

        return CliResult.ok(PluginPathsResult(paths=paths))


@cli_command("plugins.new", require_storage=False)
@plugins_app.command(name="new")
@dataclass(frozen=True)
class PluginsNew(Command[PluginNewResult]):
    """Create new plugin from template.

    Generate a plugin scaffold with manifest, entry point,
    and test files.
    """

    __operation_id__ = "plugins.new"

    name: Annotated[str, option_param(PLUGINS_NAME, command_path=PLUGINS_NEW_PATH)]
    output: Annotated[
        Path | None,
        option_param(PLUGINS_OUTPUT_DIR, command_path=PLUGINS_NEW_PATH),
    ] = None
    flags: SharedFlagsProtocol = _PLUGINS_NEW_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginNewResult]:
        """Execute plugin scaffold creation.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginNewResult]
            Created plugin info.
        """
        _ = ctx
        output_dir = self.output or Path.cwd()

        LOG.info("Creating plugin scaffold: %s in %s", self.name, output_dir)

        pattern = re.compile(r"^[a-z][a-z0-9_-]*$")
        if not pattern.match(self.name):
            return fail_invalid_plugin_name(
                "Plugin name must be lowercase alphanumeric with hyphens/underscores"
            )

        plugin_dir = create_plugin_scaffold(self.name, output_dir)

        return CliResult.ok(PluginNewResult(plugin_dir=str(plugin_dir), name=self.name))


@cli_command("plugins.test", require_storage=False)
@plugins_app.command(name="test")
@dataclass(frozen=True)
class PluginsTest(Command[PluginTestResult]):
    """Test a plugin.

    Run the test harness to validate plugin manifest, loading,
    and operation registration.
    """

    __operation_id__ = "plugins.test"

    path: Annotated[Path, option_param(PLUGINS_PATH, command_path=PLUGINS_TEST_PATH)]
    flags: SharedFlagsProtocol = _PLUGINS_TEST_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginTestResult]:
        """Execute plugin tests.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginTestResult]
            Test results.
        """
        _ = ctx
        LOG.info("Testing plugin at: %s", self.path)

        manifest_path = self.path / "plugin.json"
        if not manifest_path.exists():
            return fail_plugin_no_manifest(str(self.path))

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


@cli_command("plugins.validate", require_storage=False)
@plugins_app.command(name="validate")
@dataclass(frozen=True)
class PluginsValidate(Command[PluginValidateResult]):
    """Validate plugin manifest.

    Check that the plugin manifest is valid and compatible
    with the current CLI version.
    """

    __operation_id__ = "plugins.validate"

    path: Annotated[Path, option_param(PLUGINS_PATH, command_path=PLUGINS_VALIDATE_PATH)]
    flags: SharedFlagsProtocol = _PLUGINS_VALIDATE_FLAGS_FIELD

    def execute(self, ctx: CommandContext) -> CliResult[PluginValidateResult]:
        """Execute manifest validation.

        Parameters
        ----------
        ctx
            Command context.

        Returns
        -------
        CliResult[PluginValidateResult]
            Validation results.
        """
        _ = ctx
        LOG.info("Validating plugin at: %s", self.path)

        manifest_path = self.path / "plugin.json"
        if not manifest_path.exists():
            return fail_plugin_no_manifest(str(self.path))

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
    "PluginsDiscover",
    "PluginsDiscoverResult",
    "PluginsInfo",
    "PluginsList",
    "PluginsListResult",
    "PluginsNew",
    "PluginsPaths",
    "PluginsTest",
    "PluginsValidate",
    "plugins_app",
]
