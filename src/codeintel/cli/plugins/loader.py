"""Unified plugin loading.

Provide a unified interface for loading both manifest-based
and legacy plugins with optional sandboxing.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from codeintel.cli.plugins.discovery import DiscoveredPlugin, discover_plugins
from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest
from codeintel.cli.plugins.sandbox import PluginSandbox, SandboxConfig

LOG = logging.getLogger(__name__)


@dataclass
class LoadedPlugin:
    """A loaded plugin ready for registration.

    Parameters
    ----------
    manifest
        Plugin manifest.
    module
        Loaded Python module.
    path
        Plugin path.
    is_legacy
        Whether this is a legacy format plugin.
    """

    manifest: PluginManifest
    module: ModuleType
    path: Path
    is_legacy: bool = False

    def has_capability(self, cap: PluginCapability) -> bool:
        """Check if plugin has capability.

        Parameters
        ----------
        cap
            Capability to check.

        Returns
        -------
        bool
            True if plugin has capability.
        """
        return cap in self.manifest.capabilities


@dataclass
class PluginLoadResult:
    """Result of loading plugins.

    Parameters
    ----------
    loaded
        Successfully loaded plugins.
    failed
        Failed plugin paths with error messages.
    legacy_warnings
        Warnings about legacy plugins.
    """

    loaded: list[LoadedPlugin] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    legacy_warnings: list[str] = field(default_factory=list)


class PluginLoader:
    """Load and initialize plugins.

    Parameters
    ----------
    sandbox_enabled
        Enable sandboxing for plugins.
    allowed_capabilities
        Additional capabilities to allow if not in manifest.
    """

    def __init__(
        self,
        *,
        sandbox_enabled: bool = True,
        allowed_capabilities: set[PluginCapability] | None = None,
    ) -> None:
        """Initialize plugin loader."""
        self._sandbox_enabled = sandbox_enabled
        self._allowed_caps = allowed_capabilities or set()

    def load_all(
        self,
        search_paths: list[Path] | None = None,
    ) -> PluginLoadResult:
        """Discover and load all plugins.

        Parameters
        ----------
        search_paths
            Paths to search for plugins.

        Returns
        -------
        PluginLoadResult
            Results of loading.
        """
        result = PluginLoadResult()

        discovered = discover_plugins(search_paths)

        for plugin in discovered:
            if plugin.is_legacy:
                result.legacy_warnings.append(
                    f"Legacy plugin at {plugin.path}: {plugin.errors[0]}",
                )
                # Try loading legacy plugin
                try:
                    loaded = self._load_legacy(plugin)
                    if loaded:
                        result.loaded.append(loaded)
                        continue
                except (ImportError, OSError, AttributeError, TypeError) as e:
                    result.failed.append((plugin.path, str(e)))
                    continue

            if not plugin.valid:
                result.failed.append((plugin.path, "; ".join(plugin.errors)))
                continue

            try:
                loaded = self._load_plugin(plugin)
                result.loaded.append(loaded)
            except (ImportError, OSError, RuntimeError) as e:
                LOG.exception("Failed to load plugin %s", plugin.path)
                result.failed.append((plugin.path, str(e)))

        return result

    def load_single(
        self,
        path: Path,
    ) -> LoadedPlugin:
        """Load a single plugin.

        Parameters
        ----------
        path
            Path to plugin directory or file.

        Returns
        -------
        LoadedPlugin
            Loaded plugin.

        Raises
        ------
        ValueError
            If plugin is invalid or cannot be loaded.
        """
        manifest_path = path / "plugin.json" if path.is_dir() else None

        if manifest_path and manifest_path.exists():
            manifest = PluginManifest.load(manifest_path)
            errors = manifest.validate()
            if errors:
                msg = f"Invalid manifest: {'; '.join(errors)}"
                raise ValueError(msg)

            discovered = DiscoveredPlugin(
                path=path,
                manifest=manifest,
                valid=True,
                errors=[],
            )
            return self._load_plugin(discovered)

        # Check for legacy format
        discovered = self._check_legacy(path)
        if discovered:
            loaded = self._load_legacy(discovered)
            if loaded:
                return loaded

        msg = f"No valid plugin found at {path}"
        raise ValueError(msg)

    def _load_plugin(self, discovered: DiscoveredPlugin) -> LoadedPlugin:
        """Load a manifest-based plugin.

        Parameters
        ----------
        discovered
            Discovered plugin info.

        Returns
        -------
        LoadedPlugin
            Loaded plugin.
        """
        manifest = discovered.manifest

        if self._sandbox_enabled:
            config = SandboxConfig(
                allowed_capabilities=set(manifest.capabilities) | self._allowed_caps,
            )
            with PluginSandbox(manifest, config) as sandbox:
                module = sandbox.load_plugin()
        else:
            module = importlib.import_module(manifest.entry_point)

        return LoadedPlugin(
            manifest=manifest,
            module=module,
            path=discovered.path,
        )

    @staticmethod
    def _load_legacy(discovered: DiscoveredPlugin) -> LoadedPlugin | None:
        """Load a legacy plugin (create_plugin API).

        Parameters
        ----------
        discovered
            Discovered plugin info.

        Returns
        -------
        LoadedPlugin | None
            Loaded plugin or None if not valid.
        """
        path = discovered.path

        if path.is_file():
            # Single file plugin
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
        else:
            # Directory plugin - add to path temporarily
            sys.path.insert(0, str(path))
            try:
                module = importlib.import_module(discovered.manifest.entry_point)
            finally:
                sys.path.remove(str(path))

        # Verify it has create_plugin
        if not hasattr(module, "create_plugin"):
            return None

        return LoadedPlugin(
            manifest=discovered.manifest,
            module=module,
            path=path,
            is_legacy=True,
        )

    @staticmethod
    def _check_legacy(path: Path) -> DiscoveredPlugin | None:
        """Check if path contains a legacy plugin.

        Parameters
        ----------
        path
            Path to check.

        Returns
        -------
        DiscoveredPlugin | None
            Discovered plugin info or None.
        """
        if path.is_file() and path.suffix == ".py":
            content = path.read_text(encoding="utf-8")
            if "def create_plugin" in content:
                return DiscoveredPlugin(
                    path=path,
                    manifest=PluginManifest(
                        name=path.stem,
                        version="0.0.0",
                        api_version="0.0.0",
                        entry_point=path.stem,
                    ),
                    valid=False,
                    errors=["Legacy plugin format"],
                    is_legacy=True,
                )
        elif path.is_dir():
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                content = py_file.read_text(encoding="utf-8")
                if "def create_plugin" in content:
                    return DiscoveredPlugin(
                        path=path,
                        manifest=PluginManifest(
                            name=path.name,
                            version="0.0.0",
                            api_version="0.0.0",
                            entry_point=py_file.stem,
                        ),
                        valid=False,
                        errors=["Legacy plugin format"],
                        is_legacy=True,
                    )
        return None


def get_plugin_loader(
    *,
    sandbox_enabled: bool = True,
) -> PluginLoader:
    """Get a plugin loader instance.

    Parameters
    ----------
    sandbox_enabled
        Enable sandboxing for plugins.

    Returns
    -------
    PluginLoader
        Plugin loader instance.
    """
    return PluginLoader(sandbox_enabled=sandbox_enabled)


__all__ = [
    "LoadedPlugin",
    "PluginLoadResult",
    "PluginLoader",
    "get_plugin_loader",
]
