"""Plugin sandbox for restricted execution.

Provide a sandboxed environment for plugin execution with
limited access to system resources.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Self

from codeintel.cli.plugins.manifest import PluginCapability, PluginManifest

LOG = logging.getLogger(__name__)


# Modules that plugins can always import
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "abc",
        "collections",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "itertools",
        "json",
        "logging",
        "pathlib",
        "re",
        "typing",
        "codeintel.cli.results",
        "codeintel.cli.execution",
    },
)

# Modules that require specific capabilities
CAPABILITY_MODULES: dict[PluginCapability, frozenset[str]] = {
    PluginCapability.NETWORK_ACCESS: frozenset({"urllib", "http", "socket"}),
    PluginCapability.FILE_READ: frozenset({"io", "os.path"}),
    PluginCapability.FILE_WRITE: frozenset({"io", "os", "shutil"}),
    PluginCapability.EXECUTE_EXTERNAL: frozenset({"subprocess", "os"}),
}


@dataclass
class SandboxConfig:
    """Configuration for plugin sandbox.

    Parameters
    ----------
    allowed_capabilities
        Capabilities granted to plugin.
    timeout
        Execution timeout in seconds.
    memory_limit
        Memory limit in bytes (not enforced on all platforms).
    """

    allowed_capabilities: set[PluginCapability] = field(default_factory=set)
    timeout: float = 30.0
    memory_limit: int | None = None


class SandboxedImporter:
    """Custom importer that restricts module access.

    Parameters
    ----------
    manifest
        Plugin manifest.
    config
        Sandbox configuration.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        config: SandboxConfig,
    ) -> None:
        """Initialize importer."""
        self._manifest = manifest
        self._config = config
        self._allowed = self._compute_allowed_modules()

    def _compute_allowed_modules(self) -> frozenset[str]:
        """Compute set of allowed modules.

        Returns
        -------
        frozenset[str]
            Allowed module names.
        """
        allowed = set(ALLOWED_MODULES)

        for capability in self._config.allowed_capabilities:
            if capability in CAPABILITY_MODULES:
                allowed.update(CAPABILITY_MODULES[capability])

        return frozenset(allowed)

    def find_module(
        self,
        name: str,
        path: object = None,  # noqa: ARG002
    ) -> SandboxedImporter | None:
        """Check if module import is allowed.

        Parameters
        ----------
        name
            Module name.
        path
            Import path (unused).

        Returns
        -------
        SandboxedImporter | None
            Self if blocking import, None to allow.
        """
        # Allow importing the plugin itself
        entry_parts = self._manifest.entry_point.split(".", maxsplit=1)
        if entry_parts and name.startswith(entry_parts[0]):
            return None

        # Check if module is allowed
        root_module = name.split(".", maxsplit=1)[0]
        if root_module in self._allowed or name in self._allowed:
            return None

        # Block import
        return self

    def load_module(self, name: str) -> None:
        """Block module load.

        Parameters
        ----------
        name
            Module name.

        Raises
        ------
        ImportError
            Always raised to block import.
        """
        msg = f"Plugin '{self._manifest.name}' cannot import '{name}': missing required capability"
        raise ImportError(msg)


class PluginSandbox:
    """Sandbox environment for plugin execution.

    Parameters
    ----------
    manifest
        Plugin manifest.
    config
        Sandbox configuration.
    """

    def __init__(
        self,
        manifest: PluginManifest,
        config: SandboxConfig | None = None,
    ) -> None:
        """Initialize sandbox."""
        self._manifest = manifest
        self._config = config or SandboxConfig(
            allowed_capabilities=set(manifest.capabilities),
        )
        self._importer = SandboxedImporter(manifest, self._config)
        self._active = False

    def __enter__(self) -> Self:
        """Enter sandbox context.

        Returns
        -------
        Self
            This instance.

        Raises
        ------
        RuntimeError
            If sandbox is already active.
        """
        if self._active:
            msg = "Sandbox already active"
            raise RuntimeError(msg)

        # Install custom importer
        sys.meta_path.insert(0, self._importer)  # type: ignore[arg-type]
        self._active = True
        LOG.debug("Entered sandbox for plugin: %s", self._manifest.name)
        return self

    def __exit__(self, *args: object) -> None:
        """Exit sandbox context.

        Parameters
        ----------
        *args
            Exception info (unused).
        """
        # Remove custom importer
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self._importer)  # type: ignore[arg-type]
        self._active = False
        LOG.debug("Exited sandbox for plugin: %s", self._manifest.name)

    def load_plugin(self) -> ModuleType:
        """Load plugin module within sandbox.

        Returns
        -------
        ModuleType
            Loaded plugin module.

        Raises
        ------
        RuntimeError
            If sandbox is not active.
        """
        if not self._active:
            msg = "Sandbox not active"
            raise RuntimeError(msg)

        return importlib.import_module(self._manifest.entry_point)

    def check_capability(self, capability: PluginCapability) -> bool:
        """Check if capability is granted.

        Parameters
        ----------
        capability
            Capability to check.

        Returns
        -------
        bool
            True if granted.
        """
        return capability in self._config.allowed_capabilities


__all__ = [
    "ALLOWED_MODULES",
    "CAPABILITY_MODULES",
    "PluginSandbox",
    "SandboxConfig",
    "SandboxedImporter",
]
