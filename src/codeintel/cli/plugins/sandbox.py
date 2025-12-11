"""Plugin sandbox for restricted execution.

Provide a sandboxed environment for plugin execution with
limited access to system resources.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
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


class SandboxedImporter(MetaPathFinder, Loader):
    """Custom importer that restricts module access."""

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
            Allowed module names including capability-derived entries.
        """
        allowed = set(ALLOWED_MODULES)

        for capability in self._config.allowed_capabilities:
            if capability in CAPABILITY_MODULES:
                allowed.update(CAPABILITY_MODULES[capability])

        return frozenset(allowed)

    def _is_allowed(self, name: str) -> bool:
        """Return True when module import is permitted.

        Returns
        -------
        bool
            True if the module may be imported inside the sandbox.
        """
        entry_parts = self._manifest.entry_point.split(".", maxsplit=1)
        if entry_parts and name.startswith(entry_parts[0]):
            return True

        root_module = name.split(".", maxsplit=1)[0]
        return root_module in self._allowed or name in self._allowed

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Return spec when blocking an import.

        Returns
        -------
        ModuleSpec | None
            Module spec when import should be blocked, otherwise None.
        """
        _ = (path, target)
        if self._is_allowed(fullname):
            return None
        return ModuleSpec(fullname, self)

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        """Use default module creation semantics.

        Returns
        -------
        ModuleType | None
            None to delegate to default module creation.
        """
        _ = (self, spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        """Block module load with an ImportError."""
        msg = (
            f"Plugin '{self._manifest.name}' cannot import '{module.__name__}': "
            "missing required capability"
        )
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
        sys.meta_path.insert(0, self._importer)
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
            sys.meta_path.remove(self._importer)
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
