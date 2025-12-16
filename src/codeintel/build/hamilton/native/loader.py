"""Native Hamilton module loader with discovery and validation.

This module provides NativeModuleLoader, which handles the discovery,
validation, and loading of native Hamilton target modules. It ensures
that modules conform to the required patterns before being composed
into a Hamilton Driver.

Design Principles
-----------------
1. Modules must export at least one `t__<target>` materialize node.
2. Modules should use @tag decorators with domain, target, node_type.
3. Modules should define __all__ for explicit exports.
4. Supports domain-based filtering for incremental migration.

Example
-------
>>> loader = NativeModuleLoader()
>>> modules = loader.load_for_driver(domains={"analytics", "export"})
>>> driver = Driver({}, *modules)
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.tags import (
    TAG_DOMAIN,
    TAG_NODE_TYPE,
    TAG_TARGET,
)

if TYPE_CHECKING:
    from types import ModuleType

log = logging.getLogger(__name__)

# Known native module packages organized by domain
_NATIVE_MODULE_PACKAGES: dict[str, list[str]] = {
    "analytics": [
        "codeintel.build.hamilton.native.analytics.cfg_dfg",
        "codeintel.build.hamilton.native.analytics.coverage_functions",
        "codeintel.build.hamilton.native.analytics.data_models",
        "codeintel.build.hamilton.native.analytics.dependencies",
        "codeintel.build.hamilton.native.analytics.entrypoints",
        "codeintel.build.hamilton.native.analytics.function_history",
        "codeintel.build.hamilton.native.analytics.history_timeseries",
        "codeintel.build.hamilton.native.analytics.hotspots",
        "codeintel.build.hamilton.native.analytics.risk_factors",
        "codeintel.build.hamilton.native.analytics.subsystems",
        "codeintel.build.hamilton.native.analytics.test_graph_metrics",
    ],
    "ingestion": [
        "codeintel.build.hamilton.native.ingestion.ast",
        "codeintel.build.hamilton.native.ingestion.config",
        "codeintel.build.hamilton.native.ingestion.coverage",
        "codeintel.build.hamilton.native.ingestion.cst",
        "codeintel.build.hamilton.native.ingestion.docstrings",
        "codeintel.build.hamilton.native.ingestion.modules",
        "codeintel.build.hamilton.native.ingestion.scip",
        "codeintel.build.hamilton.native.ingestion.tests",
        "codeintel.build.hamilton.native.ingestion.typing",
    ],
    "graphs": [
        "codeintel.build.hamilton.native.graphs.call_graph_views",
    ],
    "export": [
        "codeintel.build.hamilton.native.export.export_jsonl",
        "codeintel.build.hamilton.native.export.export_parquet",
    ],
}

# Flatten all module paths for easy lookup
_ALL_NATIVE_MODULES: frozenset[str] = frozenset(
    module for modules in _NATIVE_MODULE_PACKAGES.values() for module in modules
)


@dataclass(frozen=True)
class ModuleValidationResult:
    """Result of validating a native Hamilton module.

    Attributes
    ----------
    module_path
        The fully-qualified module path.
    is_valid
        Whether the module passes all validation checks.
    target_nodes
        List of detected t__<target> node names.
    errors
        List of validation error messages.
    warnings
        List of validation warning messages.
    """

    module_path: str
    is_valid: bool
    target_nodes: tuple[str, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class NativeModuleLoader:
    """Load and validate native Hamilton target modules.

    This class provides methods to discover, validate, and load native
    Hamilton modules for composition into a Driver. It supports filtering
    by domain and target name for incremental migration.

    Attributes
    ----------
    strict
        When True, validation errors cause load_for_driver to raise.

    Examples
    --------
    >>> loader = NativeModuleLoader()
    >>> modules = loader.load_for_driver()
    >>> len(modules) > 0
    True

    >>> # Load only analytics domain
    >>> modules = loader.load_for_driver(domains={"analytics"})

    >>> # Validate a specific module
    >>> result = loader.validate_module_path("codeintel.build.hamilton.native.analytics.risk_factors")
    >>> result.is_valid
    True
    """

    strict: bool = False
    _cache: dict[str, ModuleType] = field(default_factory=dict, repr=False)

    @staticmethod
    def list_domains() -> frozenset[str]:
        """List all known native module domains.

        Returns
        -------
        frozenset[str]
            Set of domain names (analytics, ingestion, graphs, export).

        Examples
        --------
        >>> loader = NativeModuleLoader()
        >>> "analytics" in loader.list_domains()
        True
        """
        return frozenset(_NATIVE_MODULE_PACKAGES.keys())

    @staticmethod
    def list_module_paths(
        *,
        domain: str | None = None,
    ) -> list[str]:
        """List module paths, optionally filtered by domain.

        Parameters
        ----------
        domain
            If provided, only return modules in this domain.

        Returns
        -------
        list[str]
            List of fully-qualified module paths.

        Examples
        --------
        >>> loader = NativeModuleLoader()
        >>> paths = loader.list_module_paths(domain="export")
        >>> len(paths)
        2
        """
        if domain is not None:
            return list(_NATIVE_MODULE_PACKAGES.get(domain, []))
        return [
            module
            for modules in _NATIVE_MODULE_PACKAGES.values()
            for module in modules
        ]

    def discover_modules(
        self,
        *,
        domain: str | None = None,
    ) -> list[ModuleType]:
        """Discover and import native modules.

        Parameters
        ----------
        domain
            If provided, only discover modules in this domain.

        Returns
        -------
        list[ModuleType]
            List of imported module objects.

        Notes
        -----
        When strict=True and a module fails to import, the ImportError is
        re-raised. Otherwise, the error is logged and the module is skipped.

        Raises
        ------
        ImportError
            Raised when strict=True and a module cannot be imported.

        Examples
        --------
        >>> loader = NativeModuleLoader()
        >>> modules = loader.discover_modules(domain="analytics")
        >>> len(modules) > 0
        True
        """
        paths = self.list_module_paths(domain=domain)
        modules: list[ModuleType] = []

        for path in paths:
            if path in self._cache:
                modules.append(self._cache[path])
                continue

            try:
                module = importlib.import_module(path)
                self._cache[path] = module
                modules.append(module)
            except ImportError as exc:
                log.warning("Failed to import native module %s: %s", path, exc)
                if self.strict:
                    raise

        return modules

    @staticmethod
    def validate_module(module: ModuleType) -> ModuleValidationResult:
        """Validate a native Hamilton module.

        Checks that the module:
        1. Has at least one t__<target> function
        2. Functions are tagged with domain, target, node_type
        3. Has __all__ defined

        Parameters
        ----------
        module
            The module to validate.

        Returns
        -------
        ModuleValidationResult
            Validation result with errors and warnings.

        Examples
        --------
        >>> import codeintel.build.hamilton.native.analytics.risk_factors as m
        >>> loader = NativeModuleLoader()
        >>> result = loader.validate_module(m)
        >>> result.is_valid
        True
        """
        module_path = module.__name__
        errors: list[str] = []
        warnings: list[str] = []
        target_nodes: list[str] = []

        # Check for __all__
        module_all = getattr(module, "__all__", None)
        if module_all is None:
            warnings.append("Module should define __all__ for explicit exports")

        # Find t__* functions
        for name in dir(module):
            if not name.startswith("t__"):
                continue

            obj = getattr(module, name)
            if not callable(obj):
                continue

            target_nodes.append(name)

            # Check for Hamilton tags
            tags = getattr(obj, "_tags", None)
            if tags is None:
                warnings.append(f"{name}: Missing @tag decorator")
            else:
                if TAG_DOMAIN not in tags:
                    warnings.append(f"{name}: Missing '{TAG_DOMAIN}' tag")
                if TAG_TARGET not in tags:
                    warnings.append(f"{name}: Missing '{TAG_TARGET}' tag")
                if TAG_NODE_TYPE not in tags:
                    warnings.append(f"{name}: Missing '{TAG_NODE_TYPE}' tag")

        # Must have at least one target node
        if not target_nodes:
            errors.append("Module must export at least one t__<target> function")

        is_valid = len(errors) == 0

        return ModuleValidationResult(
            module_path=module_path,
            is_valid=is_valid,
            target_nodes=tuple(target_nodes),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    def validate_module_path(self, module_path: str) -> ModuleValidationResult:
        """Validate a module by its import path.

        Parameters
        ----------
        module_path
            Fully-qualified module path.

        Returns
        -------
        ModuleValidationResult
            Validation result.

        Examples
        --------
        >>> loader = NativeModuleLoader()
        >>> result = loader.validate_module_path(
        ...     "codeintel.build.hamilton.native.analytics.risk_factors"
        ... )
        >>> result.is_valid
        True
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            return ModuleValidationResult(
                module_path=module_path,
                is_valid=False,
                target_nodes=(),
                errors=(f"Failed to import: {exc}",),
            )

        return self.validate_module(module)

    def load_for_driver(
        self,
        *,
        domains: set[str] | None = None,
        exclude_targets: set[str] | None = None,
    ) -> tuple[ModuleType, ...]:
        """Load modules ready for Hamilton Driver composition.

        This method discovers, validates, and returns modules suitable
        for passing to the Hamilton Driver constructor.

        Parameters
        ----------
        domains
            If provided, only load modules from these domains.
        exclude_targets
            If provided, skip modules that only define these targets.

        Returns
        -------
        tuple[ModuleType, ...]
            Tuple of validated module objects.

        Notes
        -----
        When strict=True, ImportError is raised if a module fails to import,
        and ValueError is raised if a module fails validation.

        Raises
        ------
        ValueError
            Raised when strict=True and a module fails validation.

        Notes
        -----
        ImportError is propagated from discover_modules when strict=True.

        Examples
        --------
        >>> loader = NativeModuleLoader()
        >>> modules = loader.load_for_driver()
        >>> isinstance(modules, tuple)
        True

        >>> # Load specific domains
        >>> modules = loader.load_for_driver(domains={"analytics", "export"})
        """
        # Collect modules from requested domains
        all_modules: list[ModuleType] = []

        if domains is not None:
            for domain in domains:
                all_modules.extend(self.discover_modules(domain=domain))
        else:
            all_modules = self.discover_modules()

        # Filter and validate
        result_modules: list[ModuleType] = []
        exclude_nodes = (
            {target_node(t) for t in exclude_targets} if exclude_targets else set()
        )

        for module in all_modules:
            validation = self.validate_module(module)

            # Log warnings
            for warning in validation.warnings:
                log.debug("Module %s: %s", validation.module_path, warning)

            # Check for errors
            if not validation.is_valid:
                for error in validation.errors:
                    log.warning("Module %s: %s", validation.module_path, error)
                if self.strict:
                    msg = f"Module {validation.module_path} failed validation: {validation.errors}"
                    raise ValueError(msg)
                continue

            # Check if all targets are excluded
            if exclude_nodes:
                remaining = set(validation.target_nodes) - exclude_nodes
                if not remaining:
                    log.debug(
                        "Skipping module %s - all targets excluded",
                        validation.module_path,
                    )
                    continue

            result_modules.append(module)

        log.debug("Loaded %d native modules for driver", len(result_modules))
        return tuple(result_modules)

    def get_target_names(
        self,
        *,
        domains: set[str] | None = None,
    ) -> frozenset[str]:
        """Get all target names from native modules.

        Parameters
        ----------
        domains
            If provided, only include targets from these domains.

        Returns
        -------
        frozenset[str]
            Set of target names (without t__ prefix).

        Examples
        --------
        >>> loader = NativeModuleLoader()
        >>> names = loader.get_target_names(domains={"analytics"})
        >>> "risk_factors" in names
        True
        """
        modules = self.discover_modules() if domains is None else []
        if domains:
            for domain in domains:
                modules.extend(self.discover_modules(domain=domain))

        target_names: set[str] = set()
        for module in modules:
            for name in dir(module):
                if name.startswith("t__") and not name.endswith("__compute"):
                    # Extract target name: t__risk_factors -> risk_factors
                    target_name = name[3:]  # Remove t__ prefix
                    target_names.add(target_name)

        return frozenset(target_names)

    def clear_cache(self) -> None:
        """Clear the module import cache.

        Useful for testing or when modules may have been reloaded.
        """
        self._cache.clear()


# Module-level convenience instance
_default_loader: NativeModuleLoader | None = None


def get_loader(*, strict: bool = False) -> NativeModuleLoader:
    """Get a shared NativeModuleLoader instance.

    Parameters
    ----------
    strict
        Whether validation errors should raise exceptions.

    Returns
    -------
    NativeModuleLoader
        Shared loader instance.

    Examples
    --------
    >>> loader = get_loader()
    >>> modules = loader.load_for_driver()
    """
    global _default_loader  # noqa: PLW0603 (module-level singleton pattern)
    if _default_loader is None or _default_loader.strict != strict:
        _default_loader = NativeModuleLoader(strict=strict)
    return _default_loader


__all__ = [
    "ModuleValidationResult",
    "NativeModuleLoader",
    "get_loader",
]
