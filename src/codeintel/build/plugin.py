"""Target plugin protocol and base class.

This module defines the unified TargetPlugin protocol that all plugins
implement. Plugins are pure executors - all metadata about what they
produce and what they depend on lives in the OutputTarget definition.

Example
-------
>>> class MyPlugin(TargetPlugin):
...     plugin_name: ClassVar[str] = "my_plugin"
...     plugin_version: ClassVar[str] = "1.0.0"
...
...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
...         rows = self._compute_rows(ctx)
...         ctx.write_table("core.my_table", rows)
...         return TargetResult.succeeded(row_counts={"core.my_table": len(rows)})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from codeintel.build.context import TargetExecutionContext, TargetResult
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.metadata import CorePluginMetadata
    from codeintel.core.plugins.types.protocol import PluginMetadata

TOptions = TypeVar("TOptions")

__all__ = [
    "MetadataPlugin",
    "TargetPlugin",
    "TargetPluginProtocol",
]


@runtime_checkable
class TargetPluginProtocol(Protocol):
    """Protocol for target plugins.

    This is the minimal interface that all plugins must satisfy.
    Plugins receive everything they need via TargetExecutionContext
    and return a TargetResult.

    Class Variables
    ---------------
    plugin_name
        Unique identifier for the plugin (e.g., "ast_extract").
    plugin_version
        Semantic version string (e.g., "1.0.0").
    plugin_description
        Human-readable description of what the plugin does.
    """

    plugin_name: ClassVar[str]
    plugin_version: ClassVar[str]
    plugin_description: ClassVar[str]

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin with the given context.

        Parameters
        ----------
        ctx
            Execution context with resources, parameters, and write methods.

        Returns
        -------
        TargetResult
            Success or failure result with row counts and artifacts.
        """
        ...


class TargetPlugin(ABC):
    """Base class for target plugins.

    Provides the abstract interface for all plugins in the build system.
    Subclasses must define class variables and implement execute().

    Class Variables
    ---------------
    plugin_name
        Unique identifier for the plugin.
    plugin_version
        Semantic version for change tracking.
    plugin_description
        Human-readable description.

    Example
    -------
    >>> class RepoScanPlugin(TargetPlugin):
    ...     plugin_name = "repo_scan"
    ...     plugin_version = "3.0.0"
    ...     plugin_description = "Scan repository for Python modules."
    ...
    ...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    ...         modules = self._scan_modules(ctx.repo_root)
    ...         ctx.write_table("core.modules", modules)
    ...         return TargetResult.succeeded(row_counts={"core.modules": len(modules)})
    """

    plugin_name: ClassVar[str] = ""
    plugin_version: ClassVar[str] = "1.0.0"
    plugin_description: ClassVar[str] = ""

    @abstractmethod
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context with everything the plugin needs.

        Returns
        -------
        TargetResult
            Result indicating success/failure with row counts.
        """
        ...

    def validate_context(self, ctx: TargetExecutionContext) -> list[str]:
        """Validate that the context has everything needed.

        Override this method to add plugin-specific validation.
        The default implementation returns an empty list (no errors).

        Parameters
        ----------
        ctx
            Execution context to validate.

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        _ = (self, ctx)
        return []


class MetadataPlugin(TargetPlugin, ABC):
    """Enhanced plugin base with automatic metadata handling.

    Subclasses define `_core_metadata` and get `metadata` property
    automatically, plus standard options resolver handling. Note that
    subclasses still need to implement the abstract `execute` method.

    This base class reduces boilerplate in plugin implementations by:
    - Providing automatic metadata conversion via the `metadata` property
    - Providing automatic class variable population from core metadata
    - Providing standard options resolver handling

    Class Variables
    ---------------
    _core_metadata
        CorePluginMetadata instance defining the plugin's metadata.
        Subclasses must define this.

    Example
    -------
    >>> from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
    >>> class MyPlugin(MetadataPlugin):
    ...     _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
    ...         name="my_plugin",
    ...         version="1.0.0",
    ...         description="My plugin description",
    ...         domain=PluginDomain.ANALYTICS,
    ...         kind="compute",
    ...     )
    ...
    ...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    ...         # Plugin logic here
    ...         return TargetResult.succeeded()
    """

    _core_metadata: ClassVar[CorePluginMetadata]
    _options_resolver: PluginOptionsResolver | None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Initialize subclass by copying metadata to class variables.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to parent __init_subclass__.
        """
        super().__init_subclass__(**kwargs)

        # Copy metadata to class variables if _core_metadata is defined
        # This happens at class definition time, not at instance creation
        if hasattr(cls, "_core_metadata") and cls._core_metadata is not None:
            # Only set if not already overridden
            if "plugin_name" not in cls.__dict__:
                cls.plugin_name = cls._core_metadata.name  # pyright: ignore[reportIncompatibleVariableOverride]
            if "plugin_version" not in cls.__dict__:
                cls.plugin_version = cls._core_metadata.version  # pyright: ignore[reportIncompatibleVariableOverride]
            if "plugin_description" not in cls.__dict__:
                cls.plugin_description = cls._core_metadata.description  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        *,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        """Initialize with optional options resolver.

        Parameters
        ----------
        options_resolver
            Optional resolver for plugin configuration options.
        """
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata.

        This property automatically converts the `_core_metadata` to the
        `PluginMetadata` protocol format.

        Returns
        -------
        PluginMetadata
            Protocol-compatible metadata for registry consumers.
        """
        from codeintel.build.plugins._metadata import to_plugin_metadata  # noqa: PLC0415

        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return the canonical core metadata.

        Returns
        -------
        CorePluginMetadata
            Full core metadata definition.
        """
        return self._core_metadata

    @property
    def options_resolver(self) -> PluginOptionsResolver | None:
        """Return the options resolver if configured.

        Returns
        -------
        PluginOptionsResolver | None
            Options resolver or None if not configured.
        """
        return self._options_resolver

    def resolve_options(
        self,
        options_type: type[TOptions] | None = None,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> TOptions:
        """Resolve typed options from configuration.

        Use the options_model from _core_metadata if options_type is not
        provided. Fall back to creating a default instance if no resolver.

        Parameters
        ----------
        options_type
            Optional explicit options type. If None, uses _core_metadata.options_model.
        dynamic_overrides
            Runtime overrides to merge into options.

        Returns
        -------
        TOptions
            Resolved options instance.

        Raises
        ------
        ValueError
            If no options type is available (neither passed nor in metadata).
        """
        opts_cls = options_type or self._core_metadata.options_model
        if opts_cls is None:
            msg = f"No options type for plugin {self.plugin_name}"
            raise ValueError(msg)

        if self._options_resolver is None:
            if dynamic_overrides:
                return opts_cls(**dynamic_overrides)  # type: ignore[return-value]
            return opts_cls()  # type: ignore[return-value]

        return cast(
            "TOptions",
            self._options_resolver.get_options(
                self._core_metadata,
                opts_cls,
                dynamic_overrides=dynamic_overrides,
            ),
        )
