"""Base plugin metadata options for decorator-based plugin creation.

This module provides `BasePluginMetaOptions`, a base class for plugin
metadata options used by decorator-based plugin creation. Both analytics
and graph subsystems extend this base with domain-specific fields.

Example
-------
>>> @dataclass
... class GraphPluginMetaOptions(BasePluginMetaOptions):
...     produces_graph_kinds: tuple[GraphKind, ...] = ()
...     requires_graph_kinds: tuple[GraphKind, ...] = ()
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from codeintel.core.plugins.types.protocol import (
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginSeverity,
    PluginStage,
)
from codeintel.core.plugins.types.result import PluginResult


class BasePluginMetaOptionsInput(TypedDict, total=False):
    """Base typed keyword arguments for plugin options.

    This TypedDict defines the common fields available for all plugin
    decorators. Domain-specific options classes can extend this with
    additional fields.
    """

    name: str
    description: str
    kind: PluginKind
    stage: PluginStage
    version: str
    enabled_by_default: bool
    severity: PluginSeverity
    inputs: Sequence[PluginInputSpec]
    outputs: Sequence[PluginOutputSpec]
    provides: Sequence[str]
    requires: Sequence[str]
    depends_on: Sequence[str]
    resource_hints: PluginResourceHints | None
    requires_isolation: bool
    isolation_kind: PluginIsolation
    tags: Sequence[str]


@dataclass
class BasePluginMetaOptions:
    """Base options container for plugin metadata.

    Grouping metadata in a single object keeps decorator signatures small
    and makes future evolution easier. Domain-specific options classes
    should extend this base and add their own fields.

    Attributes
    ----------
    name
        Plugin name. Defaults to function name if not provided.
    description
        Human-readable description. Defaults to function docstring.
    kind
        Plugin kind classification.
    stage
        Processing stage for ordering.
    version
        Plugin version for cache invalidation.
    enabled_by_default
        Whether enabled when no explicit list is provided.
    severity
        How failures should be handled.
    inputs
        Required and optional input specifications.
    outputs
        Tables and artifacts produced.
    provides
        Capabilities this plugin provides.
    requires
        Capabilities this plugin needs.
    depends_on
        Explicit plugin dependencies by name.
    resource_hints
        Runtime resource hints for scheduling.
    requires_isolation
        Whether process/thread isolation is needed.
    isolation_kind
        Type of isolation required.
    tags
        Free-form tags for categorization.
    """

    name: str | None = None
    description: str | None = None
    kind: PluginKind | None = None
    stage: PluginStage | None = None
    version: str = "1.0.0"
    enabled_by_default: bool = True
    severity: PluginSeverity = "fatal"
    inputs: Sequence[PluginInputSpec] = ()
    outputs: Sequence[PluginOutputSpec] = ()
    provides: Sequence[str] = ()
    requires: Sequence[str] = ()
    depends_on: Sequence[str] = ()
    resource_hints: PluginResourceHints | None = None
    requires_isolation: bool = False
    isolation_kind: PluginIsolation = "none"
    tags: Sequence[str] = ()

    @staticmethod
    def validate_option_keys(
        allowed_keys: set[str],
        provided: Mapping[str, object],
    ) -> None:
        """Validate that only allowed option keys are provided.

        Parameters
        ----------
        allowed_keys
            Set of allowed keyword argument names.
        provided
            Dictionary of provided keyword arguments (TypedDict or dict).

        Raises
        ------
        ValueError
            If unknown keys are provided.
        """
        unknown = set(provided) - allowed_keys
        if unknown:
            message = f"Unsupported plugin option keys: {', '.join(sorted(unknown))}"
            raise ValueError(message)

    def to_base_metadata[TCtx](
        self,
        fn: Callable[[TCtx], PluginResult],
        default_kind: PluginKind = "analytics",
        default_stage: PluginStage = "other",
    ) -> PluginMetadata:
        """Convert options to base PluginMetadata.

        Parameters
        ----------
        fn
            Plugin callable used for deriving defaults (name/docstring).
        default_kind
            Default plugin kind if not specified.
        default_stage
            Default plugin stage if not specified.

        Returns
        -------
        PluginMetadata
            Metadata populated from the options and function defaults.
        """
        resolved_name = self.name or fn.__name__.replace("_", ".")

        # Use provided description, or extract summary (first line) from docstring
        if self.description is not None:
            resolved_description = self.description
        elif fn.__doc__:
            # Extract only the first line (summary) per NumPy docstring convention
            resolved_description = fn.__doc__.strip().split("\n")[0]
        else:
            resolved_description = ""

        return PluginMetadata(
            name=resolved_name,
            description=resolved_description.strip(),
            kind=self.kind or default_kind,
            stage=self.stage or default_stage,
            version=self.version,
            enabled_by_default=self.enabled_by_default,
            severity=self.severity,
            inputs=tuple(self.inputs),
            outputs=tuple(self.outputs),
            provides=tuple(self.provides),
            requires=tuple(self.requires),
            depends_on=tuple(self.depends_on),
            resource_hints=self.resource_hints,
            requires_isolation=self.requires_isolation,
            isolation_kind=self.isolation_kind,
            tags=tuple(self.tags),
        )


__all__ = [
    "BasePluginMetaOptions",
    "BasePluginMetaOptionsInput",
]
