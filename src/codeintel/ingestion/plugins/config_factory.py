"""Schema-driven configuration factory for ingestion plugins.

This module provides infrastructure for automatically building step
configuration objects from plugin context and metadata, reducing the
boilerplate code needed in each plugin.

NOTE: Imports inside functions are intentional to avoid circular dependencies.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.ingestion.core.execution_context import IngestExecutionContext
    from codeintel.ingestion.engine.service import ToolService
    from codeintel.ingestion.tracker import ChangeTracker

log = logging.getLogger(__name__)


# Default mappings from context attributes to common config field names
DEFAULT_CONTEXT_MAPPINGS: Mapping[str, str] = {
    "snapshot": "snapshot",
    "paths": "paths",
    "tool_runner": "tool_runner",
    "code_profile": "code_profile",
    "config_profile": "config_profile",
    "tools": "tools",
}


@dataclass(frozen=True)
class ConfigMapping:
    """Mapping specification for building step configs from context.

    Attributes
    ----------
    config_class
        The step config class to instantiate.
    field_map
        Mapping from config field names to context attribute names.
        If not specified, uses DEFAULT_CONTEXT_MAPPINGS for common fields.
    extra_fields
        Static extra field values to pass to config constructor.
    auto_infer
        If True, automatically infer mappings from config class fields.
    """

    config_class: type
    field_map: Mapping[str, str] = field(default_factory=dict)
    extra_fields: Mapping[str, object] = field(default_factory=dict)
    auto_infer: bool = True


@dataclass(frozen=True)
class BuildOptions:
    """Options for config factory build operations.

    Encapsulates optional parameters for ConfigFactory.build().

    Attributes
    ----------
    mapping
        Custom field mapping (config_field -> context_attr).
    extra
        Extra static values to pass to constructor.
    tracker
        Resolved change tracker.
    tool_service
        Resolved tool service.
    """

    mapping: Mapping[str, str] | None = None
    extra: Mapping[str, object] | None = None
    tracker: ChangeTracker | None = None
    tool_service: ToolService | None = None


class ConfigFactory:
    """Factory for building step configs from plugin context.

    The factory introspects config class constructors and maps context
    attributes to config fields, either using explicit mappings or
    automatic inference from field names.

    Examples
    --------
    >>> factory = ConfigFactory()
    >>> cfg = factory.build(
    ...     config_class=TypingIngestStepConfig,
    ...     ctx=plugin_context,
    ... )
    """

    def __init__(
        self,
        default_mappings: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the factory.

        Parameters
        ----------
        default_mappings
            Default context attribute mappings. Uses DEFAULT_CONTEXT_MAPPINGS
            if not specified.
        """
        self._default_mappings = dict(default_mappings or DEFAULT_CONTEXT_MAPPINGS)

    @property
    def default_mappings(self) -> Mapping[str, str]:
        """Return a copy of the default context mappings."""
        return dict(self._default_mappings)

    def build(
        self,
        config_class: type,
        ctx: IngestExecutionContext,
        options: BuildOptions | None = None,
    ) -> object:
        """Build a config instance from context.

        Parameters
        ----------
        config_class
            The config class to instantiate.
        ctx
            Plugin context providing attribute values.
        options
            Build options with mapping, extra values, and resolved services.

        Returns
        -------
        object
            Instantiated config object.
        """
        opts = options or BuildOptions()

        # Get config class fields
        fields = get_config_fields(config_class)

        # Build kwargs for constructor
        kwargs = self._apply_custom_mapping(
            fields, opts.mapping, ctx, opts.tracker, opts.tool_service
        )
        self._apply_default_mappings(fields, kwargs, ctx, opts.tracker, opts.tool_service)
        self._apply_extra_values(fields, kwargs, opts.extra)

        log.debug(
            "ConfigFactory building %s with fields: %s",
            config_class.__name__,
            list(kwargs.keys()),
        )

        return config_class(**kwargs)

    @staticmethod
    def _apply_custom_mapping(
        fields: set[str],
        mapping: Mapping[str, str] | None,
        ctx: IngestExecutionContext,
        tracker: ChangeTracker | None,
        tool_service: ToolService | None,
    ) -> dict[str, Any]:
        """Apply custom field mappings.

        Parameters
        ----------
        fields
            Config class field names.
        mapping
            Custom field mapping.
        ctx
            Plugin context.
        tracker
            Resolved change tracker.
        tool_service
            Resolved tool service.

        Returns
        -------
        dict[str, Any]
            Kwargs with mapped values.
        """
        kwargs: dict[str, Any] = {}
        if mapping:
            for config_field, ctx_attr in mapping.items():
                if config_field in fields:
                    value = _get_context_value(ctx, ctx_attr, tracker, tool_service)
                    if value is not None:
                        kwargs[config_field] = value
        return kwargs

    def _apply_default_mappings(
        self,
        fields: set[str],
        kwargs: dict[str, Any],
        ctx: IngestExecutionContext,
        tracker: ChangeTracker | None,
        tool_service: ToolService | None,
    ) -> None:
        """Apply default context attribute mappings.

        Parameters
        ----------
        fields
            Config class field names.
        kwargs
            Existing kwargs to update.
        ctx
            Plugin context.
        tracker
            Resolved change tracker.
        tool_service
            Resolved tool service.
        """
        for config_field in fields:
            if config_field in kwargs:
                continue
            if config_field in self._default_mappings:
                ctx_attr = self._default_mappings[config_field]
                value = _get_context_value(ctx, ctx_attr, tracker, tool_service)
                if value is not None:
                    kwargs[config_field] = value

    @staticmethod
    def _apply_extra_values(
        fields: set[str],
        kwargs: dict[str, Any],
        extra: Mapping[str, object] | None,
    ) -> None:
        """Apply extra static field values.

        Parameters
        ----------
        fields
            Config class field names.
        kwargs
            Existing kwargs to update.
        extra
            Extra static values.
        """
        if extra:
            kwargs.update({k: v for k, v in extra.items() if k in fields})


def get_config_fields(config_class: type) -> set[str]:
    """Get the set of field names for a config class.

    Parameters
    ----------
    config_class
        Config class to introspect.

    Returns
    -------
    set[str]
        Set of field names.
    """
    if dataclasses.is_dataclass(config_class):
        return {f.name for f in dataclasses.fields(config_class)}

    # For non-dataclass, try to get from __init__ signature
    sig = inspect.signature(config_class.__init__)
    return {
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }


def _get_context_value(
    ctx: IngestExecutionContext,
    attr_name: str,
    tracker: ChangeTracker | None,
    tool_service: ToolService | None,
) -> object | None:
    """Get a value from context by attribute name.

    Parameters
    ----------
    ctx
        Plugin context.
    attr_name
        Attribute name to retrieve.
    tracker
        Optional resolved tracker.
    tool_service
        Optional resolved tool service.

    Returns
    -------
    object | None
        Attribute value or None if not found.
    """
    # Special handling for tracker and tool_service
    if attr_name == "tracker" and tracker is not None:
        return tracker
    if attr_name == "tool_service" and tool_service is not None:
        return tool_service

    # Try direct attribute access
    if hasattr(ctx, attr_name):
        return getattr(ctx, attr_name)

    return None


def infer_config_mapping(config_class: type) -> ConfigMapping:
    """Infer a config mapping from a config class.

    Examine the config class fields and create a mapping using
    default context attribute names where they match.

    Parameters
    ----------
    config_class
        Config class to analyze.

    Returns
    -------
    ConfigMapping
        Inferred mapping for the config class.
    """
    fields = get_config_fields(config_class)

    # Build field map from defaults
    field_map: dict[str, str] = {
        field_name: DEFAULT_CONTEXT_MAPPINGS[field_name]
        for field_name in fields
        if field_name in DEFAULT_CONTEXT_MAPPINGS
    }

    return ConfigMapping(
        config_class=config_class,
        field_map=field_map,
        auto_infer=True,
    )


__all__ = [
    "DEFAULT_CONTEXT_MAPPINGS",
    "BuildOptions",
    "ConfigFactory",
    "ConfigMapping",
    "get_config_fields",
    "infer_config_mapping",
]
