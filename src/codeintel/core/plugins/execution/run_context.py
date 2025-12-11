"""Plugin run context for unified execution preparation."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from codeintel.core.plugins.execution.manifest import (
    InputHashPayload,
    compute_input_hash,
    compute_options_hash,
)

if TYPE_CHECKING:
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.metadata import CorePluginMetadata


@dataclass(frozen=True)
class PluginRunContext:
    """Context for a single plugin execution."""

    metadata: CorePluginMetadata
    options: Any | None
    upstream_state: Mapping[str, str]
    options_hash: str
    input_hash: str

    @property
    def plugin_name(self) -> str:
        """Return the plugin's canonical name.

        Returns
        -------
        str
            Plugin name.
        """
        return self.metadata.name

    @property
    def plugin_version(self) -> str:
        """Return the plugin's version.

        Returns
        -------
        str
            Plugin version string.
        """
        return self.metadata.version


@dataclass(frozen=True)
class RunContextInputs:
    """Optional inputs for plugin run context hashing."""

    repo: str = ""
    commit: str = ""
    variant: str | None = None
    scope_id: str | None = None
    extra_fields: Mapping[str, Any] | None = None


def prepare_plugin_run(
    metadata: CorePluginMetadata,
    resolver: PluginOptionsResolver,
    upstream_state: Mapping[str, str],
    *,
    dynamic_overrides: Mapping[str, Any] | None = None,
    inputs: RunContextInputs | None = None,
) -> PluginRunContext:
    """Prepare context for a plugin run.

    Returns
    -------
    PluginRunContext
        Context containing metadata, resolved options, and hashes.
    """
    context_inputs = inputs or RunContextInputs()
    if metadata.options_model is None:
        options: object | None = None
        options_hash = compute_options_hash(metadata.name, {})
    else:
        options = resolver.get_options(
            metadata,
            metadata.options_model,
            dynamic_overrides=dynamic_overrides,
        )
        options_payload = _extract_serializable_options(options)
        options_hash = compute_options_hash(metadata.name, options_payload)

    payload = InputHashPayload(
        repo=context_inputs.repo,
        commit=context_inputs.commit,
        plugin_name=metadata.name,
        version_hash=metadata.version,
        options_hash=options_hash,
        extra_fields={
            "upstream_state": dict(upstream_state),
            "variant": context_inputs.variant,
            "scope_id": context_inputs.scope_id,
            **(context_inputs.extra_fields or {}),
        },
    )
    input_hash = compute_input_hash(payload)

    return PluginRunContext(
        metadata=metadata,
        options=options,
        upstream_state=upstream_state,
        options_hash=options_hash or "",
        input_hash=input_hash,
    )


def _extract_serializable_options(options: object) -> dict[str, object]:
    """Extract serializable fields from an options object.

    Returns
    -------
    dict[str, object]
        Mapping of field names to serializable values.
    """
    if dataclasses.is_dataclass(options) and not isinstance(options, type):
        serializable: dict[str, object] = {}
        for field_info in dataclasses.fields(options):
            value = getattr(options, field_info.name)
            if _is_serializable(value):
                serializable[field_info.name] = value
        return serializable

    if hasattr(options, "model_dump"):
        model = cast("Any", options)
        dumped = model.model_dump(exclude_none=True)
        return dict(dumped)

    if hasattr(options, "dict"):
        model = cast("Any", options)
        dumped = model.dict(exclude_none=True)
        return dict(dumped)

    if isinstance(options, Mapping):
        return {k: v for k, v in options.items() if _is_serializable(v)}

    return {k: v for k, v in vars(options).items() if _is_serializable(v)}


def _is_serializable(value: object) -> bool:
    """Check if a value is JSON-serializable.

    Returns
    -------
    bool
        True when the value can be serialized to JSON.
    """
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_serializable(v) for k, v in value.items())
    return False


__all__ = [
    "PluginRunContext",
    "RunContextInputs",
    "prepare_plugin_run",
]
