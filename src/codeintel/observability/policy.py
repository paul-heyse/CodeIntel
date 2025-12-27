"""Observability policy configuration and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from codeintel.observability.semconv_keys import (
    CODEINTEL_CORRELATION_ID,
    CODEINTEL_OUTPUT_FORMAT,
    HTTP_METHOD,
    HTTP_ROUTE,
    MCP_METHOD,
    MCP_TOOL_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.core.config.settings import ObservabilitySettings

OPERATION_ATTRIBUTE_ALLOWLIST: frozenset[str] = frozenset(
    {
        CODEINTEL_CORRELATION_ID,
        CODEINTEL_OUTPUT_FORMAT,
        HTTP_METHOD,
        HTTP_ROUTE,
        MCP_METHOD,
        MCP_TOOL_NAME,
    }
)

DB_ATTRIBUTE_PREFIXES: tuple[str, ...] = ("codeintel.", "db.")


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Redaction policy for command and path values."""

    path_keep_segments: int = 1
    command_keep_segments: int = 1


@dataclass(frozen=True, slots=True)
class AttributeBudget:
    """Cardinality budget for attribute values."""

    max_list_len: int | None = None
    max_str_len: int | None = None
    cli_arg_names_max: int = 25
    http_route_max_len: int = 120
    mcp_tool_name_max_len: int = 80


@dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    """Policy controls for observability attribute shaping."""

    operation_attribute_allowlist: frozenset[str] = OPERATION_ATTRIBUTE_ALLOWLIST
    operation_attribute_overrides: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    db_attribute_prefixes: tuple[str, ...] = DB_ATTRIBUTE_PREFIXES
    budget: AttributeBudget = field(default_factory=AttributeBudget)
    redaction: RedactionPolicy = field(default_factory=RedactionPolicy)

    def operation_allowlist_for(self, component: str, operation: str) -> frozenset[str]:
        """Return allowlist for a specific component/operation combination.

        Returns
        -------
        frozenset[str]
            Attribute allowlist for the component and operation.
        """
        if not self.operation_attribute_overrides:
            return self.operation_attribute_allowlist
        key = f"{component}.{operation}"
        override = self.operation_attribute_overrides.get(key)
        if override is not None:
            return override
        component_override = self.operation_attribute_overrides.get(component)
        if component_override is not None:
            return component_override
        return self.operation_attribute_allowlist


def policy_from_settings(settings: ObservabilitySettings) -> ObservabilityPolicy:
    """Build an observability policy from runtime settings.

    Returns
    -------
    ObservabilityPolicy
        Policy with settings-derived overrides.
    """
    overrides = _normalize_overrides(settings.operation_attribute_allowlist_overrides)
    budget = AttributeBudget(
        cli_arg_names_max=settings.cli_arg_names_max,
        http_route_max_len=settings.http_route_max_len,
        mcp_tool_name_max_len=settings.mcp_tool_name_max_len,
    )
    return ObservabilityPolicy(operation_attribute_overrides=overrides, budget=budget)


def _normalize_overrides(
    overrides: tuple[tuple[str, tuple[str, ...]], ...],
) -> Mapping[str, frozenset[str]]:
    if not overrides:
        return MappingProxyType({})
    mapping: dict[str, frozenset[str]] = {}
    for key, allowlist in overrides:
        if not key:
            continue
        mapping[key] = frozenset(allowlist)
    return MappingProxyType(mapping)


__all__ = [
    "DB_ATTRIBUTE_PREFIXES",
    "OPERATION_ATTRIBUTE_ALLOWLIST",
    "AttributeBudget",
    "ObservabilityPolicy",
    "RedactionPolicy",
    "policy_from_settings",
]
