"""Shared specification types for native Hamilton target templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.native.target_decorators import TargetSpecDescriptor
from codeintel.build.schemas.column_resolution import DeferredColumns

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.tagging import TagKey, TagValue


OutputRole = Literal["contract", "internal"]


@dataclass(frozen=True, slots=True)
class ArtifactOutputSpec:
    """Specification for a file artifact output."""

    name: str
    path_template: str
    output_role: OutputRole | None = None


@dataclass(frozen=True, slots=True)
class TableOutputSpec:
    """Specification for a table output."""

    table_key: str
    node_name: str | None = None
    columns: tuple[str, ...] | DeferredColumns | None = None
    output_role: OutputRole | None = None


@dataclass(frozen=True, slots=True)
class ToolTargetSpec:
    """Specification for tool-backed target templates."""

    domain: str
    target_name: str
    spec: TargetSpecDescriptor
    artifacts: tuple[ArtifactOutputSpec, ...] = ()
    tables: tuple[TableOutputSpec, ...] = ()
    tool_tags: Mapping[TagKey, TagValue] | None = None


__all__ = [
    "ArtifactOutputSpec",
    "OutputRole",
    "TableOutputSpec",
    "ToolTargetSpec",
]
