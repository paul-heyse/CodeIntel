"""BuildSpec primitives.

These dataclasses define the stable, serialized contract produced by BuildSpec
compilation. The compiler (PR-76) is responsible for populating these from the
Hamilton DAG and canonical schema provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ImplKind = Literal["native", "wrapper"]


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset entry in a BuildSpec."""

    table_key: str
    schema_hash: str
    columns: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ArtifactOutSpec:
    """Artifact output entry for a target."""

    name: str
    kind: str | None = None
    path_template: str | None = None


@dataclass(frozen=True)
class TargetSpec:
    """Target entry in a BuildSpec."""

    name: str
    domain: str
    impl_kind: ImplKind
    deps: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    artifacts: tuple[ArtifactOutSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SemanticSpec:
    """Optional pointer to semantic layer metadata."""

    version: str | None = None


@dataclass(frozen=True)
class BuildSpec:
    """Deterministic compiled contract for the build DAG."""

    spec_version: int
    targets: tuple[TargetSpec, ...] = field(default_factory=tuple)
    datasets: tuple[DatasetSpec, ...] = field(default_factory=tuple)
    semantic: SemanticSpec | None = None
    buildspec_hash: str = ""


__all__ = [
    "ArtifactOutSpec",
    "BuildSpec",
    "DatasetSpec",
    "ImplKind",
    "SemanticSpec",
    "TargetSpec",
]
