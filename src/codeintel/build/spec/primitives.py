"""BuildSpec primitives.

These immutable structs define the stable, serialized contract produced by
BuildSpec compilation. The compiler (PR-76) is responsible for populating these
from the Hamilton DAG and canonical schema provider.
"""

from __future__ import annotations

from typing import Literal

import msgspec

ImplKind = Literal["native"]


class DatasetSpec(msgspec.Struct, frozen=True):
    """Dataset entry in a BuildSpec."""

    table_key: str
    schema_hash: str
    columns: tuple[str, ...] | None = None


class ArtifactOutSpec(msgspec.Struct, frozen=True):
    """Artifact output entry for a target."""

    name: str
    kind: str | None = None
    path_template: str | None = None


class TargetSpec(msgspec.Struct, frozen=True):
    """Target entry in a BuildSpec."""

    name: str
    domain: str
    impl_kind: ImplKind
    deps: tuple[str, ...] = msgspec.field(default_factory=tuple)
    outputs: tuple[str, ...] = msgspec.field(default_factory=tuple)
    artifacts: tuple[ArtifactOutSpec, ...] = msgspec.field(default_factory=tuple)


class SemanticSpec(msgspec.Struct, frozen=True):
    """Optional pointer to semantic layer metadata."""

    version: str | None = None


class BuildSpec(msgspec.Struct, frozen=True):
    """Deterministic compiled contract for the build DAG."""

    spec_version: int
    targets: tuple[TargetSpec, ...] = msgspec.field(default_factory=tuple)
    datasets: tuple[DatasetSpec, ...] = msgspec.field(default_factory=tuple)
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
