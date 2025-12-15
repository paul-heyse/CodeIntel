"""BuildSpec package.

BuildSpec is a deterministic, DAG-first compiled contract derived from the
Hamilton graph. It is intended to become the single metadata source for CI
gates and serving inventory.
"""

from __future__ import annotations

from codeintel.build.spec.compile import BuildSpecCompileOptions, compile_buildspec
from codeintel.build.spec.primitives import (
    ArtifactOutSpec,
    BuildSpec,
    DatasetSpec,
    ImplKind,
    SemanticSpec,
    TargetSpec,
)
from codeintel.build.spec.serdes import (
    buildspec_from_json,
    buildspec_hash,
    buildspec_to_json,
    buildspec_to_json_obj,
    canonical_json,
    ensure_buildspec_hash,
)

__all__ = [
    "ArtifactOutSpec",
    "BuildSpec",
    "BuildSpecCompileOptions",
    "DatasetSpec",
    "ImplKind",
    "SemanticSpec",
    "TargetSpec",
    "buildspec_from_json",
    "buildspec_hash",
    "buildspec_to_json",
    "buildspec_to_json_obj",
    "canonical_json",
    "compile_buildspec",
    "ensure_buildspec_hash",
]
