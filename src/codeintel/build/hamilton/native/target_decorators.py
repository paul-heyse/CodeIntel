"""Target anchor decorators for DAG-derived target specs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

from codeintel.build.hamilton.tagging import TagKey, TagValue, tag_materialize
from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
from codeintel.build.resources import (
    DEFAULT_EXECUTION,
    DEFAULT_RESOURCES,
    TargetExecution,
    TargetResources,
)
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Mapping

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


@dataclass(frozen=True, slots=True)
class TargetSpecDescriptor:
    """Spec metadata bundled for target decorators."""

    resources: TargetResources = DEFAULT_RESOURCES
    execution: TargetExecution = DEFAULT_EXECUTION
    parameters: TargetParameters = EMPTY_PARAMETERS
    estimated_duration_ms: int | None = None
    spec_version: str = "1"
    extra_tags: Mapping[TagKey, TagValue] | None = None


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def codeintel_target(
    *,
    domain: str,
    target: str,
    spec: TargetSpecDescriptor | None = None,
) -> Decorator[P, R]:
    """Build a canonical target anchor decorator for DAG-native specs.

    Returns
    -------
    Decorator[P, R]
        Decorator that applies target spec tags to a Hamilton node.
    """
    resolved = spec or TargetSpecDescriptor()
    spec_tags: dict[TagKey, TagValue] = {
        cast("TagKey", ht.TAG_TARGET_RESOURCES): _json_dumps(asdict(resolved.resources)),
        cast("TagKey", ht.TAG_TARGET_EXECUTION): _json_dumps(asdict(resolved.execution)),
        cast("TagKey", ht.TAG_TARGET_PARAMETERS): _json_dumps(resolved.parameters.as_dict()),
        cast("TagKey", ht.TAG_TARGET_SPEC_VERSION): resolved.spec_version,
    }
    if resolved.estimated_duration_ms is not None:
        spec_tags[cast("TagKey", ht.TAG_TARGET_ESTIMATED_DURATION_MS)] = str(
            resolved.estimated_duration_ms
        )

    merged: dict[TagKey, TagValue] = {
        cast("TagKey", ht.TAG_KIND): "target",
        cast("TagKey", ht.TAG_SCHEMA_REF): target,
    }
    if resolved.extra_tags:
        merged.update(resolved.extra_tags)
    merged.update(spec_tags)

    return tag_materialize(domain=domain, target=target, extra_tags=merged)


__all__ = [
    "TargetSpecDescriptor",
    "codeintel_target",
]
