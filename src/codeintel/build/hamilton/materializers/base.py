"""Shared helpers for Hamilton materializer implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.run_records import (
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from codeintel.build.targets import OutputTarget


@dataclass(frozen=True)
class MaterializationContext:
    """Resolved inputs for a materializer execution."""

    target: OutputTarget
    input_hash: str
    options_hash: str | None
    should_skip: bool


@dataclass(frozen=True)
class MaterializationContextError:
    """Error encountered while preparing materialization context."""

    message: str
    input_hash: str | None = None


def resolve_materialization_context(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
) -> MaterializationContext | MaterializationContextError:
    """Resolve materialization context from environment and graph.

    Parameters
    ----------
    env
        Build environment containing snapshot, gateway, and configuration.
    graph
        Target graph describing build dependencies.
    target_name
        Name of the target being materialized.

    Returns
    -------
    MaterializationContext | MaterializationContextError
        Resolved materialization context, or an error description.
    """
    if not isinstance(env, BuildEnv):
        return MaterializationContextError(
            message=f"Expected env to be BuildEnv, got {type(env).__name__}",
        )
    if not isinstance(graph, TargetGraph):
        return MaterializationContextError(
            message=f"Expected graph to be TargetGraph, got {type(graph).__name__}",
        )

    target = graph.get(target_name)
    if target is None:
        return MaterializationContextError(message=f"Target not found in graph: {target_name}")

    options_hash = options_hash_for_target(env, target_name)
    input_hash = compute_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        options_hash=options_hash,
        manifests=env.manifest_index,
    )
    return MaterializationContext(
        target=target,
        input_hash=input_hash,
        options_hash=options_hash,
        should_skip=should_skip_native_target(env, target, input_hash),
    )


def duration_ms(start: float) -> float:
    """Return elapsed milliseconds since start.

    Parameters
    ----------
    start
        Start time from ``time.perf_counter()``.

    Returns
    -------
    float
        Elapsed time in milliseconds.
    """
    return (time.perf_counter() - start) * 1000


def manifest_row_count(env: BuildEnv, *, target_name: str) -> int | None:
    """Return the row count recorded in an existing manifest if present.

    Parameters
    ----------
    env
        Build environment containing manifest index.
    target_name
        Target name to lookup in the manifest index.

    Returns
    -------
    int | None
        Row count from the manifest, or None if unavailable.
    """
    index = env.manifest_index
    if index is None:
        return None
    manifest = index.get(target_name)
    if manifest is None:
        return None
    return manifest.row_count


__all__ = [
    "MaterializationContext",
    "MaterializationContextError",
    "duration_ms",
    "manifest_row_count",
    "resolve_materialization_context",
]
