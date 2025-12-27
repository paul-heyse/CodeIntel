"""Shared helpers for Hamilton materializer implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv

if TYPE_CHECKING:
    from codeintel.build.hamilton.dag_catalog import TargetDescriptor


@dataclass(frozen=True)
class MaterializationContext:
    """Resolved inputs for a materializer execution."""

    target: TargetDescriptor
    input_hash: str | None
    options_hash: str | None


@dataclass(frozen=True)
class MaterializationContextError:
    """Error encountered while preparing materialization context."""

    message: str
    input_hash: str | None = None


def resolve_materialization_context(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    target_name: str,
) -> MaterializationContext | MaterializationContextError:
    """Resolve materialization context from environment and catalog.

    Parameters
    ----------
    env
        Build environment containing snapshot, gateway, and configuration.
    catalog
        DAG catalog describing build dependencies.
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
    if not isinstance(catalog, DagCatalog):
        return MaterializationContextError(
            message=f"Expected catalog to be DagCatalog, got {type(catalog).__name__}",
        )

    target = catalog.get(target_name)
    if target is None:
        return MaterializationContextError(message=f"Target not found in catalog: {target_name}")

    return MaterializationContext(
        target=target,
        input_hash=None,
        options_hash=None,
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


__all__ = [
    "MaterializationContext",
    "MaterializationContextError",
    "duration_ms",
    "resolve_materialization_context",
]
