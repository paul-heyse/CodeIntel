"""Native Hamilton implementation for profiles target.

This module provides the Hamilton native nodes for aggregated profiles:
- `t__profiles__compute`: Pure compute node for profiles
- `t__profiles`: Materialize node that writes all tables

Phase 4: Analytics domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import tag

from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import CatalogService

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class ProfilesResult:
    """Result from profiles computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    error
        Error message if computation failed.
    """

    success: bool
    error: str | None = None


@tag(domain="analytics", target="profiles", node_type="compute")
def t__profiles__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__symbol_uses: TargetRunRecord,
) -> ProfilesResult:
    """Build aggregated profiles for functions, files, and modules.

    This is a compute node that calls the profile builders
    which handle both computation and persistence internally.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    t__call_graph
        Upstream call_graph target result (for dependency).
    t__symbol_uses
        Upstream symbol_uses target result (for dependency).

    Returns
    -------
    ProfilesResult
        Result indicating success or failure.

    Notes
    -----
    The profiles include:
    - Function profiles (combining metrics, effects, contracts, etc.)
    - File profiles (aggregating function profiles)
    - Module profiles (aggregating file profiles)
    """
    if t__call_graph.status != "succeeded":
        return ProfilesResult(
            success=False,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    if t__symbol_uses.status != "succeeded":
        return ProfilesResult(
            success=False,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )

    try:
        # Load catalog for module info
        try:
            catalog = CatalogService.from_db(
                env.gateway,
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
            )
        except (RuntimeError, ValueError) as exc:
            log.warning("Failed to load catalog: %s", exc)
            catalog = None

        # Build profiles (handle persistence internally)
        build_function_profile(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
            module_map=None,
        )
        build_file_profile(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
        )
        build_module_profile(
            env.gateway,
            env.snapshot,
            catalog_provider=catalog,
        )

        return ProfilesResult(success=True)

    except Exception as exc:
        log.exception("Profiles computation failed")
        return ProfilesResult(
            success=False,
            error=str(exc),
        )


@tag(domain="analytics", target="profiles", node_type="materialize")
def t__profiles(
    env: BuildEnv,
    graph: TargetGraph,
    t__profiles__compute: ProfilesResult,
) -> TargetRunRecord:
    """Materialize profiles target.

    This is the entry point for the profiles target. The actual
    computation and persistence happens in the compute node.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    t__profiles__compute
        Computed profiles result from the compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.

    Notes
    -----
    This node materializes the following tables:
    - analytics.function_profile
    - analytics.file_profile
    - analytics.module_profile
    """
    executor = NativeTargetExecutor.for_target(env, graph, "profiles")

    if executor.should_skip():
        return executor.skip()

    if not t__profiles__compute.success:
        return executor.fail(RuntimeError(t__profiles__compute.error or "Profiles failed"))

    def compute() -> dict[str, int]:
        # Profiles are persisted during compute - return empty counts
        return {
            "analytics.function_profile": 0,
            "analytics.file_profile": 0,
            "analytics.module_profile": 0,
        }

    return executor.execute(compute)


__all__ = [
    "ProfilesResult",
    "t__profiles",
    "t__profiles__compute",
]
