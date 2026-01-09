"""Execution context for Acero plans or table fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.columnar import compute_config as columnar_compute_config
from codeintel.core.columnar import profiles as columnar_profiles
from codeintel.core.config.settings import ArrowScanSettings
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.runtime.loader import load_runtime_settings

if TYPE_CHECKING:
    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.config.settings import ColumnarRuntimeSettings


RuntimeProfile = columnar_profiles.RuntimeProfile


DEV_FAST = columnar_profiles.DEV_FAST
DEV_DETERMINISTIC = columnar_profiles.DEV_DETERMINISTIC
CI_STABLE = columnar_profiles.CI_STABLE
PROD_THROUGHPUT = columnar_profiles.PROD_THROUGHPUT


def runtime_profile_registry() -> dict[str, RuntimeProfile]:
    """Return the runtime profile registry.

    Returns
    -------
    dict[str, RuntimeProfile]
        Mapping of profile names to RuntimeProfile defaults.
    """
    return columnar_profiles.runtime_profile_registry()


def resolve_runtime_profile(name: str | None) -> RuntimeProfile | None:
    """Resolve a runtime profile by name.

    Parameters
    ----------
    name
        Runtime profile name (case-insensitive).

    Returns
    -------
    RuntimeProfile | None
        Runtime profile when resolved, otherwise None.
    """
    return columnar_profiles.resolve_runtime_profile(name)


def runtime_profile_from_settings(
    settings: ColumnarRuntimeSettings | None,
) -> RuntimeProfile | None:
    """Return the runtime profile derived from columnar runtime settings.

    Returns
    -------
    RuntimeProfile | None
        Resolved runtime profile when configured.
    """
    return columnar_compute_config.resolve_runtime_profile_from_settings(settings)


def resolve_runtime_profile_for_context(
    ctx: ExecutionContext | None,
) -> RuntimeProfile | None:
    """Return the runtime profile for an optional execution context.

    Returns
    -------
    RuntimeProfile | None
        Runtime profile from the context or runtime settings.
    """
    if ctx is not None:
        return ctx.runtime_profile
    settings = load_runtime_settings().columnar
    return runtime_profile_from_settings(settings)


def resolve_arrow_scan_settings(
    ctx: ExecutionContext | None,
) -> ArrowScanSettings:
    """Return Arrow scan settings for an optional execution context.

    Returns
    -------
    ArrowScanSettings
        Scan settings from the context or runtime defaults.
    """
    if ctx is not None and ctx.scan_settings is not None:
        return ctx.scan_settings
    return load_runtime_settings().build.arrow_scan


def resolve_execution_context(ctx: ExecutionContext | None) -> ExecutionContext:
    """Return an execution context with runtime defaults applied.

    Returns
    -------
    ExecutionContext
        Execution context with runtime profile defaults applied.
    """
    if ctx is not None:
        return ctx
    profile = resolve_runtime_profile_for_context(None)
    return ExecutionContext(
        runtime_profile=profile,
        scan_settings=resolve_arrow_scan_settings(None),
    )


def resolve_columnar_context(
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> ExecutionContext | None:
    """Resolve a columnar execution context from runtime or columnar contexts.

    Returns
    -------
    ExecutionContext | None
        Columnar execution context when available.
    """
    if ctx is None:
        return None
    if isinstance(ctx, ExecutionContext):
        return ctx
    if isinstance(ctx, RuntimeExecutionContext):
        profile = runtime_profile_from_settings(ctx.columnar_settings)
        return ExecutionContext(
            runtime_profile=profile,
            scan_settings=ctx.settings.build.arrow_scan,
        )
    return None


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Execution context for Acero plans or table fallbacks."""

    use_threads: bool = True
    determinism: DedupeTier = "stable_set"
    combine_chunks: bool = True
    provenance: bool = False
    runtime_profile: RuntimeProfile | None = None
    scan_settings: ArrowScanSettings | None = None

    def resolve_use_threads(self) -> bool:
        """Return the resolved plan use_threads setting.

        Returns
        -------
        bool
            Effective plan use_threads toggle.
        """
        resolved = self.use_threads
        profile = self.runtime_profile
        if profile is not None:
            resolved = profile.resolve_plan_use_threads(default=resolved)
        return resolved

    def resolve_determinism(self) -> DedupeTier:
        """Return the resolved determinism tier.

        Returns
        -------
        DedupeTier
            Effective determinism tier after applying runtime profile defaults.
        """
        resolved = self.determinism
        profile = self.runtime_profile
        if profile is not None:
            resolved = profile.resolve_determinism(default=resolved)
        return resolved
