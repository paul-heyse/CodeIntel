"""Runtime settings and bundle types."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.config.settings import (
    BuildSettings,
    CliSettings,
    HamiltonExecutionSettings,
    ObservabilitySettings,
    ServingSettings,
)
from codeintel.core.runtime.primitives import RuntimePrimitives
from codeintel.core.runtime.variants import VariantConfig


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Bundle of runtime settings for build, execution, and serving."""

    build: BuildSettings
    cli: CliSettings
    execution: HamiltonExecutionSettings
    serving: ServingSettings
    observability: ObservabilitySettings
    variants: VariantConfig


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    """Runtime primitives and settings resolved for an entrypoint."""

    primitives: RuntimePrimitives
    settings: RuntimeSettings


__all__ = ["RuntimeBundle", "RuntimeSettings"]
