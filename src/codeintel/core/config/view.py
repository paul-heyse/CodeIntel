"""Typed configuration access layer for runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.config.settings import (
    BuildSettings,
    CliSettings,
    HamiltonExecutionSettings,
    ObservabilitySettings,
    ServingSettings,
)
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.core.runtime.variants import VariantConfig

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.core.runtime import RuntimeSettings


@dataclass(frozen=True, slots=True)
class SettingsView:
    """Typed view over runtime settings with validation and defaults."""

    build: BuildSettings
    execution: HamiltonExecutionSettings
    serving: ServingSettings | None = None
    observability: ObservabilitySettings | None = None
    cli: CliSettings | None = None
    variants: VariantConfig | None = None

    @classmethod
    def from_runtime(cls, settings: RuntimeSettings | None = None) -> SettingsView:
        """Create a settings view from runtime settings.

        Parameters
        ----------
        settings
            Optional runtime settings bundle. When omitted, settings are loaded
            from the runtime loader.

        Returns
        -------
        SettingsView
            View bound to runtime settings.
        """
        resolved = settings or load_runtime_settings()
        return cls(
            build=resolved.build,
            execution=resolved.execution,
            serving=resolved.serving,
            observability=resolved.observability,
            cli=resolved.cli,
            variants=resolved.variants,
        )

    @classmethod
    def from_build_env(cls, env: BuildEnv) -> SettingsView:
        """Create a settings view from a BuildEnv instance.

        Parameters
        ----------
        env
            BuildEnv that carries build and execution settings.

        Returns
        -------
        SettingsView
            View bound to build settings.
        """
        observability = (
            env.execution_context.observability_settings
            if env.execution_context is not None
            else None
        )
        return cls(
            build=env.settings,
            execution=env.execution_settings,
            observability=observability,
            variants=env.variants,
        )

    def require_serving(self) -> ServingSettings:
        """Return serving settings or raise if unavailable.

        Returns
        -------
        ServingSettings
            Serving settings for the current view.

        Raises
        ------
        ValueError
            If serving settings are not available.
        """
        if self.serving is None:
            msg = "Serving settings are not available in this SettingsView"
            raise ValueError(msg)
        return self.serving

    def require_observability(self) -> ObservabilitySettings:
        """Return observability settings or raise if unavailable.

        Returns
        -------
        ObservabilitySettings
            Observability settings for the current view.

        Raises
        ------
        ValueError
            If observability settings are not available.
        """
        if self.observability is None:
            msg = "Observability settings are not available in this SettingsView"
            raise ValueError(msg)
        return self.observability

    @staticmethod
    def resolve_validation_profile(
        *,
        default_profile: str | None,
        config_mode: object,
    ) -> str | None:
        """Normalize validation profile configuration.

        Returns
        -------
        str | None
            Normalized validation profile, or None when validation is disabled.
        """
        if not isinstance(config_mode, str):
            return default_profile
        normalized = config_mode.strip().lower()
        if normalized in {"strict", "lenient"}:
            return normalized
        if normalized in {"off", "none", ""}:
            return None
        return default_profile

    @staticmethod
    def resolve_min_rows(
        *,
        table_key: str,
        base_min_rows: int,
        overrides: Mapping[str, int] | None,
    ) -> int:
        """Resolve minimum row validation threshold for a table.

        Returns
        -------
        int
            Minimum row threshold for the given table.
        """
        if not overrides or not isinstance(overrides, Mapping):
            return base_min_rows
        override = overrides.get(table_key)
        if isinstance(override, int) and override >= 0:
            return override
        return base_min_rows


__all__ = ["SettingsView"]
