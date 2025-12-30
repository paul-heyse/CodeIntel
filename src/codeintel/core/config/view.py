"""Typed configuration access layer for runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from codeintel.core.config.settings import (
    BuildSettings,
    CliSettings,
    HamiltonExecutionSettings,
    IcebergSettings,
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
        normalized_build = _normalize_build_settings(resolved.build)
        return cls(
            build=normalized_build,
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
        normalized_build = _normalize_build_settings(env.settings)
        return cls(
            build=normalized_build,
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


def _normalize_build_settings(settings: BuildSettings) -> BuildSettings:
    normalized_iceberg = _normalize_iceberg_settings(settings.iceberg)
    if normalized_iceberg == settings.iceberg:
        return settings
    return replace(settings, iceberg=normalized_iceberg)


def _normalize_iceberg_settings(settings: IcebergSettings) -> IcebergSettings:
    catalog_name = settings.catalog_name.strip() if settings.catalog_name else "default"
    catalog_type = settings.catalog_type.strip().lower() if settings.catalog_type else None
    catalog_uri = settings.catalog_uri.strip() if settings.catalog_uri else None
    catalog_warehouse = settings.catalog_warehouse.strip() if settings.catalog_warehouse else None
    read_ref = settings.read_ref.strip() if settings.read_ref else None
    io_impl = settings.io_impl.strip() if settings.io_impl else None
    location_provider_impl = (
        settings.location_provider_impl.strip() if settings.location_provider_impl else None
    )
    write_data_path = settings.write_data_path.strip() if settings.write_data_path else None
    write_metadata_path = (
        settings.write_metadata_path.strip() if settings.write_metadata_path else None
    )

    enforced_prefixes = tuple(
        prefix.strip()
        for prefix in settings.enforced_table_prefixes
        if isinstance(prefix, str) and prefix.strip()
    )

    catalog_properties = _normalize_pairs(settings.catalog_properties)
    io_options = _normalize_pairs(settings.io_options)

    return IcebergSettings(
        read_enabled=bool(settings.read_enabled),
        write_enabled=bool(settings.write_enabled),
        tombstones_enabled=bool(settings.tombstones_enabled),
        flight_enabled=bool(settings.flight_enabled),
        read_ref=read_ref or None,
        enforced_table_prefixes=enforced_prefixes,
        catalog_name=catalog_name or "default",
        catalog_type=catalog_type or None,
        catalog_uri=catalog_uri or None,
        catalog_warehouse=catalog_warehouse or None,
        catalog_properties=catalog_properties,
        config_path=settings.config_path,
        io_impl=io_impl or None,
        io_options=io_options,
        location_provider_impl=location_provider_impl or None,
        write_data_path=write_data_path or None,
        write_metadata_path=write_metadata_path or None,
        object_store_partitioned_paths=settings.object_store_partitioned_paths,
    )


def _normalize_pairs(value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    normalized: dict[str, str] = {}
    for key, raw_value in value:
        if not key:
            continue
        if not isinstance(key, str):
            continue
        key_str = key.strip()
        if not key_str:
            continue
        if not isinstance(raw_value, str):
            raw_value = str(raw_value)
        val = raw_value.strip()
        normalized[key_str] = val
    return tuple(sorted(normalized.items()))


__all__ = ["SettingsView"]
