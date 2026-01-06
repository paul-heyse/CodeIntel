"""TOML-based configuration for build targets.

This module provides a global configuration system for tuning parameters.
Configuration is loaded from config/codeintel.build.toml in the project root.

The config file uses TOML format with sections for each module and
per-target overrides.

Example config file (config/codeintel.build.toml):
```toml
[analytics]
max_parallel_workers = 4

[ingestion]
skip_empty_files = true
```

Example
-------
>>> from codeintel.build.config import load_build_config
>>> config = load_build_config(Path("/path/to/project"))
>>> entrypoint_params = config.parameters_for("entrypoints")
>>> max_entries = entrypoint_params.get_typed("max_entries", int, default=2000)
"""

from __future__ import annotations

import copy
import logging
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, overload

import msgspec

from codeintel.build.parameters import TargetParameters

_T = TypeVar("_T")

log = logging.getLogger(__name__)

__all__ = [
    "BuildConfig",
    "BuildConfigOverrides",
    "BuildConfigStack",
    "ConfigSection",
    "load_build_config",
]


CONFIG_FILE_NAME = "config/codeintel.build.toml"

_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "analytics",
        "ingestion",
        "graphs",
        "export",
        "views",
        "hamilton",
        "scope",
        "telemetry",
        "variants",
    }
)

_ALLOWED_SCHEMA_DRIFT_MODES: frozenset[str] = frozenset({"off", "warn", "strict"})


class _HamiltonConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    graph_backend: str | None = None
    schema_drift_mode: str | None = None


class _TelemetryHooksConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    enable_telemetry: bool | None = None
    enable_io_telemetry: bool | None = None
    enable_progress: bool | None = None
    enable_timing: bool | None = None
    telemetry_output_path: str | None = None
    io_telemetry_output_path: str | None = None
    progress_style: str | None = None
    progress_desc: str | None = None
    println_enabled: bool | None = None
    println_verbosity: int | None = None
    println_node_filter: list[str] | str | None = None
    typecheck_enabled: bool | None = None
    typecheck_inputs: bool | None = None
    typecheck_outputs: bool | None = None
    graceful_errors_enabled: bool | None = None
    graceful_try_all_parallel: bool | None = None
    graceful_allow_injection: bool | None = None
    pdb_enabled: bool | None = None
    pdb_before: bool | None = None
    pdb_during: bool | None = None
    pdb_after: bool | None = None
    pdb_node_filter: list[str] | str | None = None
    event_stream_enabled: bool | None = None
    event_stream_path: str | None = None
    cache_logger_level: str | None = None
    cache_logger_path: str | None = None
    hang_watchdog_enabled: bool | None = None
    hang_watchdog_timeout_s: float | None = None
    hang_watchdog_repeat: bool | None = None
    hang_watchdog_path: str | None = None
    display_all_functions_enabled: bool | None = None
    display_all_functions_path: str | None = None
    visualize_execution_enabled: bool | None = None
    visualize_execution_path: str | None = None
    ddog_enabled: bool | None = None
    ddog_root_name: str | None = None
    ddog_service: str | None = None
    ddog_include_causal_links: bool | None = None


class _TelemetryTrackerConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    enabled: bool | None = None
    project_id: str | int | None = None
    username: str | None = None
    dag_name: str | None = None
    api_url: str | None = None
    ui_url: str | None = None
    capture_data_statistics: bool | None = None
    max_list_length: int | None = None
    max_dict_length: int | None = None
    config_uri: str | None = None
    tags: dict[str, str] | None = None


class _TelemetryConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    hooks: _TelemetryHooksConfig | None = None
    hamilton_tracker: _TelemetryTrackerConfig | None = None


@dataclass(frozen=True)
class ConfigSection:
    """A section of configuration (module or target level).

    Attributes
    ----------
    name
        Section name (e.g., "analytics", "analytics.function_types").
    values
        Key-value pairs in this section.
    """

    name: str
    values: dict[str, Any] = field(default_factory=dict)

    @overload
    def get(self, key: str) -> object | None: ...

    @overload
    def get(self, key: str, default: _T) -> object | _T: ...

    def get(self, key: str, default: object | None = None) -> object:
        """Get a value from this section.

        Parameters
        ----------
        key
            Configuration key.
        default
            Default value if not found.

        Returns
        -------
        object
            Value or default.
        """
        return self.values.get(key, default)

    def as_parameters(self) -> TargetParameters:
        """Convert section to TargetParameters.

        Returns
        -------
        TargetParameters
            Parameters from this section.
        """
        return TargetParameters(dict(self.values))


@dataclass
class BuildConfig:
    """Global build configuration loaded from TOML.

    Attributes
    ----------
    config_path
        Path to the config file (if loaded from file).
    sections
        Configuration sections by name.
    _raw
        Raw TOML data.
    """

    config_path: Path | None = None
    sections: dict[str, ConfigSection] = field(default_factory=dict)
    _raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> BuildConfig:
        """Create empty configuration (all defaults).

        Returns
        -------
        BuildConfig
            Empty config.
        """
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any], config_path: Path | None = None) -> BuildConfig:
        """Create config from dict.

        Parameters
        ----------
        data
            TOML-like dict structure.
        config_path
            Optional path for reference.

        Returns
        -------
        BuildConfig
            Parsed config.
        """
        config = cls(config_path=config_path, _raw=data)

        def parse_sections(prefix: str, d: dict[str, Any]) -> None:
            section_values: dict[str, Any] = {}

            for key, value in d.items():
                if isinstance(value, dict):
                    section_name = f"{prefix}.{key}" if prefix else key
                    parse_sections(section_name, value)
                else:
                    section_values[key] = value

            if section_values:
                section_name = prefix
                config.sections[section_name] = ConfigSection(section_name, section_values)

        parse_sections("", data)
        return config

    @overload
    def get(self, key: str) -> object | None: ...

    @overload
    def get(self, key: str, default: _T) -> object | _T: ...

    def get(self, key: str, default: object | None = None) -> object:
        """Get a top-level config value.

        Parameters
        ----------
        key
            Configuration key (dot-separated path).
        default
            Default value if not found.

        Returns
        -------
        object
            Value or default.
        """
        parts = key.split(".")
        current: dict[str, object] | object = self._raw

        for part in parts:
            if not isinstance(current, dict):
                return default
            current = current.get(part)
            if current is None:
                return default

        return current

    def schema_drift_mode(self) -> str:
        """Return the configured schema drift mode.

        Returns
        -------
        str
            Normalized drift mode ("off", "warn", or "strict").

        Raises
        ------
        TypeError
            If the configuration value is not a string.
        ValueError
            If the value is not an allowed schema drift mode.
        """
        raw = self.get("hamilton.schema_drift_mode")
        if raw is None:
            return "warn"
        if not isinstance(raw, str):
            msg = "hamilton.schema_drift_mode must be a string"
            raise TypeError(msg)
        normalized = raw.strip().lower()
        if normalized in _ALLOWED_SCHEMA_DRIFT_MODES:
            return normalized
        msg = (
            "hamilton.schema_drift_mode must be one of: "
            f"{', '.join(sorted(_ALLOWED_SCHEMA_DRIFT_MODES))}"
        )
        raise ValueError(msg)

    def raw_data(self) -> dict[str, Any]:
        """Return a shallow copy of raw TOML data.

        Returns
        -------
        dict[str, Any]
            Raw configuration data as a new dictionary.
        """
        return dict(self._raw)

    def get_section(self, name: str) -> ConfigSection | None:
        """Get a configuration section by name.

        Parameters
        ----------
        name
            Section name (e.g., "analytics.function_types").

        Returns
        -------
        ConfigSection | None
            Section if found.
        """
        return self.sections.get(name)

    @staticmethod
    def _merge_section_values(
        *,
        result_values: dict[str, Any],
        section: ConfigSection | None,
    ) -> None:
        if section is None:
            return
        result_values.update(section.values)

    def _merge_nested_sections(
        self,
        *,
        result_values: dict[str, Any],
        prefix: str,
    ) -> None:
        base_prefix = f"{prefix}."
        nested_sections = sorted(
            (name, section)
            for name, section in self.sections.items()
            if name.startswith(base_prefix)
        )
        for name, section in nested_sections:
            remainder = name[len(base_prefix) :]
            if not remainder:
                continue
            parts = remainder.split(".")
            self._merge_nested_values(
                result_values=result_values,
                parts=parts,
                values=section.values,
            )

    @staticmethod
    def _merge_nested_values(
        *,
        result_values: dict[str, Any],
        parts: list[str],
        values: Mapping[str, Any],
    ) -> None:
        cursor: dict[str, Any] = result_values
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                nested: dict[str, Any] = {}
                cursor[part] = nested
                cursor = nested
            else:
                cursor = existing

        leaf_key = parts[-1]
        existing_leaf = cursor.get(leaf_key)
        if not isinstance(existing_leaf, dict):
            cursor[leaf_key] = dict(values)
        else:
            existing_leaf.update(values)

    def parameters_for(self, target_name: str) -> TargetParameters:
        """Get parameters for a target, merging module and target sections.

        Looks up configuration in this order:
        1. Module-level section (e.g., "analytics")
        2. Target-level section (e.g., "analytics.function_types")

        Target-level values override module-level.

        Parameters
        ----------
        target_name
            Target name to get parameters for.

        Returns
        -------
        TargetParameters
            Merged parameters.
        """
        modules = ["ingestion", "graphs", "analytics", "export", "views"]

        result_values: dict[str, Any] = {}
        defaults = DEFAULT_PARAMETERS.get(target_name)
        if defaults:
            result_values.update(copy.deepcopy(defaults))

        for module in modules:
            self._merge_section_values(
                result_values=result_values,
                section=self.sections.get(module),
            )

            target_prefix = f"{module}.{target_name}"
            target_section = self.sections.get(target_prefix)
            if target_section is not None:
                result_values.update(target_section.values)
                self._merge_nested_sections(result_values=result_values, prefix=target_prefix)

        return TargetParameters(result_values)


@dataclass(frozen=True)
class BuildConfigOverrides:
    """Explicit per-target parameter overrides for a build run."""

    per_target: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def for_target(self, target_name: str) -> Mapping[str, object]:
        """Return overrides for a target name.

        Returns
        -------
        Mapping[str, object]
            Overrides for the target, or an empty mapping.
        """
        return self.per_target.get(target_name, {})

    def is_empty(self) -> bool:
        """Return True when no overrides are configured.

        Returns
        -------
        bool
            True when no overrides are present.
        """
        return not self.per_target


@dataclass
class BuildConfigStack(BuildConfig):
    """BuildConfig wrapper that overlays explicit per-target overrides."""

    overrides: BuildConfigOverrides | None = None

    @classmethod
    def from_base(
        cls,
        base: BuildConfig,
        *,
        overrides: BuildConfigOverrides | None = None,
    ) -> BuildConfigStack:
        """Create a BuildConfigStack from an existing BuildConfig.

        Parameters
        ----------
        base
            Base BuildConfig instance to copy.
        overrides
            Optional per-target override mappings.

        Returns
        -------
        BuildConfigStack
            BuildConfigStack with overlayed run overrides.
        """
        return cls(
            config_path=base.config_path,
            sections=dict(base.sections),
            _raw=base.raw_data(),
            overrides=overrides,
        )

    def parameters_for(self, target_name: str) -> TargetParameters:
        """Return parameters with overrides applied.

        Parameters
        ----------
        target_name
            Target name to get parameters for.

        Returns
        -------
        TargetParameters
            Parameters merged with explicit overrides.
        """
        base = super().parameters_for(target_name)
        if not self.overrides or self.overrides.is_empty():
            return base
        overrides = self.overrides.for_target(target_name)
        if not overrides:
            return base
        return base.merge(TargetParameters(dict(overrides)))


def load_build_config(project_root: Path) -> BuildConfig:
    """Load build configuration from project root.

    Looks for config/codeintel.build.toml in the project root.
    Returns empty config if file doesn't exist.

    Parameters
    ----------
    project_root
        Project root directory.

    Returns
    -------
    BuildConfig
        Loaded or empty configuration.

    Raises
    ------
    TypeError
        Raised when configuration data does not match expected types.
    ValueError
        Raised when configuration data is invalid or cannot be read.
    """
    config_path = project_root / CONFIG_FILE_NAME

    if not config_path.exists():
        log.debug("No build config found at %s, using defaults", config_path)
        return BuildConfig.empty()

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        _validate_config_data(data, config_path=config_path)
        log.info("build.config.validation.ok config_path=%s", config_path)
        return BuildConfig.from_dict(data, config_path)
    except tomllib.TOMLDecodeError as exc:
        log.exception("build.config.validation.fail config_path=%s", config_path)
        raise ValueError(_format_config_error("Failed to parse", config_path, exc)) from exc
    except (TypeError, ValueError):
        log.exception("build.config.validation.fail config_path=%s", config_path)
        raise
    except OSError as exc:
        log.exception("build.config.validation.fail config_path=%s", config_path)
        raise ValueError(_format_config_error("Failed to read", config_path, exc)) from exc


def _format_config_error(action: str, config_path: Path, exc: Exception) -> str:
    return f"{action} build config {config_path}: {exc}"


def _validate_config_data(data: Mapping[str, Any], *, config_path: Path) -> None:
    if not isinstance(data, Mapping):
        msg = f"Build config must be a mapping; got {type(data).__name__}"
        raise TypeError(msg)
    _reject_seeded_dataset_config(data, config_path=config_path)
    unknown = sorted(set(data) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        msg = f"Unknown build config sections: {', '.join(unknown)}"
        raise ValueError(msg)

    for section_name in ("analytics", "ingestion", "graphs", "export", "views"):
        _validate_module_section(data, section_name)

    _validate_scope_section(data.get("scope"))
    _validate_variants_section(data.get("variants"))
    _validate_hamilton_section(data.get("hamilton"), config_path=config_path)
    _validate_telemetry_section(data.get("telemetry"), config_path=config_path)


def _decode_hamilton_section(section: Mapping[str, Any], *, config_path: Path) -> None:
    try:
        msgspec.convert(section, type=_HamiltonConfig, strict=True)
    except msgspec.ValidationError as exc:
        msg = f"hamilton section invalid in {config_path}: {exc}"
        raise ValueError(msg) from exc
    _validate_schema_drift_mode(section.get("schema_drift_mode"), config_path=config_path)


def _decode_telemetry_section(section: Mapping[str, Any], *, config_path: Path) -> None:
    try:
        msgspec.convert(section, type=_TelemetryConfig, strict=True)
    except msgspec.ValidationError as exc:
        msg = f"telemetry section invalid in {config_path}: {exc}"
        raise ValueError(msg) from exc


def _validate_module_section(data: Mapping[str, Any], section_name: str) -> None:
    section = data.get(section_name)
    if section is None:
        return
    if not isinstance(section, Mapping):
        msg = f"{section_name} section must be a mapping"
        raise TypeError(msg)


def _validate_scope_section(scope: object) -> None:
    if scope is None:
        return
    if not isinstance(scope, Mapping):
        msg = "scope section must be a mapping"
        raise TypeError(msg)
    scope_paths = scope.get("scope_paths")
    if scope_paths is None:
        return
    if not isinstance(scope_paths, list) or not all(
        isinstance(value, str) for value in scope_paths
    ):
        msg = "scope.scope_paths must be a list of strings"
        raise TypeError(msg)


def _validate_variants_section(variants: object) -> None:
    if variants is None:
        return
    if not isinstance(variants, Mapping):
        msg = "variants section must be a mapping"
        raise TypeError(msg)


def _validate_hamilton_section(hamilton: object, *, config_path: Path) -> None:
    if hamilton is None:
        return
    if not isinstance(hamilton, Mapping):
        msg = "hamilton section must be a mapping"
        raise TypeError(msg)
    _decode_hamilton_section(hamilton, config_path=config_path)


def _validate_telemetry_section(telemetry: object, *, config_path: Path) -> None:
    if telemetry is None:
        return
    if not isinstance(telemetry, Mapping):
        msg = "telemetry section must be a mapping"
        raise TypeError(msg)
    _decode_telemetry_section(telemetry, config_path=config_path)


def _validate_schema_drift_mode(raw: object, *, config_path: Path) -> None:
    if raw is None:
        return
    if not isinstance(raw, str):
        msg = f"hamilton.schema_drift_mode must be a string in {config_path}"
        raise TypeError(msg)
    normalized = raw.strip().lower()
    if normalized in _ALLOWED_SCHEMA_DRIFT_MODES:
        return
    msg = (
        "hamilton.schema_drift_mode must be one of: "
        f"{', '.join(sorted(_ALLOWED_SCHEMA_DRIFT_MODES))} in {config_path}"
    )
    raise ValueError(msg)


def _reject_seeded_dataset_config(data: Mapping[str, Any], *, config_path: Path) -> None:
    prohibited = {"ci_seeded_datasets", "seed_suite_manifest_path"}
    top_level = sorted(prohibited.intersection(data.keys()))
    if top_level:
        msg = f"Seeded datasets are not supported; remove {', '.join(top_level)} from {config_path}"
        raise ValueError(msg)
    hamilton = data.get("hamilton")
    if not isinstance(hamilton, Mapping):
        return
    nested = sorted(prohibited.intersection(hamilton.keys()))
    if nested:
        msg = (
            "Seeded datasets are not supported; remove "
            f"{', '.join(nested)} from [hamilton] in {config_path}"
        )
        raise ValueError(msg)


DEFAULT_PARAMETERS: dict[str, dict[str, Any]] = {
    "subsystems": {
        "min_modules_per_subsystem": 2,
        "max_subsystems": 50,
    },
    "semantic_roles": {
        "min_confidence": 0.7,
    },
    "data_models": {
        "include_private": False,
    },
    "graph_metrics": {
        "enable_extended_metrics": True,
    },
    "external_deps": {
        "include_stdlib": False,
    },
    "entrypoints": {
        "detect_http": True,
        "detect_cli": True,
        "detect_grpc": False,
    },
}


def get_default_parameters(target_name: str) -> TargetParameters:
    """Get default parameters for a target.

    Parameters
    ----------
    target_name
        Target name.

    Returns
    -------
    TargetParameters
        Default parameters or empty.
    """
    defaults = DEFAULT_PARAMETERS.get(target_name, {})
    return TargetParameters(defaults)


def _require_str_field(data: Mapping[str, object], key: str, *, ctx: str) -> str:
    raw = data.get(key)
    if isinstance(raw, str) and raw:
        return raw
    msg = f"{ctx} missing {key}"
    raise TypeError(msg)
