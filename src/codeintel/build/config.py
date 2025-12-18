"""TOML-based configuration for build targets.

This module provides a global configuration system for tuning parameters.
Configuration is loaded from codeintel.build.toml in the project root.

The config file uses TOML format with sections for each module and
per-target overrides.

Example config file (codeintel.build.toml):
```toml
[analytics.hotspots]
max_commits = 2000

[analytics.function_history]
max_history_days = 365
min_lines_threshold = 1

[analytics.profiles]
include_ownership = true
max_parallel_workers = 4

[ingestion]
skip_empty_files = true
```

Example
-------
>>> from codeintel.build.config import load_build_config
>>> config = load_build_config(Path("/path/to/project"))
>>> hotspot_params = config.parameters_for("hotspots")
>>> max_commits = hotspot_params.get("max_commits", int, default=2000)
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, overload

from codeintel.build.parameters import TargetParameters

if TYPE_CHECKING:
    from pathlib import Path

_T = TypeVar("_T")

log = logging.getLogger(__name__)

__all__ = [
    "BuildConfig",
    "ConfigSection",
    "load_build_config",
]


CONFIG_FILE_NAME = "codeintel.build.toml"


@dataclass(frozen=True)
class ConfigSection:
    """A section of configuration (module or target level).

    Attributes
    ----------
    name
        Section name (e.g., "analytics", "analytics.hotspots").
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

    def get_section(self, name: str) -> ConfigSection | None:
        """Get a configuration section by name.

        Parameters
        ----------
        name
            Section name (e.g., "analytics.hotspots").

        Returns
        -------
        ConfigSection | None
            Section if found.
        """
        return self.sections.get(name)

    def parameters_for(self, target_name: str) -> TargetParameters:
        """Get parameters for a target, merging module and target sections.

        Looks up configuration in this order:
        1. Module-level section (e.g., "analytics")
        2. Target-level section (e.g., "analytics.hotspots")

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
        modules = ["ingestion", "graphs", "analytics", "export"]

        result_values: dict[str, Any] = {}

        def merge_nested_sections(*, prefix: str) -> None:
            """Merge nested section values under a target section into dict parameters.

            For example, given a TOML structure like:

            [analytics.function_metrics.features]
            eager_hydration = true

            the nested section name ``analytics.function_metrics.features`` is merged into:
            ``{"features": {"eager_hydration": True}}``.
            """
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
                    cursor[leaf_key] = dict(section.values)
                else:
                    existing_leaf.update(section.values)

        for module in modules:
            module_section = self.sections.get(module)
            if module_section:
                result_values.update(module_section.values)

            target_section = self.sections.get(f"{module}.{target_name}")
            if target_section:
                result_values.update(target_section.values)
                merge_nested_sections(prefix=f"{module}.{target_name}")

        return TargetParameters(result_values)

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


def load_build_config(project_root: Path) -> BuildConfig:
    """Load build configuration from project root.

    Looks for codeintel.build.toml in the project root.
    Returns empty config if file doesn't exist.

    Parameters
    ----------
    project_root
        Project root directory.

    Returns
    -------
    BuildConfig
        Loaded or empty configuration.
    """
    config_path = project_root / CONFIG_FILE_NAME

    if not config_path.exists():
        log.debug("No build config found at %s, using defaults", config_path)
        return BuildConfig.empty()

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        log.info("Loaded build config from %s", config_path)
        return BuildConfig.from_dict(data, config_path)
    except tomllib.TOMLDecodeError as e:
        log.warning("Failed to parse build config %s: %s", config_path, e)
        return BuildConfig.empty()
    except OSError as e:
        log.warning("Failed to read build config %s: %s", config_path, e)
        return BuildConfig.empty()


DEFAULT_PARAMETERS: dict[str, dict[str, Any]] = {
    "hotspots": {
        "max_commits": 2000,
    },
    "function_history": {
        "max_history_days": 365,
        "min_lines_threshold": 1,
        "default_branch": "HEAD",
    },
    "history_timeseries": {
        "days_back": 90,
        "bucket_days": 7,
    },
    "profiles": {
        "include_ownership": True,
        "compute_trends": True,
    },
    "subsystems": {
        "min_modules_per_subsystem": 2,
        "max_subsystems": 50,
    },
    "semantic_roles": {
        "min_confidence": 0.7,
    },
    "coverage_functions": {
        "min_coverage_threshold": 0.0,
    },
    "data_models": {
        "include_private": False,
    },
    "graph_metrics": {
        "enable_extended_metrics": True,
    },
    "risk_factors": {
        "weights": {
            "complexity": 0.3,
            "coverage": 0.2,
            "churn": 0.2,
            "coupling": 0.15,
            "age": 0.15,
        },
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
