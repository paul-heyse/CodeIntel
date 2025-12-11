"""Profile configuration file loader."""

from __future__ import annotations

import importlib
import logging
import tomllib
from pathlib import Path
from typing import Any, Protocol, cast

from codeintel.core.plugins.execution.profiles import ExecutionProfile, register_profile


class _YamlModule(Protocol):
    def safe_load(self, stream: object) -> dict[str, Any] | list[Any] | None:
        ...


yaml = cast("_YamlModule", importlib.import_module("yaml"))

log = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    TypeError
        If the YAML content is not a mapping.
    """
    with path.open(encoding="utf8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        message = f"YAML profile at {path} must be a mapping"
        raise TypeError(message)
    return cast("dict[str, Any]", data)


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed TOML content.
    """
    with path.open("rb") as file:
        return tomllib.load(file)


def load_profile_from_file(path: Path) -> ExecutionProfile:
    """Load an execution profile from a configuration file.

    Returns
    -------
    ExecutionProfile
        Loaded profile instance.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(path)
    elif suffix == ".toml":
        data = _load_toml(path)
    else:
        message = f"Unsupported profile file format: {suffix}"
        raise ValueError(message)

    return _parse_profile_data(data)


def _parse_profile_data(data: dict[str, Any]) -> ExecutionProfile:
    """Parse profile data from loaded configuration.

    Returns
    -------
    ExecutionProfile
        Constructed profile instance.

    Raises
    ------
    ValueError
        If required profile fields are missing.
    """
    profile_section = data.get("profile")
    if not profile_section:
        message = "Profile configuration must have a 'profile' section"
        raise ValueError(message)

    name = profile_section.get("name")
    description = profile_section.get("description")
    if not name or not description:
        message = "Profile must have 'name' and 'description'"
        raise ValueError(message)

    shared = data.get("shared", {})
    plugins = data.get("plugins", {})

    return ExecutionProfile(
        name=name,
        description=description,
        plugin_options=plugins,
        shared_options=shared,
        metadata={
            "version": profile_section.get("version", "0.0.0"),
            "source": "file",
        },
    )


def load_profiles_from_directory(
    directory: Path,
    *,
    register: bool = True,
) -> list[ExecutionProfile]:
    """Load all profile files from a directory.

    Returns
    -------
    list[ExecutionProfile]
        Loaded profiles.
    """
    profiles: list[ExecutionProfile] = []
    for path in directory.iterdir():
        if path.suffix.lower() in {".yaml", ".yml", ".toml"}:
            try:
                profile = load_profile_from_file(path)
                if register:
                    try:
                        register_profile(profile)
                    except ValueError as exc:
                        log.warning("Could not register profile %s: %s", profile.name, exc)
                profiles.append(profile)
                log.info("Loaded profile: %s from %s", profile.name, path)
            except (ValueError, OSError) as exc:
                log.warning("Failed to load profile from %s: %s", path, exc)
    return profiles


__all__ = [
    "load_profile_from_file",
    "load_profiles_from_directory",
]
