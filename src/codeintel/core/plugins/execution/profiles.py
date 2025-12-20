"""Execution profile definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ExecutionProfile:
    """Named execution profile configuration."""

    name: str
    description: str
    plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    shared_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return option overrides for a specific plugin.

        Returns
        -------
        Mapping[str, Any] | None
            Option overrides, or None if not configured.
        """
        return self.plugin_options.get(plugin_name)

    def get_effective_options(self, plugin_name: str) -> dict[str, Any]:
        """Return merged shared + plugin-specific options.

        Returns
        -------
        dict[str, Any]
            Combined options dictionary.
        """
        result = dict(self.shared_options)
        plugin_opts = self.plugin_options.get(plugin_name)
        if plugin_opts:
            result.update(plugin_opts)
        return result


FAST_PROFILE = ExecutionProfile(
    name="fast",
    description="Quick iteration profile for local development.",
    plugin_options={
        "analytics.function_metrics": {
            "include_graph_metrics": False,
            "include_coverage_metrics": False,
        },
        "analytics.type_coverage": {
            "include_private": False,
        },
        "graphs.callgraph": {
            "use_libcst": False,
            "resolve_imports": False,
            "include_external_calls": False,
        },
        "graphs.import_graph": {
            "include_stdlib": False,
            "include_third_party": False,
        },
        "ingest.scip_python": {
            "include_references": False,
            "include_implementations": False,
            "timeout_seconds": 120,
        },
        "ingest.modules": {
            "include_tests": False,
        },
    },
    metadata={"version": "1.0.0", "category": "builtin"},
)

FULL_PROFILE = ExecutionProfile(
    name="full",
    description="Comprehensive analysis profile for thorough code intelligence.",
    plugin_options={
        "analytics.function_metrics": {
            "include_graph_metrics": True,
            "include_coverage_metrics": True,
        },
        "graphs.callgraph": {
            "use_libcst": True,
            "resolve_imports": True,
            "include_external_calls": True,
        },
        "graphs.import_graph": {
            "include_stdlib": True,
            "include_third_party": True,
        },
        "ingest.scip_python": {
            "include_references": True,
            "include_implementations": True,
            "timeout_seconds": 600,
        },
    },
    metadata={"version": "1.0.0", "category": "builtin"},
)

DEFAULT_PROFILE_NAME = FULL_PROFILE.name
_LEGACY_DEFAULT_PROFILE_ALIAS = "default"

CI_PROFILE = ExecutionProfile(
    name="ci",
    description="CI/CD profile optimized for automated pipelines.",
    plugin_options={
        "analytics.function_metrics": {
            "include_graph_metrics": True,
            "include_coverage_metrics": True,
        },
        "analytics.type_coverage": {
            "include_private": True,
        },
        "graphs.callgraph": {
            "use_libcst": True,
            "resolve_imports": True,
            "include_external_calls": False,
        },
        "graphs.import_graph": {
            "include_stdlib": False,
            "include_third_party": False,
        },
        "ingest.scip_python": {
            "include_references": True,
            "include_implementations": True,
            "timeout_seconds": 300,
        },
    },
    metadata={"version": "1.0.0", "category": "builtin"},
)

BUILTIN_PROFILES: dict[str, ExecutionProfile] = {
    "fast": FAST_PROFILE,
    "full": FULL_PROFILE,
    "ci": CI_PROFILE,
}

_custom_profiles: dict[str, ExecutionProfile] = {}


def register_profile(profile: ExecutionProfile) -> None:
    """Register a custom execution profile.

    Raises
    ------
    ValueError
        If a builtin or existing profile is re-registered.
    """
    if profile.name in BUILTIN_PROFILES:
        message = f"Cannot override builtin profile: {profile.name}"
        raise ValueError(message)
    if profile.name in _custom_profiles:
        message = f"Profile already registered: {profile.name}"
        raise ValueError(message)
    _custom_profiles[profile.name] = profile


def get_profile(name: str) -> ExecutionProfile:
    """Return a profile by name.

    Returns
    -------
    ExecutionProfile
        The requested profile instance.

    Raises
    ------
    KeyError
        When the profile name is unknown.
    """
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    if name == _LEGACY_DEFAULT_PROFILE_ALIAS:
        return FULL_PROFILE
    if name in _custom_profiles:
        return _custom_profiles[name]
    available = list(BUILTIN_PROFILES.keys()) + list(_custom_profiles.keys())
    message = f"Unknown profile: {name}. Available: {available}"
    raise KeyError(message)


def list_profiles() -> tuple[str, ...]:
    """Return names of all available profiles.

    Returns
    -------
    tuple[str, ...]
        Profile names.
    """
    return tuple(BUILTIN_PROFILES.keys()) + tuple(_custom_profiles.keys())


__all__ = [
    "CI_PROFILE",
    "DEFAULT_PROFILE_NAME",
    "FAST_PROFILE",
    "FULL_PROFILE",
    "ExecutionProfile",
    "get_profile",
    "list_profiles",
    "register_profile",
]
