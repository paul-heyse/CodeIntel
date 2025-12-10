# Phase 4: Profile Integration Implementation Plan

> **Scope**: Wire up ProfiledConfigSource and define execution profiles
> **Duration**: 2-3 days
> **Risk Level**: Medium (integrates with execution paths)
> **Depends On**: Phase 3 (Full Rollout)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Task 1: Profile Definition Module](#3-task-1-profile-definition-module)
4. [Task 2: Profile Configuration Files](#4-task-2-profile-configuration-files)
5. [Task 3: Build Run Configuration](#5-task-3-build-run-configuration)
6. [Task 4: Executor Integration](#6-task-4-executor-integration)
7. [Task 5: Profile CLI Support](#7-task-5-profile-cli-support)
8. [Verification](#8-verification)
9. [Rollback Plan](#9-rollback-plan)

---

## 1. Overview

Phase 4 completes the options infrastructure by:

1. **Defining execution profiles** - Named configurations (fast, full, ci)
2. **Creating profile config files** - YAML/TOML profile definitions
3. **Wiring ProfiledConfigSource** - Load and merge configuration layers
4. **Integrating with executors** - Pass resolved options to plugins
5. **Adding CLI profile selection** - `--profile fast` flag support

### Profile Semantics

| Profile | Purpose | Characteristics |
|---------|---------|-----------------|
| `fast` | Quick iteration | Skip graph metrics, minimal coverage, fast parsing |
| `full` | Comprehensive analysis | All features enabled, high fidelity |
| `ci` | CI/CD pipelines | Balanced, optimized for CI resources |

---

## 2. Prerequisites

Verify Phase 3 is complete:

```bash
# Verify all metadata is registered
uv run python -c "
from codeintel.core.plugins.registry.all_metadata import ALL_PLUGIN_METADATA
print(f'Registered plugins: {len(ALL_PLUGIN_METADATA)}')
for m in ALL_PLUGIN_METADATA:
    print(f'  - {m.name} ({m.domain.value})')
"

# Verify global index works
uv run python -c "
from codeintel.core.plugins.registry.all_metadata import get_global_registry_index
index = get_global_registry_index()
print(f'Capabilities: {len(index.all_capabilities())}')
print(f'Tables: {len(index.all_tables())}')
"
```

---

## 3. Task 1: Profile Definition Module

### 3.1 Create `core/plugins/execution/profiles.py`

```python
# File: src/codeintel/core/plugins/execution/profiles.py
"""Execution profile definitions.

This module defines named execution profiles that configure plugin behavior
for different use cases (quick iteration, comprehensive analysis, CI/CD).

Architecture
------------
Profiles are immutable configuration bundles that provide:
- Plugin-specific option overrides
- Shared semantic meaning (what "fast" means across plugins)
- Extensibility for custom profiles

Usage
-----
>>> from codeintel.core.plugins.execution.profiles import (
...     get_profile,
...     FAST_PROFILE,
...     FULL_PROFILE,
... )
>>> profile = get_profile("fast")
>>> analytics_opts = profile.get_plugin_options("analytics.function_metrics")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ExecutionProfile:
    """Named execution profile configuration.

    An execution profile provides plugin-specific option overrides that
    define a coherent execution mode (fast, full, ci).

    Attributes
    ----------
    name
        Profile identifier ("fast", "full", "ci", or custom).
    description
        Human-readable description of the profile's purpose.
    plugin_options
        Mapping from plugin name to option overrides.
    shared_options
        Options applied to all plugins (e.g., scope_paths).
    metadata
        Additional profile metadata (author, version, etc.).

    Examples
    --------
    >>> profile = ExecutionProfile(
    ...     name="fast",
    ...     description="Quick iteration profile.",
    ...     plugin_options={
    ...         "analytics.function_metrics": {
    ...             "include_graph_metrics": False,
    ...         },
    ...     },
    ... )
    >>> profile.get_plugin_options("analytics.function_metrics")
    {'include_graph_metrics': False}
    """

    name: str
    description: str
    plugin_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    shared_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def get_plugin_options(self, plugin_name: str) -> Mapping[str, Any] | None:
        """Return option overrides for a specific plugin.

        Parameters
        ----------
        plugin_name
            Canonical plugin name.

        Returns
        -------
        Mapping[str, Any] | None
            Option overrides, or None if no overrides for this plugin.
        """
        return self.plugin_options.get(plugin_name)

    def get_effective_options(
        self,
        plugin_name: str,
    ) -> dict[str, Any]:
        """Return merged shared + plugin-specific options.

        Parameters
        ----------
        plugin_name
            Canonical plugin name.

        Returns
        -------
        dict[str, Any]
            Merged options (shared options + plugin-specific overrides).
        """
        result = dict(self.shared_options)
        plugin_opts = self.plugin_options.get(plugin_name)
        if plugin_opts:
            result.update(plugin_opts)
        return result


# =============================================================================
# Built-in Profiles
# =============================================================================

FAST_PROFILE = ExecutionProfile(
    name="fast",
    description="Quick iteration profile for local development.",
    plugin_options={
        # Analytics
        "analytics.function_metrics": {
            "include_graph_metrics": False,
            "include_coverage_metrics": False,
        },
        "analytics.type_coverage": {
            "include_private": False,
        },
        "analytics.docstring_metrics": {
            "include_examples": False,
        },
        "analytics.risk_profile": {
            "include_historical": False,
        },
        "analytics.hotspot_analysis": {
            "max_iterations": 10,
        },
        # Graphs
        "graphs.callgraph": {
            "use_libcst": False,
            "resolve_imports": False,
            "include_external_calls": False,
        },
        "graphs.import_graph": {
            "include_stdlib": False,
            "include_third_party": False,
        },
        "graphs.dataflow": {
            "max_depth": 3,
        },
        "graphs.cfg": {
            "simplify_branches": True,
        },
        # Ingestion
        "ingest.scip_python": {
            "include_references": False,
            "include_implementations": False,
            "timeout_seconds": 120,
        },
        "ingest.modules": {
            "include_tests": False,
        },
        "ingest.goid_builder": {
            "extract_docstrings": False,
        },
    },
    shared_options={},
    metadata={"version": "1.0.0", "category": "builtin"},
)

FULL_PROFILE = ExecutionProfile(
    name="full",
    description="Comprehensive analysis profile for thorough code intelligence.",
    plugin_options={
        # Analytics - all features enabled (use defaults)
        "analytics.function_metrics": {
            "include_graph_metrics": True,
            "include_coverage_metrics": True,
        },
        # Graphs - all features enabled (use defaults)
        "graphs.callgraph": {
            "use_libcst": True,
            "resolve_imports": True,
            "include_external_calls": True,
        },
        "graphs.import_graph": {
            "include_stdlib": True,
            "include_third_party": True,
        },
        # Ingestion - all features enabled (use defaults)
        "ingest.scip_python": {
            "include_references": True,
            "include_implementations": True,
            "timeout_seconds": 600,
        },
    },
    shared_options={},
    metadata={"version": "1.0.0", "category": "builtin"},
)

CI_PROFILE = ExecutionProfile(
    name="ci",
    description="CI/CD profile optimized for automated pipelines.",
    plugin_options={
        # Analytics - balanced settings
        "analytics.function_metrics": {
            "include_graph_metrics": True,
            "include_coverage_metrics": True,  # CI usually has coverage
        },
        "analytics.type_coverage": {
            "include_private": True,
        },
        # Graphs - moderate settings
        "graphs.callgraph": {
            "use_libcst": True,
            "resolve_imports": True,
            "include_external_calls": False,  # Focus on internal code
        },
        "graphs.import_graph": {
            "include_stdlib": False,
            "include_third_party": False,
        },
        # Ingestion - moderate timeouts
        "ingest.scip_python": {
            "include_references": True,
            "include_implementations": True,
            "timeout_seconds": 300,
        },
    },
    shared_options={},
    metadata={"version": "1.0.0", "category": "builtin"},
)

# =============================================================================
# Profile Registry
# =============================================================================

BUILTIN_PROFILES: dict[str, ExecutionProfile] = {
    "fast": FAST_PROFILE,
    "full": FULL_PROFILE,
    "ci": CI_PROFILE,
}

_custom_profiles: dict[str, ExecutionProfile] = {}


def register_profile(profile: ExecutionProfile) -> None:
    """Register a custom execution profile.

    Parameters
    ----------
    profile
        Profile to register.

    Raises
    ------
    ValueError
        If a profile with the same name already exists.
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

    Parameters
    ----------
    name
        Profile name.

    Returns
    -------
    ExecutionProfile
        The requested profile.

    Raises
    ------
    KeyError
        If no profile with the given name exists.
    """
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
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
    "ExecutionProfile",
    "FAST_PROFILE",
    "FULL_PROFILE",
    "get_profile",
    "list_profiles",
    "register_profile",
]
```

### 3.2 Test File: `tests/core/plugins/test_profiles.py`

```python
# File: tests/core/plugins/test_profiles.py
"""Tests for execution profiles."""

from __future__ import annotations

import pytest

from codeintel.core.plugins.execution.profiles import (
    CI_PROFILE,
    ExecutionProfile,
    FAST_PROFILE,
    FULL_PROFILE,
    get_profile,
    list_profiles,
    register_profile,
)


class TestExecutionProfile:
    """Tests for ExecutionProfile dataclass."""

    def test_get_plugin_options_existing(self) -> None:
        """Verify get_plugin_options returns options for configured plugin."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
            plugin_options={
                "plugin.a": {"key": "value"},
            },
        )
        opts = profile.get_plugin_options("plugin.a")
        assert opts == {"key": "value"}

    def test_get_plugin_options_missing(self) -> None:
        """Verify get_plugin_options returns None for unconfigured plugin."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
        )
        assert profile.get_plugin_options("unknown") is None

    def test_get_effective_options_merges_shared(self) -> None:
        """Verify get_effective_options merges shared and plugin options."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
            shared_options={"scope_paths": ["src/"]},
            plugin_options={
                "plugin.a": {"enabled": True},
            },
        )
        opts = profile.get_effective_options("plugin.a")
        assert opts == {"scope_paths": ["src/"], "enabled": True}

    def test_plugin_options_override_shared(self) -> None:
        """Verify plugin options override shared options."""
        profile = ExecutionProfile(
            name="test",
            description="Test profile.",
            shared_options={"key": "shared"},
            plugin_options={
                "plugin.a": {"key": "plugin"},
            },
        )
        opts = profile.get_effective_options("plugin.a")
        assert opts["key"] == "plugin"


class TestBuiltinProfiles:
    """Tests for builtin profiles."""

    def test_fast_profile_disables_expensive_features(self) -> None:
        """Verify fast profile disables expensive computations."""
        opts = FAST_PROFILE.get_plugin_options("analytics.function_metrics")
        assert opts is not None
        assert opts.get("include_graph_metrics") is False

    def test_full_profile_enables_all_features(self) -> None:
        """Verify full profile enables all features."""
        opts = FULL_PROFILE.get_plugin_options("analytics.function_metrics")
        assert opts is not None
        assert opts.get("include_graph_metrics") is True

    def test_ci_profile_has_balanced_settings(self) -> None:
        """Verify CI profile has balanced settings."""
        opts = CI_PROFILE.get_plugin_options("graphs.callgraph")
        assert opts is not None
        assert opts.get("include_external_calls") is False  # Focus internal


class TestProfileRegistry:
    """Tests for profile registry functions."""

    def test_get_profile_builtin(self) -> None:
        """Verify get_profile returns builtin profiles."""
        assert get_profile("fast") is FAST_PROFILE
        assert get_profile("full") is FULL_PROFILE
        assert get_profile("ci") is CI_PROFILE

    def test_get_profile_unknown_raises(self) -> None:
        """Verify get_profile raises for unknown profiles."""
        with pytest.raises(KeyError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_list_profiles_includes_builtins(self) -> None:
        """Verify list_profiles includes builtin profiles."""
        profiles = list_profiles()
        assert "fast" in profiles
        assert "full" in profiles
        assert "ci" in profiles

    def test_register_profile_custom(self) -> None:
        """Verify custom profile can be registered."""
        custom = ExecutionProfile(
            name="test_custom_unique",
            description="Custom test profile.",
        )
        register_profile(custom)
        assert get_profile("test_custom_unique") is custom

    def test_register_profile_duplicate_raises(self) -> None:
        """Verify registering duplicate profile raises."""
        custom = ExecutionProfile(
            name="test_dup",
            description="Test.",
        )
        register_profile(custom)
        with pytest.raises(ValueError, match="already registered"):
            register_profile(custom)

    def test_cannot_override_builtin(self) -> None:
        """Verify builtin profiles cannot be overridden."""
        fake_fast = ExecutionProfile(
            name="fast",
            description="Fake fast.",
        )
        with pytest.raises(ValueError, match="Cannot override builtin"):
            register_profile(fake_fast)
```

---

## 4. Task 2: Profile Configuration Files

### 4.1 Create Profile YAML Schema

```yaml
# File: config/profiles/schema.yaml
# JSON Schema for profile configuration files

$schema: "https://json-schema.org/draft/2020-12/schema"
$id: "https://codeintel.dev/schemas/profile-config.schema.json"
title: "CodeIntel Profile Configuration"
description: "Schema for plugin profile configuration files"
type: object
properties:
  profile:
    type: object
    properties:
      name:
        type: string
        description: "Profile identifier"
      description:
        type: string
        description: "Human-readable description"
      version:
        type: string
        description: "Profile version (semver)"
    required: [name, description]
  
  shared:
    type: object
    description: "Options applied to all plugins"
    additionalProperties: true
  
  plugins:
    type: object
    description: "Plugin-specific option overrides"
    additionalProperties:
      type: object
      additionalProperties: true

required: [profile]
```

### 4.2 Create Example Profile Files

```yaml
# File: config/profiles/fast.yaml
# Fast execution profile - optimized for quick iteration

profile:
  name: fast
  description: "Quick iteration profile for local development"
  version: "1.0.0"

shared:
  # Shared options applied to all plugins
  # scope_paths: ["src/"]  # Uncomment to limit scope

plugins:
  # Analytics plugins
  analytics.function_metrics:
    include_graph_metrics: false
    include_coverage_metrics: false

  analytics.type_coverage:
    include_private: false

  analytics.docstring_metrics:
    include_examples: false

  # Graph plugins
  graphs.callgraph:
    use_libcst: false
    resolve_imports: false
    include_external_calls: false

  graphs.import_graph:
    include_stdlib: false
    include_third_party: false

  # Ingestion plugins
  ingest.scip_python:
    include_references: false
    include_implementations: false
    timeout_seconds: 120

  ingest.modules:
    include_tests: false
```

```yaml
# File: config/profiles/full.yaml
# Full execution profile - comprehensive analysis

profile:
  name: full
  description: "Comprehensive analysis profile"
  version: "1.0.0"

shared: {}

plugins:
  analytics.function_metrics:
    include_graph_metrics: true
    include_coverage_metrics: true

  graphs.callgraph:
    use_libcst: true
    resolve_imports: true
    include_external_calls: true

  graphs.import_graph:
    include_stdlib: true
    include_third_party: true

  ingest.scip_python:
    include_references: true
    include_implementations: true
    timeout_seconds: 600
```

### 4.3 Create Profile Loader

```python
# File: src/codeintel/core/plugins/execution/profile_loader.py
"""Profile configuration file loader.

This module provides utilities for loading execution profiles from
YAML or TOML configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from codeintel.core.plugins.execution.profiles import (
    ExecutionProfile,
    register_profile,
)

log = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    path
        Path to YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.
    """
    import yaml  # type: ignore[import-untyped]

    with path.open() as f:
        return yaml.safe_load(f)


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML configuration file.

    Parameters
    ----------
    path
        Path to TOML file.

    Returns
    -------
    dict[str, Any]
        Parsed TOML content.
    """
    import tomllib

    with path.open("rb") as f:
        return tomllib.load(f)


def load_profile_from_file(path: Path) -> ExecutionProfile:
    """Load an execution profile from a configuration file.

    Parameters
    ----------
    path
        Path to profile configuration file (YAML or TOML).

    Returns
    -------
    ExecutionProfile
        Loaded profile.

    Raises
    ------
    ValueError
        If the file format is not supported or content is invalid.
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

    Parameters
    ----------
    data
        Parsed configuration data.

    Returns
    -------
    ExecutionProfile
        Constructed profile.

    Raises
    ------
    ValueError
        If required fields are missing.
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

    Parameters
    ----------
    directory
        Directory containing profile configuration files.
    register
        Whether to register loaded profiles globally.

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
                    except ValueError as e:
                        log.warning("Could not register profile %s: %s", profile.name, e)
                profiles.append(profile)
                log.info("Loaded profile: %s from %s", profile.name, path)
            except (ValueError, OSError) as e:
                log.warning("Failed to load profile from %s: %s", path, e)

    return profiles


__all__ = [
    "load_profile_from_file",
    "load_profiles_from_directory",
]
```

---

## 5. Task 3: Build Run Configuration

### 5.1 Create `BuildRunConfig`

```python
# File: src/codeintel/build/config.py
"""Build run configuration.

This module provides `BuildRunConfig`, the top-level configuration object
for a single build run, integrating profiles, CLI overrides, and scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.core.plugins.execution.options import (
    PluginConfigBundle,
    PluginOptionsResolver,
    ProfiledConfigSource,
)
from codeintel.core.plugins.execution.profiles import (
    ExecutionProfile,
    get_profile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class BuildRunConfig:
    """Configuration for a single build/analytics run.

    This is the top-level configuration object that ties together:
    - Execution profile selection
    - CLI option overrides
    - Scope restrictions (paths)
    - Output configuration

    Attributes
    ----------
    profile_name
        Name of the execution profile ("fast", "full", "ci", or custom).
    cli_overrides
        Plugin options from CLI flags.
    scope_paths
        If set, restrict processing to these paths.
    output_dir
        Directory for build outputs.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Examples
    --------
    >>> config = BuildRunConfig(
    ...     profile_name="fast",
    ...     scope_paths=["src/codeintel/"],
    ... )
    >>> resolver = config.build_options_resolver()
    """

    profile_name: str = "full"
    cli_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    scope_paths: list[str] | None = None
    output_dir: Path | None = None
    repo: str = ""
    commit: str = ""

    # Base configuration (from config files)
    base_config: PluginConfigBundle | None = None

    def get_profile(self) -> ExecutionProfile:
        """Return the selected execution profile.

        Returns
        -------
        ExecutionProfile
            The profile for this run.
        """
        return get_profile(self.profile_name)

    def build_options_resolver(self) -> PluginOptionsResolver:
        """Build a PluginOptionsResolver for this run configuration.

        Creates a ProfiledConfigSource with:
        - Base layer: base_config (from files)
        - Profile layer: selected profile's plugin_options
        - CLI layer: cli_overrides

        Returns
        -------
        PluginOptionsResolver
            Resolver configured for this run.
        """
        profile = self.get_profile()

        # Apply scope_paths to shared options if set
        profile_opts = dict(profile.plugin_options)
        if self.scope_paths:
            # Add scope_paths to each plugin's options
            for plugin_name in profile_opts:
                opts = dict(profile_opts[plugin_name])
                if "scope_paths" not in opts:
                    opts["scope_paths"] = self.scope_paths
                profile_opts[plugin_name] = opts

        config_source = ProfiledConfigSource(
            base=self.base_config or PluginConfigBundle(),
            profile=PluginConfigBundle(plugin_options=profile_opts),
            cli=PluginConfigBundle(plugin_options=dict(self.cli_overrides)),
            active_profile_name=self.profile_name,
        )

        return PluginOptionsResolver(config_source)


def create_build_config(
    *,
    profile: str = "full",
    scope_paths: list[str] | None = None,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    repo: str = "",
    commit: str = "",
    base_config_path: Path | None = None,
) -> BuildRunConfig:
    """Create a BuildRunConfig from parameters.

    Parameters
    ----------
    profile
        Profile name.
    scope_paths
        Scope restriction paths.
    cli_overrides
        CLI option overrides.
    output_dir
        Output directory.
    repo
        Repository identifier.
    commit
        Commit SHA.
    base_config_path
        Path to base configuration file.

    Returns
    -------
    BuildRunConfig
        Configured build run.
    """
    base_config = None
    if base_config_path and base_config_path.exists():
        base_config = _load_base_config(base_config_path)

    return BuildRunConfig(
        profile_name=profile,
        cli_overrides=cli_overrides or {},
        scope_paths=scope_paths,
        output_dir=output_dir,
        repo=repo,
        commit=commit,
        base_config=base_config,
    )


def _load_base_config(path: Path) -> PluginConfigBundle:
    """Load base configuration from file.

    Parameters
    ----------
    path
        Path to configuration file.

    Returns
    -------
    PluginConfigBundle
        Loaded configuration bundle.
    """
    import yaml  # type: ignore[import-untyped]

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    plugins = data.get("plugins", {})
    return PluginConfigBundle(plugin_options=plugins)


__all__ = [
    "BuildRunConfig",
    "create_build_config",
]
```

### 5.2 Test File: `tests/build/test_build_config.py`

```python
# File: tests/build/test_build_config.py
"""Tests for BuildRunConfig."""

from __future__ import annotations

import pytest

from codeintel.build.config import BuildRunConfig, create_build_config
from codeintel.core.plugins.execution.profiles import FAST_PROFILE


class TestBuildRunConfig:
    """Tests for BuildRunConfig."""

    def test_default_profile_is_full(self) -> None:
        """Verify default profile is 'full'."""
        config = BuildRunConfig()
        assert config.profile_name == "full"

    def test_get_profile(self) -> None:
        """Verify get_profile returns correct profile."""
        config = BuildRunConfig(profile_name="fast")
        profile = config.get_profile()
        assert profile is FAST_PROFILE

    def test_build_options_resolver(self) -> None:
        """Verify build_options_resolver creates working resolver."""
        config = BuildRunConfig(profile_name="fast")
        resolver = config.build_options_resolver()
        opts = resolver.config_source.get_plugin_options("analytics.function_metrics")
        assert opts is not None
        assert opts.get("include_graph_metrics") is False

    def test_scope_paths_propagated(self) -> None:
        """Verify scope_paths are added to plugin options."""
        config = BuildRunConfig(
            profile_name="fast",
            scope_paths=["src/"],
        )
        resolver = config.build_options_resolver()
        opts = resolver.config_source.get_plugin_options("analytics.function_metrics")
        assert opts is not None
        assert opts.get("scope_paths") == ["src/"]

    def test_cli_overrides(self) -> None:
        """Verify CLI overrides take precedence."""
        config = BuildRunConfig(
            profile_name="fast",
            cli_overrides={
                "analytics.function_metrics": {
                    "include_graph_metrics": True,  # Override fast profile
                },
            },
        )
        resolver = config.build_options_resolver()
        opts = resolver.config_source.get_plugin_options("analytics.function_metrics")
        assert opts is not None
        assert opts.get("include_graph_metrics") is True  # CLI wins


class TestCreateBuildConfig:
    """Tests for create_build_config helper."""

    def test_creates_config_with_defaults(self) -> None:
        """Verify create_build_config creates config with defaults."""
        config = create_build_config()
        assert config.profile_name == "full"
        assert config.scope_paths is None

    def test_creates_config_with_params(self) -> None:
        """Verify create_build_config respects parameters."""
        config = create_build_config(
            profile="fast",
            scope_paths=["src/"],
            repo="owner/repo",
            commit="abc123",
        )
        assert config.profile_name == "fast"
        assert config.scope_paths == ["src/"]
        assert config.repo == "owner/repo"
        assert config.commit == "abc123"
```

---

## 6. Task 4: Executor Integration

### 6.1 Update Build Executor

```python
# File: src/codeintel/build/executor.py (modifications)
# Add profile-aware execution support

"""Build executor with profile support.

Modifications to integrate BuildRunConfig and options resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.config import BuildRunConfig
    from codeintel.build.plugin import TargetPlugin
    from codeintel.core.plugins.execution.options import PluginOptionsResolver


class BuildExecutor:
    """Execute build plugins with profile-aware configuration.

    Attributes
    ----------
    config
        Build run configuration.
    resolver
        Options resolver built from config.
    """

    def __init__(self, config: BuildRunConfig) -> None:
        """Initialize executor with configuration.

        Parameters
        ----------
        config
            Build run configuration.
        """
        self._config = config
        self._resolver = config.build_options_resolver()

    @property
    def options_resolver(self) -> PluginOptionsResolver:
        """Return the options resolver.

        Returns
        -------
        PluginOptionsResolver
            Resolver for plugin options.
        """
        return self._resolver

    def configure_plugin(self, plugin: TargetPlugin) -> TargetPlugin:
        """Configure a plugin with the options resolver.

        If the plugin supports options resolution (has __init__ with
        options_resolver parameter), create a new instance with the
        resolver injected.

        Parameters
        ----------
        plugin
            Plugin to configure.

        Returns
        -------
        TargetPlugin
            Configured plugin (may be same instance or new).
        """
        # Check if plugin supports options resolver injection
        if hasattr(plugin.__class__, "__init__"):
            import inspect
            sig = inspect.signature(plugin.__class__.__init__)
            if "options_resolver" in sig.parameters:
                # Create new instance with resolver
                return plugin.__class__(options_resolver=self._resolver)
        return plugin

    # ... rest of executor implementation ...
```

---

## 7. Task 5: Profile CLI Support

### 7.1 Add Profile CLI Arguments

```python
# File: src/codeintel/cli/profile_args.py
"""Profile CLI argument definitions.

This module provides CLI argument handling for profile selection
and option overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import cyclopts

from codeintel.build.config import create_build_config
from codeintel.core.plugins.execution.profiles import list_profiles

ProfileArg = Annotated[
    str,
    cyclopts.Parameter(
        name=["--profile", "-p"],
        help="Execution profile (fast, full, ci, or custom)",
    ),
]

ScopePathsArg = Annotated[
    list[str] | None,
    cyclopts.Parameter(
        name=["--scope", "-s"],
        help="Limit processing to these paths",
    ),
]


def parse_cli_overrides(override_strs: list[str]) -> dict[str, dict[str, Any]]:
    """Parse CLI option override strings.

    Format: "plugin.name:key=value"

    Parameters
    ----------
    override_strs
        Override strings from CLI.

    Returns
    -------
    dict[str, dict[str, Any]]
        Parsed overrides by plugin name.

    Examples
    --------
    >>> parse_cli_overrides([
    ...     "analytics.function_metrics:include_graph_metrics=false",
    ...     "graphs.callgraph:use_libcst=true",
    ... ])
    {
        'analytics.function_metrics': {'include_graph_metrics': False},
        'graphs.callgraph': {'use_libcst': True},
    }
    """
    result: dict[str, dict[str, Any]] = {}

    for override in override_strs:
        if ":" not in override or "=" not in override:
            continue

        plugin_part, kv_part = override.split(":", 1)
        key, value_str = kv_part.split("=", 1)

        # Parse value
        value: Any
        if value_str.lower() == "true":
            value = True
        elif value_str.lower() == "false":
            value = False
        elif value_str.isdigit():
            value = int(value_str)
        else:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str

        if plugin_part not in result:
            result[plugin_part] = {}
        result[plugin_part][key] = value

    return result


def build_config_from_cli(
    profile: str = "full",
    scope_paths: list[str] | None = None,
    overrides: list[str] | None = None,
    output_dir: Path | None = None,
    repo: str = "",
    commit: str = "",
    base_config: Path | None = None,
) -> Any:  # Returns BuildRunConfig
    """Build a BuildRunConfig from CLI arguments.

    Parameters
    ----------
    profile
        Profile name.
    scope_paths
        Scope paths.
    overrides
        CLI override strings.
    output_dir
        Output directory.
    repo
        Repository.
    commit
        Commit.
    base_config
        Base config file path.

    Returns
    -------
    BuildRunConfig
        Configured build run.
    """
    cli_overrides = parse_cli_overrides(overrides or [])

    return create_build_config(
        profile=profile,
        scope_paths=scope_paths,
        cli_overrides=cli_overrides,
        output_dir=output_dir,
        repo=repo,
        commit=commit,
        base_config_path=base_config,
    )


__all__ = [
    "ProfileArg",
    "ScopePathsArg",
    "build_config_from_cli",
    "parse_cli_overrides",
]
```

### 7.2 Example CLI Integration

```python
# Example: Add to existing CLI command
# File: src/codeintel/cli/commands/build.py (example integration)

from codeintel.cli.profile_args import (
    ProfileArg,
    ScopePathsArg,
    build_config_from_cli,
)

@app.command()
def build(
    repo: str,
    commit: str,
    profile: ProfileArg = "full",
    scope: ScopePathsArg = None,
    override: list[str] | None = None,
) -> None:
    """Run build with profile configuration.

    Examples
    --------
    # Fast profile for quick iteration
    codeintel build owner/repo abc123 --profile fast

    # Full profile with scope restriction
    codeintel build owner/repo abc123 --profile full --scope src/

    # Custom overrides
    codeintel build owner/repo abc123 \\
        --override "analytics.function_metrics:include_graph_metrics=false"
    """
    config = build_config_from_cli(
        profile=profile,
        scope_paths=scope,
        overrides=override,
        repo=repo,
        commit=commit,
    )

    # Use config.build_options_resolver() in execution
    # ...
```

---

## 8. Verification

### 8.1 Run Quality Checks

```bash
# Format and lint
uv run ruff format \
    src/codeintel/core/plugins/execution/profiles.py \
    src/codeintel/core/plugins/execution/profile_loader.py \
    src/codeintel/build/config.py \
    src/codeintel/cli/profile_args.py

uv run ruff check --fix \
    src/codeintel/core/plugins/execution/ \
    src/codeintel/build/ \
    src/codeintel/cli/

# Type checking
uv run pyright \
    src/codeintel/core/plugins/execution/profiles.py \
    src/codeintel/build/config.py
```

### 8.2 Run Tests

```bash
# Run profile tests
uv run pytest tests/core/plugins/test_profiles.py -v
uv run pytest tests/build/test_build_config.py -v

# Run integration tests
uv run pytest tests/build/ tests/cli/ -v -k profile
```

### 8.3 Verification Checklist

- [ ] All builtin profiles (fast, full, ci) are defined
- [ ] Profiles can be loaded from YAML/TOML files
- [ ] BuildRunConfig integrates with ProfiledConfigSource
- [ ] CLI profile selection works
- [ ] CLI option overrides work
- [ ] Scope paths propagate to plugins
- [ ] Profile settings flow through to plugin execution

---

## 9. Rollback Plan

Phase 4 changes can be rolled back by:

1. **Revert execution modules**:
   - `src/codeintel/core/plugins/execution/profiles.py`
   - `src/codeintel/core/plugins/execution/profile_loader.py`
2. **Revert build config**:
   - `src/codeintel/build/config.py`
3. **Revert CLI changes**:
   - `src/codeintel/cli/profile_args.py`
4. **Remove profile configuration files**:
   - `config/profiles/*.yaml`
5. **Delete test files**

---

**Next Steps**: After Phase 4 is complete, proceed to Phase 5 (Skip/Manifest Integration) to enable execution skipping based on input hash comparison.
