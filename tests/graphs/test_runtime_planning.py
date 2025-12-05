"""Extended tests for graph runtime planning module.

This module provides additional test coverage for the planning module,
focusing on specific paths not covered by test_runtime.py:

- Plugin options resolution and merging
- Severity and timeout resolution with overrides
- Target resolution from various sources
- Plugin settings hash computation
- Execution input preparation
- Scope override via run options
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Final

import pytest

from codeintel.config.steps_graphs import (
    GraphMetricsStepConfig,
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.core.plugins.types.protocol import PluginResourceHints, PluginSeverity
from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import get_graph_registry, register_graph_plugin
from codeintel.graphs.runtime import planning
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    GraphPluginRunOptions,
    PlanCoordinates,
    plan_graph_plugin_run,
)
from tests._helpers.assertions import assert_cannot_setattr
from tests._helpers.factories import make_snapshot

# Constants
EXPECTED_HASH_LENGTH: Final = 16
TIMEOUT_DEFAULT_MS: Final = 10000
TIMEOUT_OVERRIDE_MS: Final = 5000
RETRY_MAX_ATTEMPTS: Final = 3
_PLANNING_PRIVATES: Final = planning.__dict__
RESOLVE_PLUGIN_OPTIONS_MAP = _PLANNING_PRIVATES["_resolve_plugin_options_map"]
EFFECTIVE_SEVERITY = _PLANNING_PRIVATES["_effective_severity"]
EFFECTIVE_TIMEOUT = _PLANNING_PRIVATES["_effective_timeout"]
RESOLVE_TARGET = _PLANNING_PRIVATES["_resolve_target"]
BUILD_PLUGIN_SETTINGS = _PLANNING_PRIVATES["_build_plugin_settings"]


# Test Helpers


@dataclass
class PluginConfig:
    """Configuration for constructing test planning plugins."""

    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    severity: PluginSeverity = "fatal"
    resource_hints: PluginResourceHints | None = None
    options_default: object | None = None


PLUGIN_CONFIG_FIELDS: Final = {field.name for field in fields(PluginConfig)}


def _resolve_plugin_config(
    config: PluginConfig | None, overrides: dict[str, object]
) -> PluginConfig:
    """Merge a base plugin config with validated overrides.

    Parameters
    ----------
    config
        Base plugin configuration or None for defaults.
    overrides
        Override values keyed by PluginConfig field names.

    Returns
    -------
    PluginConfig
        Combined plugin configuration.

    Raises
    ------
    ValueError
        If overrides contain unsupported keys.
    """
    unknown_keys = set(overrides) - PLUGIN_CONFIG_FIELDS
    if unknown_keys:
        message = f"Unsupported plugin config overrides: {sorted(unknown_keys)}"
        raise ValueError(message)
    base_config = config or PluginConfig()
    if not overrides:
        return base_config
    return replace(base_config, **overrides)


class _PluginRegistrar:
    """Context manager for registering and cleaning up test plugins."""

    def __init__(self, plugins: list[GraphPluginProtocol]) -> None:
        """Initialize with plugins to register.

        Parameters
        ----------
        plugins
            Plugins to register.
        """
        self._plugins = plugins
        self._registry = get_graph_registry()

    def __enter__(self) -> None:
        """Register plugins on entry."""
        for plugin in self._plugins:
            register_graph_plugin(plugin)

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Unregister plugins on exit."""
        for plugin in self._plugins:
            self._registry.unregister(plugin.metadata.name)


def _make_test_plugin(
    name: str, *, config: PluginConfig | None = None, **overrides: object
) -> GraphPluginProtocol:
    """Create a configurable test plugin for planning tests.

    Parameters
    ----------
    name
        Plugin name.
    config
        Base configuration for plugin metadata and defaults.
    **overrides
        Overrides for plugin configuration fields.

    Returns
    -------
    GraphPluginProtocol
        Configured test plugin.
    """
    plugin_config = _resolve_plugin_config(config, overrides)

    def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
        return PluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
        depends_on=plugin_config.depends_on,
        provides=plugin_config.provides,
        severity=plugin_config.severity,
        resource_hints=plugin_config.resource_hints,
        options_default=plugin_config.options_default,
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def test_resolve_plugin_options_map_uses_default() -> None:
    """Plugin default options used when no config or runtime options."""
    plugin = _make_test_plugin("opt_default", options_default={"default_key": "value"})

    resolved = RESOLVE_PLUGIN_OPTIONS_MAP(
        plugins=[plugin],
        cfg_options=None,
        runtime_options=None,
    )

    assert resolved[plugin.metadata.name] == {"default_key": "value"}


def test_resolve_plugin_options_map_config_overrides_default() -> None:
    """Config options override plugin defaults."""
    plugin = _make_test_plugin("opt_cfg", options_default={"key": "default"})

    resolved = RESOLVE_PLUGIN_OPTIONS_MAP(
        plugins=[plugin],
        cfg_options={"opt_cfg": {"key": "from_config"}},
        runtime_options=None,
    )

    assert resolved["opt_cfg"] == {"key": "from_config"}


def test_resolve_plugin_options_map_runtime_overrides_config() -> None:
    """Runtime options override both config and default options."""
    plugin = _make_test_plugin("opt_runtime", options_default={"key": "default"})

    resolved = RESOLVE_PLUGIN_OPTIONS_MAP(
        plugins=[plugin],
        cfg_options={"opt_runtime": {"key": "from_config"}},
        runtime_options={"opt_runtime": {"key": "from_runtime"}},
    )

    assert resolved["opt_runtime"] == {"key": "from_runtime"}


def test_resolve_plugin_options_map_merges_multiple_plugins() -> None:
    """Options resolved correctly for multiple plugins."""
    plugin_a = _make_test_plugin("opt_a", options_default={"a": 1})
    plugin_b = _make_test_plugin("opt_b", options_default=None)

    resolved = RESOLVE_PLUGIN_OPTIONS_MAP(
        plugins=[plugin_a, plugin_b],
        cfg_options={"opt_b": {"b": 2}},
        runtime_options=None,
    )

    assert resolved["opt_a"] == {"a": 1}
    assert resolved["opt_b"] == {"b": 2}


def test_resolve_plugin_options_map_unknown_plugin_raises() -> None:
    """Raise ValueError when options provided for unknown plugin."""
    plugin = _make_test_plugin("known_plugin")

    with pytest.raises(ValueError, match="unknown graph plugins"):
        RESOLVE_PLUGIN_OPTIONS_MAP(
            plugins=[plugin],
            cfg_options={"unknown_plugin": {"key": "value"}},
            runtime_options=None,
        )


def test_resolve_plugin_options_map_unknown_runtime_plugin_raises() -> None:
    """Raise ValueError when runtime options provided for unknown plugin."""
    plugin = _make_test_plugin("known_plugin")

    with pytest.raises(ValueError, match="unknown graph plugins"):
        RESOLVE_PLUGIN_OPTIONS_MAP(
            plugins=[plugin],
            cfg_options=None,
            runtime_options={"unknown_runtime": {"key": "value"}},
        )


def test_effective_severity_uses_policy_default() -> None:
    """Effective severity uses policy default when no override."""
    plugin = _make_test_plugin("sev_default", severity="fatal")
    policy = GraphPluginPolicy(default_severity="soft_fail")

    severity = EFFECTIVE_SEVERITY(plugin, policy)

    assert severity == "soft_fail"


def test_effective_severity_uses_override() -> None:
    """Severity override in policy takes precedence."""
    plugin = _make_test_plugin("sev_override", severity="fatal")
    policy = GraphPluginPolicy(
        default_severity="soft_fail",
        severity_overrides={"sev_override": "skip_on_error"},
    )

    severity = EFFECTIVE_SEVERITY(plugin, policy)

    assert severity == "skip_on_error"


def test_effective_timeout_uses_policy_override() -> None:
    """Timeout override in policy takes precedence."""
    plugin = _make_test_plugin("timeout_override")
    policy = GraphPluginPolicy(
        timeouts_ms={"timeout_override": TIMEOUT_OVERRIDE_MS},
    )

    timeout = EFFECTIVE_TIMEOUT(plugin, policy)

    assert timeout == TIMEOUT_OVERRIDE_MS


def test_effective_timeout_uses_resource_hints() -> None:
    """Timeout from plugin resource hints used when no policy override."""
    hints = PluginResourceHints(max_runtime_ms=TIMEOUT_DEFAULT_MS)
    plugin = _make_test_plugin("timeout_hints", resource_hints=hints)
    policy = GraphPluginPolicy()

    timeout = EFFECTIVE_TIMEOUT(plugin, policy)

    assert timeout == TIMEOUT_DEFAULT_MS


def test_effective_timeout_none_when_no_hints_or_override() -> None:
    """Timeout is None when no hints and no policy override."""
    plugin = _make_test_plugin("timeout_none")
    policy = GraphPluginPolicy()

    timeout = EFFECTIVE_TIMEOUT(plugin, policy)

    assert timeout is None


def test_resolve_target_from_cfg() -> None:
    """Target resolved from configuration."""
    cfg = GraphMetricsStepConfig(repo="cfg/repo", commit="cfg_commit")

    repo, commit = RESOLVE_TARGET(
        cfg=cfg,
        runtime_snapshot=None,
        target=None,
    )

    assert repo == "cfg/repo"
    assert commit == "cfg_commit"


def test_resolve_target_from_explicit_tuple() -> None:
    """Explicit target tuple takes precedence over config."""
    repo, commit = RESOLVE_TARGET(
        cfg=None,  # No config
        runtime_snapshot=None,
        target=("explicit/repo", "explicit_commit"),
    )

    assert repo == "explicit/repo"
    assert commit == "explicit_commit"


def test_resolve_target_from_runtime_snapshot() -> None:
    """Target resolved from runtime snapshot when no explicit target."""
    snapshot = make_snapshot(repo="snapshot/repo", commit="snap_commit")

    repo, commit = RESOLVE_TARGET(
        cfg=None,
        runtime_snapshot=snapshot,
        target=None,
    )

    assert repo == "snapshot/repo"
    assert commit == "snap_commit"


def test_resolve_target_missing_raises() -> None:
    """Raise ValueError when no target source available."""
    with pytest.raises(ValueError, match="missing snapshot"):
        RESOLVE_TARGET(
            cfg=None,
            runtime_snapshot=None,
            target=None,
        )


def test_build_plugin_settings_computes_hashes() -> None:
    """Plugin settings include computed input and options hashes."""
    plugin = _make_test_plugin("hash_settings")
    policy = GraphPluginPolicy()
    coords = PlanCoordinates(
        repo="test/repo",
        commit="abc123",
        scope=GraphRunScope(),
    )
    options = {"key": "value"}

    settings = BUILD_PLUGIN_SETTINGS(plugin, policy, coords, options)

    assert settings.name == "hash_settings"
    assert settings.input_hash is not None
    assert len(settings.input_hash) == EXPECTED_HASH_LENGTH
    assert settings.options_hash is not None
    assert len(settings.options_hash) == EXPECTED_HASH_LENGTH


def test_build_plugin_settings_includes_policy_values() -> None:
    """Plugin settings reflect policy configuration."""
    plugin = _make_test_plugin("policy_settings")
    policy = GraphPluginPolicy(
        default_severity="skip_on_error",
        fail_fast=False,
        timeouts_ms={"policy_settings": TIMEOUT_OVERRIDE_MS},
        retries={"policy_settings": GraphPluginRetryPolicy(max_attempts=3)},
    )
    coords = PlanCoordinates(
        repo="test/repo",
        commit="abc123",
        scope=GraphRunScope(),
    )

    settings = BUILD_PLUGIN_SETTINGS(plugin, policy, coords, None)

    assert settings.severity == "skip_on_error"
    assert settings.timeout_ms == TIMEOUT_OVERRIDE_MS
    assert settings.fail_fast is False
    assert settings.retry_cfg.max_attempts == RETRY_MAX_ATTEMPTS


def test_build_plugin_settings_includes_version_hash() -> None:
    """Plugin settings include plugin version hash."""
    plugin = _make_test_plugin("version_settings")
    policy = GraphPluginPolicy()
    coords = PlanCoordinates(
        repo="test/repo",
        commit="abc123",
        scope=GraphRunScope(),
    )

    settings = BUILD_PLUGIN_SETTINGS(plugin, policy, coords, None)

    assert settings.version_hash == plugin.metadata.version_hash


def test_plan_graph_plugin_run_basic(tmp_path: Path) -> None:
    """Basic plan generation produces valid execution plan."""
    plugin = _make_test_plugin("basic_plan")
    snapshot = make_snapshot(repo="plan/repo", commit="plan_commit", repo_root=tmp_path)

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )

        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.plan_id
        assert plan.run_id
        assert plan.repo == "plan/repo"
        assert plan.commit == "plan_commit"
        assert len(plan.plugins) == 1
        assert plan.plugins[0].metadata.name == "basic_plan"


def test_plan_graph_plugin_run_with_scope_override() -> None:
    """Run options scope overrides config scope."""
    plugin = _make_test_plugin("scope_plan")
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin]):
        run_options = GraphPluginRunOptions(
            scope=GraphRunScope(paths=("custom/path/",), modules=("custom.module",)),
        )
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
            run_options=run_options,
        )

        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.scope.paths == ("custom/path/",)
        assert plan.scope.modules == ("custom.module",)


def test_plan_graph_plugin_run_with_plugin_options() -> None:
    """Plugin options included in plan."""
    plugin = _make_test_plugin("options_plan", options_default={"default": True})
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin]):
        run_options = GraphPluginRunOptions(
            plugin_options={"options_plan": {"override": True}},
        )
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
            run_options=run_options,
        )

        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.options_by_plugin["options_plan"] == {"override": True}


def test_plan_graph_plugin_run_with_prior_manifest() -> None:
    """Prior manifest included in plan for skip detection."""
    plugin = _make_test_plugin("manifest_plan")
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin]):
        prior = {"manifest_plan": {"input_hash": "prior_hash"}}
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
            prior_manifest=prior,
        )

        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.prior_manifest is not None
        assert "manifest_plan" in plan.prior_manifest


def test_plan_graph_plugin_run_includes_settings() -> None:
    """Plan includes settings for each plugin."""
    plugin = _make_test_plugin("settings_plan")
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(default_severity="soft_fail"),
        )

        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert "settings_plan" in plan.settings_by_plugin
        settings = plan.settings_by_plugin["settings_plan"]
        assert settings.severity == "soft_fail"


def test_plan_graph_plugin_run_with_dependencies() -> None:
    """Plan orders plugins by dependencies."""
    plugin_a = _make_test_plugin("dep_a", provides=("capability_a",))
    plugin_b = _make_test_plugin("dep_b", depends_on=("dep_a",))
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin_a, plugin_b]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )

        plan = plan_graph_plugin_run(
            plugin_names=["dep_a", "dep_b"],
            context=context,
        )

        # dep_a should come before dep_b
        names = plan.ordered_names
        assert names.index("dep_a") < names.index("dep_b")


def test_plan_graph_plugin_run_with_telemetry() -> None:
    """Plan includes telemetry manager."""
    plugin = _make_test_plugin("telemetry_plan")
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )

        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        assert plan.telemetry is not None


def test_graph_plugin_execution_plan_dep_graph() -> None:
    """Plan includes dependency graph mapping."""
    plugin_a = _make_test_plugin("graph_a", provides=("cap_a",))
    plugin_b = _make_test_plugin("graph_b", depends_on=("graph_a",))
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin_a, plugin_b]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )

        plan = plan_graph_plugin_run(
            plugin_names=["graph_a", "graph_b"],
            context=context,
        )

        assert "graph_b" in plan.dep_graph
        assert "graph_a" in plan.dep_graph["graph_b"]


def test_graph_plugin_execution_plan_skipped_plugins() -> None:
    """Plan records skipped plugins."""
    plugin = _make_test_plugin("skip_test")
    snapshot = make_snapshot(repo="plan/repo", commit="abc")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )

        # Request unknown plugin - should be skipped
        plan = plan_graph_plugin_run(
            plugin_names=["skip_test", "nonexistent_plugin"],
            context=context,
        )

        skipped_names = [s.name for s in plan.skipped_plugins]
        assert "nonexistent_plugin" in skipped_names


def test_plan_coordinates_frozen() -> None:
    """PlanCoordinates is frozen (immutable)."""
    coords = PlanCoordinates(
        repo="test/repo",
        commit="abc123",
        scope=GraphRunScope(),
    )

    assert_cannot_setattr(coords, "repo", "other/repo")


def test_plan_coordinates_equality() -> None:
    """PlanCoordinates supports equality comparison."""
    coords1 = PlanCoordinates(
        repo="test/repo",
        commit="abc123",
        scope=GraphRunScope(paths=("src/",)),
    )

    coords2 = PlanCoordinates(
        repo="test/repo",
        commit="abc123",
        scope=GraphRunScope(paths=("src/",)),
    )

    assert coords1 == coords2
