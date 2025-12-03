"""Tests for graph runtime planning and manifest.

This module tests the execution planning infrastructure including
plan generation, dependency resolution, policy application, and
manifest-based skip detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.graphs.core.context import GraphExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.registry import get_graph_registry, register_graph_plugin
from codeintel.graphs.core.result import GraphPluginResult
from codeintel.graphs.runtime.manifest import (
    GraphPluginManifest,
    InputHashPayload,
    ManifestState,
    RecordParams,
    compute_input_hash,
    compute_options_hash,
    dry_run_record,
    is_unchanged,
    skip_record,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    GraphPluginRunOptions,
    plan_graph_plugin_run,
)
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.gateway import open_ingestion_gateway_with_macros

EXPECTED_PLUGIN_COUNT: Final = 3
EXPECTED_TIMEOUT_MS: Final = 5000
EXPECTED_HASH_LENGTH: Final = 16


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
    name: str,
    *,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
    severity: str = "fatal",
) -> GraphPluginProtocol:
    """Create a test plugin for planning tests.

    Parameters
    ----------
    name
        Plugin name.
    depends_on
        Plugin dependencies.
    provides
        Capabilities provided.
    severity
        Failure severity.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphExecutionContext) -> GraphPluginResult:
        return GraphPluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
        depends_on=depends_on,
        provides=provides,
        severity=severity,
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


# =============================================================================
# Planning Tests
# =============================================================================


def test_plan_graph_plugin_run_basic(tmp_path: Path) -> None:
    """Basic plan generation produces valid execution plan.

    Raises
    ------
    AssertionError
        If plan is not generated correctly.
    """
    plugin = _make_test_plugin("basic_plan_plugin")
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        if not plan.plan_id:
            msg = "Expected plan_id to be set"
            raise AssertionError(msg)
        if not plan.run_id:
            msg = "Expected run_id to be set"
            raise AssertionError(msg)
        if plan.repo != "demo/repo":
            msg = f"Expected repo 'demo/repo', got '{plan.repo}'"
            raise AssertionError(msg)
        if plan.commit != "deadbeef":
            msg = f"Expected commit 'deadbeef', got '{plan.commit}'"
            raise AssertionError(msg)
        if len(plan.plugins) != 1:
            msg = f"Expected 1 plugin, got {len(plan.plugins)}"
            raise AssertionError(msg)


def test_plan_with_dependencies() -> None:
    """Dependency resolution orders plugins correctly.

    Raises
    ------
    AssertionError
        If dependencies are not resolved correctly.
    """
    # Plugin B depends on Plugin A
    plugin_a = _make_test_plugin("dep_a", provides=("capability_a",))
    plugin_b = _make_test_plugin("dep_b", depends_on=("dep_a",))

    with _PluginRegistrar([plugin_a, plugin_b]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=["dep_a", "dep_b"],
            context=context,
        )

        # A should come before B in ordered names
        names = plan.ordered_names
        if "dep_a" not in names or "dep_b" not in names:
            msg = f"Expected both plugins in plan, got {names}"
            raise AssertionError(msg)
        if names.index("dep_a") >= names.index("dep_b"):
            msg = f"Expected dep_a before dep_b, got order {names}"
            raise AssertionError(msg)


def test_plan_with_custom_policy() -> None:
    """Custom policy settings are applied to plan.

    Raises
    ------
    AssertionError
        If policy is not applied correctly.
    """
    plugin = _make_test_plugin("policy_plugin")

    with _PluginRegistrar([plugin]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(
                default_severity="soft_fail",
                fail_fast=False,
                timeouts_ms={"policy_plugin": EXPECTED_TIMEOUT_MS},
                retries={"policy_plugin": GraphPluginRetryPolicy(max_attempts=3)},
            ),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        settings = plan.settings_by_plugin.get(plugin.metadata.name)
        if settings is None:
            msg = "Expected settings for plugin"
            raise AssertionError(msg)
        if settings.severity != "soft_fail":
            msg = f"Expected severity 'soft_fail', got '{settings.severity}'"
            raise AssertionError(msg)
        if settings.timeout_ms != EXPECTED_TIMEOUT_MS:
            msg = f"Expected timeout {EXPECTED_TIMEOUT_MS}, got {settings.timeout_ms}"
            raise AssertionError(msg)
        if settings.retry_cfg.max_attempts != EXPECTED_PLUGIN_COUNT:
            msg = f"Expected 3 max_attempts, got {settings.retry_cfg.max_attempts}"
            raise AssertionError(msg)


def test_plugin_execution_settings_hashes() -> None:
    """Execution settings include computed hashes.

    Raises
    ------
    AssertionError
        If hashes are not computed correctly.
    """
    plugin = _make_test_plugin("hash_plugin")

    with _PluginRegistrar([plugin]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        context = GraphPlanContext(
            runtime_snapshot=snapshot,
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        settings = plan.settings_by_plugin.get(plugin.metadata.name)
        if settings is None:
            msg = "Expected settings for plugin"
            raise AssertionError(msg)
        if not settings.input_hash:
            msg = "Expected input_hash to be computed"
            raise AssertionError(msg)
        # Version hash may be None if not set on metadata
        # Options hash may be None if no options


def test_plan_with_explicit_target() -> None:
    """Plan can use explicit target tuple instead of snapshot.

    Raises
    ------
    AssertionError
        If target is not used correctly.
    """
    plugin = _make_test_plugin("target_plugin")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            target=("explicit/repo", "explicit_commit"),
            policy=GraphPluginPolicy(),
        )
        plan = plan_graph_plugin_run(
            plugin_names=[plugin.metadata.name],
            context=context,
        )

        if plan.repo != "explicit/repo":
            msg = f"Expected repo 'explicit/repo', got '{plan.repo}'"
            raise AssertionError(msg)
        if plan.commit != "explicit_commit":
            msg = f"Expected commit 'explicit_commit', got '{plan.commit}'"
            raise AssertionError(msg)


def test_plan_missing_target_raises() -> None:
    """Plan without any target source raises ValueError."""
    plugin = _make_test_plugin("no_target_plugin")

    with _PluginRegistrar([plugin]):
        context = GraphPlanContext(
            policy=GraphPluginPolicy(),
        )
        with pytest.raises(ValueError, match="missing snapshot"):
            plan_graph_plugin_run(
                plugin_names=[plugin.metadata.name],
                context=context,
            )


def test_plan_with_run_options() -> None:
    """Runtime options override config settings.

    Raises
    ------
    AssertionError
        If runtime options are not applied.
    """
    plugin = _make_test_plugin("options_plugin")

    with _PluginRegistrar([plugin]):
        snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path())
        run_options = GraphPluginRunOptions(
            scope=GraphRunScope(paths=["src/"], modules=["mymodule"]),
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

        if plan.scope.paths != ["src/"]:
            msg = f"Expected scope paths ['src/'], got {plan.scope.paths}"
            raise AssertionError(msg)


# =============================================================================
# Manifest Tests
# =============================================================================


def test_compute_input_hash_deterministic() -> None:
    """Input hash is deterministic for same inputs.

    Raises
    ------
    AssertionError
        If hash is not deterministic.
    """
    payload = InputHashPayload(
        repo="demo/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash="opt123",
    )

    hash1 = compute_input_hash(payload)
    hash2 = compute_input_hash(payload)

    if hash1 != hash2:
        msg = f"Expected same hash, got '{hash1}' and '{hash2}'"
        raise AssertionError(msg)
    if len(hash1) != EXPECTED_HASH_LENGTH:
        msg = f"Expected {EXPECTED_HASH_LENGTH}-char hash, got {len(hash1)}"
        raise AssertionError(msg)


def test_compute_input_hash_varies_with_inputs() -> None:
    """Input hash varies when inputs change.

    Raises
    ------
    AssertionError
        If hash does not vary.
    """
    payload1 = InputHashPayload(
        repo="demo/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash="opt123",
    )
    payload2 = InputHashPayload(
        repo="demo/repo",
        commit="different_commit",  # Changed
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash="opt123",
    )

    hash1 = compute_input_hash(payload1)
    hash2 = compute_input_hash(payload2)

    if hash1 == hash2:
        msg = "Expected different hashes for different commits"
        raise AssertionError(msg)


def test_compute_options_hash_with_options() -> None:
    """Options hash is computed when options are provided.

    Raises
    ------
    AssertionError
        If options hash is not computed.
    """
    plugin = _make_test_plugin("opt_hash_plugin")
    options = {"key": "value", "number": 42}

    hash_val = compute_options_hash(plugin, options)

    if hash_val is None:
        msg = "Expected options hash to be computed"
        raise AssertionError(msg)
    if len(hash_val) != EXPECTED_HASH_LENGTH:
        msg = f"Expected {EXPECTED_HASH_LENGTH}-char hash, got {len(hash_val)}"
        raise AssertionError(msg)


def test_compute_options_hash_none_returns_none() -> None:
    """Options hash is None when options are None.

    Raises
    ------
    AssertionError
        If hash is not None.
    """
    plugin = _make_test_plugin("no_opt_plugin")

    hash_val = compute_options_hash(plugin, None)

    if hash_val is not None:
        msg = f"Expected None, got '{hash_val}'"
        raise AssertionError(msg)


def test_is_unchanged_when_hashes_match() -> None:
    """Skip detection returns True when hashes match.

    Raises
    ------
    AssertionError
        If unchanged is not detected.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    try:
        apply_all_schemas(gateway.con)

        prior_manifest = {
            "test_plugin": {
                "input_hash": "abc123",
                "options_hash": "opt456",
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash="opt456",
        )

        result = is_unchanged(prior_manifest, state)

        if not result:
            msg = "Expected is_unchanged to return True for matching hashes"
            raise AssertionError(msg)
    finally:
        gateway.close()


def test_is_unchanged_when_hashes_differ() -> None:
    """Skip detection returns False when hashes differ.

    Raises
    ------
    AssertionError
        If change is not detected.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    try:
        apply_all_schemas(gateway.con)

        prior_manifest = {
            "test_plugin": {
                "input_hash": "old_hash",
                "options_hash": "opt456",
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="new_hash",  # Different
            options_hash="opt456",
        )

        result = is_unchanged(prior_manifest, state)

        if result:
            msg = "Expected is_unchanged to return False for different hashes"
            raise AssertionError(msg)
    finally:
        gateway.close()


def test_is_unchanged_no_prior_manifest() -> None:
    """Skip detection returns False when no prior manifest.

    Raises
    ------
    AssertionError
        If False is not returned.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    try:
        apply_all_schemas(gateway.con)

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash="opt456",
        )

        result = is_unchanged(None, state)

        if result:
            msg = "Expected is_unchanged to return False for no prior manifest"
            raise AssertionError(msg)
    finally:
        gateway.close()


def test_dry_run_record() -> None:
    """Dry run mode produces skipped record with correct reason.

    Raises
    ------
    AssertionError
        If record is not correct.
    """
    plugin = _make_test_plugin("dry_run_plugin")
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=1000,
        version_hash="v1",
        input_hash="inp123",
        options_hash="opt456",
        options=None,
    )

    record = dry_run_record(plugin=plugin, params=params, run_id="test-run")

    if record.status != "skipped":
        msg = f"Expected status 'skipped', got '{record.status}'"
        raise AssertionError(msg)
    if record.meta.get("skipped_reason") != "dry_run":
        msg = f"Expected skipped_reason 'dry_run', got '{record.meta.get('skipped_reason')}'"
        raise AssertionError(msg)
    if record.name != "dry_run_plugin":
        msg = f"Expected name 'dry_run_plugin', got '{record.name}'"
        raise AssertionError(msg)


def test_skip_record() -> None:
    """Skip record includes reason and metadata.

    Raises
    ------
    AssertionError
        If record is not correct.
    """
    plugin = _make_test_plugin("skip_plugin")
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=1000,
        version_hash="v1",
        input_hash="inp123",
        options_hash="opt456",
        options=None,
    )

    record = skip_record(plugin=plugin, params=params, reason="unchanged", run_id="test-run")

    if record.status != "skipped":
        msg = f"Expected status 'skipped', got '{record.status}'"
        raise AssertionError(msg)
    if record.meta.get("skipped_reason") != "unchanged":
        msg = f"Expected skipped_reason 'unchanged', got '{record.meta.get('skipped_reason')}'"
        raise AssertionError(msg)


def test_graph_plugin_manifest_record() -> None:
    """Manifest records execution metadata correctly.

    Raises
    ------
    AssertionError
        If metadata is not recorded correctly.
    """
    manifest = GraphPluginManifest()

    manifest.record(
        plugin_name="test_plugin",
        input_hash="inp123",
        options_hash="opt456",
        version_hash="v1",
        row_counts={"table1": 100},
    )

    entries = manifest.to_dict()
    if "test_plugin" not in entries:
        msg = "Expected 'test_plugin' in manifest entries"
        raise AssertionError(msg)

    entry = entries["test_plugin"]
    if entry.get("input_hash") != "inp123":
        msg = f"Expected input_hash 'inp123', got '{entry.get('input_hash')}'"
        raise AssertionError(msg)
    if entry.get("row_counts") != {"table1": 100}:
        msg = f"Expected row_counts, got '{entry.get('row_counts')}'"
        raise AssertionError(msg)


def test_record_params_defaults() -> None:
    """RecordParams has correct defaults.

    Raises
    ------
    AssertionError
        If defaults are incorrect.
    """
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=None,
        version_hash=None,
        input_hash=None,
        options_hash=None,
        options=None,
    )

    if params.requires_isolation:
        msg = "Expected requires_isolation to default to False"
        raise AssertionError(msg)
    if params.policy_fail_fast is not True:
        msg = "Expected policy_fail_fast to default to True"
        raise AssertionError(msg)
