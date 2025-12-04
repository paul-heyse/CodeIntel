"""Extended tests for graph runtime manifest module.

This module provides additional test coverage for the manifest module,
focusing on specific paths not covered by test_runtime.py:

- Hash computation with scope paths and modules
- Options hash serialization failure handling
- Manifest state and unchanged detection edge cases
- GraphPluginManifest record and retrieval
- InputHashPayload immutability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginMetadata,
    GraphPluginProtocol,
)
from codeintel.graphs.core.result import PluginResult
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
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.assertions import assert_cannot_setattr
from tests._helpers.gateway import open_ingestion_gateway_with_macros

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Constants
EXPECTED_HASH_LENGTH: Final = 16
CUSTOM_TIMEOUT_MS: Final = 5000


# Test Helpers


def _make_test_plugin(name: str) -> GraphPluginProtocol:
    """Create a minimal test plugin.

    Parameters
    ----------
    name
        Plugin name.

    Returns
    -------
    GraphPluginProtocol
        Test plugin instance.
    """

    def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
        return PluginResult.ok()

    metadata = GraphPluginMetadata(
        name=name,
        description=f"Test plugin {name}",
        kind="builder",
        stage="goid",
    )

    return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


def _make_gateway() -> StorageGateway:
    """Create a gateway for manifest state tests.

    Returns
    -------
    StorageGateway
        Configured gateway.
    """
    gateway = open_ingestion_gateway_with_macros(
        apply_schema=True, ensure_views=True, validate_schema=True
    )
    apply_all_schemas(gateway.con)
    return gateway


def test_compute_input_hash_scope_paths_included() -> None:
    """Scope paths affect the computed input hash."""
    payload_no_paths = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(paths=(), modules=()),
        options_hash=None,
    )

    payload_with_paths = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(paths=("src/", "lib/"), modules=()),
        options_hash=None,
    )

    hash_no_paths = compute_input_hash(payload_no_paths)
    hash_with_paths = compute_input_hash(payload_with_paths)

    assert hash_no_paths != hash_with_paths
    assert len(hash_no_paths) == EXPECTED_HASH_LENGTH
    assert len(hash_with_paths) == EXPECTED_HASH_LENGTH


def test_compute_input_hash_scope_modules_included() -> None:
    """Scope modules affect the computed input hash."""
    payload_no_modules = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(paths=(), modules=()),
        options_hash=None,
    )

    payload_with_modules = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(paths=(), modules=("mypackage.core", "mypackage.utils")),
        options_hash=None,
    )

    hash_no_modules = compute_input_hash(payload_no_modules)
    hash_with_modules = compute_input_hash(payload_with_modules)

    assert hash_no_modules != hash_with_modules


def test_compute_input_hash_deterministic() -> None:
    """Input hash is deterministic for identical inputs."""
    payload = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(paths=("src/",), modules=("mod",)),
        options_hash="opt_hash",
    )

    hash1 = compute_input_hash(payload)
    hash2 = compute_input_hash(payload)

    assert hash1 == hash2


def test_compute_input_hash_varies_with_commit() -> None:
    """Input hash changes when commit changes."""
    payload1 = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash=None,
    )

    payload2 = InputHashPayload(
        repo="test/repo",
        commit="def456",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash=None,
    )

    assert compute_input_hash(payload1) != compute_input_hash(payload2)


def test_compute_options_hash_with_dict_options() -> None:
    """Options hash computed for dictionary options."""
    plugin = _make_test_plugin("opt_plugin")
    options = {"key": "value", "count": 42, "enabled": True}

    hash_val = compute_options_hash(plugin, options)

    assert hash_val is not None
    assert len(hash_val) == EXPECTED_HASH_LENGTH


def test_compute_options_hash_none_returns_none() -> None:
    """Options hash is None when options are None."""
    plugin = _make_test_plugin("none_opt_plugin")

    hash_val = compute_options_hash(plugin, None)

    assert hash_val is None


def test_compute_options_hash_deterministic() -> None:
    """Options hash is deterministic for same options."""
    plugin = _make_test_plugin("det_opt_plugin")
    options = {"alpha": 1, "beta": 2}

    hash1 = compute_options_hash(plugin, options)
    hash2 = compute_options_hash(plugin, options)

    assert hash1 == hash2


def test_compute_options_hash_serialization_failure_returns_none() -> None:
    """Non-serializable options return None without raising."""
    plugin = _make_test_plugin("fail_serial_plugin")

    class NonSerializable:
        def __str__(self) -> str:
            msg = "no string representation"
            raise ValueError(msg)

    options = {"obj": NonSerializable()}

    hash_val = compute_options_hash(plugin, options)

    assert hash_val is None


def test_compute_options_hash_varies_with_options() -> None:
    """Options hash changes when options change."""
    plugin = _make_test_plugin("vary_opt_plugin")

    hash1 = compute_options_hash(plugin, {"value": 1})
    hash2 = compute_options_hash(plugin, {"value": 2})

    assert hash1 != hash2


def test_is_unchanged_when_hashes_match() -> None:
    """Return True when input and options hashes match."""
    gateway = _make_gateway()
    try:
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

        assert is_unchanged(prior_manifest, state)
    finally:
        gateway.close()


def test_is_unchanged_when_input_hash_differs() -> None:
    """Return False when input hashes differ."""
    gateway = _make_gateway()
    try:
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
            input_hash="new_hash",
            options_hash="opt456",
        )

        assert not is_unchanged(prior_manifest, state)
    finally:
        gateway.close()


def test_is_unchanged_when_options_hash_differs() -> None:
    """Return False when options hashes differ."""
    gateway = _make_gateway()
    try:
        prior_manifest = {
            "test_plugin": {
                "input_hash": "abc123",
                "options_hash": "old_opt",
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash="new_opt",
        )

        assert not is_unchanged(prior_manifest, state)
    finally:
        gateway.close()


def test_is_unchanged_missing_input_hash_returns_false() -> None:
    """Return False when current state has None input hash."""
    gateway = _make_gateway()
    try:
        prior_manifest = {
            "test_plugin": {
                "input_hash": "abc123",
                "options_hash": None,
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash=None,
            options_hash=None,
        )

        assert not is_unchanged(prior_manifest, state)
    finally:
        gateway.close()


def test_is_unchanged_missing_prior_input_hash_returns_false() -> None:
    """Return False when prior manifest has None input hash."""
    gateway = _make_gateway()
    try:
        prior_manifest = {
            "test_plugin": {
                "input_hash": None,
                "options_hash": None,
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash=None,
        )

        assert not is_unchanged(prior_manifest, state)
    finally:
        gateway.close()


def test_is_unchanged_missing_plugin_in_prior_returns_false() -> None:
    """Return False when plugin not in prior manifest."""
    gateway = _make_gateway()
    try:
        prior_manifest = {
            "other_plugin": {
                "input_hash": "abc123",
                "options_hash": None,
            }
        }

        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash=None,
        )

        assert not is_unchanged(prior_manifest, state)
    finally:
        gateway.close()


def test_is_unchanged_no_prior_manifest() -> None:
    """Return False when prior manifest is None."""
    gateway = _make_gateway()
    try:
        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=(),
            gateway=gateway,
            repo="demo/repo",
            commit="deadbeef",
            input_hash="abc123",
            options_hash=None,
        )

        assert not is_unchanged(None, state)
    finally:
        gateway.close()


def test_dry_run_record_creates_skipped_status() -> None:
    """Dry run record has skipped status with dry_run reason."""
    plugin = _make_test_plugin("dry_run_plugin")
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=1000,
        version_hash="v1",
        input_hash="inp123",
        options_hash="opt456",
        options={"key": "value"},
    )

    record = dry_run_record(plugin=plugin, params=params)

    assert record.status == "skipped"
    assert record.plugin_name == "dry_run_plugin"
    assert record.meta.get("skipped_reason") == "dry_run"
    assert record.meta.get("input_hash") == "inp123"
    assert record.meta.get("options_hash") == "opt456"
    assert record.duration_ms == 0.0
    assert record.attempts == 0


def test_skip_record_creates_skipped_status_with_reason() -> None:
    """Skip record has skipped status with custom reason."""
    plugin = _make_test_plugin("skip_plugin")
    params = RecordParams(
        severity="fatal",
        timeout_ms=None,
        version_hash="v2",
        input_hash="inp789",
        options_hash=None,
        options=None,
    )

    record = skip_record(plugin=plugin, params=params, reason="unchanged")

    assert record.status == "skipped"
    assert record.plugin_name == "skip_plugin"
    assert record.meta.get("skipped_reason") == "unchanged"
    assert record.meta.get("version_hash") == "v2"


def test_skip_record_with_custom_reason() -> None:
    """Skip record accepts custom skip reasons."""
    plugin = _make_test_plugin("custom_skip_plugin")
    params = RecordParams(
        severity="skip_on_error",
        timeout_ms=500,
        version_hash=None,
        input_hash=None,
        options_hash=None,
        options=None,
    )

    record = skip_record(plugin=plugin, params=params, reason="dependency_missing")

    assert record.meta.get("skipped_reason") == "dependency_missing"


def test_graph_plugin_manifest_record_and_to_dict() -> None:
    """Manifest records plugin data and returns as dict."""
    manifest = GraphPluginManifest()

    manifest.record(
        plugin_name="plugin_a",
        input_hash="inp_a",
        options_hash="opt_a",
        version_hash="v1",
        row_counts={"table1": 100, "table2": 50},
    )

    manifest.record(
        plugin_name="plugin_b",
        input_hash="inp_b",
        options_hash=None,
        version_hash="v2",
        row_counts=None,
    )

    entries = manifest.to_dict()

    assert "plugin_a" in entries
    assert "plugin_b" in entries

    entry_a = entries["plugin_a"]
    assert entry_a.get("input_hash") == "inp_a"
    assert entry_a.get("options_hash") == "opt_a"
    assert entry_a.get("version_hash") == "v1"
    assert entry_a.get("row_counts") == {"table1": 100, "table2": 50}
    assert "recorded_at" in entry_a

    entry_b = entries["plugin_b"]
    assert entry_b.get("input_hash") == "inp_b"
    assert entry_b.get("options_hash") is None
    assert entry_b.get("row_counts") is None


def test_graph_plugin_manifest_overwrite_existing() -> None:
    """Recording same plugin name overwrites previous entry."""
    manifest = GraphPluginManifest()

    manifest.record(
        plugin_name="plugin",
        input_hash="old_hash",
        options_hash=None,
        version_hash="v1",
        row_counts={"t": 10},
    )

    manifest.record(
        plugin_name="plugin",
        input_hash="new_hash",
        options_hash="new_opt",
        version_hash="v2",
        row_counts={"t": 20},
    )

    entries = manifest.to_dict()

    assert len(entries) == 1
    assert entries["plugin"]["input_hash"] == "new_hash"
    assert entries["plugin"]["row_counts"] == {"t": 20}


def test_graph_plugin_manifest_empty() -> None:
    """Empty manifest returns empty dict."""
    manifest = GraphPluginManifest()

    assert manifest.to_dict() == {}


def test_record_params_defaults() -> None:
    """RecordParams has correct default values."""
    params = RecordParams(
        severity="soft_fail",
        timeout_ms=None,
        version_hash=None,
        input_hash=None,
        options_hash=None,
        options=None,
    )

    assert params.requires_isolation is False
    assert params.isolation_kind is None
    assert params.policy_fail_fast is True


def test_record_params_custom_values() -> None:
    """RecordParams accepts custom values for all fields."""
    params = RecordParams(
        severity="fatal",
        timeout_ms=CUSTOM_TIMEOUT_MS,
        version_hash="v3",
        input_hash="inp",
        options_hash="opt",
        options={"config": True},
        requires_isolation=True,
        isolation_kind="process",
        policy_fail_fast=False,
    )

    assert params.severity == "fatal"
    assert params.timeout_ms == CUSTOM_TIMEOUT_MS
    assert params.requires_isolation is True
    assert params.isolation_kind == "process"
    assert params.policy_fail_fast is False


def test_input_hash_payload_frozen() -> None:
    """InputHashPayload is frozen (immutable)."""
    payload = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash=None,
    )

    assert_cannot_setattr(payload, "repo", "other/repo")


def test_input_hash_payload_equality() -> None:
    """InputHashPayload supports equality comparison."""
    payload1 = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash=None,
    )

    payload2 = InputHashPayload(
        repo="test/repo",
        commit="abc123",
        plugin_name="test_plugin",
        version_hash="v1",
        scope=GraphRunScope(),
        options_hash=None,
    )

    assert payload1 == payload2


def test_manifest_state_frozen() -> None:
    """ManifestState is frozen (immutable)."""
    gateway = _make_gateway()
    try:
        state = ManifestState(
            plugin_name="test_plugin",
            row_count_tables=("table1",),
            gateway=gateway,
            repo="demo/repo",
            commit="abc123",
            input_hash="inp",
            options_hash="opt",
        )

        assert_cannot_setattr(state, "plugin_name", "other")
    finally:
        gateway.close()
