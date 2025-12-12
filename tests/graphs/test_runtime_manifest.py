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
from codeintel.graphs.runtime.manifest import (
    GraphPluginManifest,
    InputHashPayload,
    ManifestState,
    compute_input_hash,
    compute_options_hash,
    is_unchanged,
)
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_not_equal,
    expect_true,
)
from tests._helpers.fakes.graph_plugins import make_graph_plugin

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


EXPECTED_HASH_LENGTH: Final = 16


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

    expect_not_equal(hash_no_paths, hash_with_paths)
    expect_length(hash_no_paths, EXPECTED_HASH_LENGTH)
    expect_length(hash_with_paths, EXPECTED_HASH_LENGTH)


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

    expect_not_equal(hash_no_modules, hash_with_modules)


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

    expect_equal(hash1, hash2)


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

    expect_not_equal(compute_input_hash(payload1), compute_input_hash(payload2))


def test_compute_options_hash_with_dict_options() -> None:
    """Options hash computed for dictionary options."""
    plugin = make_graph_plugin("opt_plugin")
    options = {"key": "value", "count": 42, "enabled": True}

    hash_val = compute_options_hash(plugin, options)

    expect_is_not_none(hash_val, message="Expected hash value for serializable options")
    if hash_val is None:
        return

    expect_length(hash_val, EXPECTED_HASH_LENGTH)


def test_compute_options_hash_none_returns_none() -> None:
    """Options hash is None when options are None."""
    plugin = make_graph_plugin("none_opt_plugin")

    hash_val = compute_options_hash(plugin, None)

    expect_is_none(hash_val)


def test_compute_options_hash_deterministic() -> None:
    """Options hash is deterministic for same options."""
    plugin = make_graph_plugin("det_opt_plugin")
    options = {"alpha": 1, "beta": 2}

    hash1 = compute_options_hash(plugin, options)
    hash2 = compute_options_hash(plugin, options)

    expect_equal(hash1, hash2)


def test_compute_options_hash_serialization_failure_returns_none() -> None:
    """Non-serializable options return None without raising."""
    plugin = make_graph_plugin("fail_serial_plugin")

    class NonSerializable:
        def __str__(self) -> str:
            msg = "no string representation"
            raise ValueError(msg)

    options = {"obj": NonSerializable()}

    hash_val = compute_options_hash(plugin, options)

    expect_is_none(hash_val)


def test_compute_options_hash_varies_with_options() -> None:
    """Options hash changes when options change."""
    plugin = make_graph_plugin("vary_opt_plugin")

    hash1 = compute_options_hash(plugin, {"value": 1})
    hash2 = compute_options_hash(plugin, {"value": 2})

    expect_not_equal(hash1, hash2)


def test_is_unchanged_when_hashes_match(graph_gateway: StorageGateway) -> None:
    """Return True when input and options hashes match."""
    prior_manifest = {
        "test_plugin": {
            "input_hash": "abc123",
            "options_hash": "opt456",
        }
    }

    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="abc123",
        options_hash="opt456",
    )

    expect_true(is_unchanged(prior_manifest, state))


def test_is_unchanged_when_input_hash_differs(graph_gateway: StorageGateway) -> None:
    """Return False when input hashes differ."""
    prior_manifest = {
        "test_plugin": {
            "input_hash": "old_hash",
            "options_hash": "opt456",
        }
    }

    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="new_hash",
        options_hash="opt456",
    )

    expect_true(not is_unchanged(prior_manifest, state))


def test_is_unchanged_when_options_hash_differs(graph_gateway: StorageGateway) -> None:
    """Return False when options hashes differ."""
    prior_manifest = {
        "test_plugin": {
            "input_hash": "abc123",
            "options_hash": "old_opt",
        }
    }

    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="abc123",
        options_hash="new_opt",
    )

    expect_true(not is_unchanged(prior_manifest, state))


def test_is_unchanged_missing_input_hash_returns_false(graph_gateway: StorageGateway) -> None:
    """Return False when current state has None input hash."""
    prior_manifest = {
        "test_plugin": {
            "input_hash": "abc123",
            "options_hash": None,
        }
    }

    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash=None,
        options_hash=None,
    )

    expect_true(not is_unchanged(prior_manifest, state))


def test_is_unchanged_missing_prior_input_hash_returns_false(
    graph_gateway: StorageGateway,
) -> None:
    """Return False when prior manifest has None input hash."""
    prior_manifest = {
        "test_plugin": {
            "input_hash": None,
            "options_hash": None,
        }
    }

    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="abc123",
        options_hash=None,
    )

    expect_true(not is_unchanged(prior_manifest, state))


def test_is_unchanged_missing_plugin_in_prior_returns_false(
    graph_gateway: StorageGateway,
) -> None:
    """Return False when plugin not in prior manifest."""
    prior_manifest = {
        "other_plugin": {
            "input_hash": "abc123",
            "options_hash": None,
        }
    }

    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="abc123",
        options_hash=None,
    )

    expect_true(not is_unchanged(prior_manifest, state))


def test_is_unchanged_no_prior_manifest(graph_gateway: StorageGateway) -> None:
    """Return False when prior manifest is None."""
    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=(),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="abc123",
        options_hash=None,
    )

    expect_true(not is_unchanged(None, state))


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

    expect_in("plugin_a", entries)
    expect_in("plugin_b", entries)

    entry_a = entries["plugin_a"]
    expect_equal(entry_a.get("input_hash"), "inp_a")
    expect_equal(entry_a.get("options_hash"), "opt_a")
    expect_equal(entry_a.get("version_hash"), "v1")
    expect_equal(entry_a.get("row_counts"), {"table1": 100, "table2": 50})
    expect_in("recorded_at", entry_a)

    entry_b = entries["plugin_b"]
    expect_equal(entry_b.get("input_hash"), "inp_b")
    expect_is_none(entry_b.get("options_hash"))
    expect_is_none(entry_b.get("row_counts"))


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

    expect_length(entries, 1)
    expect_equal(entries["plugin"]["input_hash"], "new_hash")
    expect_equal(entries["plugin"]["row_counts"], {"t": 20})


def test_graph_plugin_manifest_empty() -> None:
    """Empty manifest returns empty dict."""
    manifest = GraphPluginManifest()

    expect_equal(manifest.to_dict(), {})


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

    expect_equal(payload1, payload2)


def test_manifest_state_frozen(graph_gateway: StorageGateway) -> None:
    """ManifestState is frozen (immutable)."""
    state = ManifestState(
        plugin_name="test_plugin",
        row_count_tables=("table1",),
        gateway=graph_gateway,
        repo="demo/repo",
        commit="abc123",
        input_hash="inp",
        options_hash="opt",
    )

    assert_cannot_setattr(state, "plugin_name", "other")
