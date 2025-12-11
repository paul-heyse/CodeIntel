"""Tests for manifest infrastructure extensions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.core.plugins.execution.manifest import (
    ManifestQuery,
    build_upstream_state_from_records,
    compute_scope_id,
)
from codeintel.core.plugins.execution.manifest_store import InMemoryManifestStore
from codeintel.core.plugins.types.result import PluginExecutionRecord
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)

SCOPE_HASH_LENGTH = 16


class TestComputeScopeId:
    """Tests for compute_scope_id."""

    @staticmethod
    def test_none_returns_none() -> None:
        """Verify None paths returns None."""
        expect_equal(compute_scope_id(None), None)

    @staticmethod
    def test_empty_list_returns_none() -> None:
        """Verify empty list returns None."""
        expect_equal(compute_scope_id([]), None)

    @staticmethod
    def test_paths_return_hash() -> None:
        """Verify paths return a 16-char hash."""
        result = compute_scope_id(["src/", "lib/"])
        observed = expect_is_not_none(result)
        expect_equal(len(observed), SCOPE_HASH_LENGTH)

    @staticmethod
    def test_order_independent() -> None:
        """Verify hash is order-independent."""
        hash_one = compute_scope_id(["src/", "lib/"])
        hash_two = compute_scope_id(["lib/", "src/"])
        expect_equal(hash_one, hash_two)

    @staticmethod
    def test_different_paths_different_hash() -> None:
        """Verify different paths produce different hashes."""
        hash_one = compute_scope_id(["src/"])
        hash_two = compute_scope_id(["lib/"])
        expect_true(hash_one != hash_two)


@pytest.fixture
def manifest_store() -> InMemoryManifestStore:
    """Create a test manifest store.

    Returns
    -------
    InMemoryManifestStore
        Manifest store instance for tests.
    """
    return InMemoryManifestStore()


@pytest.fixture
def provider_lookup() -> dict[str, str]:
    """Create a test provider lookup.

    Returns
    -------
    dict[str, str]
        Capability-to-provider mapping.
    """
    return {
        "core.goids": "graphs.goid_builder",
        "graph.callgraph": "graphs.callgraph",
    }


class TestBuildUpstreamState:
    """Tests for build_upstream_state_from_records."""

    @staticmethod
    def test_empty_capabilities(
        manifest_store: InMemoryManifestStore, provider_lookup: dict[str, str]
    ) -> None:
        """Verify empty capabilities returns empty state."""
        state = build_upstream_state_from_records(
            required_capabilities=(),
            provider_lookup=provider_lookup,
            manifest_store=manifest_store,
            query=ManifestQuery(
                repo="owner/repo",
                commit="abc123",
                scope_id=None,
                variant=None,
            ),
        )
        expect_equal(state, {})

    @staticmethod
    def test_missing_provider(manifest_store: InMemoryManifestStore) -> None:
        """Verify missing provider is skipped."""
        state = build_upstream_state_from_records(
            required_capabilities=("unknown.capability",),
            provider_lookup={},
            manifest_store=manifest_store,
            query=ManifestQuery(
                repo="owner/repo",
                commit="abc123",
                scope_id=None,
                variant=None,
            ),
        )
        expect_equal(state, {})

    @staticmethod
    def test_with_records(
        manifest_store: InMemoryManifestStore, provider_lookup: dict[str, str]
    ) -> None:
        """Verify state is populated from records."""
        now = datetime.now(tz=UTC)
        record = PluginExecutionRecord(
            plugin_name="graphs.goid_builder",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": "hash123",
            },
        )
        manifest_store.append_record(record)

        state = build_upstream_state_from_records(
            required_capabilities=("core.goids",),
            provider_lookup=provider_lookup,
            manifest_store=manifest_store,
            query=ManifestQuery(
                repo="owner/repo",
                commit="abc123",
                scope_id=None,
                variant=None,
            ),
        )
        expect_equal(state, {"core.goids": "hash123"})
