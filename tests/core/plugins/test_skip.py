"""Tests for skip decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from codeintel.core.plugins.execution.manifest_store import InMemoryManifestStore
from codeintel.core.plugins.execution.options import EmptyConfigSource, PluginOptionsResolver
from codeintel.core.plugins.execution.run_context import (
    PluginRunContext,
    RunContextInputs,
    prepare_plugin_run,
)
from codeintel.core.plugins.execution.skip import (
    SkipDecision,
    create_skip_execution_record,
    should_skip_plugin,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.result import PluginExecutionRecord
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_is_not_none,
    expect_true,
)


@dataclass(frozen=True)
class TestOptions:
    """Test options."""

    value: int = 10


@pytest.fixture
def sample_metadata() -> CorePluginMetadata:
    """Create sample metadata.

    Returns
    -------
    CorePluginMetadata
        Sample metadata instance.
    """
    return CorePluginMetadata(
        name="test.plugin",
        version="1.0.0",
        description="Test plugin.",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        options_model=TestOptions,
    )


@pytest.fixture
def resolver() -> PluginOptionsResolver:
    """Create options resolver.

    Returns
    -------
    PluginOptionsResolver
        Resolver configured with an empty source.
    """
    return PluginOptionsResolver(EmptyConfigSource())


@pytest.fixture
def run_inputs() -> RunContextInputs:
    """Create run inputs for hashing.

    Returns
    -------
    RunContextInputs
        Inputs containing repo and commit identifiers.
    """
    return RunContextInputs(repo="owner/repo", commit="abc123", variant="fast")


@pytest.fixture
def run_context(
    sample_metadata: CorePluginMetadata,
    resolver: PluginOptionsResolver,
    run_inputs: RunContextInputs,
) -> PluginRunContext:
    """Create run context.

    Returns
    -------
    PluginRunContext
        Prepared run context with options and hashes.
    """
    return prepare_plugin_run(
        metadata=sample_metadata,
        resolver=resolver,
        upstream_state={"test.input": "upstream123"},
        inputs=run_inputs,
    )


@pytest.fixture
def manifest_store() -> InMemoryManifestStore:
    """Create manifest store.

    Returns
    -------
    InMemoryManifestStore
        In-memory manifest store for tests.
    """
    return InMemoryManifestStore()


class TestSkipDecision:
    """Tests for SkipDecision helpers."""

    @staticmethod
    def test_execute_factory() -> None:
        """Verify execute factory sets fields."""
        decision = SkipDecision.execute("run it")
        expect_false(decision.should_skip)
        expect_equal(decision.reason, "run it")

    @staticmethod
    def test_skip_factory(sample_metadata: CorePluginMetadata) -> None:
        """Verify skip factory sets fields."""
        now = datetime.now(tz=UTC)
        prior = PluginExecutionRecord(
            plugin_name=sample_metadata.name,
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=1.0,
        )
        decision = SkipDecision.skip("skip it", prior)
        expect_true(decision.should_skip)
        expect_equal(decision.reason, "skip it")
        expect_equal(decision.prior_record, prior)


class TestShouldSkipPlugin:
    """Tests for should_skip_plugin."""

    @staticmethod
    def test_force_always_executes(
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
        run_inputs: RunContextInputs,
    ) -> None:
        """Verify force=True always returns execute."""
        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            inputs=run_inputs,
            force=True,
        )
        expect_false(decision.should_skip)
        expect_true("force" in decision.reason)

    @staticmethod
    def test_no_prior_record_executes(
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
        run_inputs: RunContextInputs,
    ) -> None:
        """Verify no prior record returns execute."""
        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            inputs=run_inputs,
        )
        expect_false(decision.should_skip)
        expect_true("no prior" in decision.reason)

    @staticmethod
    def test_prior_failed_executes(
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
        run_inputs: RunContextInputs,
    ) -> None:
        """Verify prior failed record returns execute."""
        now = datetime.now(tz=UTC)
        failed_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="failed",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            error="Some error",
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": run_inputs.variant,
                "input_hash": run_context.input_hash,
            },
        )
        manifest_store.append_record(failed_record)

        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            inputs=run_inputs,
        )
        expect_false(decision.should_skip)
        expect_true("failed" in decision.reason)

    @staticmethod
    def test_same_hash_skips(
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
        run_inputs: RunContextInputs,
    ) -> None:
        """Verify matching input_hash returns skip."""
        now = datetime.now(tz=UTC)
        prior_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": run_inputs.variant,
                "input_hash": run_context.input_hash,
            },
        )
        manifest_store.append_record(prior_record)

        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            inputs=run_inputs,
        )
        expect_true(decision.should_skip)
        expect_true("unchanged" in decision.reason)
        expect_is_not_none(decision.prior_record)

    @staticmethod
    def test_different_hash_executes(
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
        run_inputs: RunContextInputs,
    ) -> None:
        """Verify different input_hash returns execute."""
        now = datetime.now(tz=UTC)
        prior_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": run_inputs.variant,
                "input_hash": "different_hash",
            },
        )
        manifest_store.append_record(prior_record)

        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            inputs=run_inputs,
        )
        expect_false(decision.should_skip)
        expect_true("changed" in decision.reason)


class TestCreateSkipExecutionRecord:
    """Tests for create_skip_execution_record."""

    @staticmethod
    def test_creates_skipped_record(
        run_context: PluginRunContext,
        run_inputs: RunContextInputs,
    ) -> None:
        """Verify skipped record is created correctly."""
        now = datetime.now(tz=UTC)
        prior_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={"input_hash": "prior123"},
        )

        record = create_skip_execution_record(
            run_context=run_context,
            prior_record=prior_record,
            inputs=run_inputs,
        )

        expect_equal(record.plugin_name, "test.plugin")
        expect_equal(record.status, "skipped")
        expect_equal(record.duration_ms, 0.0)
        expect_equal(record.meta["skip_reason"], "input_hash_unchanged")
