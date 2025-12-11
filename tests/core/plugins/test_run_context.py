"""Tests for PluginRunContext and prepare_plugin_run."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.plugins.execution.options import (
    EmptyConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.execution.run_context import (
    PluginRunContext,
    RunContextInputs,
    prepare_plugin_run,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from tests._helpers.assertions import (
    expect_equal,
    expect_is_not_none,
    expect_true,
)


@dataclass(frozen=True)
class TestOptions:
    """Test options model."""

    threshold: float = 0.5
    enabled: bool = True


def _sample_metadata() -> CorePluginMetadata:
    """Create sample metadata for testing.

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
        provides=("test.output",),
        requires=("test.input",),
        options_model=TestOptions,
    )


def _resolver() -> PluginOptionsResolver:
    """Create an options resolver with empty config.

    Returns
    -------
    PluginOptionsResolver
        Resolver that returns default options.
    """
    return PluginOptionsResolver(EmptyConfigSource())


class TestPreparePluginRun:
    """Tests for prepare_plugin_run."""

    @staticmethod
    def test_creates_context_with_defaults() -> None:
        """Verify context is created with default options."""
        ctx = prepare_plugin_run(
            metadata=_sample_metadata(),
            resolver=_resolver(),
            upstream_state={},
            inputs=RunContextInputs(repo="owner/repo", commit="abc123"),
        )
        expect_true(isinstance(ctx, PluginRunContext))
        expect_equal(ctx.metadata.name, "test.plugin")
        options = expect_is_not_none(ctx.options)
        expect_equal(options.threshold, 0.5)

    @staticmethod
    def test_computes_hashes() -> None:
        """Verify hashes are computed."""
        ctx = prepare_plugin_run(
            metadata=_sample_metadata(),
            resolver=_resolver(),
            upstream_state={"test.input": "upstream123"},
            inputs=RunContextInputs(repo="owner/repo", commit="abc123"),
        )
        expect_equal(len(ctx.options_hash), 16)
        expect_equal(len(ctx.input_hash), 16)

    @staticmethod
    def test_different_upstream_produces_different_hash() -> None:
        """Verify different upstream state produces different input hash."""
        ctx_one = prepare_plugin_run(
            metadata=_sample_metadata(),
            resolver=_resolver(),
            upstream_state={"test.input": "upstream1"},
            inputs=RunContextInputs(repo="owner/repo", commit="abc123"),
        )
        ctx_two = prepare_plugin_run(
            metadata=_sample_metadata(),
            resolver=_resolver(),
            upstream_state={"test.input": "upstream2"},
            inputs=RunContextInputs(repo="owner/repo", commit="abc123"),
        )
        expect_equal(ctx_one.options_hash, ctx_two.options_hash)
        expect_true(ctx_one.input_hash != ctx_two.input_hash)

    @staticmethod
    def test_plugin_name_property() -> None:
        """Verify plugin_name property."""
        ctx = prepare_plugin_run(
            metadata=_sample_metadata(),
            resolver=_resolver(),
            upstream_state={},
        )
        expect_equal(ctx.plugin_name, "test.plugin")

    @staticmethod
    def test_metadata_without_options_model() -> None:
        """Verify context works without options model."""
        metadata = CorePluginMetadata(
            name="test.no_options",
            version="1.0.0",
            description="Test.",
            domain=PluginDomain.ANALYTICS,
            kind="metric",
            options_model=None,
        )
        ctx = prepare_plugin_run(
            metadata=metadata,
            resolver=_resolver(),
            upstream_state={},
        )
        expect_equal(ctx.options, None)
        expect_equal(len(ctx.options_hash), 16)
