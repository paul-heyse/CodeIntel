"""Tests for GraphPluginMetadataConfig and create_graph_metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from codeintel.core.plugins.types.protocol import PluginResourceHints
from codeintel.graphs.core.protocol import (
    GraphPluginMetadataConfig,
    create_graph_metadata,
)
from codeintel.graphs.engine import GraphKind
from tests._helpers.assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from codeintel.graphs.core.protocol import (
        GraphPluginKind,
        GraphPluginStage,
    )


@pytest.mark.parametrize(
    ("name", "kind", "stage", "config", "expected_produces_graphs"),
    [
        (
            "callgraph",
            "builder",
            "edges",
            GraphPluginMetadataConfig(
                produces_graph_kinds=(GraphKind.CALL_GRAPH, GraphKind.SYMBOL),
            ),
            ("GraphKind.CALL_GRAPH", "GraphKind.SYMBOL"),
        ),
        (
            "goid_builder",
            "builder",
            "goid",
            GraphPluginMetadataConfig(),
            (),
        ),
    ],
)
def test_create_graph_metadata_success_paths(
    name: str,
    kind: GraphPluginKind,
    stage: GraphPluginStage,
    config: GraphPluginMetadataConfig,
    expected_produces_graphs: tuple[str, ...],
) -> None:
    """Validate successful metadata creation with and without graph kinds."""
    metadata = create_graph_metadata(
        name=name,
        description="desc",
        kind=kind,
        stage=stage,
        config=config,
    )

    expect_equal(metadata.produces_graphs, expected_produces_graphs)
    expect_equal(metadata.produces_graph_kinds, config.produces_graph_kinds)
    if stage != "goid":
        expect_equal(metadata.requires_graphs, tuple(str(g) for g in config.requires_graph_kinds))


@pytest.mark.parametrize(
    ("kind", "stage", "config", "error_match"),
    [
        ("metric", "core", GraphPluginMetadataConfig(), "requires_graph_kinds"),
        (
            "validation",
            "validation",
            GraphPluginMetadataConfig(produces_graph_kinds=(GraphKind.CALL_GRAPH,)),
            "must not declare produces_graph_kinds",
        ),
        (
            "builder",
            "edges",
            GraphPluginMetadataConfig(scope_aware=True),
            "supported_scopes",
        ),
    ],
)
def test_create_graph_metadata_errors(
    kind: GraphPluginKind,
    stage: GraphPluginStage,
    config: GraphPluginMetadataConfig,
    error_match: str,
) -> None:
    """Ensure invalid configurations raise ValueError with helpful messages."""
    with pytest.raises(ValueError, match=error_match):
        create_graph_metadata(
            name="invalid",
            description="invalid",
            kind=kind,
            stage=stage,
            config=config,
        )


def test_scope_normalization_when_not_scope_aware() -> None:
    """Scopes are cleared when scope awareness is disabled."""
    config = GraphPluginMetadataConfig(
        scope_aware=False,
        supported_scopes=("function",),
        produces_graph_kinds=(GraphKind.CALL_GRAPH,),
    )

    metadata = create_graph_metadata(
        name="callgraph",
        description="Ignore scopes when not scope aware",
        kind="builder",
        stage="edges",
        config=config,
    )

    expect_equal(metadata.supported_scopes, ())
    expect_true(metadata.scope_aware is False)


def test_resource_cache_and_options_passthrough() -> None:
    """Resource hints, cache hints, and options model/default propagate to metadata."""

    class _Options(BaseModel):
        threshold: int = 1

    config = GraphPluginMetadataConfig(
        produces_graph_kinds=(GraphKind.CALL_GRAPH,),
        requires_graph_kinds=(GraphKind.IMPORT_GRAPH,),
        resource_hints=PluginResourceHints(
            max_runtime_ms=111,
            max_memory_mb=222,
            cpu_intensive=True,
            io_intensive=True,
            requires_gpu=True,
            priority=3,
        ),
        cache_populates=("cache_pop",),
        cache_consumes=("cache_con",),
        options_model=_Options,
        options_default={"threshold": 7},
    )

    metadata = create_graph_metadata(
        name="callgraph",
        description="with resource hints",
        kind="builder",
        stage="edges",
        config=config,
    )

    expect_equal(metadata.cache_populates, ("cache_pop",))
    expect_equal(metadata.cache_consumes, ("cache_con",))
    expect_equal(metadata.options_model, _Options)
    expect_equal(metadata.options_default, {"threshold": 7})
    expect_true(metadata.resource_hints is not None)
    expect_equal(metadata.resource_hints, config.resource_hints)
