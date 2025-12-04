"""Tests for secondary graph metrics plugins.

This module tests the plugin definitions, getters, and protocol compliance
for the secondary graph metrics plugins from
`codeintel.graphs.plugins.metrics.secondary`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from codeintel.graphs.core.protocol import GraphPluginProtocol
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.plugins.metrics.secondary import (
    cfg_metrics_plugin,
    config_graph_metrics_plugin,
    dfg_metrics_plugin,
    get_cfg_metrics_plugin,
    get_config_graph_metrics_plugin,
    get_dfg_metrics_plugin,
    get_graph_stats_plugin,
    get_subsystem_agreement_plugin,
    get_subsystem_graph_metrics_plugin,
    get_symbol_graph_metrics_functions_plugin,
    get_symbol_graph_metrics_modules_plugin,
    get_test_graph_metrics_plugin,
    graph_stats_plugin,
    subsystem_agreement_plugin,
    subsystem_graph_metrics_plugin,
    symbol_graph_metrics_functions_plugin,
    symbol_graph_metrics_modules_plugin,
    test_graph_metrics_plugin,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_PLUGIN_COUNT: Final[int] = 10
CFG_STAGE: Final[str] = "cfg"
DFG_STAGE: Final[str] = "dfg"
TEST_STAGE: Final[str] = "test"
SUBSYSTEM_STAGE: Final[str] = "subsystem"
SYMBOL_STAGE: Final[str] = "symbol"
CONFIG_STAGE: Final[str] = "config"
STATS_STAGE: Final[str] = "stats"


# ===========================================================================
# Plugin Instances Tests
# ===========================================================================


def test_cfg_metrics_plugin_protocol() -> None:
    """CFG metrics plugin implements GraphPluginProtocol."""
    assert isinstance(cfg_metrics_plugin, GraphPluginProtocol)


def test_dfg_metrics_plugin_protocol() -> None:
    """DFG metrics plugin implements GraphPluginProtocol."""
    assert isinstance(dfg_metrics_plugin, GraphPluginProtocol)


def test_test_graph_metrics_plugin_protocol() -> None:
    """Test graph metrics plugin implements GraphPluginProtocol."""
    assert isinstance(test_graph_metrics_plugin, GraphPluginProtocol)


def test_subsystem_graph_metrics_plugin_protocol() -> None:
    """Subsystem graph metrics plugin implements GraphPluginProtocol."""
    assert isinstance(subsystem_graph_metrics_plugin, GraphPluginProtocol)


def test_symbol_graph_metrics_modules_plugin_protocol() -> None:
    """Symbol graph metrics modules plugin implements GraphPluginProtocol."""
    assert isinstance(symbol_graph_metrics_modules_plugin, GraphPluginProtocol)


def test_symbol_graph_metrics_functions_plugin_protocol() -> None:
    """Symbol graph metrics functions plugin implements GraphPluginProtocol."""
    assert isinstance(symbol_graph_metrics_functions_plugin, GraphPluginProtocol)


def test_config_graph_metrics_plugin_protocol() -> None:
    """Config graph metrics plugin implements GraphPluginProtocol."""
    assert isinstance(config_graph_metrics_plugin, GraphPluginProtocol)


def test_subsystem_agreement_plugin_protocol() -> None:
    """Subsystem agreement plugin implements GraphPluginProtocol."""
    assert isinstance(subsystem_agreement_plugin, GraphPluginProtocol)


def test_graph_stats_plugin_protocol() -> None:
    """Graph stats plugin implements GraphPluginProtocol."""
    assert isinstance(graph_stats_plugin, GraphPluginProtocol)


# ===========================================================================
# Plugin Getter Tests
# ===========================================================================


def test_get_cfg_metrics_plugin() -> None:
    """get_cfg_metrics_plugin returns cfg_metrics_plugin."""
    result = get_cfg_metrics_plugin()
    assert result is cfg_metrics_plugin


def test_get_dfg_metrics_plugin() -> None:
    """get_dfg_metrics_plugin returns dfg_metrics_plugin."""
    result = get_dfg_metrics_plugin()
    assert result is dfg_metrics_plugin


def test_get_test_graph_metrics_plugin() -> None:
    """get_test_graph_metrics_plugin returns test_graph_metrics_plugin."""
    result = get_test_graph_metrics_plugin()
    assert result is test_graph_metrics_plugin


def test_get_subsystem_graph_metrics_plugin() -> None:
    """get_subsystem_graph_metrics_plugin returns subsystem_graph_metrics_plugin."""
    result = get_subsystem_graph_metrics_plugin()
    assert result is subsystem_graph_metrics_plugin


def test_get_symbol_graph_metrics_modules_plugin() -> None:
    """get_symbol_graph_metrics_modules_plugin returns symbol_graph_metrics_modules_plugin."""
    result = get_symbol_graph_metrics_modules_plugin()
    assert result is symbol_graph_metrics_modules_plugin


def test_get_symbol_graph_metrics_functions_plugin() -> None:
    """get_symbol_graph_metrics_functions_plugin returns symbol_graph_metrics_functions_plugin."""
    result = get_symbol_graph_metrics_functions_plugin()
    assert result is symbol_graph_metrics_functions_plugin


def test_get_config_graph_metrics_plugin() -> None:
    """get_config_graph_metrics_plugin returns config_graph_metrics_plugin."""
    result = get_config_graph_metrics_plugin()
    assert result is config_graph_metrics_plugin


def test_get_subsystem_agreement_plugin() -> None:
    """get_subsystem_agreement_plugin returns subsystem_agreement_plugin."""
    result = get_subsystem_agreement_plugin()
    assert result is subsystem_agreement_plugin


def test_get_graph_stats_plugin() -> None:
    """get_graph_stats_plugin returns graph_stats_plugin."""
    result = get_graph_stats_plugin()
    assert result is graph_stats_plugin


# ===========================================================================
# Plugin Metadata Tests - Names
# ===========================================================================


def test_cfg_metrics_plugin_name() -> None:
    """CFG metrics plugin has correct name."""
    assert cfg_metrics_plugin.metadata.name == "cfg_metrics"


def test_dfg_metrics_plugin_name() -> None:
    """DFG metrics plugin has correct name."""
    assert dfg_metrics_plugin.metadata.name == "dfg_metrics"


def test_test_graph_metrics_plugin_name() -> None:
    """Test graph metrics plugin has correct name."""
    assert test_graph_metrics_plugin.metadata.name == "test_graph_metrics"


def test_subsystem_graph_metrics_plugin_name() -> None:
    """Subsystem graph metrics plugin has correct name."""
    assert subsystem_graph_metrics_plugin.metadata.name == "subsystem_graph_metrics"


def test_symbol_graph_metrics_modules_plugin_name() -> None:
    """Symbol graph metrics modules plugin has correct name."""
    assert symbol_graph_metrics_modules_plugin.metadata.name == "symbol_graph_metrics_modules"


def test_symbol_graph_metrics_functions_plugin_name() -> None:
    """Symbol graph metrics functions plugin has correct name."""
    assert symbol_graph_metrics_functions_plugin.metadata.name == "symbol_graph_metrics_functions"


def test_config_graph_metrics_plugin_name() -> None:
    """Config graph metrics plugin has correct name."""
    assert config_graph_metrics_plugin.metadata.name == "config_graph_metrics"


def test_subsystem_agreement_plugin_name() -> None:
    """Subsystem agreement plugin has correct name."""
    assert subsystem_agreement_plugin.metadata.name == "subsystem_agreement"


def test_graph_stats_plugin_name() -> None:
    """Graph stats plugin has correct name."""
    assert graph_stats_plugin.metadata.name == "graph_stats"


# ===========================================================================
# Plugin Metadata Tests - Stages
# ===========================================================================


def test_cfg_metrics_plugin_stage() -> None:
    """CFG metrics plugin has cfg stage."""
    assert cfg_metrics_plugin.metadata.stage == CFG_STAGE


def test_dfg_metrics_plugin_stage() -> None:
    """DFG metrics plugin has dfg stage."""
    assert dfg_metrics_plugin.metadata.stage == DFG_STAGE


def test_test_graph_metrics_plugin_stage() -> None:
    """Test graph metrics plugin has test stage."""
    assert test_graph_metrics_plugin.metadata.stage == TEST_STAGE


def test_subsystem_graph_metrics_plugin_stage() -> None:
    """Subsystem graph metrics plugin has subsystem stage."""
    assert subsystem_graph_metrics_plugin.metadata.stage == SUBSYSTEM_STAGE


def test_symbol_graph_metrics_modules_plugin_stage() -> None:
    """Symbol graph metrics modules plugin has symbol stage."""
    assert symbol_graph_metrics_modules_plugin.metadata.stage == SYMBOL_STAGE


def test_symbol_graph_metrics_functions_plugin_stage() -> None:
    """Symbol graph metrics functions plugin has symbol stage."""
    assert symbol_graph_metrics_functions_plugin.metadata.stage == SYMBOL_STAGE


def test_config_graph_metrics_plugin_stage() -> None:
    """Config graph metrics plugin has config stage."""
    assert config_graph_metrics_plugin.metadata.stage == CONFIG_STAGE


def test_subsystem_agreement_plugin_stage() -> None:
    """Subsystem agreement plugin has subsystem stage."""
    assert subsystem_agreement_plugin.metadata.stage == SUBSYSTEM_STAGE


def test_graph_stats_plugin_stage() -> None:
    """Graph stats plugin has stats stage."""
    assert graph_stats_plugin.metadata.stage == STATS_STAGE


# ===========================================================================
# Plugin Metadata Tests - Dependencies
# ===========================================================================


def test_cfg_metrics_depends_on_cfg_dfg_builder() -> None:
    """CFG metrics plugin depends on cfg_dfg_builder."""
    assert "cfg_dfg_builder" in cfg_metrics_plugin.metadata.depends_on


def test_dfg_metrics_depends_on_cfg_dfg_builder() -> None:
    """DFG metrics plugin depends on cfg_dfg_builder."""
    assert "cfg_dfg_builder" in dfg_metrics_plugin.metadata.depends_on


def test_test_graph_metrics_depends_on_callgraph_builder() -> None:
    """Test graph metrics plugin depends on callgraph_builder."""
    assert "callgraph_builder" in test_graph_metrics_plugin.metadata.depends_on


def test_subsystem_graph_metrics_depends_on_import_graph_builder() -> None:
    """Subsystem graph metrics plugin depends on import_graph_builder."""
    assert "import_graph_builder" in subsystem_graph_metrics_plugin.metadata.depends_on


def test_graph_stats_depends_on_multiple_builders() -> None:
    """Graph stats plugin depends on multiple builders."""
    deps = graph_stats_plugin.metadata.depends_on
    assert "callgraph_builder" in deps
    assert "import_graph_builder" in deps


# ===========================================================================
# Plugin Metadata Tests - Provides
# ===========================================================================


def test_cfg_metrics_provides() -> None:
    """CFG metrics plugin provides cfg_metrics."""
    assert "cfg_metrics" in cfg_metrics_plugin.metadata.provides


def test_dfg_metrics_provides() -> None:
    """DFG metrics plugin provides dfg_metrics."""
    assert "dfg_metrics" in dfg_metrics_plugin.metadata.provides


def test_test_graph_metrics_provides() -> None:
    """Test graph metrics plugin provides test_metrics."""
    assert "test_metrics" in test_graph_metrics_plugin.metadata.provides


def test_subsystem_graph_metrics_provides() -> None:
    """Subsystem graph metrics plugin provides subsystem_metrics."""
    assert "subsystem_metrics" in subsystem_graph_metrics_plugin.metadata.provides


def test_graph_stats_provides() -> None:
    """Graph stats plugin provides graph_stats."""
    assert "graph_stats" in graph_stats_plugin.metadata.provides


# ===========================================================================
# Plugin Metadata Tests - Required Graph Kinds
# ===========================================================================


def test_cfg_metrics_requires_cfg_graph() -> None:
    """CFG metrics plugin requires CFG_GRAPH."""
    assert GraphKind.CFG_GRAPH in cfg_metrics_plugin.metadata.requires_graph_kinds


def test_dfg_metrics_requires_cfg_graph() -> None:
    """DFG metrics plugin requires CFG_GRAPH (DFG is part of CFG data)."""
    assert GraphKind.CFG_GRAPH in dfg_metrics_plugin.metadata.requires_graph_kinds


def test_test_graph_metrics_requires_call_graph() -> None:
    """Test graph metrics plugin requires CALL_GRAPH."""
    assert GraphKind.CALL_GRAPH in test_graph_metrics_plugin.metadata.requires_graph_kinds


def test_subsystem_graph_metrics_requires_import_graph() -> None:
    """Subsystem graph metrics plugin requires IMPORT_GRAPH."""
    assert GraphKind.IMPORT_GRAPH in subsystem_graph_metrics_plugin.metadata.requires_graph_kinds


def test_graph_stats_requires_multiple_graphs() -> None:
    """Graph stats plugin requires both CALL_GRAPH and IMPORT_GRAPH."""
    kinds = graph_stats_plugin.metadata.requires_graph_kinds
    assert GraphKind.CALL_GRAPH in kinds
    assert GraphKind.IMPORT_GRAPH in kinds


# ===========================================================================
# Plugin Metadata Tests - Produces Tables
# ===========================================================================


def test_test_graph_metrics_produces_tables() -> None:
    """Test graph metrics plugin produces expected tables."""
    tables = test_graph_metrics_plugin.metadata.produces_tables
    assert "analytics.test_graph_metrics_tests" in tables
    assert "analytics.test_graph_metrics_functions" in tables


def test_subsystem_graph_metrics_produces_tables() -> None:
    """Subsystem graph metrics plugin produces expected table."""
    tables = subsystem_graph_metrics_plugin.metadata.produces_tables
    assert "analytics.subsystem_graph_metrics" in tables


def test_symbol_graph_metrics_modules_produces_tables() -> None:
    """Symbol graph metrics modules plugin produces expected table."""
    tables = symbol_graph_metrics_modules_plugin.metadata.produces_tables
    assert "analytics.symbol_graph_metrics_modules" in tables


def test_symbol_graph_metrics_functions_produces_tables() -> None:
    """Symbol graph metrics functions plugin produces expected table."""
    tables = symbol_graph_metrics_functions_plugin.metadata.produces_tables
    assert "analytics.symbol_graph_metrics_functions" in tables


def test_config_graph_metrics_produces_tables() -> None:
    """Config graph metrics plugin produces expected tables."""
    tables = config_graph_metrics_plugin.metadata.produces_tables
    assert "analytics.config_graph_metrics_keys" in tables
    assert "analytics.config_graph_metrics_modules" in tables


def test_subsystem_agreement_produces_tables() -> None:
    """Subsystem agreement plugin produces expected table."""
    tables = subsystem_agreement_plugin.metadata.produces_tables
    assert "analytics.subsystem_agreement" in tables


def test_graph_stats_produces_tables() -> None:
    """Graph stats plugin produces expected table."""
    tables = graph_stats_plugin.metadata.produces_tables
    assert "analytics.graph_stats" in tables


# ===========================================================================
# Plugin Kind Tests
# ===========================================================================


def test_all_plugins_are_metric_kind() -> None:
    """All secondary plugins are metric kind."""
    all_plugins: tuple[GraphPluginProtocol, ...] = (
        cfg_metrics_plugin,
        dfg_metrics_plugin,
        test_graph_metrics_plugin,
        subsystem_graph_metrics_plugin,
        symbol_graph_metrics_modules_plugin,
        symbol_graph_metrics_functions_plugin,
        config_graph_metrics_plugin,
        subsystem_agreement_plugin,
        graph_stats_plugin,
    )
    for plugin in all_plugins:
        assert plugin.metadata.kind == "metric"


# ===========================================================================
# Parametrized Plugin Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("getter", "expected_name"),
    [
        (get_cfg_metrics_plugin, "cfg_metrics"),
        (get_dfg_metrics_plugin, "dfg_metrics"),
        (get_test_graph_metrics_plugin, "test_graph_metrics"),
        (get_subsystem_graph_metrics_plugin, "subsystem_graph_metrics"),
        (get_symbol_graph_metrics_modules_plugin, "symbol_graph_metrics_modules"),
        (get_symbol_graph_metrics_functions_plugin, "symbol_graph_metrics_functions"),
        (get_config_graph_metrics_plugin, "config_graph_metrics"),
        (get_subsystem_agreement_plugin, "subsystem_agreement"),
        (get_graph_stats_plugin, "graph_stats"),
    ],
)
def test_getter_returns_correct_plugin_name(
    getter: Callable[[], GraphPluginProtocol], expected_name: str
) -> None:
    """Getter returns plugin with correct name."""
    plugin = getter()
    assert plugin.metadata.name == expected_name


@pytest.mark.parametrize(
    ("plugin_name", "plugin"),
    [
        ("cfg_metrics", cfg_metrics_plugin),
        ("dfg_metrics", dfg_metrics_plugin),
        ("test_graph_metrics", test_graph_metrics_plugin),
        ("subsystem_graph_metrics", subsystem_graph_metrics_plugin),
        ("symbol_graph_metrics_modules", symbol_graph_metrics_modules_plugin),
        ("symbol_graph_metrics_functions", symbol_graph_metrics_functions_plugin),
        ("config_graph_metrics", config_graph_metrics_plugin),
        ("subsystem_agreement", subsystem_agreement_plugin),
        ("graph_stats", graph_stats_plugin),
    ],
)
def test_plugin_has_description(plugin_name: str, plugin: GraphPluginProtocol) -> None:
    """Each plugin has a non-empty description."""
    # Name is used in message only
    _ = plugin_name
    assert plugin.metadata.description


@pytest.mark.parametrize(
    ("plugin_name", "plugin"),
    [
        ("cfg_metrics", cfg_metrics_plugin),
        ("dfg_metrics", dfg_metrics_plugin),
        ("test_graph_metrics", test_graph_metrics_plugin),
        ("subsystem_graph_metrics", subsystem_graph_metrics_plugin),
        ("symbol_graph_metrics_modules", symbol_graph_metrics_modules_plugin),
        ("symbol_graph_metrics_functions", symbol_graph_metrics_functions_plugin),
        ("config_graph_metrics", config_graph_metrics_plugin),
        ("subsystem_agreement", subsystem_agreement_plugin),
        ("graph_stats", graph_stats_plugin),
    ],
)
def test_plugin_has_dependencies(plugin_name: str, plugin: GraphPluginProtocol) -> None:
    """Each plugin has at least one dependency."""
    _ = plugin_name  # Used for error reporting
    assert len(plugin.metadata.depends_on) >= 1


@pytest.mark.parametrize(
    ("plugin_name", "plugin"),
    [
        ("cfg_metrics", cfg_metrics_plugin),
        ("dfg_metrics", dfg_metrics_plugin),
        ("test_graph_metrics", test_graph_metrics_plugin),
        ("subsystem_graph_metrics", subsystem_graph_metrics_plugin),
        ("symbol_graph_metrics_modules", symbol_graph_metrics_modules_plugin),
        ("symbol_graph_metrics_functions", symbol_graph_metrics_functions_plugin),
        ("config_graph_metrics", config_graph_metrics_plugin),
        ("subsystem_agreement", subsystem_agreement_plugin),
        ("graph_stats", graph_stats_plugin),
    ],
)
def test_plugin_provides_capabilities(plugin_name: str, plugin: GraphPluginProtocol) -> None:
    """Each plugin provides at least one capability."""
    _ = plugin_name
    assert len(plugin.metadata.provides) >= 1

