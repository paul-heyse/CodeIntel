"""Tests for core graph metrics plugins.

This module tests the core graph metrics plugins including their
configuration, metadata, and basic execution paths.
"""

from __future__ import annotations

from codeintel.graphs.core.protocol import GraphPluginProtocol
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.plugins.metrics.core import (
    core_graph_metrics_plugin,
    function_ext_metrics_plugin,
    get_core_graph_metrics_plugin,
    get_function_ext_metrics_plugin,
    get_module_ext_metrics_plugin,
    module_ext_metrics_plugin,
)


def test_core_graph_metrics_plugin_metadata() -> None:
    """Core graph metrics plugin has correct metadata.

    Raises
    ------
    AssertionError
        If metadata is incorrect.
    """
    plugin = core_graph_metrics_plugin
    meta = plugin.metadata

    if meta.name != "core_graph_metrics":
        msg = f"Expected name 'core_graph_metrics', got '{meta.name}'"
        raise AssertionError(msg)
    if meta.kind != "metric":
        msg = f"Expected kind 'metric', got '{meta.kind}'"
        raise AssertionError(msg)
    if meta.stage != "core":
        msg = f"Expected stage 'core', got '{meta.stage}'"
        raise AssertionError(msg)


def test_core_graph_metrics_plugin_dependencies() -> None:
    """Core graph metrics plugin has correct dependencies.

    Raises
    ------
    AssertionError
        If dependencies are incorrect.
    """
    plugin = core_graph_metrics_plugin
    meta = plugin.metadata

    if "callgraph_builder" not in meta.depends_on:
        msg = "Expected 'callgraph_builder' in dependencies"
        raise AssertionError(msg)
    if "import_graph_builder" not in meta.depends_on:
        msg = "Expected 'import_graph_builder' in dependencies"
        raise AssertionError(msg)


def test_core_graph_metrics_plugin_produces_tables() -> None:
    """Core graph metrics plugin declares correct output tables.

    Raises
    ------
    AssertionError
        If produces_tables is incorrect.
    """
    plugin = core_graph_metrics_plugin
    meta = plugin.metadata

    if "analytics.graph_metrics_functions" not in meta.produces_tables:
        msg = "Expected 'analytics.graph_metrics_functions' in produces_tables"
        raise AssertionError(msg)
    if "analytics.graph_metrics_modules" not in meta.produces_tables:
        msg = "Expected 'analytics.graph_metrics_modules' in produces_tables"
        raise AssertionError(msg)


def test_core_graph_metrics_plugin_requires_graphs() -> None:
    """Core graph metrics plugin requires correct graph kinds.

    Raises
    ------
    AssertionError
        If requires_graph_kinds is incorrect.
    """
    plugin = core_graph_metrics_plugin
    meta = plugin.metadata

    if GraphKind.CALL_GRAPH not in meta.requires_graph_kinds:
        msg = "Expected CALL_GRAPH in requires_graph_kinds"
        raise AssertionError(msg)
    if GraphKind.IMPORT_GRAPH not in meta.requires_graph_kinds:
        msg = "Expected IMPORT_GRAPH in requires_graph_kinds"
        raise AssertionError(msg)


def test_function_ext_metrics_plugin_metadata() -> None:
    """Function ext metrics plugin has correct metadata.

    Raises
    ------
    AssertionError
        If metadata is incorrect.
    """
    plugin = function_ext_metrics_plugin
    meta = plugin.metadata

    if meta.name != "graph_metrics_functions_ext":
        msg = f"Expected name 'graph_metrics_functions_ext', got '{meta.name}'"
        raise AssertionError(msg)
    if meta.kind != "metric":
        msg = f"Expected kind 'metric', got '{meta.kind}'"
        raise AssertionError(msg)
    if "callgraph_builder" not in meta.depends_on:
        msg = "Expected 'callgraph_builder' in dependencies"
        raise AssertionError(msg)


def test_function_ext_metrics_plugin_produces_tables() -> None:
    """Function ext metrics plugin declares correct output tables.

    Raises
    ------
    AssertionError
        If produces_tables is incorrect.
    """
    plugin = function_ext_metrics_plugin
    meta = plugin.metadata

    if "analytics.graph_metrics_functions_ext" not in meta.produces_tables:
        msg = "Expected 'analytics.graph_metrics_functions_ext' in produces_tables"
        raise AssertionError(msg)


def test_module_ext_metrics_plugin_metadata() -> None:
    """Module ext metrics plugin has correct metadata.

    Raises
    ------
    AssertionError
        If metadata is incorrect.
    """
    plugin = module_ext_metrics_plugin
    meta = plugin.metadata

    if meta.name != "graph_metrics_modules_ext":
        msg = f"Expected name 'graph_metrics_modules_ext', got '{meta.name}'"
        raise AssertionError(msg)
    if meta.kind != "metric":
        msg = f"Expected kind 'metric', got '{meta.kind}'"
        raise AssertionError(msg)
    if "import_graph_builder" not in meta.depends_on:
        msg = "Expected 'import_graph_builder' in dependencies"
        raise AssertionError(msg)


def test_module_ext_metrics_plugin_produces_tables() -> None:
    """Module ext metrics plugin declares correct output tables.

    Raises
    ------
    AssertionError
        If produces_tables is incorrect.
    """
    plugin = module_ext_metrics_plugin
    meta = plugin.metadata

    if "analytics.graph_metrics_modules_ext" not in meta.produces_tables:
        msg = "Expected 'analytics.graph_metrics_modules_ext' in produces_tables"
        raise AssertionError(msg)


def test_module_ext_metrics_plugin_requires_graphs() -> None:
    """Module ext metrics plugin requires correct graph kinds.

    Raises
    ------
    AssertionError
        If requires_graph_kinds is incorrect.
    """
    plugin = module_ext_metrics_plugin
    meta = plugin.metadata

    if GraphKind.IMPORT_GRAPH not in meta.requires_graph_kinds:
        msg = "Expected IMPORT_GRAPH in requires_graph_kinds"
        raise AssertionError(msg)


def test_get_core_graph_metrics_plugin_returns_instance() -> None:
    """get_core_graph_metrics_plugin returns the plugin instance.

    Raises
    ------
    AssertionError
        If getter fails.
    """
    plugin = get_core_graph_metrics_plugin()

    if plugin is not core_graph_metrics_plugin:
        msg = "Expected getter to return same instance"
        raise AssertionError(msg)
    if plugin.metadata.name != "core_graph_metrics":
        msg = f"Expected name 'core_graph_metrics', got '{plugin.metadata.name}'"
        raise AssertionError(msg)


def test_get_function_ext_metrics_plugin_returns_instance() -> None:
    """get_function_ext_metrics_plugin returns the plugin instance.

    Raises
    ------
    AssertionError
        If getter fails.
    """
    plugin = get_function_ext_metrics_plugin()

    if plugin is not function_ext_metrics_plugin:
        msg = "Expected getter to return same instance"
        raise AssertionError(msg)


def test_get_module_ext_metrics_plugin_returns_instance() -> None:
    """get_module_ext_metrics_plugin returns the plugin instance.

    Raises
    ------
    AssertionError
        If getter fails.
    """
    plugin = get_module_ext_metrics_plugin()

    if plugin is not module_ext_metrics_plugin:
        msg = "Expected getter to return same instance"
        raise AssertionError(msg)


def test_plugins_provide_capabilities() -> None:
    """All metrics plugins provide their capabilities.

    Raises
    ------
    AssertionError
        If provides is incorrect.
    """
    if "core_metrics" not in core_graph_metrics_plugin.metadata.provides:
        msg = "Expected core_graph_metrics to provide 'core_metrics'"
        raise AssertionError(msg)
    if "function_ext_metrics" not in function_ext_metrics_plugin.metadata.provides:
        msg = "Expected function_ext_metrics to provide 'function_ext_metrics'"
        raise AssertionError(msg)
    if "module_ext_metrics" not in module_ext_metrics_plugin.metadata.provides:
        msg = "Expected module_ext_metrics to provide 'module_ext_metrics'"
        raise AssertionError(msg)


def test_plugins_implement_protocol() -> None:
    """All metrics plugins implement GraphPluginProtocol.

    Raises
    ------
    TypeError
        If plugin does not implement protocol.
    """
    if not isinstance(core_graph_metrics_plugin, GraphPluginProtocol):
        msg = "core_graph_metrics_plugin should implement GraphPluginProtocol"
        raise TypeError(msg)
    if not isinstance(function_ext_metrics_plugin, GraphPluginProtocol):
        msg = "function_ext_metrics_plugin should implement GraphPluginProtocol"
        raise TypeError(msg)
    if not isinstance(module_ext_metrics_plugin, GraphPluginProtocol):
        msg = "module_ext_metrics_plugin should implement GraphPluginProtocol"
        raise TypeError(msg)
