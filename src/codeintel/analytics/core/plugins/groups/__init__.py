"""Plugin groups for bundling related plugins.

This package provides `PluginGroup` for organizing plugins into
logical bundles that can be enabled/disabled together.

Groups
------
FUNCTION_PLUGINS
    Function-level analytics plugins.
GRAPH_PLUGINS
    Graph metric plugins.
RISK_PLUGINS
    Risk scoring plugins.

Example
-------
>>> from codeintel.analytics.core.plugins.groups import FUNCTION_PLUGINS
>>> plugins = FUNCTION_PLUGINS.get_plugins(registry)
"""

from __future__ import annotations

from codeintel.analytics.core.plugins.groups.protocol import PluginGroup

# Define standard groups
FUNCTION_PLUGINS = PluginGroup(
    name="functions",
    description="Function-level analytics plugins",
    plugins=(
        "functions.metrics",
        "functions.ast_features",
    ),
    default_order="dependency",
)

GRAPH_PLUGINS = PluginGroup(
    name="graphs",
    description="Graph metric plugins",
    plugins=(
        "graph_metrics.core",
        "graph_metrics.centrality",
        "graph_metrics.modules",
    ),
    default_order="dependency",
)

RISK_PLUGINS = PluginGroup(
    name="risk",
    description="Risk scoring plugins",
    plugins=(
        "risk.function_risk",
        "risk.hotspots",
    ),
    default_order="dependency",
    requires=("functions", "graphs"),
)

ALL_GROUPS = {
    "functions": FUNCTION_PLUGINS,
    "graphs": GRAPH_PLUGINS,
    "risk": RISK_PLUGINS,
}

__all__ = [
    "ALL_GROUPS",
    "FUNCTION_PLUGINS",
    "GRAPH_PLUGINS",
    "RISK_PLUGINS",
    "PluginGroup",
]
