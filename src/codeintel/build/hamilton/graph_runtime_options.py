"""Build-side configuration loader for GraphRuntimeOptions.

Graph runtime options are used by analytics targets that need access to graph engines
(call graph, import graph, CFG/DFG views, etc.). These options should be loaded from
``env.config.parameters_for(target_name)`` to avoid plan/execution drift.
"""

from __future__ import annotations

from dataclasses import replace

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.graphs.runtime import GraphRuntimeOptions


def load_graph_runtime_options(
    env: BuildEnv,
    *,
    target_name: str,
) -> GraphRuntimeOptions:
    """Load GraphRuntimeOptions from BuildEnv configuration.

    Parameters
    ----------
    env
        Build environment providing snapshot and configuration.
    target_name
        Target name whose configuration section should be loaded.

    Returns
    -------
    GraphRuntimeOptions
        Runtime options with snapshot defaults normalized.
    """
    options = load_target_options(env, target_name=target_name, options_type=GraphRuntimeOptions)
    return replace(options, snapshot=env.snapshot)


__all__ = [
    "load_graph_runtime_options",
]
