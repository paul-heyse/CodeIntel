"""Runtime containers for Hamilton execution."""

from __future__ import annotations

from dataclasses import dataclass

import hamilton.driver as h_driver

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.core.hamilton.tag_query import TagQuery


@dataclass(frozen=True)
class HamiltonRuntime:
    """Bundled Hamilton Driver and DagCatalog for build execution.

    This dataclass provides convenient access to both the Hamilton Driver
    (for DAG execution) and the DagCatalog (for target metadata lookup).

    Attributes
    ----------
    dr
        Hamilton Driver configured with the appropriate node module.
    catalog
        Immutable catalog derived from the Hamilton graph.
    tag_query
        Cached tag-filter query helper bound to the Driver.

    Examples
    --------
    >>> runtime = build_driver()
    >>> node = runtime.catalog.target_node("function_metrics")
    >>> result = runtime.dr.execute(
    ...     [node],
    ...     inputs={"env": env, "catalog": runtime.catalog},
    ... )
    """

    dr: h_driver.Driver
    catalog: DagCatalog
    tag_query: TagQuery


__all__ = ["HamiltonRuntime"]
