"""Runtime containers for Hamilton execution."""

from __future__ import annotations

from dataclasses import dataclass

import hamilton.driver as h_driver

from codeintel.build.hamilton.dag_catalog import DagCatalog


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


__all__ = ["HamiltonRuntime"]
