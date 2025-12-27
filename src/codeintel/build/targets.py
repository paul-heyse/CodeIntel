"""Target metadata helpers for the build system."""

from __future__ import annotations

from typing import Literal

from codeintel.build.hamilton.dag_catalog import TargetDescriptor

if TYPE_CHECKING:
    from codeintel.build.hamilton.dag_catalog import DagCatalog

TargetModule = Literal["ingestion", "graphs", "analytics", "export"]
"""Classification of which target module produces an output."""


def get_target_by_table(
    table_key: str,
    *,
    catalog: DagCatalog,
) -> TargetDescriptor | None:
    """Return the first target producing a given table key.

    Returns
    -------
    TargetDescriptor | None
        Producing target when found, otherwise None.
    """
    output = catalog.table_outputs.get(table_key)
    if output is None:
        return None
    return catalog.targets.get(output.producer_target)


__all__ = [
    "TargetDescriptor",
    "TargetModule",
    "get_target_by_table",
]
