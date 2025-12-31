"""Legacy SQL-based view materialization (retired)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from hamilton.driver import Driver

    from codeintel.core.hamilton.tag_query import TagQuery
    from codeintel.storage.gateway.protocol import MinimalGateway


@dataclass(frozen=True, slots=True)
class ViewMaterializationOptions:
    """Options controlling view materialization.

    Notes
    -----
    SQLGlot-driven view materialization is retired. This type remains to preserve
    backward-compatible signatures while callers migrate to Hamilton view outputs.
    """

    overwrite: bool = True
    strict: bool = False
    dr: Driver | None = None
    tag_query: TagQuery | None = None


def materialize_registered_views(
    gateway: MinimalGateway,
    *,
    modules: tuple[ModuleType, ...],
    options: ViewMaterializationOptions | None = None,
) -> dict[str, str]:
    """Reject SQLGlot view materialization.

    Parameters
    ----------
    gateway
        Storage gateway (unused).
    modules
        View builder modules (unused).
    options
        Legacy options (unused).

    Raises
    ------
    RuntimeError
        Always raised because SQLGlot view materialization has been retired.
    """
    _ = (gateway, modules, options)
    msg = (
        "SQLGlot view materialization has been retired. "
        "Use Hamilton-native view outputs and dataset-backed snapshots instead."
    )
    raise RuntimeError(msg)


__all__ = ["ViewMaterializationOptions", "materialize_registered_views"]
