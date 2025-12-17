"""Target graph access and lightweight registry helpers.

Hamilton-First Architecture
---------------------------
In the Hamilton-first architecture, the **Hamilton DAG is the single source of truth**
for target dependencies and the executable graph.

Target metadata (contracts/resources/execution policy/descriptions) is declared alongside the
native Hamilton target implementations (via ``TARGET_SPECS``), and collected into a canonical
catalog by :func:`codeintel.build.target_catalog.load_target_specs`.

Use :func:`get_target_graph` to obtain the singleton :class:`~codeintel.build.targets.TargetGraph`
whose dependency edges are derived from the Hamilton DAG.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from codeintel.build.target_catalog import load_target_catalog

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.targets import OutputTarget, TargetGraph
    from codeintel.core.schemas.primitives import TableSchema

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_target_graph() -> TargetGraph:
    """Get the singleton target graph with Hamilton-derived dependencies.

    Returns
    -------
    TargetGraph
        The singleton target graph with Hamilton-derived dependencies.

    Raises
    ------
    TypeError
        If the Hamilton driver factory does not expose a callable
        ``build_driver`` function.
    """
    driver_factory_mod: ModuleType = importlib.import_module(
        "codeintel.build.hamilton.driver_factory"
    )
    build_driver_fn_raw = getattr(driver_factory_mod, "build_driver", None)
    if not callable(build_driver_fn_raw):
        msg = "codeintel.build.hamilton.driver_factory.build_driver is missing or not callable"
        raise TypeError(msg)

    build_driver_fn = cast("Callable[[], HamiltonRuntime]", build_driver_fn_raw)
    runtime = build_driver_fn()
    return runtime.graph


def derive_schemas_from_targets(
    targets: tuple[OutputTarget, ...],
) -> dict[str, TableSchema]:
    """Derive table schema mapping from target contracts.

    Parameters
    ----------
    targets
        Tuple of OutputTargets to extract schemas from.

    Returns
    -------
    dict[str, TableSchema]
        Mapping of table key to TableSchema.
    """
    schemas: dict[str, TableSchema] = {}

    for target in targets:
        for table in target.contract.tables:
            key = table.fq_name
            if key in schemas:
                log.warning(
                    "Duplicate schema for %s from targets %s",
                    key,
                    target.name,
                )
            schemas[key] = table

    return schemas


def get_all_target_table_keys(targets: tuple[OutputTarget, ...] | None = None) -> frozenset[str]:
    """Return all table keys declared by any target.

    Parameters
    ----------
    targets
        Optional targets to extract keys from (defaults to the canonical target catalog).

    Returns
    -------
    frozenset[str]
        Set of all table keys from target contracts.
    """
    if targets is None:
        return load_target_catalog().all_table_keys

    keys: set[str] = set()
    for target in targets:
        keys.update(target.table_keys)
    return frozenset(keys)


def get_target_by_table(
    table_key: str,
    *,
    targets: tuple[OutputTarget, ...] | None = None,
) -> OutputTarget | None:
    """Find the target that produces a given table.

    Parameters
    ----------
    table_key
        Fully-qualified table name.
    targets
        Optional set of targets to search (defaults to the canonical target catalog).

    Returns
    -------
    OutputTarget | None
        Target that produces this table, or None.
    """
    if targets is None:
        return load_target_catalog().target_for_table_key(table_key)

    for target in targets:
        if table_key in target.table_keys:
            return target
    return None


__all__ = [
    "derive_schemas_from_targets",
    "get_all_target_table_keys",
    "get_target_by_table",
    "get_target_graph",
]
