"""Mini seed harness for deterministic schema compilation.

The harness creates empty LazyFrames from declared schemas to seed q__ inputs
for schema inference without relying on DuckDB.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.conversion import table_to_lazyframe
from codeintel.core.schemas.arrow_gen import arrow_schema_from_table_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from codeintel.core.schemas.provider import SchemaProvider


def qparam_to_table_key(qparam: str) -> str:
    """Convert a q__ parameter name into a table key.

    Parameters
    ----------
    qparam
        Parameter name in the form ``q__schema__table``.

    Returns
    -------
    str
        Table key in the form ``schema.table``.

    Raises
    ------
    ValueError
        If qparam does not start with ``q__`` or cannot be parsed.
    """
    if not qparam.startswith("q__"):
        msg = f"Expected q__ parameter, got: {qparam}"
        raise ValueError(msg)
    payload = qparam.removeprefix("q__")
    schema, rest = payload.split("__", 1)
    return f"{schema}.{rest}"


def extract_qparams_from_callable(fn: Callable[..., Any]) -> set[str]:
    """Return q__ parameter names declared by a callable.

    Parameters
    ----------
    fn
        Callable to inspect.

    Returns
    -------
    set[str]
        Set of parameter names that begin with ``q__``.
    """
    return {name for name in inspect.signature(fn).parameters if name.startswith("q__")}


def extract_qparams_for_target_module(target: str, module: ModuleType) -> set[str]:
    """Union q__ parameters across functions belonging to a target module.

    This avoids depending on Hamilton internals while remaining robust to a
    target being split into multiple Hamilton nodes.

    Parameters
    ----------
    target
        Target name (e.g., "risk_factors").
    module
        Python module containing target node functions.

    Returns
    -------
    set[str]
        Union of q__ parameter names across compute functions for the target.
    """
    prefix = f"t__{target}__"
    qparams: set[str] = set()
    for name, obj in vars(module).items():
        if not name.startswith(prefix) or not callable(obj):
            continue
        qparams |= extract_qparams_from_callable(obj)
    return qparams


@dataclass
class MiniSeedHarness:
    """Seed upstream empty tables and build q__ LazyFrame inputs for compute execution."""

    schema_provider: SchemaProvider
    _seeded: dict[str, pl.LazyFrame] = field(default_factory=dict)

    def seed_table(self, table_key: str) -> pl.LazyFrame:
        """Create and cache an empty LazyFrame for the declared schema.

        Returns
        -------
        pl.LazyFrame
            Empty LazyFrame for the requested schema.
        """
        cached = self._seeded.get(table_key)
        if cached is not None:
            return cached
        table_schema = self.schema_provider.require_table_schema(table_key)
        arrow_schema = arrow_schema_from_table_schema(table_schema=table_schema)
        table = pa.Table.from_batches([], schema=arrow_schema)
        frame = table_to_lazyframe(table)
        self._seeded[table_key] = frame
        return frame

    def seeded_table_keys(self) -> tuple[str, ...]:
        """Return seeded table keys in deterministic order.

        Returns
        -------
        tuple[str, ...]
            Seeded table keys sorted lexicographically.
        """
        return tuple(sorted(self._seeded))

    def seed_input(self, qparam: str) -> pl.LazyFrame:
        """Return an empty LazyFrame for a q__ parameter.

        Parameters
        ----------
        qparam
            q__ parameter name in the form ``q__schema__table``.

        Returns
        -------
        pl.LazyFrame
            Empty LazyFrame for the referenced upstream schema.
        """
        table_key = qparam_to_table_key(qparam)
        return self.seed_table(table_key)

    def build_inputs(self, qparams: set[str]) -> Mapping[str, pl.LazyFrame]:
        """Build a deterministic mapping of q__ inputs for compute execution.

        Parameters
        ----------
        qparams
            Set of q__ parameter names.

        Returns
        -------
        Mapping[str, pl.LazyFrame]
            Mapping from q__ parameter name to empty LazyFrames.
        """
        return {q: self.seed_input(q) for q in sorted(qparams)}


__all__ = [
    "MiniSeedHarness",
    "extract_qparams_for_target_module",
    "extract_qparams_from_callable",
    "qparam_to_table_key",
]
