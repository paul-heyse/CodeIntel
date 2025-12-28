"""Polars view registry for semantic serving."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.hamilton.semantic_tags import TAG_TABLE_KEY, get_semantic_view_tags

if TYPE_CHECKING:
    from types import ModuleType

    import polars as pl

    type ViewBuilder = Callable[[ViewInputs], pl.LazyFrame]
else:
    type ViewBuilder = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class ViewInputs:
    """Helper for resolving source tables inside Polars view builders."""

    loader: Callable[[str, str | None], object]

    def table(self, table_key: str, *, row_index: str | None = None) -> object:
        """Return a lazy frame for a table key, with optional row index injection.

        Returns
        -------
        object
            Lazy frame or equivalent table handle.
        """
        return self.loader(table_key, row_index)


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """View spec for Polars-based semantic views."""

    table_key: str
    builder: ViewBuilder
    tags: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ViewRegistry:
    """Registry of Polars view builders keyed by table_key."""

    specs: Mapping[str, ViewSpec]

    def get(self, table_key: str) -> ViewSpec | None:
        """Return a view spec by table key, if registered.

        Returns
        -------
        ViewSpec | None
            Registered view spec or None.
        """
        return self.specs.get(table_key)

    @classmethod
    def load(cls, *, modules: tuple[ModuleType, ...]) -> ViewRegistry:
        """Discover and load view specs from modules.

        Returns
        -------
        ViewRegistry
            Registry containing all discovered specs.
        """
        specs = discover_view_specs(modules=modules)
        by_table = {spec.table_key: spec for spec in specs}
        return cls(specs=by_table)


def view_spec_modules() -> tuple[ModuleType, ...]:
    """Return modules to scan for Polars view specs.

    Returns
    -------
    tuple[ModuleType, ...]
        Modules containing Polars view definitions.
    """
    return (importlib.import_module("codeintel.serving.semantic.polars_views"),)


def discover_view_specs(*, modules: tuple[ModuleType, ...]) -> tuple[ViewSpec, ...]:
    """Discover Polars view specs from tagged callables in modules.

    Returns
    -------
    tuple[ViewSpec, ...]
        Discovered view specs sorted by table key.
    """
    discovered: list[ViewSpec] = []
    for module in modules:
        for value in vars(module).values():
            if not inspect.isfunction(value):
                continue
            if value.__module__ != module.__name__:
                continue
            tags = get_semantic_view_tags(value)
            if not tags:
                continue
            table_key = tags.get(TAG_TABLE_KEY)
            if not isinstance(table_key, str) or not table_key:
                continue
            discovered.append(ViewSpec(table_key=table_key, builder=value, tags=tags))
    discovered.sort(key=lambda spec: spec.table_key)
    return tuple(discovered)


__all__ = [
    "ViewInputs",
    "ViewRegistry",
    "ViewSpec",
    "discover_view_specs",
    "view_spec_modules",
]
