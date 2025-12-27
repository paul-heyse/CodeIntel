"""Validate TagQuery tag_filter discovery semantics."""

from __future__ import annotations

import types
from collections.abc import Callable, Iterable
from typing import ParamSpec, TypeVar, cast

import hamilton.driver as h_driver
from hamilton.function_modifiers import tag

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import semantic_view
from codeintel.core.hamilton.tag_filters import (
    tf_artifacts,
    tf_datasets,
    tf_savers,
    tf_semantic_views,
)
from codeintel.core.hamilton.tag_query import TagQuery

P = ParamSpec("P")
R = TypeVar("R")
TagDecorator = Callable[[Callable[P, R]], Callable[P, R]]
TagFactory = Callable[..., TagDecorator]
_TAG_ANY = cast("TagFactory", tag)


@tag(node_type=ht.NODE_TYPE_DATASET, table_key="core.modules")
def dataset_modules() -> int:
    """Return sentinel value for dataset module node.

    Returns
    -------
    int
        Sentinel value for testing.
    """
    return 1


@tag(node_type=ht.NODE_TYPE_DATASET, table_key="core.functions")
def dataset_functions() -> int:
    """Return sentinel value for dataset function node.

    Returns
    -------
    int
        Sentinel value for testing.
    """
    return 1


@tag(node_type=ht.NODE_TYPE_ARTIFACT, artifact="semantic_registry")
def artifact_semantic_registry() -> int:
    """Return sentinel value for semantic registry artifact node.

    Returns
    -------
    int
        Sentinel value for testing.
    """
    return 1


_SAVER_TAGS = cast(
    "dict[str, object]",
    {
        "hamilton.data_saver": True,
        "output_role": "contract",
        "hamilton.data_saver.sink": "duckdb",
    },
)


@_TAG_ANY(**_SAVER_TAGS)
def saver_duckdb() -> int:
    """Return sentinel value for a duckdb saver node.

    Returns
    -------
    int
        Sentinel value for testing.
    """
    return 1


@semantic_view(
    semantic_id="sv_example",
    table_key="docs.v_example",
    entity="example",
    grain="example",
)
def semantic_view_example() -> int:
    """Return sentinel value for semantic view node.

    Returns
    -------
    int
        Sentinel value for testing.
    """
    return 1


def _build_driver() -> h_driver.Driver:
    module = types.ModuleType("tag_filter_fixture")
    for fn in (
        dataset_modules,
        dataset_functions,
        artifact_semantic_registry,
        saver_duckdb,
        semantic_view_example,
    ):
        setattr(module, fn.__name__, fn)
    return h_driver.Builder().with_modules(module).build()


def _names(variables: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for variable in variables:
        if isinstance(variable, str):
            names.add(variable)
            continue
        name = getattr(variable, "name", None)
        names.add(str(name) if name is not None else str(variable))
    return names


def test_tag_filters_discover_nodes() -> None:
    """Discover nodes with tag filters using TagQuery."""
    dr = _build_driver()
    tag_query = TagQuery(dr)

    datasets = _names(tag_query.query(tf_datasets()))
    assert datasets == {"dataset_functions", "dataset_modules"}

    modules_only = _names(tag_query.query(tf_datasets(table_key="core.modules")))
    assert modules_only == {"dataset_modules"}

    artifacts = _names(tag_query.query(tf_artifacts()))
    assert artifacts == {"artifact_semantic_registry"}

    savers = _names(tag_query.query(tf_savers(role="contract", sink="duckdb")))
    assert savers == {"saver_duckdb"}

    semantic_views = _names(tag_query.query(tf_semantic_views()))
    assert semantic_views == {"semantic_view_example"}
