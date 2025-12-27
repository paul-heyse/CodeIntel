"""Validate resolve_from_config changes DAG shape for features."""

from __future__ import annotations

from collections.abc import Iterable

import hamilton.driver as h_driver

from codeintel.build.hamilton.native.analytics import tables_functions


def _names(variables: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for variable in variables:
        name = getattr(variable, "name", None)
        names.add(str(name) if name is not None else str(variable))
    return names


def _build_driver(feature_sets: dict[str, tuple[str, ...]]) -> h_driver.Driver:
    config = {
        "df_backend": "polars",
        "clean_mode": "lenient",
        "null_policy": "preserve",
        "max_loc_clip": 10_000,
        "feature_sets": feature_sets,
    }
    return (
        h_driver.Builder()
        .with_config(config)
        .with_modules(tables_functions)
        .allow_module_overrides()
        .build()
    )


def test_resolve_from_config_changes_dag_shape() -> None:
    """Ensure feature subDAGs appear only when configured."""
    table_key = tables_functions.FUNCTION_METRICS_TABLE_KEY
    with_features = _build_driver({table_key: ("loc_squared",)})
    without_features = _build_driver({})

    names_with = _names(with_features.list_available_variables())
    names_without = _names(without_features.list_available_variables())

    namespace_prefix = f"feat__{table_key}"

    assert "function_metrics__table" in names_with
    assert "function_metrics__table" in names_without
    assert any(namespace_prefix in name for name in names_with)
    assert not any(namespace_prefix in name for name in names_without)
    assert names_with != names_without
