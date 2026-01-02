"""Validate pipe_input step gating by clean_mode."""

from __future__ import annotations

from collections.abc import Iterable

import hamilton.driver as h_driver

from codeintel.build.hamilton.native.analytics import function_types


def _names(variables: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for variable in variables:
        name = getattr(variable, "name", None)
        names.add(str(name) if name is not None else str(variable))
    return names


def _build_driver(clean_mode: str) -> h_driver.Driver:
    config = {
        "hamilton.enable_power_user_mode": True,
        "df_backend": "polars_lazy",
        "clean_mode": clean_mode,
        "null_policy": "preserve",
        "max_loc_clip": 10_000,
        "feature_sets": {},
    }
    return (
        h_driver.Builder()
        .with_config(config)
        .with_modules(function_types)
        .allow_module_overrides()
        .build()
    )


def _has_prep_step(names: set[str], step_name: str) -> bool:
    tokens = (f"prep.{step_name}", f"prep__{step_name}")
    return any(any(token in name for token in tokens) for name in names)


def test_pipe_input_step_gating() -> None:
    """Ensure prep pipeline steps appear only when enabled."""
    strict_driver = _build_driver("strict")
    off_driver = _build_driver("off")

    strict_names = _names(strict_driver.list_available_variables())
    off_names = _names(off_driver.list_available_variables())

    assert _has_prep_step(strict_names, "nulls")
    assert _has_prep_step(strict_names, "with_drop_bad_rows")
    assert not _has_prep_step(off_names, "nulls")
    assert not _has_prep_step(off_names, "with_drop_bad_rows")
