"""Variant configuration loader for Hamilton build runs."""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.column_ops import allowed_ops_by_table
from codeintel.core.runtime.variants import VariantConfig


def variant_config_from_build_config(config: BuildConfig) -> VariantConfig:
    """Load VariantConfig from BuildConfig.

    Parameters
    ----------
    config
        Build configuration loaded from TOML.

    Returns
    -------
    VariantConfig
        Parsed variant configuration.

    Raises
    ------
    TypeError
        If the variants section is not a mapping.
    """
    raw_variants = config.get("variants", {})
    if raw_variants is None:
        raw_variants = {}
    if not isinstance(raw_variants, Mapping):
        msg = "variants section must be a mapping"
        raise TypeError(msg)
    return VariantConfig.from_mapping(raw_variants).validate(
        allowed_ops=allowed_ops_by_table(),
    )


__all__ = ["variant_config_from_build_config"]
