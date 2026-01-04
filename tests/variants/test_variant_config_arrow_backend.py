"""Tests for the Arrow backend variant configuration."""

from __future__ import annotations

import pytest

from codeintel.core.runtime import VariantConfig


def test_variant_config_accepts_arrow_backend() -> None:
    """Arrow backend should be accepted in variant configuration."""
    config = VariantConfig.from_mapping({"df_backend": "arrow"})

    assert config.df_backend == "arrow"


def test_variant_config_rejects_features_for_arrow_backend() -> None:
    """Arrow backend should reject feature sets until with_columns support exists."""
    config = VariantConfig.from_mapping(
        {
            "df_backend": "arrow",
            "feature_sets": {"analytics.some_table": ("feature_a",)},
        }
    )

    with pytest.raises(ValueError, match="feature_sets are not supported"):
        config.validate()
