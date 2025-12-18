"""GraphFeatureFlags feature flag loading and validation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import GraphFeatureFlags

if TYPE_CHECKING:
    from collections.abc import Callable


def _with_env(overrides: dict[str, str], func: Callable[[], None]) -> None:
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        func()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_graph_feature_flags_from_env() -> None:
    """from_env should parse graph feature flags when provided."""

    def _run() -> None:
        expected_limit = 25
        flags = GraphFeatureFlags.from_env()
        if flags.eager_hydration is not True:
            message = "eager_hydration should be parsed as True"
            raise AssertionError(message)
        if flags.community_detection_limit != expected_limit:
            message = "community_detection_limit should be parsed from env"
            raise AssertionError(message)
        if flags.validation_strict is not True:
            message = "validation_strict should be parsed as True"
            raise AssertionError(message)

    _with_env(
        {
            "CODEINTEL_GRAPH_EAGER": "1",
            "CODEINTEL_GRAPH_COMMUNITY_LIMIT": "25",
            "CODEINTEL_GRAPH_VALIDATION_STRICT": "true",
        },
        _run,
    )


def test_graph_feature_flags_reject_invalid_values() -> None:
    """Invalid feature flags should raise during parsing."""

    def _run_invalid() -> None:
        with pytest.raises(ValueError, match="graph feature flag"):
            GraphFeatureFlags.from_env()

    _with_env(
        {
            "CODEINTEL_GRAPH_COMMUNITY_LIMIT": "-3",
        },
        _run_invalid,
    )
