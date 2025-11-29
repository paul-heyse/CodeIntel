"""ServingConfig feature flag loading and validation."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from codeintel.config.serving_models import ServingConfig


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


def test_serving_config_from_env_graph_features(tmp_path: Path) -> None:
    """from_env should parse graph feature flags when provided."""

    def _run() -> None:
        cfg = ServingConfig.from_env()
        expected_limit = 25
        if cfg.graph_features.eager_hydration is not True:
            message = "eager_hydration should be parsed as True"
            raise AssertionError(message)
        if cfg.graph_features.community_detection_limit != expected_limit:
            message = "community_detection_limit should be parsed from env"
            raise AssertionError(message)
        if cfg.graph_features.validation_strict is not True:
            message = "validation_strict should be parsed as True"
            raise AssertionError(message)

    _with_env(
        {
            "CODEINTEL_REPO_ROOT": str(tmp_path),
            "CODEINTEL_GRAPH_EAGER": "1",
            "CODEINTEL_GRAPH_COMMUNITY_LIMIT": "25",
            "CODEINTEL_GRAPH_VALIDATION_STRICT": "true",
        },
        _run,
    )


def test_serving_config_validates_graph_features(tmp_path: Path) -> None:
    """Invalid feature flags should raise during validation."""

    def _run_invalid() -> None:
        with pytest.raises(ValueError, match="community_detection_limit"):
            ServingConfig.from_env()

    _with_env(
        {
            "CODEINTEL_REPO_ROOT": str(tmp_path),
            "CODEINTEL_GRAPH_COMMUNITY_LIMIT": "-3",
        },
        _run_invalid,
    )
