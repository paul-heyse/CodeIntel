"""OpenTelemetry config validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.observability.config_validation import validate_otel_config_file


def test_validate_otel_config_file_missing(tmp_path: Path) -> None:
    """Missing config files should raise FileNotFoundError."""
    path = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        validate_otel_config_file(path)


def test_validate_otel_config_file_empty(tmp_path: Path) -> None:
    """Empty config files should raise ValueError."""
    path = tmp_path / "otel.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        validate_otel_config_file(path)


def test_validate_otel_config_file_requires_root_keys(tmp_path: Path) -> None:
    """Config files without required sections should raise ValueError."""
    path = tmp_path / "otel.yaml"
    path.write_text("foo: bar", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required sections"):
        validate_otel_config_file(path)


def test_validate_otel_config_file_accepts_yaml(tmp_path: Path) -> None:
    """YAML config files should parse successfully."""
    path = tmp_path / "otel.yaml"
    path.write_text("service: {}", encoding="utf-8")
    payload = validate_otel_config_file(path)
    assert "service" in payload


def test_validate_otel_config_file_accepts_json(tmp_path: Path) -> None:
    """JSON config files should parse successfully."""
    path = tmp_path / "otel.json"
    path.write_text('{"service": {}}', encoding="utf-8")
    payload = validate_otel_config_file(path)
    assert "service" in payload


def test_validate_otel_config_file_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON config files should raise ValueError."""
    path = tmp_path / "otel.json"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        validate_otel_config_file(path)
