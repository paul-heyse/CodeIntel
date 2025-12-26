"""Validation helpers for OpenTelemetry config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping

_REQUIRED_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "service",
        "traces",
        "metrics",
        "logs",
        "exporters",
        "processors",
    }
)


def validate_otel_config_file(path: Path) -> Mapping[str, object]:
    """Validate and load an OpenTelemetry SDK config file."""
    if not path.exists():
        message = f"OpenTelemetry config file not found: {path}"
        raise FileNotFoundError(message)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        message = f"OpenTelemetry config file is empty: {path}"
        raise ValueError(message)
    payload = _parse_config_payload(path, text)
    _validate_root_keys(payload, path)
    return payload


def _parse_config_payload(path: Path, text: str) -> Mapping[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _parse_json_payload(text, path)
    return _parse_yaml_payload(text, path)


def _parse_json_payload(text: str, path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        message = f"Invalid JSON in OpenTelemetry config file {path}: {exc}"
        raise ValueError(message) from exc
    if not isinstance(payload, dict):
        message = f"OpenTelemetry config file {path} must contain a mapping"
        raise ValueError(message)
    return payload


def _parse_yaml_payload(text: str, path: Path) -> Mapping[str, object]:
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        message = f"Invalid YAML in OpenTelemetry config file {path}: {exc}"
        raise ValueError(message) from exc
    if not isinstance(payload, dict):
        message = f"OpenTelemetry config file {path} must contain a mapping"
        raise ValueError(message)
    return payload


def _validate_root_keys(payload: Mapping[str, object], path: Path) -> None:
    if any(key in payload for key in _REQUIRED_ROOT_KEYS):
        return
    message = (
        "OpenTelemetry config file is missing required sections; "
        f"expected one of {_REQUIRED_ROOT_KEYS} in {path}"
    )
    raise ValueError(message)


__all__ = ["validate_otel_config_file"]
