"""OpenTelemetry SDK config file loading and application."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import yaml

LOG = logging.getLogger(__name__)

_REQUIRED_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "receivers",
        "service",
        "traces",
        "metrics",
        "logs",
        "exporters",
        "processors",
    }
)


def load_otel_config_file(path: Path) -> Mapping[str, object]:
    """Validate and load an OpenTelemetry SDK config file.

    Returns
    -------
    Mapping[str, object]
        Parsed configuration payload.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not path.exists():
        message = f"OpenTelemetry config file not found: {path}"
        raise FileNotFoundError(message)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    payload = _parse_config_payload(path, text)
    _validate_root_keys(payload, path)
    return payload


def apply_otel_config_file(path: Path) -> bool:
    """Apply an OpenTelemetry SDK configuration file, when supported.

    Returns
    -------
    bool
        True when the SDK configurator was applied.

    Raises
    ------
    TypeError
        If the SDK configurator entrypoint is unavailable or incompatible.
    """
    otel_config = _load_module(
        "opentelemetry.sdk._configuration",
        label="OpenTelemetry SDK configuration",
    )

    configure = getattr(otel_config, "configure", None)
    if not callable(configure):
        message = "OpenTelemetry SDK configurator entrypoint is unavailable"
        raise TypeError(message)
    try:
        configure(config_file=str(path))
    except TypeError as exc:
        message = "OpenTelemetry SDK configurator does not accept config_file"
        raise TypeError(message) from exc
    except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover
        LOG.warning("Failed to apply OpenTelemetry config: %s", exc)
        return False
    return True


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
        raise TypeError(message)
    return payload


def _parse_yaml_payload(text: str, path: Path) -> Mapping[str, object]:
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        message = f"Invalid YAML in OpenTelemetry config file {path}: {exc}"
        raise ValueError(message) from exc
    if not isinstance(payload, dict):
        message = f"OpenTelemetry config file {path} must contain a mapping"
        raise TypeError(message)
    return payload


def _validate_root_keys(payload: Mapping[str, object], path: Path) -> None:
    if not payload:
        return
    if any(key in payload for key in _REQUIRED_ROOT_KEYS):
        return
    LOG.warning("OpenTelemetry config file %s missing standard sections; continuing", path)


def _load_module(module_name: str, *, label: str) -> object:
    try:
        module = __import__(module_name, fromlist=["_"])
    except ImportError as exc:
        message = f"{label} module is unavailable: {exc}"
        raise RuntimeError(message) from exc
    return module


__all__ = ["apply_otel_config_file", "load_otel_config_file"]
