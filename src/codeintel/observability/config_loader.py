"""OpenTelemetry SDK config file loading and application."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

import yaml

LOG = logging.getLogger(__name__)

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
    ValueError
        If the config file is empty or invalid.
    """
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


def apply_otel_config_file(path: Path) -> bool:
    """Apply an OpenTelemetry SDK configuration file, when supported.

    Returns
    -------
    bool
        True when a compatible configurator was applied.
    """
    otel_config = _load_module(
        "opentelemetry.sdk._configuration",
        label="OpenTelemetry SDK configuration",
    )

    candidates: list[Callable[..., object]] = []
    for name in ("configure", "configure_otel", "initialize", "init", "load"):
        value = getattr(otel_config, name, None)
        if callable(value):
            candidates.append(value)

    configurator_cls = getattr(otel_config, "Configurator", None)
    if configurator_cls is not None:
        configurator = configurator_cls()
        configure = getattr(configurator, "configure", None)
        if callable(configure):
            candidates.append(configure)

    for candidate in candidates:
        if _call_configurator(candidate, path):
            return True

    LOG.warning("No compatible OpenTelemetry configuration entrypoint found")
    return False


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
    if any(key in payload for key in _REQUIRED_ROOT_KEYS):
        return
    message = (
        "OpenTelemetry config file is missing required sections; "
        f"expected one of {_REQUIRED_ROOT_KEYS} in {path}"
    )
    raise ValueError(message)


def _load_module(module_name: str, *, label: str) -> object:
    try:
        module = __import__(module_name, fromlist=["_"])
    except ImportError as exc:
        message = f"{label} module is unavailable: {exc}"
        raise RuntimeError(message) from exc
    return module


def _call_configurator(func: Callable[..., object], path: Path) -> bool:
    config_value = str(path)
    attempts: tuple[Callable[[], object], ...] = (
        partial(func, config_file=config_value),
        partial(func, config_file_path=config_value),
        partial(func, path=config_value),
        partial(func, config_value),
        partial(func),
    )

    for attempt in attempts:
        try:
            attempt()
        except TypeError:
            continue
        except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover
            LOG.warning("Failed to apply OpenTelemetry config: %s", exc)
            return False
        else:
            return True

    return False


__all__ = ["apply_otel_config_file", "load_otel_config_file"]
