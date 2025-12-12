"""Tests for ConfigService."""

from __future__ import annotations

from pathlib import Path

from codeintel.cli.config import ConfigService
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_not_equal,
    expect_true,
)


def test_config_service_load_defaults() -> None:
    """Verify load() returns config with defaults."""
    service = ConfigService.load(validate=False)

    expect_is_not_none(service.config)
    expect_in("defaults", service.sources)


def test_config_service_load_with_overrides() -> None:
    """Verify load() applies CLI overrides."""
    service = ConfigService.load(
        cli_overrides={"color": False},
        validate=False,
    )

    expect_true(not service.config.color)


def test_config_service_with_overrides() -> None:
    """Verify with_overrides creates new service."""
    service = ConfigService.load(validate=False)
    original_color = service.config.color

    modified = service.with_overrides(color=not original_color)

    expect_not_equal(modified.config.color, original_color)

    expect_equal(service.config.color, original_color)


def test_config_service_sources_tracking() -> None:
    """Verify sources are tracked correctly."""
    service = ConfigService.load(validate=False)

    expect_true(len(service.sources) >= 1)


def test_config_service_cyclopts_config_chain() -> None:
    """Verify get_cyclopts_config_chain returns list with two elements."""
    chain = ConfigService.get_cyclopts_config_chain()

    expect_is_instance(chain, list)

    expect_equal(len(chain), 2)


def test_config_service_cyclopts_config_chain_first_is_callable() -> None:
    """Verify first element in config chain is callable (TOML loader)."""
    chain = ConfigService.get_cyclopts_config_chain()

    expect_true(callable(chain[0]))


def test_config_service_get_toml_config_path_returns_path_or_none() -> None:
    """Verify get_toml_config_path returns Path or None."""
    path = ConfigService.get_toml_config_path()

    if path is not None:
        expect_is_instance(path, Path)


def test_config_service_load_from_explicit_path(tmp_path: Path) -> None:
    """Verify load() uses explicit config path."""
    config_file = tmp_path / "test-config.yaml"
    config_file.write_text("color: false\nlog_level: DEBUG\n")

    service = ConfigService.load(config_path=config_file, validate=False)

    expect_true(not service.config.color)
    expect_equal(service.config.log_level, "DEBUG")

    sources_str = " ".join(service.sources)
    expect_in("file:", sources_str)
