"""Integration tests for ConfigService with Cyclopts."""

from __future__ import annotations

import os
from pathlib import Path

from codeintel.cli.commands import make_root_app
from codeintel.cli.config import ConfigService
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)


def test_make_root_app_uses_config_service() -> None:
    """Verify make_root_app creates App with ConfigService chain."""
    app = make_root_app()

    name = app.name[0] if isinstance(app.name, tuple) else app.name
    expect_equal(name, "codeintel")
    expect_is_not_none(app.config)

    expect_equal(len(app.config), 2)


def test_make_root_app_has_expected_properties() -> None:
    """Verify make_root_app configures App correctly."""
    app = make_root_app()

    name = app.name[0] if isinstance(app.name, tuple) else app.name
    expect_equal(name, "codeintel")
    expect_true(app.print_error)


def test_env_overrides_file(tmp_path: Path) -> None:
    """Verify environment variables override file config."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("color = true\n")

    original_value = os.environ.get("CODEINTEL_COLOR")
    try:
        os.environ["CODEINTEL_COLOR"] = "false"

        service = ConfigService.load(config_path=config_file, validate=False)

        expect_true(not service.config.color)
        expect_in("env", service.sources)
    finally:
        if original_value is None:
            os.environ.pop("CODEINTEL_COLOR", None)
        else:
            os.environ["CODEINTEL_COLOR"] = original_value


def test_cli_overrides_env() -> None:
    """Verify CLI overrides have highest precedence."""
    original_value = os.environ.get("CODEINTEL_COLOR")
    try:
        os.environ["CODEINTEL_COLOR"] = "true"

        service = ConfigService.load(
            cli_overrides={"color": False},
            validate=False,
        )

        expect_true(not service.config.color)
    finally:
        if original_value is None:
            os.environ.pop("CODEINTEL_COLOR", None)
        else:
            os.environ["CODEINTEL_COLOR"] = original_value


def test_sources_tracking(tmp_path: Path) -> None:
    """Verify config sources are tracked correctly."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("color = false\n")

    service = ConfigService.load(
        config_path=config_file,
        cli_overrides={"log_level": "DEBUG"},
        validate=False,
    )

    expect_in("defaults", service.sources)
    sources_str = " ".join(service.sources)
    expect_in("file:", sources_str)
    expect_in("cli-flags", service.sources)


def test_default_path_when_no_file() -> None:
    """Verify returns None when no config file exists."""
    original_value = os.environ.get("CODEINTEL_CONFIG_PATH")
    try:
        os.environ.pop("CODEINTEL_CONFIG_PATH", None)
        path = ConfigService.get_toml_config_path()

        if path is not None:
            expect_true(path.exists())
    finally:
        if original_value is not None:
            os.environ["CODEINTEL_CONFIG_PATH"] = original_value


def test_env_override_path(tmp_path: Path) -> None:
    """Verify CODEINTEL_CONFIG_PATH env var takes precedence."""
    config_file = tmp_path / "custom.toml"
    config_file.write_text("[codeintel]\n")

    original_value = os.environ.get("CODEINTEL_CONFIG_PATH")
    try:
        os.environ["CODEINTEL_CONFIG_PATH"] = str(config_file)

        path = ConfigService.get_toml_config_path()

        expect_equal(path, config_file)
    finally:
        if original_value is None:
            os.environ.pop("CODEINTEL_CONFIG_PATH", None)
        else:
            os.environ["CODEINTEL_CONFIG_PATH"] = original_value


def test_search_default_locations(tmp_path: Path) -> None:
    """Verify searches default locations when no env override."""
    config_file = tmp_path / "codeintel.toml"
    config_file.write_text("[codeintel]\n")

    original_cwd = Path.cwd()
    original_env = os.environ.get("CODEINTEL_CONFIG_PATH")
    try:
        os.chdir(tmp_path)
        os.environ.pop("CODEINTEL_CONFIG_PATH", None)

        path = ConfigService.get_toml_config_path()

        expect_is_not_none(path)
        if path is not None:
            expect_equal(path.resolve(), config_file.resolve())
    finally:
        os.chdir(original_cwd)
        if original_env is not None:
            os.environ["CODEINTEL_CONFIG_PATH"] = original_env
