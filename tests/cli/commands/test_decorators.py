"""Tests for @cli_command decorator.

Verify the decorator correctly:
- Generates __call__ methods
- Registers operations with the NEW registry
- Extracts parameters from dataclass fields
- Handles output format resolution
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest

from codeintel.cli.commands.decorators import (
    CommandConfig,
    cli_command,
    extract_params,
    get_output_format,
    get_path_field,
)
from codeintel.cli.core import CliResult
from codeintel.cli.execution.registry import get_registry, reset_registry
from codeintel.cli.rendering.types import OutputFormat

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext


def _force_setattr(target: object, name: str, value: object) -> None:
    """Attempt to set an attribute on a frozen dataclass for testing."""
    setattr(target, name, value)


@pytest.fixture(autouse=True)
def _reset_registries() -> None:
    """Reset registry before each test to ensure isolation."""
    reset_registry()


def _dummy_handler(ctx: CommandContext) -> CliResult[dict[str, bool]]:
    """Return success result for testing.

    Parameters
    ----------
    ctx
        Command context.

    Returns
    -------
    CliResult[dict[str, bool]]
        Success result with test data.
    """
    _ = ctx.operation_id
    return CliResult.ok({"test": True})


EXPECTED_COUNT_ZERO = 0
EXPECTED_COUNT_ONE = 1


def _verify_true(*, condition: bool, message: str) -> None:
    """Verify that condition is true.

    Parameters
    ----------
    condition
        Condition to check.
    message
        Error message if condition is false.

    Raises
    ------
    AssertionError
        If condition is false.
    """
    if not condition:
        raise AssertionError(message)


def _verify_false(*, condition: bool, message: str) -> None:
    """Verify that condition is false.

    Parameters
    ----------
    condition
        Condition to check.
    message
        Error message if condition is true.

    Raises
    ------
    AssertionError
        If condition is true.
    """
    if condition:
        raise AssertionError(message)


def _verify_equal(actual: object, expected: object, message: str) -> None:
    """Verify two values are equal.

    Parameters
    ----------
    actual
        Actual value.
    expected
        Expected value.
    message
        Error message prefix.

    Raises
    ------
    AssertionError
        If values are not equal.
    """
    if actual != expected:
        full_msg = f"{message}: expected {expected!r}, got {actual!r}"
        raise AssertionError(full_msg)


class _StrContainer(Protocol):
    def __contains__(self, item: str, /) -> bool: ...


def _verify_in(item: str, container: _StrContainer, message: str) -> None:
    """Verify item is in container.

    Parameters
    ----------
    item
        Item to find.
    container
        Container to search.
    message
        Error message prefix.

    Raises
    ------
    AssertionError
        If item is not in container.
    """
    if item not in container:
        full_msg = f"{message}: {item!r} not in {container!r}"
        raise AssertionError(full_msg)


def _verify_is_not_none(value: object, message: str) -> None:
    """Verify value is not None.

    Parameters
    ----------
    value
        Value to check.
    message
        Error message if value is None.

    Raises
    ------
    AssertionError
        If value is None.
    """
    if value is None:
        raise AssertionError(message)


def test_cli_command_generates_call_method() -> None:
    """Verify decorator generates __call__ method on the class."""
    cfg = CommandConfig(require_runtime=False, require_gateway=False)

    @cli_command("test.generates_call", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """Test command."""

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    cmd = TestCommand()
    _verify_true(
        condition=callable(cmd),
        message="Decorator should generate __call__ method",
    )


def test_cli_command_registers_operation() -> None:
    """Verify decorator registers operation with NEW registry."""
    cfg = CommandConfig(require_runtime=False, require_gateway=False)

    @cli_command("test.registers_op", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """Test command."""

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    _ = TestCommand

    registry = get_registry()
    _verify_in(
        "test.registers_op",
        registry,
        "Operation should be registered with NEW registry",
    )


def test_cli_command_uses_docstring_as_description() -> None:
    """Verify decorator uses class docstring as operation description."""
    cfg = CommandConfig(require_runtime=False, require_gateway=False)

    @cli_command("test.docstring", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """Operation description from docstring."""

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    _ = TestCommand

    registry = get_registry()
    spec = registry.get("test.docstring")
    _verify_is_not_none(spec, "Operation spec should exist")
    _verify_equal(
        spec.description if spec else None,
        "Operation description from docstring.",
        "Description should match docstring",
    )


def test_cli_command_uses_first_line_of_multiline_docstring() -> None:
    """Verify decorator extracts first line from multi-line docstring."""
    cfg = CommandConfig(require_runtime=False, require_gateway=False)

    @cli_command("test.multiline", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """First line only.

        Additional documentation that should be ignored.
        """

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    _ = TestCommand

    registry = get_registry()
    spec = registry.get("test.multiline")
    _verify_is_not_none(spec, "Operation spec should exist")
    _verify_equal(
        spec.description if spec else None,
        "First line only.",
        "Description should be first line only",
    )


def test_cli_command_extracts_group_from_operation_id() -> None:
    """Verify decorator extracts group from operation_id prefix."""
    cfg = CommandConfig(require_runtime=False, require_gateway=False)

    @cli_command("mygroup.myop", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """Test command."""

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    _ = TestCommand

    registry = get_registry()
    spec = registry.get("mygroup.myop")
    _verify_is_not_none(spec, "Operation spec should exist")
    _verify_equal(
        spec.group if spec else None,
        "mygroup",
        "Group should be extracted from operation_id",
    )


def test_cli_command_with_custom_description() -> None:
    """Verify decorator uses explicit description over docstring."""
    cfg = CommandConfig(
        require_runtime=False,
        require_gateway=False,
        description="Custom description",
    )

    @cli_command("test.custom_desc", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """Docstring should be ignored."""

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    _ = TestCommand

    registry = get_registry()
    spec = registry.get("test.custom_desc")
    _verify_is_not_none(spec, "Operation spec should exist")
    _verify_equal(
        spec.description if spec else None,
        "Custom description",
        "Description should use explicit value",
    )


def test_cli_command_sets_resource_requirements() -> None:
    """Verify decorator correctly sets resource requirements from config."""
    cfg = CommandConfig(
        require_runtime=False,
        require_gateway=True,
        require_graph_runtime=True,
    )

    @cli_command("test.resources", handler=_dummy_handler, config=cfg)
    @dataclass
    class TestCommand:
        """Test command."""

        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    _ = TestCommand

    registry = get_registry()
    spec = registry.get("test.resources")
    _verify_is_not_none(spec, "Operation spec should exist")
    if spec:
        _verify_false(
            condition=spec.require_runtime,
            message="require_runtime should be False",
        )
        _verify_true(
            condition=spec.require_gateway,
            message="require_gateway should be True",
        )
        _verify_true(
            condition=spec.require_graph_runtime,
            message="require_graph_runtime should be True",
        )


def test_extract_params_from_dataclass() -> None:
    """Extract non-infrastructure params from dataclass."""

    @dataclass
    class TestCommand:
        name: str = "test"
        count: int = 10
        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0

    cmd = TestCommand(name="custom", count=42)
    params = extract_params(cmd)

    _verify_in("name", params, "name should be in params")
    _verify_in("count", params, "count should be in params")
    _verify_equal(params["name"], "custom", "name value should match")
    _verify_equal(params["count"], 42, "count value should match")


def test_extract_params_excludes_infrastructure_fields() -> None:
    """Infrastructure fields should not be in params."""

    @dataclass
    class TestCommand:
        name: str = "test"
        output_format: OutputFormat = OutputFormat.TEXT
        verbose: int = 0
        project_root: Path | None = None
        db_path: Path | None = None

    cmd = TestCommand()
    params = extract_params(cmd)

    _verify_in("name", params, "name should be in params")
    _verify_false(
        condition="output_format" in params,
        message="output_format should not be in params",
    )
    _verify_false(
        condition="verbose" in params,
        message="verbose should not be in params",
    )
    _verify_false(
        condition="project_root" in params,
        message="project_root should not be in params",
    )
    _verify_false(
        condition="db_path" in params,
        message="db_path should not be in params",
    )


def test_extract_params_from_non_dataclass() -> None:
    """Non-dataclass returns empty params dict."""

    class NotADataclass:
        name = "test"

    obj = NotADataclass()
    params = extract_params(obj)

    _verify_equal(params, {}, "Non-dataclass should return empty dict")


def test_get_output_format_from_field() -> None:
    """Get output format from explicit field."""

    @dataclass
    class TestCommand:
        output_format: OutputFormat = OutputFormat.JSON

    cmd = TestCommand()
    fmt = get_output_format(cmd)
    _verify_equal(fmt, OutputFormat.JSON, "Should get JSON format from field")


def test_get_output_format_from_json_flag() -> None:
    """Get JSON format from --json flag."""

    @dataclass
    class TestCommand:
        json: bool = True

    cmd = TestCommand()
    fmt = get_output_format(cmd)
    _verify_equal(fmt, OutputFormat.JSON, "Should get JSON format from flag")


def test_get_output_format_defaults_to_text() -> None:
    """Default to TEXT format when no format specified."""

    @dataclass
    class TestCommand:
        name: str = "test"

    cmd = TestCommand()
    fmt = get_output_format(cmd)
    _verify_equal(fmt, OutputFormat.TEXT, "Should default to TEXT format")


def test_get_output_format_prefers_field_over_flag() -> None:
    """Explicit output_format field takes precedence over json flag."""

    @dataclass
    class TestCommand:
        output_format: OutputFormat = OutputFormat.JSONL
        json: bool = True

    cmd = TestCommand()
    fmt = get_output_format(cmd)
    _verify_equal(fmt, OutputFormat.JSONL, "output_format should take precedence")


def test_get_path_field_returns_path() -> None:
    """Get Path value from command field."""

    @dataclass
    class TestCommand:
        project: Path | None = Path("/test/path")

    cmd = TestCommand()
    path = get_path_field(cmd, "project")
    _verify_equal(path, Path("/test/path"), "Should return Path value")


def test_get_path_field_converts_string() -> None:
    """Convert string value to Path."""

    @dataclass
    class TestCommand:
        project: str = "/string/path"

    cmd = TestCommand()
    path = get_path_field(cmd, "project")
    _verify_equal(path, Path("/string/path"), "Should convert string to Path")


def test_get_path_field_returns_none_when_missing() -> None:
    """Return None when field doesn't exist."""

    @dataclass
    class TestCommand:
        name: str = "test"

    cmd = TestCommand()
    path = get_path_field(cmd, "project", "project_root")
    _verify_equal(path, None, "Should return None when field missing")


def test_get_path_field_tries_multiple_names() -> None:
    """Try multiple field names in order."""

    @dataclass
    class TestCommand:
        project_root: Path | None = Path("/root/path")

    cmd = TestCommand()

    path = get_path_field(cmd, "project", "project_root")
    _verify_equal(path, Path("/root/path"), "Should find project_root")


def test_get_path_field_returns_none_for_none_value() -> None:
    """Return None when field exists but value is None."""

    @dataclass
    class TestCommand:
        project: Path | None = None

    cmd = TestCommand()
    path = get_path_field(cmd, "project")
    _verify_equal(path, None, "Should return None for None value")


def test_command_config_defaults() -> None:
    """CommandConfig has sensible defaults."""
    cfg = CommandConfig()

    _verify_true(
        condition=cfg.require_runtime,
        message="require_runtime should default to True",
    )
    _verify_true(
        condition=cfg.require_gateway,
        message="require_gateway should default to True",
    )
    _verify_false(
        condition=cfg.require_graph_runtime,
        message="require_graph_runtime should default to False",
    )
    _verify_equal(cfg.description, None, "description should default to None")


def test_command_config_is_frozen() -> None:
    """CommandConfig is immutable."""
    cfg = CommandConfig()

    with pytest.raises(AttributeError):
        _force_setattr(cfg, "require_runtime", value=False)
