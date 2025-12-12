"""Unit-level validation of the scoped help patch helper."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Annotated

import cyclopts.help as help_pkg
import cyclopts.help.help as help_mod
from cyclopts import App, Parameter

from codeintel.cli.commands import help_utils
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_not_equal,
    expect_not_in,
    expect_true,
)

_DISPLAY_DEFAULT_ATTR = "_DisplayDefault"
DISPLAY_DEFAULT_CLS = getattr(help_utils, _DISPLAY_DEFAULT_ATTR)
apply_help_patch = help_utils.apply_help_patch
build_patched_app = help_utils.build_patched_app
create_parameter_help_panel = help_utils.create_parameter_help_panel


@dataclass
class Options:
    """Simple options used to exercise help rendering."""

    flag: Annotated[bool, Parameter(name="--flag", help="Example flag.")] = False
    mode: Annotated[str | None, Parameter(name="--mode", help="Optional mode.")] = None


class Mode(Enum):
    """Enum used to verify choice rendering."""

    FAST = "fast"
    SLOW = "slow"


@dataclass
class NestedOptions:
    """Nested options used to ensure deep defaults are patched."""

    inner: Annotated[str | None, Parameter(name="--inner", help="Inner option.")] = None


@dataclass
class OuterOptions:
    """Wrap nested options to exercise grouped defaults."""

    nested: Annotated[NestedOptions, Parameter(name="*")] = field(default_factory=NestedOptions)
    toggle: Annotated[bool, Parameter(name="--toggle", help="Nested toggle.")] = True


def _make_app() -> App:
    local_app = App()

    @local_app.command
    def cmd(
        options: Annotated[Options | None, Parameter(name="*")] = None,
    ) -> None:
        _ = options or Options()

    @local_app.command
    def positional(
        arg: Annotated[int | None, Parameter(help="Positional argument.")] = None,
    ) -> None:
        _ = arg

    @local_app.command
    def enum_command(
        mode: Annotated[
            Mode, Parameter(name="--mode", help="Enum mode.", show_choices=True)
        ] = Mode.FAST,
    ) -> None:
        _ = mode

    @local_app.command
    def nested(
        outer: Annotated[OuterOptions | None, Parameter(name="*")] = None,
    ) -> None:
        _ = outer or OuterOptions()

    @local_app.command
    def custom_default(
        flag: Annotated[
            bool | None,
            Parameter(
                name="--flag",
                help="Flag with custom show_default.",
                show_default=lambda current: f"custom-default ({current})",
            ),
        ] = None,
    ) -> None:
        _ = False if flag is None else flag

    return local_app


def _render_help(app: App, args: list[str]) -> str:
    stdout = StringIO()
    with redirect_stdout(stdout):
        app(args, result_action="return_value", exit_on_error=False, print_error=False)
    return stdout.getvalue().lower()


def test_patched_app_help_with_missing_metadata() -> None:
    """Patched app should render help without crashing and with readable defaults."""
    app = build_patched_app(_make_app)

    output = _render_help(app, ["cmd", "--help"])
    expect_in("usage", output)
    expect_in("--flag", output)
    expect_in("false", output)
    expect_in("--mode", output)
    expect_in("(none)", output)
    expect_not_in("simplenamespace", output)


def test_help_renders_positional_defaults() -> None:
    """Positional arguments should render readable defaults."""
    app = build_patched_app(_make_app)

    output = _render_help(app, ["positional", "--help"])
    expect_in("positional argument", output)
    expect_in("(none)", output)
    expect_not_in("simplenamespace", output)


def test_help_renders_enum_choices_and_default() -> None:
    """Enums should display choices and defaults clearly."""
    app = build_patched_app(_make_app)

    output = _render_help(app, ["enum-command", "--help"])
    expect_in("--mode", output)
    expect_in("fast", output)
    expect_in("slow", output)
    expect_not_in("simplenamespace", output)


def test_help_renders_nested_defaults() -> None:
    """Nested grouped options should render patched defaults."""
    app = build_patched_app(_make_app)

    output = _render_help(app, ["nested", "--help"])
    expect_in("--inner", output)
    expect_in("(none)", output)
    expect_in("--toggle", output)
    expect_in("true", output)
    expect_not_in("simplenamespace", output)


def test_help_renders_custom_show_default_callable() -> None:
    """Callable show_default outputs should be preserved in help."""
    app = build_patched_app(_make_app)

    output = _render_help(app, ["custom-default", "--help"])
    expect_in("--flag", output)
    expect_in("custom-default", output)
    expect_not_in("simplenamespace", output)


def test_argument_collection_type_preserved() -> None:
    """Patched collection should preserve ArgumentCollection type when possible."""
    app = build_patched_app(_make_app)

    stdout = StringIO()
    with redirect_stdout(stdout):
        app.help_print()

    output = stdout.getvalue().lower()
    expect_in("usage", output)
    expect_not_in("simplenamespace", output)


def test_all_cyclopts_locations_are_patched() -> None:
    """Verify the help patch covers all Cyclopts import locations.

    This test ensures that our patch is applied to all locations where
    Cyclopts imports ``create_parameter_help_panel``, preventing the
    Python import aliasing issue where some code paths would use the
    unpatched original function.
    """
    apply_help_patch()

    expect_true(help_mod.create_parameter_help_panel is create_parameter_help_panel)
    expect_true(help_pkg.create_parameter_help_panel is create_parameter_help_panel)


def test_display_default_repr_is_clean() -> None:
    """Verify _DisplayDefault produces clean repr output.

    The ``_DisplayDefault`` class replaces ``SimpleNamespace`` for rendering
    defaults in help output. It must produce clean repr/str output without
    wrapper text like ``namespace(name='...')``.
    """
    none_default = DISPLAY_DEFAULT_CLS("(none)")

    expect_equal(repr(none_default), "(none)")
    expect_equal(str(none_default), "(none)")
    expect_equal(none_default.name, "(none)")

    expect_false(bool(none_default))

    expect_not_equal(none_default, None)

    expect_not_in(none_default, (None, "empty"))


def test_display_default_equality() -> None:
    """Verify _DisplayDefault equality semantics."""
    dd1 = DISPLAY_DEFAULT_CLS("(none)")
    dd2 = DISPLAY_DEFAULT_CLS("(none)")
    dd3 = DISPLAY_DEFAULT_CLS("other")

    expect_equal(dd1, dd2)

    expect_not_equal(dd1, dd3)

    expect_not_equal(dd1, "some string")
    arbitrary_int = 123
    expect_not_equal(dd1, arbitrary_int)
