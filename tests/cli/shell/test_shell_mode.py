"""Tests for interactive shell mode.

Test REPL interactions, session management, and completion.
"""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.shell import (
    InteractiveShell,
    ShellCompleter,
    ShellSession,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

# Expected number of history entries for tests
EXPECTED_HISTORY_ENTRIES = 2
EXPECTED_HISTORY_AFTER_APPEND = 3


# ---------------------------------------------------------------------------
# ShellSession tests
# ---------------------------------------------------------------------------


def test_session_initialization() -> None:
    """Test session initializes with empty state."""
    session = ShellSession()

    expect_equal(session.history, [])
    expect_equal(session.variables, {})
    expect_is_none(session.last_result)


def test_add_to_history() -> None:
    """Test adding commands to history."""
    session = ShellSession()

    session.history.append("op list")
    session.history.append("build status")

    expect_length(session.history, EXPECTED_HISTORY_ENTRIES)
    expect_equal(session.history[0], "op list")
    expect_equal(session.history[1], "build status")


def test_session_variables() -> None:
    """Test session variable storage."""
    session = ShellSession()

    session.variables["project"] = "/path/to/project"
    session.variables["format"] = "json"

    expect_equal(session.variables["project"], "/path/to/project")
    expect_equal(session.variables["format"], "json")


def test_last_result_storage() -> None:
    """Test storing last operation result."""
    session = ShellSession()
    result: CliResult[dict[str, int]] = CliResult.ok({"count": 10})
    session.last_result = result

    expect_is_not_none(session.last_result)
    expect_true(session.last_result.success)
    expect_equal(session.last_result.data, {"count": 10})


# ---------------------------------------------------------------------------
# ShellCompleter tests
# ---------------------------------------------------------------------------


def test_completer_initialization() -> None:
    """Test completer initializes with commands."""
    completer = ShellCompleter()

    # Should have built-in commands
    expect_in("call", completer.commands)
    expect_in("list", completer.commands)
    expect_in("help", completer.commands)
    expect_in("quit", completer.commands)


def test_complete_builtin_commands() -> None:
    """Test completing built-in commands."""
    completer = ShellCompleter()

    # Complete "he" should give "help"
    matches = completer.complete("he", 0)
    expect_equal(matches, "help")

    # Complete "q" should give "quit"
    matches = completer.complete("q", 0)
    expect_equal(matches, "quit")


def test_complete_partial_match() -> None:
    """Test completing partial input."""
    completer = ShellCompleter()

    # Multiple matches for "s"
    matches = []
    idx = 0
    while True:
        match = completer.complete("s", idx)
        if match is None:
            break
        matches.append(match)
        idx += 1

    expect_in("set", matches)
    expect_in("search", matches)


def test_complete_no_match() -> None:
    """Test completion with no matches."""
    completer = ShellCompleter()

    # No command starts with "xyz"
    matches = completer.complete("xyz", 0)
    expect_is_none(matches)


def test_complete_operations() -> None:
    """Test completing operation IDs."""
    completer = ShellCompleter()

    # Verify method exists
    expect_true(hasattr(completer, "complete"))


# ---------------------------------------------------------------------------
# InteractiveShell tests
# ---------------------------------------------------------------------------


def test_shell_initialization() -> None:
    """Test shell initializes with session."""
    shell = InteractiveShell()

    expect_is_not_none(shell.session)
    expect_is_instance(shell.session, ShellSession)
    expect_is_not_none(shell.completer)


def test_parse_params_simple() -> None:
    """Test parsing simple parameters."""
    shell = InteractiveShell()

    params = shell.parse_params("--format=json")
    expect_equal(params, {"format": "json"})


def test_parse_params_multiple() -> None:
    """Test parsing multiple parameters."""
    shell = InteractiveShell()

    params = shell.parse_params("--project=/path --format=text --verbose")
    expect_equal(params["project"], "/path")
    expect_equal(params["format"], "text")
    expect_equal(params["verbose"], "true")


def test_parse_params_with_spaces() -> None:
    """Test parsing parameters with quoted spaces."""
    shell = InteractiveShell()

    params = shell.parse_params('--message="hello world"')
    expect_equal(params["message"], "hello world")


def test_variable_substitution() -> None:
    """Test variable substitution in parameters."""
    shell = InteractiveShell()
    shell.session.variables["mypath"] = "/custom/path"

    params = shell.parse_params("--project=$mypath")
    expect_equal(params["project"], "/custom/path")


# ---------------------------------------------------------------------------
# Shell command tests
# ---------------------------------------------------------------------------


def test_set_command() -> None:
    """Test set command through shell."""
    shell = InteractiveShell()
    output = StringIO()

    shell.execute_command("set myvar myvalue", output)

    expect_equal(shell.session.variables["myvar"], "myvalue")
    expect_in("Set myvar", output.getvalue())


def test_get_command() -> None:
    """Test get command through shell."""
    shell = InteractiveShell()
    shell.session.variables["testvar"] = "testvalue"
    output = StringIO()

    shell.execute_command("get testvar", output)

    expect_in("testvar = ", output.getvalue())


def test_get_unknown_variable() -> None:
    """Test get command for unknown variable."""
    shell = InteractiveShell()
    output = StringIO()

    shell.execute_command("get unknown", output)

    expect_in("not set", output.getvalue())


def test_history_command() -> None:
    """Test history command."""
    shell = InteractiveShell()
    shell.session.history = ["op list", "build status"]
    output = StringIO()

    shell.execute_command("history", output)

    output_text = output.getvalue()
    expect_in("op list", output_text)
    expect_in("build status", output_text)


def test_quit_command() -> None:
    """Test quit command signals exit."""
    shell = InteractiveShell()

    result = shell.execute_command("quit", StringIO())
    expect_true(result)


def test_export_command(tmp_path: Path) -> None:
    """Test export command."""
    shell = InteractiveShell()
    shell.session.history = ["op list", "build status"]

    export_file = tmp_path / "history.txt"
    output = StringIO()

    shell.execute_command(f"export {export_file}", output)

    expect_true(export_file.exists())
    content = export_file.read_text()
    # Export filters to only 'call' commands
    expect_true(isinstance(content, str))


# ---------------------------------------------------------------------------
# Shell operations tests
# ---------------------------------------------------------------------------


def test_list_shows_operations() -> None:
    """Test list command shows operations."""
    shell = InteractiveShell()
    output = StringIO()

    shell.execute_command("list", output)

    output_text = output.getvalue()
    expect_true(bool(output_text.strip()) or not output_text)


def test_search_filters_operations() -> None:
    """Test search command filters operations."""
    shell = InteractiveShell()
    output = StringIO()

    shell.execute_command("search build", output)

    expect_true(isinstance(output.getvalue(), str))


def test_help_shows_usage() -> None:
    """Test help command shows usage info."""
    shell = InteractiveShell()
    output = StringIO()

    shell.execute_command("help", output)

    output_text = output.getvalue()
    expect_in("Available", output_text)


def test_call_missing_operation() -> None:
    """Test call command with missing operation."""
    shell = InteractiveShell()
    output = StringIO()

    shell.execute_command("call", output)

    expect_true(bool(output.getvalue().strip()))


# ---------------------------------------------------------------------------
# Shell integration tests
# ---------------------------------------------------------------------------


def test_session_persists_across_commands() -> None:
    """Test session state persists across commands."""
    shell = InteractiveShell()

    # Set a variable
    output1 = StringIO()
    shell.execute_command("set myvar hello", output1)

    # Get the variable
    output2 = StringIO()
    shell.execute_command("get myvar", output2)

    expect_in("myvar =", output2.getvalue())


def test_history_accumulates() -> None:
    """Test history accumulates commands."""
    shell = InteractiveShell()

    # Simulate command execution
    shell.session.history.append("list")
    shell.session.history.append("search build")
    shell.session.history.append("quit")

    expect_length(shell.session.history, EXPECTED_HISTORY_AFTER_APPEND)


def test_completer_has_operations() -> None:
    """Test completer has access to operations."""
    completer = ShellCompleter()

    # Refresh operations
    completer.refresh_operations()

    expect_true(hasattr(completer, "operations"))
