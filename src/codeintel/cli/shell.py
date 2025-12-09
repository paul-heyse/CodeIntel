"""Interactive shell for CLI operations.

Provide REPL-style interaction with the CLI, maintaining session state
and providing rich completion for exploratory usage.
"""

from __future__ import annotations

import contextlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from codeintel.cli.execution import get_executor
from codeintel.cli.introspection import list_all_operations, search_operations
from codeintel.cli.operation_registry import get_operation_registry

if TYPE_CHECKING:
    from typing import TextIO

# Optional readline import for tab completion
try:
    import readline as _readline

    _READLINE_AVAILABLE = True
except ImportError:
    _readline = None  # type: ignore[assignment]
    _READLINE_AVAILABLE = False


@dataclass
class ShellSession:
    """Interactive shell session state.

    Parameters
    ----------
    history
        Command history.
    variables
        Session variables.
    last_result
        Last operation result.
    """

    history: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    last_result: Any = None


class ShellCompleter:
    """Tab completion for shell commands.

    Parameters
    ----------
    session
        Shell session (optional).

    Attributes
    ----------
    commands
        List of available shell commands.
    operations
        List of registered operation IDs.
    """

    #: Available shell commands
    commands: ClassVar[list[str]] = [
        "call",
        "list",
        "search",
        "help",
        "history",
        "set",
        "get",
        "export",
        "quit",
        "exit",
    ]

    def __init__(self, session: ShellSession | None = None) -> None:
        """Initialize completer."""
        self._session = session or ShellSession()
        self.operations: list[str] = []
        self.refresh_operations()

    def refresh_operations(self) -> None:
        """Refresh operation list from registry."""
        registry = get_operation_registry()
        self.operations = [spec.operation_id for spec in registry.list_operations()]

    def complete(self, text: str, state: int) -> str | None:
        """Complete text.

        Parameters
        ----------
        text
            Text to complete.
        state
            Completion state.

        Returns
        -------
        str | None
            Completion or None.
        """
        buffer = _readline.get_line_buffer() if _READLINE_AVAILABLE and _readline else ""
        parts = buffer.split()

        if not parts or (len(parts) == 1 and not buffer.endswith(" ")):
            # Complete command
            matches = [c for c in self.commands if c.startswith(text)]
        elif parts[0] == "call":
            # Complete operation ID
            matches = [op for op in self.operations if op.startswith(text)]
        else:
            matches = []

        if state < len(matches):
            return matches[state]
        return None


class InteractiveShell:
    """Interactive CLI shell.

    Parameters
    ----------
    session
        Shell session state.

    Attributes
    ----------
    session
        Shell session state.
    completer
        Tab completion provider.
    """

    def __init__(self, session: ShellSession | None = None) -> None:
        """Initialize shell."""
        self.session = session or ShellSession()
        self.completer = ShellCompleter(self.session)
        self._running = False

    def run(self) -> None:
        """Run interactive shell."""
        self._setup_readline()
        self._print_banner()
        self._running = True

        while self._running:
            try:
                line = input("codeintel> ").strip()
                if line:
                    should_exit = self.execute_command(line)
                    if should_exit:
                        break
            except EOFError:
                print()  # noqa: T201
                break
            except KeyboardInterrupt:
                print()  # noqa: T201

    def _setup_readline(self) -> None:
        """Set up readline for completion and history."""
        if not _READLINE_AVAILABLE or _readline is None:
            return

        _readline.set_completer(self.completer.complete)
        _readline.parse_and_bind("tab: complete")

        history_file = Path.home() / ".codeintel" / "shell_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            _readline.read_history_file(history_file)

    @staticmethod
    def _print_banner() -> None:
        """Print welcome banner."""
        print("CodeIntel Interactive Shell")  # noqa: T201
        print("Type 'help' for commands, 'quit' to exit")  # noqa: T201
        print()  # noqa: T201

    def execute_command(self, line: str, output: TextIO | None = None) -> bool:
        """Execute shell command.

        Parameters
        ----------
        line
            Command line to execute.
        output
            Optional output stream (defaults to stdout).

        Returns
        -------
        bool
            True if shell should exit, False otherwise.
        """
        self.session.history.append(line)

        try:
            parts = shlex.split(line)
        except ValueError as e:
            self._print(f"Parse error: {e}", output)
            return False

        if not parts:
            return False

        cmd = parts[0]
        args = parts[1:]

        handlers = {
            "call": self._cmd_call,
            "list": self._cmd_list,
            "search": self._cmd_search,
            "help": self._cmd_help,
            "history": self._cmd_history,
            "set": self._cmd_set,
            "get": self._cmd_get,
            "export": self._cmd_export,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            return handler(args, output)
        self._print(f"Unknown command: {cmd}", output)
        self._print("Type 'help' for available commands", output)
        return False

    @staticmethod
    def _print(msg: str, output: TextIO | None = None) -> None:
        """Print message to output stream.

        Parameters
        ----------
        msg
            Message to print.
        output
            Optional output stream.
        """
        if output is not None:
            output.write(msg + "\n")
        else:
            print(msg)  # noqa: T201

    def _cmd_call(self, args: list[str], output: TextIO | None = None) -> bool:
        """Execute operation.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        if not args:
            self._print("Usage: call <operation_id> [--param=value ...]", output)
            return False

        operation_id = args[0]
        params = self.parse_params(" ".join(args[1:]))

        registry = get_operation_registry()
        spec = registry.get(operation_id)

        if spec is None:
            self._print(f"Unknown operation: {operation_id}", output)
            return False

        executor = get_executor()
        result = executor.execute(spec, params, render=False)

        if result.result.success and result.result.data is not None:
            data = result.result.data
            self.session.last_result = data if isinstance(data, dict) else None
            self._print(json.dumps(data, indent=2, default=str), output)
        elif result.result.error:
            error_msg = result.result.error.detail if result.result.error else "Unknown error"
            self._print(f"Error: {error_msg}", output)
        return False

    def _cmd_list(self, args: list[str], output: TextIO | None = None) -> bool:
        """List operations.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        _ = args  # Unused
        operations = list_all_operations()
        for info in sorted(operations, key=lambda x: x.operation_id):
            self._print(f"  {info.operation_id:30} {info.description}", output)
        return False

    def _cmd_search(self, args: list[str], output: TextIO | None = None) -> bool:
        """Search operations.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        if not args:
            self._print("Usage: search <query>", output)
            return False

        query = " ".join(args)
        results = search_operations(query)
        if not results:
            self._print(f"No operations matching: {query}", output)
            return False

        for info in results:
            self._print(f"  {info.operation_id}: {info.description}", output)
        return False

    def _cmd_help(self, args: list[str], output: TextIO | None = None) -> bool:
        """Show help.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        _ = args  # Unused
        self._print("Available commands:", output)
        self._print("  call <operation> [params]  Execute operation", output)
        self._print("  list                       List all operations", output)
        self._print("  search <query>             Search operations", output)
        self._print("  set <name> <value>         Set session variable", output)
        self._print("  get <name>                 Get session variable", output)
        self._print("  history                    Show command history", output)
        self._print("  export [file]              Export session as script", output)
        self._print("  help                       Show this help", output)
        self._print("  quit                       Exit shell", output)
        return False

    def _cmd_history(self, args: list[str], output: TextIO | None = None) -> bool:
        """Show history.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        _ = args  # Unused
        for i, cmd in enumerate(self.session.history[-20:], 1):
            self._print(f"  {i:3d}  {cmd}", output)
        return False

    def _cmd_set(self, args: list[str], output: TextIO | None = None) -> bool:
        """Set session variable.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        min_set_args = 2
        if len(args) < min_set_args:
            self._print("Usage: set <name> <value>", output)
            return False

        name = args[0]
        value_str = " ".join(args[1:])

        # Try to parse as JSON
        try:
            value: Any = json.loads(value_str)
        except json.JSONDecodeError:
            value = value_str

        self.session.variables[name] = value
        self._print(f"Set {name} = {value!r}", output)
        return False

    def _cmd_get(self, args: list[str], output: TextIO | None = None) -> bool:
        """Get session variable.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        if not args:
            # Show all variables
            for name, value in self.session.variables.items():
                self._print(f"  {name} = {value!r}", output)
            return False

        name = args[0]
        value = self.session.variables.get(name)
        if value is not None:
            self._print(f"{name} = {value!r}", output)
        else:
            self._print(f"Variable '{name}' not set", output)
        return False

    def _cmd_export(self, args: list[str], output: TextIO | None = None) -> bool:
        """Export session as script.

        Returns
        -------
        bool
            Always False (continue shell).
        """
        lines = ["#!/usr/bin/env bash", "# Exported from codeintel shell", ""]

        min_call_parts = 2
        for cmd in self.session.history:
            if cmd.startswith("call "):
                parts = shlex.split(cmd)
                if len(parts) >= min_call_parts:
                    lines.append(f"codeintel op call {' '.join(parts[1:])}")

        script = "\n".join(lines)

        if args:
            path = Path(args[0])
            path.write_text(script, encoding="utf-8")
            self._print(f"Exported to {path}", output)
        else:
            self._print(script, output)
        return False

    def _cmd_quit(self, args: list[str], output: TextIO | None = None) -> bool:
        """Exit shell.

        Returns
        -------
        bool
            Always True (exit shell).
        """
        _ = args  # Unused
        _ = output  # Unused
        self._running = False
        return True

    def parse_params(self, arg_string: str) -> dict[str, Any]:
        """Parse parameters from command line string.

        Parameters
        ----------
        arg_string
            Parameter string (e.g., "--key=value --other=test").

        Returns
        -------
        dict[str, Any]
            Parsed parameters.
        """
        params: dict[str, Any] = {}

        try:
            args = shlex.split(arg_string)
        except ValueError:
            return params

        for arg in args:
            if "=" in arg:
                key, value_str = arg.split("=", 1)
                key = key.lstrip("-")

                # Try JSON parse
                try:
                    value: Any = json.loads(value_str)
                except json.JSONDecodeError:
                    value = value_str

                params[key] = value
            elif arg.startswith("--"):
                # Flag without value
                key = arg.lstrip("-")
                params[key] = "true"

        # Substitute session variables
        for key, value in list(params.items()):
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in self.session.variables:
                    params[key] = self.session.variables[var_name]

        return params


def start_shell() -> None:
    """Start interactive shell."""
    shell = InteractiveShell()
    shell.run()


__all__ = [
    "InteractiveShell",
    "ShellCompleter",
    "ShellSession",
    "start_shell",
]
