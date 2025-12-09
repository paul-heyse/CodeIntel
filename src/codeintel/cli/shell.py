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
from typing import Any

from codeintel.cli.executor import get_executor
from codeintel.cli.introspection import list_all_operations, search_operations
from codeintel.cli.operation_registry import get_operation_registry

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
    last_result: dict[str, Any] | None = None


class ShellCompleter:
    """Tab completion for shell commands.

    Parameters
    ----------
    session
        Shell session.
    """

    def __init__(self, session: ShellSession) -> None:
        """Initialize completer."""
        self._session = session
        self._operations: list[str] = []
        self._refresh_operations()

    def _refresh_operations(self) -> None:
        """Refresh operation list."""
        registry = get_operation_registry()
        self._operations = [spec.operation_id for spec in registry.list_operations()]

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
            commands = [
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
            matches = [c for c in commands if c.startswith(text)]
        elif parts[0] == "call":
            # Complete operation ID
            matches = [op for op in self._operations if op.startswith(text)]
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
    """

    def __init__(self, session: ShellSession | None = None) -> None:
        """Initialize shell."""
        self._session = session or ShellSession()
        self._completer = ShellCompleter(self._session)
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
                    self._execute_command(line)
            except EOFError:
                print()  # noqa: T201
                break
            except KeyboardInterrupt:
                print()  # noqa: T201

    def _setup_readline(self) -> None:
        """Set up readline for completion and history."""
        if not _READLINE_AVAILABLE or _readline is None:
            return

        _readline.set_completer(self._completer.complete)
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

    def _execute_command(self, line: str) -> None:
        """Execute shell command.

        Parameters
        ----------
        line
            Command line.
        """
        self._session.history.append(line)

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"Parse error: {e}")  # noqa: T201
            return

        if not parts:
            return

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
            handler(args)
        else:
            print(f"Unknown command: {cmd}")  # noqa: T201
            print("Type 'help' for available commands")  # noqa: T201

    def _cmd_call(self, args: list[str]) -> None:
        """Execute operation."""
        if not args:
            print("Usage: call <operation_id> [--param=value ...]")  # noqa: T201
            return

        operation_id = args[0]
        params = self._parse_params(args[1:])

        registry = get_operation_registry()
        spec = registry.get(operation_id)

        if spec is None:
            print(f"Unknown operation: {operation_id}")  # noqa: T201
            return

        executor = get_executor()
        result = executor.execute(spec, params, render=False)

        if result.result.success and result.result.data is not None:
            data = result.result.data
            self._session.last_result = data if isinstance(data, dict) else None
            print(json.dumps(data, indent=2, default=str))  # noqa: T201
        elif result.result.error:
            error_msg = result.result.error.detail if result.result.error else "Unknown error"
            print(f"Error: {error_msg}")  # noqa: T201

    @staticmethod
    def _cmd_list(args: list[str]) -> None:
        """List operations."""
        _ = args  # Unused
        operations = list_all_operations()
        for info in sorted(operations, key=lambda x: x.operation_id):
            print(f"  {info.operation_id:30} {info.description}")  # noqa: T201

    @staticmethod
    def _cmd_search(args: list[str]) -> None:
        """Search operations."""
        if not args:
            print("Usage: search <query>")  # noqa: T201
            return

        query = " ".join(args)
        results = search_operations(query)
        if not results:
            print(f"No operations matching: {query}")  # noqa: T201
            return

        for info in results:
            print(f"  {info.operation_id}: {info.description}")  # noqa: T201

    @staticmethod
    def _cmd_help(args: list[str]) -> None:
        """Show help."""
        _ = args  # Unused
        print("Commands:")  # noqa: T201
        print("  call <operation> [params]  Execute operation")  # noqa: T201
        print("  list                       List all operations")  # noqa: T201
        print("  search <query>             Search operations")  # noqa: T201
        print("  set <name> <value>         Set session variable")  # noqa: T201
        print("  get <name>                 Get session variable")  # noqa: T201
        print("  history                    Show command history")  # noqa: T201
        print("  export [file]              Export session as script")  # noqa: T201
        print("  help                       Show this help")  # noqa: T201
        print("  quit                       Exit shell")  # noqa: T201

    def _cmd_history(self, args: list[str]) -> None:
        """Show history."""
        _ = args  # Unused
        for i, cmd in enumerate(self._session.history[-20:], 1):
            print(f"  {i:3d}  {cmd}")  # noqa: T201

    def _cmd_set(self, args: list[str]) -> None:
        """Set session variable."""
        if len(args) < 2:  # noqa: PLR2004
            print("Usage: set <name> <value>")  # noqa: T201
            return

        name = args[0]
        value_str = " ".join(args[1:])

        # Try to parse as JSON
        try:
            value: Any = json.loads(value_str)
        except json.JSONDecodeError:
            value = value_str

        self._session.variables[name] = value
        print(f"Set {name} = {value!r}")  # noqa: T201

    def _cmd_get(self, args: list[str]) -> None:
        """Get session variable."""
        if not args:
            # Show all variables
            for name, value in self._session.variables.items():
                print(f"  {name} = {value!r}")  # noqa: T201
            return

        name = args[0]
        value = self._session.variables.get(name)
        if value is not None:
            print(f"{name} = {value!r}")  # noqa: T201
        else:
            print(f"Variable not set: {name}")  # noqa: T201

    def _cmd_export(self, args: list[str]) -> None:
        """Export session as script."""
        lines = ["#!/usr/bin/env bash", "# Exported from codeintel shell", ""]

        for cmd in self._session.history:
            if cmd.startswith("call "):
                parts = shlex.split(cmd)
                if len(parts) >= 2:  # noqa: PLR2004
                    lines.append(f"codeintel op call {' '.join(parts[1:])}")

        script = "\n".join(lines)

        if args:
            path = Path(args[0])
            path.write_text(script, encoding="utf-8")
            print(f"Exported to {path}")  # noqa: T201
        else:
            print(script)  # noqa: T201

    def _cmd_quit(self, args: list[str]) -> None:
        """Exit shell."""
        _ = args  # Unused
        self._running = False

    def _parse_params(self, args: list[str]) -> dict[str, Any]:
        """Parse parameters from command line.

        Parameters
        ----------
        args
            Parameter arguments.

        Returns
        -------
        dict[str, Any]
            Parsed parameters.
        """
        params: dict[str, Any] = {}

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

        # Substitute session variables
        for key, value in list(params.items()):
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in self._session.variables:
                    params[key] = self._session.variables[var_name]

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
