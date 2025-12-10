"""Interactive shell infrastructure.

This package provides:

- ``InteractiveShell``: Full interactive shell experience
- ``ShellSession``: Session management
- ``ShellCompleter``: Command completion in shell mode
"""

from __future__ import annotations

from codeintel.cli.shell._shell import (
    InteractiveShell,
    ShellCompleter,
    ShellSession,
    start_shell,
)

__all__ = [
    "InteractiveShell",
    "ShellCompleter",
    "ShellSession",
    "start_shell",
]
