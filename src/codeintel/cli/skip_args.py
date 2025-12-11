"""Skip control CLI arguments."""

from __future__ import annotations

from typing import Annotated

import cyclopts

ForceArg = Annotated[
    bool,
    cyclopts.Parameter(
        name=["--force", "-f"],
        help="Force execution, skip nothing",
    ),
]

DryRunArg = Annotated[
    bool,
    cyclopts.Parameter(
        name=["--dry-run", "-n"],
        help="Show what would be executed/skipped without running",
    ),
]


__all__ = ["DryRunArg", "ForceArg"]
