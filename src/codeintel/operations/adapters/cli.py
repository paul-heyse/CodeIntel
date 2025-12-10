"""CLI adapter for operations.

Generates cyclopts commands from the operation registry,
enabling CLI access to all registered operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cyclopts import App

from codeintel.operations.registry import OperationRegistry, get_default_registry

if TYPE_CHECKING:
    from codeintel.operations.base import OperationSpec


LOG = logging.getLogger(__name__)


@dataclass
class CliAdapter:
    """Adapt operations to cyclopts CLI commands.

    The adapter iterates over the operation registry and generates
    CLI commands for each operation, grouping them into sub-apps.

    Parameters
    ----------
    root_app
        Root cyclopts application.
    registry
        Operation registry (defaults to global).
    _group_apps
        Cache of sub-apps by group name.
    """

    root_app: App
    registry: OperationRegistry = field(default_factory=get_default_registry)
    _group_apps: dict[str, App] = field(default_factory=dict)

    def register_all(self) -> None:
        """Register all operations from the registry.

        Creates sub-apps for each operation group and registers
        commands within them.
        """
        for spec in self.registry.list_operations(include_hidden=False):
            self._register_operation(spec)

    def register_group(self, group: str) -> None:
        """Register all operations in a specific group.

        Parameters
        ----------
        group
            Group name (e.g., "jobs", "datasets").
        """
        for spec in self.registry.list_operations(group=group, include_hidden=False):
            self._register_operation(spec)

    def _register_operation(self, spec: OperationSpec) -> None:
        """Register a single operation as a CLI command.

        Parameters
        ----------
        spec
            Operation specification.
        """
        _ = self  # Instance method for adapter pattern
        # Placeholder - full implementation will create commands
        LOG.debug("Would register CLI command: %s", spec.operation_id)


def register_operations_with_app(app: App) -> CliAdapter:
    """Register all operations with a cyclopts app.

    Parameters
    ----------
    app
        Root cyclopts application.

    Returns
    -------
    CliAdapter
        The configured adapter.
    """
    adapter = CliAdapter(app)
    adapter.register_all()
    return adapter


__all__ = [
    "CliAdapter",
    "register_operations_with_app",
]
