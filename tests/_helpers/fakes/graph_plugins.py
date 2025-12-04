"""Test helpers for graph plugin testing.

This module provides helpers for creating and managing graph plugins in tests:

- `GraphPluginBuilder`: Fluent builder for configurable test plugins
- `plugin_registrar`: Context manager for scoped plugin registration
- `make_functional_plugin`: Simple factory for basic test plugins

These helpers eliminate boilerplate setup code and ensure consistent
plugin behavior across tests.

Example
-------
>>> from tests._helpers.fakes.graph_plugins import GraphPluginBuilder, plugin_registrar
>>>
>>> plugin = GraphPluginBuilder(name="test.plugin").with_row_counts({"t": 5}).build()
>>> with plugin_registrar([plugin]):
...     # Plugin is registered during this block
...     pass
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.plugins.protocol import PluginSeverity
from codeintel.core.plugins.result import PluginResult
from codeintel.graphs.core.protocol import (
    FunctionalGraphPlugin,
    GraphPluginKind,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginStage,
)
from codeintel.graphs.core.registry import (
    get_graph_registry,
    register_graph_plugin,
)

if TYPE_CHECKING:
    from codeintel.graphs.core.context import GraphPluginExecutionContext


@dataclass
class GraphPluginBuilder:
    """Fluent builder for creating test graph plugins.

    Provides a configurable builder pattern for creating graph plugins
    with various behaviors for testing different scenarios.

    Attributes
    ----------
    name : str
        Plugin name (required).
    description : str
        Plugin description.
    kind : GraphPluginKind
        Plugin kind (builder, metric, validation).
    stage : GraphPluginStage
        Plugin stage in the pipeline.
    succeed : bool
        Whether execution should succeed.
    row_counts : dict[str, int] | None
        Row counts to return on success.
    exception_type : type[Exception] | None
        Exception type to raise during execution.
    exception_message : str
        Message for raised exception.
    delay_ms : int
        Delay in milliseconds before returning.
    depends_on : tuple[str, ...]
        Plugin dependencies.
    provides : tuple[str, ...]
        Capabilities provided by the plugin.
    requires : tuple[str, ...]
        Capabilities required by the plugin.
    produces_tables : tuple[str, ...]
        Tables populated by the plugin.
    severity : PluginSeverity
        Failure severity level.
    error_message : str
        Error message when succeed=False.
    """

    name: str
    description: str = ""
    kind: GraphPluginKind = "builder"
    stage: GraphPluginStage = "goid"
    succeed: bool = True
    row_counts: dict[str, int] | None = None
    exception_type: type[Exception] | None = None
    exception_message: str = "Test exception"
    delay_ms: int = 0
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    produces_tables: tuple[str, ...] = ()
    severity: PluginSeverity = "fatal"
    error_message: str = "Plugin failed"

    def with_description(self, description: str) -> GraphPluginBuilder:
        """Set plugin description.

        Parameters
        ----------
        description
            Plugin description.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.description = description
        return self

    def with_kind(self, kind: GraphPluginKind) -> GraphPluginBuilder:
        """Set plugin kind.

        Parameters
        ----------
        kind
            Plugin kind.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.kind = kind
        return self

    def with_stage(self, stage: GraphPluginStage) -> GraphPluginBuilder:
        """Set plugin stage.

        Parameters
        ----------
        stage
            Plugin stage.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.stage = stage
        return self

    def succeeding(self) -> GraphPluginBuilder:
        """Configure plugin to succeed.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.succeed = True
        return self

    def failing(self, error_message: str = "Plugin failed") -> GraphPluginBuilder:
        """Configure plugin to fail.

        Parameters
        ----------
        error_message
            Error message to return.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.succeed = False
        self.error_message = error_message
        return self

    def with_row_counts(self, row_counts: dict[str, int]) -> GraphPluginBuilder:
        """Set row counts to return on success.

        Parameters
        ----------
        row_counts
            Mapping of table names to row counts.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.row_counts = row_counts
        return self

    def raising(
        self,
        exception_type: type[Exception],
        message: str = "Test exception",
    ) -> GraphPluginBuilder:
        """Configure plugin to raise an exception.

        Parameters
        ----------
        exception_type
            Exception type to raise.
        message
            Exception message.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.exception_type = exception_type
        self.exception_message = message
        return self

    def with_delay(self, delay_ms: int) -> GraphPluginBuilder:
        """Set execution delay.

        Parameters
        ----------
        delay_ms
            Delay in milliseconds.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.delay_ms = delay_ms
        return self

    def with_dependencies(self, *depends_on: str) -> GraphPluginBuilder:
        """Set plugin dependencies.

        Parameters
        ----------
        depends_on
            Plugin names this plugin depends on.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.depends_on = depends_on
        return self

    def with_provides(self, *provides: str) -> GraphPluginBuilder:
        """Set capabilities provided.

        Parameters
        ----------
        provides
            Capability names.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.provides = provides
        return self

    def with_requires(self, *requires: str) -> GraphPluginBuilder:
        """Set required capabilities.

        Parameters
        ----------
        requires
            Required capability names.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.requires = requires
        return self

    def with_produces_tables(self, *tables: str) -> GraphPluginBuilder:
        """Set produced tables.

        Parameters
        ----------
        tables
            Table names this plugin produces.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.produces_tables = tables
        return self

    def with_severity(self, severity: PluginSeverity) -> GraphPluginBuilder:
        """Set failure severity.

        Parameters
        ----------
        severity
            Severity level.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.severity = severity
        return self

    def build(self) -> GraphPluginProtocol:
        """Build the configured plugin.

        Returns
        -------
        GraphPluginProtocol
            Configured plugin instance.
        """
        # Capture values for closure
        succeed = self.succeed
        row_counts = self.row_counts
        exception_type = self.exception_type
        exception_message = self.exception_message
        delay_ms = self.delay_ms
        error_message = self.error_message
        plugin_name = self.name

        def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            if exception_type is not None:
                msg = f"{exception_message} from {plugin_name}"
                raise exception_type(msg)
            if succeed:
                return PluginResult.ok(row_counts=row_counts)
            return PluginResult.fail(error_message)

        description = self.description or f"Test plugin {self.name}"
        metadata = GraphPluginMetadata(
            name=self.name,
            description=description,
            kind=self.kind,
            stage=self.stage,
            severity=self.severity,
            depends_on=self.depends_on,
            provides=self.provides,
            requires=self.requires,
            produces_tables=self.produces_tables,
        )

        return FunctionalGraphPlugin(_metadata=metadata, _execute_fn=execute)


@contextlib.contextmanager
def plugin_registrar(
    plugins: Sequence[GraphPluginProtocol],
) -> Iterator[None]:
    """Context manager for scoped plugin registration.

    Registers plugins on entry and unregisters them on exit.
    This ensures test plugins don't leak into the global registry.

    Parameters
    ----------
    plugins
        Plugins to register.

    Yields
    ------
    None
        Control returns to the with block.

    Example
    -------
    >>> plugin = GraphPluginBuilder(name="test").build()
    >>> with plugin_registrar([plugin]):
    ...     # Plugin is registered
    ...     pass
    >>> # Plugin is unregistered
    """
    registry = get_graph_registry()

    # Register all plugins
    for plugin in plugins:
        register_graph_plugin(plugin)

    try:
        yield
    finally:
        # Unregister all plugins, suppressing KeyError if already removed
        for plugin in plugins:
            with contextlib.suppress(KeyError):
                registry.unregister(plugin.metadata.name)


def make_functional_plugin(
    name: str,
    *,
    succeed: bool = True,
    row_counts: dict[str, int] | None = None,
    depends_on: tuple[str, ...] = (),
    provides: tuple[str, ...] = (),
) -> GraphPluginProtocol:
    """Create a simple test plugin with minimal configuration.

    This is a convenience function for creating basic test plugins
    without using the full builder pattern. For more configuration options,
    use `GraphPluginBuilder` directly.

    Parameters
    ----------
    name
        Plugin name.
    succeed
        Whether execution should succeed.
    row_counts
        Row counts to return on success.
    depends_on
        Plugin dependencies.
    provides
        Capabilities provided.

    Returns
    -------
    GraphPluginProtocol
        Configured plugin instance.
    """
    builder = GraphPluginBuilder(
        name=name,
        succeed=succeed,
        row_counts=row_counts,
        depends_on=depends_on,
        provides=provides,
    )
    return builder.build()


__all__ = [
    "GraphPluginBuilder",
    "make_functional_plugin",
    "plugin_registrar",
]
