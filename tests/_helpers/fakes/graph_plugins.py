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
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.core.plugins.types.result import PluginResult
from codeintel.graphs.core.protocol import (
    GraphPluginMetadata,
)
from codeintel.graphs.core.registry import (
    get_graph_registry,
    register_graph_plugin,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from codeintel.core.plugins.types.protocol import PluginResourceHints, PluginSeverity
    from codeintel.graphs.core.context import GraphPluginExecutionContext
    from codeintel.graphs.core.protocol import (
        GraphPluginKind,
        GraphPluginProtocol,
        GraphPluginStage,
    )


@dataclass
class FakeGraphPlugin:
    """Simple fake plugin implementing GraphPluginProtocol.

    This is a test double for creating graph plugins in tests without
    depending on production plugin infrastructure.

    Attributes
    ----------
    _metadata
        Plugin metadata.
    _execute_fn
        Function that performs the plugin's work.
    """

    _metadata: GraphPluginMetadata
    _execute_fn: Callable[[GraphPluginExecutionContext], PluginResult]

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata describing the plugin.
        """
        return self._metadata

    def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        PluginResult
            Result from the execute function.
        """
        return self._execute_fn(ctx)


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
    resource_hints : PluginResourceHints | None
        Optional resource hints attached to metadata.
    options_default : object | None
        Default options included in metadata.
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
    input_hash : str | None
        Input hash to return on success.
    options_hash : str | None
        Options hash to return on success.
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
    resource_hints: PluginResourceHints | None = None
    options_default: object | None = None
    succeed: bool = True
    row_counts: dict[str, int] | None = None
    exception_type: type[Exception] | None = None
    exception_message: str = "Test exception"
    delay_ms: int = 0
    input_hash: str | None = None
    options_hash: str | None = None
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

    def with_resource_hints(self, hints: PluginResourceHints | None) -> GraphPluginBuilder:
        """Set resource hints for metadata.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.resource_hints = hints
        return self

    def with_options_default(self, options: object | None) -> GraphPluginBuilder:
        """Set default options for metadata.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.options_default = options
        return self

    def with_input_hash(self, input_hash: str | None) -> GraphPluginBuilder:
        """Set input hash returned on success.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.input_hash = input_hash
        return self

    def with_options_hash(self, options_hash: str | None) -> GraphPluginBuilder:
        """Set options hash returned on success.

        Returns
        -------
        GraphPluginBuilder
            Self for chaining.
        """
        self.options_hash = options_hash
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
        result_input_hash = self.input_hash
        result_options_hash = self.options_hash

        def execute(_ctx: GraphPluginExecutionContext) -> PluginResult:
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            if exception_type is not None:
                msg = f"{exception_message} from {plugin_name}"
                raise exception_type(msg)
            if succeed:
                return PluginResult.ok(
                    row_counts=row_counts,
                    input_hash=result_input_hash,
                    options_hash=result_options_hash,
                )
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
            resource_hints=self.resource_hints,
            options_default=self.options_default,
        )

        return FakeGraphPlugin(_metadata=metadata, _execute_fn=execute)


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
        if registry.contains(plugin.metadata.name):
            registry.unregister(plugin.metadata.name)
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


def make_graph_plugin(
    name: str,
    *,
    prefix: str = "",
    metadata: Mapping[str, object] | None = None,
    runtime: Mapping[str, object] | None = None,
) -> GraphPluginProtocol:
    """Create a configurable graph plugin using the fluent builder.

    This helper centralizes common plugin setup for tests to avoid
    bespoke builders in individual modules.

    Parameters
    ----------
    name
        Plugin name (without prefix).
    prefix
        Optional prefix to prepend to the name (useful for test isolation).
    metadata
        Optional mapping of metadata attributes (kind, stage, depends_on,
        provides, requires, produces_tables, options_default, severity,
        resource_hints).
    runtime
        Optional mapping of runtime behaviors (delay_ms, input_hash,
        options_hash, row_counts, succeed, exception_type).

    Returns
    -------
    GraphPluginProtocol
        Configured plugin instance.
    """
    metadata = metadata or {}
    runtime = runtime or {}
    builder = GraphPluginBuilder(name=f"{prefix}{name}")

    metadata_handlers: list[
        tuple[object, Callable[[GraphPluginBuilder, object], GraphPluginBuilder]]
    ] = [
        (metadata.get("kind"), lambda b, v: b.with_kind(cast("GraphPluginKind", v))),
        (
            metadata.get("stage"),
            lambda b, v: b.with_stage(cast("GraphPluginStage", v)),
        ),
        (
            metadata.get("depends_on"),
            lambda b, v: b.with_dependencies(*cast("tuple[str, ...]", v)),
        ),
        (
            metadata.get("provides"),
            lambda b, v: b.with_provides(*cast("tuple[str, ...]", v)),
        ),
        (
            metadata.get("requires"),
            lambda b, v: b.with_requires(*cast("tuple[str, ...]", v)),
        ),
        (
            metadata.get("produces_tables"),
            lambda b, v: b.with_produces_tables(*cast("tuple[str, ...]", v)),
        ),
        (
            metadata.get("options_default"),
            lambda b, v: b.with_options_default(v),
        ),
        (
            metadata.get("severity"),
            lambda b, v: b.with_severity(cast("PluginSeverity", v)),
        ),
        (
            metadata.get("resource_hints"),
            lambda b, v: b.with_resource_hints(cast("PluginResourceHints | None", v)),
        ),
    ]

    for value, handler in metadata_handlers:
        if value:
            builder = handler(builder, value)

    runtime_handlers: list[
        tuple[object, Callable[[GraphPluginBuilder, object], GraphPluginBuilder]]
    ] = [
        (runtime.get("delay_ms"), lambda b, v: b.with_delay(cast("int", v))),
        (runtime.get("input_hash"), lambda b, v: b.with_input_hash(cast("str", v))),
        (runtime.get("options_hash"), lambda b, v: b.with_options_hash(cast("str", v))),
        (runtime.get("row_counts"), lambda b, v: b.with_row_counts(cast("dict[str, int]", v))),
        (
            runtime.get("exception_type"),
            lambda b, v: b.raising(
                cast("type[Exception]", v),
                str(runtime.get("exception_message"))
                if runtime.get("exception_message")
                else "Test exception",
            ),
        ),
    ]

    for value, handler in runtime_handlers:
        if value:
            builder = handler(builder, value)

    succeed = runtime.get("succeed")
    if isinstance(succeed, bool) and not succeed:
        error_message = runtime.get("error_message")
        builder = builder.failing(str(error_message) if error_message else "Plugin failed")

    return builder.build()


__all__ = [
    "FakeGraphPlugin",
    "GraphPluginBuilder",
    "make_functional_plugin",
    "make_graph_plugin",
    "plugin_registrar",
]
