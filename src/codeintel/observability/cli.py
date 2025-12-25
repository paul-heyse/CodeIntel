"""CLI telemetry wrapper for Cyclopts-based invocations."""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args, get_origin

from cyclopts import App
from cyclopts.exceptions import CycloptsError

from codeintel.cli.errors import (
    CliError,
    OutputFormat,
    StorageConnectionError,
    StorageError,
    StorageQueryError,
    StorageSchemaError,
    StructuredCliError,
    handle_cli_error,
)
from codeintel.cli.execution.bootstrap import bootstrap_cli
from codeintel.cli.resolution.errors import ResolutionError
from codeintel.core.errors.base import CodeIntelError
from codeintel.core.runtime.loader import load_runtime_settings
from codeintel.observability.otel import (
    ObservabilityRuntime,
    get_observability,
    shutdown_observability,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager
    from inspect import BoundArguments

    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Span, Tracer

log = logging.getLogger(__name__)
_CLI_CAPTURE_MODE_NAMES_ONLY = "names-only"
_CLI_CAPTURE_MODE_ALLOWLIST = "allowlist"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Invocation context injected into Cyclopts commands."""

    invocation_id: str
    command_chain: tuple[str, ...]
    start_ns: int
    logger: logging.Logger
    tracer: Tracer | None
    meter: Meter | None


@dataclass(slots=True)
class _InvocationState:
    invocation_id: str
    start_ns: int
    start_ts: float
    command_chain: tuple[str, ...] = ()
    arg_names: tuple[str, ...] = ()
    is_parse_error: bool = False
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class _InvocationOptions:
    output_format: OutputFormat
    cli_enabled: bool
    arg_capture_mode: str
    arg_allowlist: set[str]


def run_cli_with_telemetry(
    app: App,
    *,
    output_format: OutputFormat,
    argv: Sequence[str] | None = None,
) -> None:
    """Run a Cyclopts app with a unified telemetry wrapper.

    Raises
    ------
    SystemExit
        Raised with the resolved CLI exit code.
    """
    state = _InvocationState(
        invocation_id=uuid.uuid4().hex,
        start_ns=time.time_ns(),
        start_ts=time.perf_counter(),
    )
    settings = load_runtime_settings().observability
    options = _InvocationOptions(
        output_format=output_format,
        cli_enabled=settings.cli_enabled,
        arg_capture_mode=_normalize_capture_mode(settings.cli_args_capture_mode),
        arg_allowlist=_normalize_allowlist(settings.cli_args_allowlist),
    )
    obs = get_observability()
    span_cm = _span_context(obs) if options.cli_enabled else nullcontext(None)
    exit_code = 1
    span: Span | None = None
    try:
        with span_cm as active_span:
            span = active_span
            exit_code = _execute_invocation(
                app,
                argv=argv,
                span=span,
                state=state,
                options=options,
            )
            _set_span_context(
                span,
                invocation_id=state.invocation_id,
                command_chain=state.command_chain,
                arg_names=state.arg_names,
            )
    finally:
        _finalize_span(span, state=state, exit_code=exit_code)
        shutdown_observability()
    raise SystemExit(exit_code)


def _execute_invocation(
    app: App,
    *,
    argv: Sequence[str] | None,
    span: Span | None,
    state: _InvocationState,
    options: _InvocationOptions,
) -> int:
    try:
        command, bound, ignored = app.parse_args(
            argv,
            exit_on_error=False,
            print_error=options.output_format == OutputFormat.TEXT,
        )
        bound_args: BoundArguments = bound
        state.command_chain = _command_chain(command)
        if options.cli_enabled:
            run_ctx = _build_run_context(state)
            _inject_run_context(bound_args, ignored, run_ctx)
            state.arg_names = _safe_arg_names(
                bound_args,
                ignored,
                capture_mode=options.arg_capture_mode,
                allowlist=options.arg_allowlist,
                enabled=True,
            )
        else:
            state.arg_names = ()
        verbosity = _resolve_verbosity(bound_args)
        bootstrap_cli(verbosity=verbosity)
        result = _invoke_command(command, bound_args)
        return _default_exit_code_from_result(result)
    except SystemExit as exc:
        state.error_type = type(exc).__name__
        return _exit_code_from_system_exit(exc)
    except CycloptsError as exc:
        state.is_parse_error = True
        state.error_type = type(exc).__name__
        if span is not None:
            span.record_exception(exc)
        return handle_cli_error(exc, sys.stderr, output_format=options.output_format)
    except (
        CliError,
        StructuredCliError,
        ResolutionError,
        StorageConnectionError,
        StorageError,
        StorageQueryError,
        StorageSchemaError,
        CodeIntelError,
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        state.error_type = type(exc).__name__
        if span is not None:
            span.record_exception(exc)
        return handle_cli_error(exc, sys.stderr, output_format=options.output_format)


def _invoke_command(command: Callable[..., object], bound: BoundArguments) -> object:
    args = getattr(bound, "args", None)
    kwargs = getattr(bound, "kwargs", None)
    if args is not None and kwargs is not None:
        return command(*args, **kwargs)
    arguments = getattr(bound, "arguments", {})
    return command(**arguments)


def _set_span_context(
    span: Span | None,
    *,
    invocation_id: str,
    command_chain: tuple[str, ...],
    arg_names: tuple[str, ...],
) -> None:
    if span is None:
        return
    span.set_attribute("cli.invocation_id", invocation_id)
    span.set_attribute("cli.command", ".".join(command_chain) if command_chain else "<unknown>")
    span.set_attribute("cli.arg_count", len(arg_names))
    if arg_names:
        span.set_attribute("cli.arg_names", [*arg_names])


def _finalize_span(span: Span | None, *, state: _InvocationState, exit_code: int) -> None:
    if span is None:
        return
    duration_ms = (time.perf_counter() - state.start_ts) * 1000
    span.set_attribute("cli.exit_code", exit_code)
    span.set_attribute("cli.duration_ms", duration_ms)
    span.set_attribute("cli.is_parse_error", state.is_parse_error)
    if state.error_type is not None:
        span.set_attribute("cli.error_type", state.error_type)


def _resolve_verbosity(bound: BoundArguments) -> int:
    arguments = getattr(bound, "arguments", {})
    flags = arguments.get("flags")
    if flags is not None and hasattr(flags, "verbose"):
        verbose = flags.verbose
        if isinstance(verbose, int):
            return verbose
    verbose = arguments.get("verbose")
    if isinstance(verbose, int):
        return verbose
    return 0


def _safe_arg_names(
    bound: BoundArguments,
    ignored: Mapping[str, object] | None,
    *,
    capture_mode: str,
    allowlist: set[str],
    enabled: bool,
) -> tuple[str, ...]:
    if not enabled:
        return ()
    arguments = getattr(bound, "arguments", {})
    ignored_names = set(ignored or {})
    names = [name for name in arguments if name not in ignored_names]
    if capture_mode == _CLI_CAPTURE_MODE_ALLOWLIST:
        return tuple(name for name in names if name in allowlist)
    return tuple(names)


def _inject_run_context(
    bound: BoundArguments,
    ignored: Mapping[str, object] | None,
    ctx: RunContext,
) -> None:
    if not ignored:
        return
    arguments = getattr(bound, "arguments", None)
    if arguments is None:
        return
    for name, annotation in ignored.items():
        if _is_run_context_annotation(annotation):
            arguments[name] = ctx
    flags = arguments.get("flags")
    if flags is not None and hasattr(flags, "run_context"):
        try:
            arguments["flags"] = dataclasses.replace(flags, run_context=ctx)
        except TypeError:
            log.debug("Unable to replace flags for run_context injection", exc_info=True)


def _is_run_context_annotation(annotation: object) -> bool:
    if annotation is RunContext:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    args = get_args(annotation)
    return bool(args) and args[0] is RunContext


def _command_chain(command: object) -> tuple[str, ...]:
    name = getattr(command, "__qualname__", None) or getattr(command, "__name__", None)
    if name is None:
        return (type(command).__name__,)
    return tuple(part for part in name.split(".") if part)


def _default_exit_code_from_result(result: object) -> int:
    if result is None:
        return 0
    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        return result
    if isinstance(result, str):
        _write_result(result)
        return 0
    _write_result(result)
    return 0


def _write_result(result: object) -> None:
    sys.stdout.write(f"{result}\n")


def _exit_code_from_system_exit(exc: SystemExit) -> int:
    return exc.code if isinstance(exc.code, int) else 1


def _span_context(obs: ObservabilityRuntime) -> AbstractContextManager[Span | None]:
    if obs.enabled and obs.tracer is not None:
        return obs.tracer.start_as_current_span("cli.invocation")
    return nullcontext(None)


def _build_run_context(state: _InvocationState) -> RunContext:
    obs = get_observability()
    return RunContext(
        invocation_id=state.invocation_id,
        command_chain=state.command_chain,
        start_ns=state.start_ns,
        logger=log,
        tracer=obs.tracer if obs.enabled else None,
        meter=obs.meter if obs.enabled else None,
    )


def _normalize_capture_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == _CLI_CAPTURE_MODE_ALLOWLIST:
        return _CLI_CAPTURE_MODE_ALLOWLIST
    return _CLI_CAPTURE_MODE_NAMES_ONLY


def _normalize_allowlist(values: tuple[str, ...]) -> set[str]:
    return {value.strip() for value in values if value.strip()}


__all__ = ["RunContext", "run_cli_with_telemetry"]
