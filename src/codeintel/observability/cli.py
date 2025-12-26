"""CLI telemetry wrapper for Cyclopts-based invocations."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, get_args, get_origin
from weakref import WeakKeyDictionary

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
from codeintel.observability.attribute_sanitizer import limit_cli_arg_names
from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.policy import ObservabilityPolicy
from codeintel.observability.runtime import (
    ObservabilityRuntime,
    get_observability,
    shutdown_observability,
)
from codeintel.observability.semconv_keys import (
    CLI_ARG_COUNT,
    CLI_ARG_NAMES,
    CLI_COMMAND,
    CLI_DURATION_MS,
    CLI_ERROR_TYPE,
    CLI_EXIT_CODE,
    CLI_INVOCATION_ID,
    CLI_IS_PARSE_ERROR,
    CLI_PARSE_DURATION_MS,
)
from codeintel.observability.test_mode import should_shutdown_observability_per_command

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager
    from inspect import BoundArguments

    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span, Tracer

    from codeintel.core.runtime import RuntimeSettings

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
    parse_duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class _InvocationOptions:
    output_format: OutputFormat
    cli_enabled: bool
    arg_capture_mode: str
    arg_allowlist: set[str]


@dataclass(slots=True)
class _CliInstruments:
    invocation_count: Counter
    invocation_duration_ms: Histogram
    parse_duration_ms: Histogram
    parse_errors: Counter


_CLI_INSTRUMENTS: WeakKeyDictionary[Meter, _CliInstruments] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class _InvocationContext:
    state: _InvocationState
    options: _InvocationOptions
    bootstrap: Callable[[int], object]


@dataclass(frozen=True, slots=True)
class RunCliTelemetryDeps:
    """Dependency overrides for CLI telemetry execution."""

    load_settings: Callable[[], RuntimeSettings] = load_runtime_settings
    bootstrap: Callable[[int], object] = bootstrap_cli
    shutdown: Callable[[], object] = shutdown_observability
    get_observability: Callable[[], ObservabilityRuntime] = get_observability


def run_cli_with_telemetry(
    app: App,
    *,
    output_format: OutputFormat,
    argv: Sequence[str] | None = None,
    deps: RunCliTelemetryDeps | None = None,
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
    resolved_deps = deps or RunCliTelemetryDeps()
    settings = resolved_deps.load_settings().observability
    options = _InvocationOptions(
        output_format=output_format,
        cli_enabled=settings.cli_enabled,
        arg_capture_mode=_normalize_capture_mode(settings.cli_args_capture_mode),
        arg_allowlist=normalize_allowlist(settings.cli_args_allowlist),
    )
    obs = resolved_deps.get_observability()
    span_cm = _span_context(obs) if options.cli_enabled else nullcontext(None)
    exit_code = 1
    span: Span | None = None
    try:
        with span_cm as active_span:
            span = active_span
            invocation_ctx = _InvocationContext(
                state=state,
                options=options,
                bootstrap=resolved_deps.bootstrap,
            )
            exit_code = _execute_invocation(
                app,
                argv=argv,
                span=span,
                context=invocation_ctx,
            )
            _set_span_context(
                span,
                invocation_id=state.invocation_id,
                command_chain=state.command_chain,
                arg_names=state.arg_names,
                policy=obs.policy,
            )
    finally:
        _finalize_span(span, state=state, exit_code=exit_code)
        _record_cli_metrics(
            obs,
            state=state,
            exit_code=exit_code,
            enabled=options.cli_enabled,
        )
        _emit_cli_log(state, exit_code=exit_code, enabled=options.cli_enabled)
        if should_shutdown_observability_per_command():
            resolved_deps.shutdown()
    raise SystemExit(exit_code)


def _execute_invocation(
    app: App,
    *,
    argv: Sequence[str] | None,
    span: Span | None,
    context: _InvocationContext,
) -> int:
    parse_start = time.perf_counter()
    try:
        command, bound, ignored = app.parse_args(
            argv,
            exit_on_error=False,
            print_error=context.options.output_format == OutputFormat.TEXT,
        )
        context.state.parse_duration_ms = (time.perf_counter() - parse_start) * 1000
        bound_args: BoundArguments = bound
        context.state.command_chain = _command_chain(command)
        if context.options.cli_enabled:
            run_ctx = _build_run_context(context.state)
            _inject_run_context(bound_args, ignored, run_ctx)
            context.state.arg_names = _safe_arg_names(
                bound_args,
                ignored,
                capture_mode=context.options.arg_capture_mode,
                allowlist=context.options.arg_allowlist,
                enabled=True,
            )
        else:
            context.state.arg_names = ()
        verbosity = _resolve_verbosity(bound_args)
        context.bootstrap(verbosity)
        result = _invoke_command(command, bound_args)
        return _default_exit_code_from_result(result)
    except SystemExit as exc:
        context.state.error_type = type(exc).__name__
        return _exit_code_from_system_exit(exc)
    except CycloptsError as exc:
        context.state.parse_duration_ms = (time.perf_counter() - parse_start) * 1000
        context.state.is_parse_error = True
        context.state.error_type = type(exc).__name__
        if span is not None:
            span.record_exception(exc)
        return handle_cli_error(exc, sys.stderr, output_format=context.options.output_format)
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
        if context.state.parse_duration_ms is None:
            context.state.parse_duration_ms = (time.perf_counter() - parse_start) * 1000
        context.state.error_type = type(exc).__name__
        if span is not None:
            span.record_exception(exc)
        return handle_cli_error(exc, sys.stderr, output_format=context.options.output_format)


def _invoke_command(command: Callable[..., object], bound: BoundArguments) -> object:
    args = getattr(bound, "args", None)
    kwargs = getattr(bound, "kwargs", None)
    if args is not None and kwargs is not None:
        result = command(*args, **kwargs)
        if dataclasses.is_dataclass(result) and callable(result):
            return result()
        return result
    arguments = getattr(bound, "arguments", {})
    result = command(**arguments)
    if dataclasses.is_dataclass(result) and callable(result):
        return result()
    return result


def _set_span_context(
    span: Span | None,
    *,
    invocation_id: str,
    command_chain: tuple[str, ...],
    arg_names: tuple[str, ...],
    policy: ObservabilityPolicy,
) -> None:
    if span is None:
        return
    bounded_arg_names = limit_cli_arg_names(
        arg_names,
        max_len=policy.budget.cli_arg_names_max,
    )
    normalizer = build_attribute_normalizer(policy)
    attrs: dict[str, object] = {
        CLI_INVOCATION_ID: invocation_id,
        CLI_COMMAND: ".".join(command_chain) if command_chain else "<unknown>",
        CLI_ARG_COUNT: len(arg_names),
    }
    if bounded_arg_names:
        attrs[CLI_ARG_NAMES] = [*bounded_arg_names]
    for key, value in normalizer.normalize(attrs).items():
        span.set_attribute(key, value)


def _command_label(command_chain: tuple[str, ...]) -> str:
    if command_chain:
        return ".".join(command_chain)
    return "<unknown>"


def _finalize_span(span: Span | None, *, state: _InvocationState, exit_code: int) -> None:
    if span is None:
        return
    duration_ms = (time.perf_counter() - state.start_ts) * 1000
    normalizer = build_attribute_normalizer(get_observability().policy)
    attrs: dict[str, object] = {
        CLI_EXIT_CODE: exit_code,
        CLI_DURATION_MS: duration_ms,
        CLI_IS_PARSE_ERROR: state.is_parse_error,
    }
    if state.parse_duration_ms is not None:
        attrs[CLI_PARSE_DURATION_MS] = state.parse_duration_ms
    if state.error_type is not None:
        attrs[CLI_ERROR_TYPE] = state.error_type
    for key, value in normalizer.normalize(attrs).items():
        span.set_attribute(key, value)


def _get_cli_instruments(meter: Meter) -> _CliInstruments:
    instruments = _CLI_INSTRUMENTS.get(meter)
    if instruments is not None:
        return instruments
    instruments = _CliInstruments(
        invocation_count=meter.create_counter(
            "codeintel.cli.invocations",
            unit="1",
            description="Count of CLI invocations by command and outcome",
        ),
        invocation_duration_ms=meter.create_histogram(
            "codeintel.cli.invocation.duration_ms",
            unit="ms",
            description="Duration of CLI invocations (ms)",
        ),
        parse_duration_ms=meter.create_histogram(
            "codeintel.cli.parse.duration_ms",
            unit="ms",
            description="Argument parsing duration for CLI invocations (ms)",
        ),
        parse_errors=meter.create_counter(
            "codeintel.cli.parse.errors",
            unit="1",
            description="Count of CLI parse errors by command",
        ),
    )
    _CLI_INSTRUMENTS[meter] = instruments
    return instruments


def _record_cli_metrics(
    obs: ObservabilityRuntime,
    *,
    state: _InvocationState,
    exit_code: int,
    enabled: bool,
) -> None:
    if not enabled or not obs.enabled or obs.meter is None:
        return
    instruments = _get_cli_instruments(obs.meter)
    command = _command_label(state.command_chain)
    attrs: dict[str, str | bool | int | float] = {
        CLI_COMMAND: command,
        CLI_EXIT_CODE: exit_code,
        CLI_IS_PARSE_ERROR: state.is_parse_error,
    }
    if state.error_type is not None:
        attrs[CLI_ERROR_TYPE] = state.error_type
    instruments.invocation_count.add(1, attributes=attrs)
    invocation_duration_ms = (time.perf_counter() - state.start_ts) * 1000
    instruments.invocation_duration_ms.record(invocation_duration_ms, attributes=attrs)
    if state.parse_duration_ms is not None:
        instruments.parse_duration_ms.record(
            state.parse_duration_ms,
            attributes={CLI_COMMAND: command},
        )
    if state.is_parse_error:
        error_attrs: dict[str, str | bool | int | float] = {CLI_COMMAND: command}
        if state.error_type is not None:
            error_attrs[CLI_ERROR_TYPE] = state.error_type
        instruments.parse_errors.add(1, attributes=error_attrs)


def _emit_cli_log(state: _InvocationState, *, exit_code: int, enabled: bool) -> None:
    if not enabled:
        return
    payload = {
        "event": "cli.invocation",
        "invocation_id": state.invocation_id,
        "command": _command_label(state.command_chain),
        "exit_code": exit_code,
        "duration_ms": (time.perf_counter() - state.start_ts) * 1000,
        "parse_duration_ms": state.parse_duration_ms,
        "is_parse_error": state.is_parse_error,
        "error_type": state.error_type,
    }
    message = json.dumps(payload, sort_keys=True)
    if state.is_parse_error:
        log.warning("cli.parse_error %s", message)
    else:
        log.info("cli.invocation %s", message)


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


def flatten_arg_names(arguments: Mapping[str, object], ignored_names: set[str]) -> list[str]:
    """Flatten bound arguments into a list of parameter names.

    Parameters
    ----------
    arguments
        Bound arguments mapping.
    ignored_names
        Argument names to ignore.

    Returns
    -------
    list[str]
        Flattened argument names.
    """
    names: list[str] = []
    for name, value in arguments.items():
        if name in ignored_names:
            continue
        if name == "flags" and dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                if field.name == "run_context":
                    continue
                names.append(f"flags.{field.name}")
            continue
        names.append(name)
    return list(dict.fromkeys(names))


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
    names = flatten_arg_names(arguments, ignored_names)
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


def normalize_allowlist(values: tuple[str, ...]) -> set[str]:
    """Normalize allowlist entries by expanding shared flag prefixes.

    Parameters
    ----------
    values
        Raw allowlist values from settings.

    Returns
    -------
    set[str]
        Normalized allowlist entries.
    """
    normalized = {value.strip() for value in values if value.strip()}
    expanded: set[str] = set()
    for value in normalized:
        expanded.add(value)
        if "." not in value:
            expanded.add(f"flags.{value}")
    return expanded


__all__ = [
    "RunContext",
    "flatten_arg_names",
    "normalize_allowlist",
    "run_cli_with_telemetry",
]
