"""Utilities for building Typer command wrappers with grouped options.

Typer 0.20 does not ship a dependency injection helper like ``Depends``.
This module provides a lightweight adapter that keeps the full CLI surface
while allowing command handlers to receive a small number of grouped
dataclasses. It works by setting a custom ``__signature__`` on a wrapper
function so Typer sees the desired options, while the handler itself
receives a bundled mapping of parsed CLI values.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from typer.models import ArgumentInfo, OptionInfo


@dataclass(frozen=True)
class OptionSpec:
    """Describe a single Typer-exposed parameter."""

    name: str
    annotation: object
    default: object


def wrap_command(
    handler: Callable[..., object],
    option_specs: Sequence[OptionSpec],
    *,
    bundle: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    name: str | None = None,
) -> Callable[..., object]:
    """Create a Typer-friendly wrapper around a command handler.

    Parameters
    ----------
    handler
        The underlying function that executes the command. It should accept
        keyword arguments matching the output of ``bundle`` (or the raw
        CLI kwargs if ``bundle`` is ``None``).
    option_specs
        Descriptions of the CLI parameters to expose. Each spec contributes
        a keyword-only parameter to the wrapper's signature so Typer can
        render flags and parse values correctly.
    bundle
        Optional function that maps the raw CLI kwargs into a new mapping
        suitable for the handler. This allows grouping options into
        dataclasses without changing the CLI surface.
    name
        Optional name for the generated wrapper; defaults to the handler's
        ``__name__``.

    Returns
    -------
    Callable[..., Any]
        A wrapper function with a custom signature Typer can inspect.
    """

    def _normalized_annotation(annotation: object, default: object) -> object:
        if isinstance(annotation, (OptionInfo, ArgumentInfo)):
            if isinstance(default, (OptionInfo, ArgumentInfo)):
                if default.default not in {inspect.Signature.empty, ..., None}:  # type: ignore[comparison-overlap]
                    return type(default.default)
                return str
            if default not in {inspect.Signature.empty, ..., None}:  # type: ignore[comparison-overlap]
                return type(default)
            return str
        return annotation

    def command_wrapper(**cli_kwargs: object) -> object:
        handler_kwargs = bundle(cli_kwargs) if bundle is not None else cli_kwargs
        return handler(**handler_kwargs)

    parameters = [
        inspect.Parameter(
            spec.name,
            inspect.Parameter.KEYWORD_ONLY,
            default=spec.default,
            annotation=_normalized_annotation(spec.annotation, spec.default),
        )
        for spec in option_specs
    ]

    command_wrapper.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    command_wrapper.__name__ = name or handler.__name__
    command_wrapper.__qualname__ = command_wrapper.__name__
    command_wrapper.__doc__ = handler.__doc__
    return command_wrapper


def option_specs_from_kwargs(
    kwargs: Mapping[str, tuple[object, object]],
) -> Iterable[OptionSpec]:
    """Build OptionSpec objects from a mapping of (annotation, default) pairs.

    Returns
    -------
    Iterable[OptionSpec]
        OptionSpec instances ready for wrapper construction.
    """
    return (
        OptionSpec(name=key, annotation=annotation, default=default)
        for key, (annotation, default) in kwargs.items()
    )
