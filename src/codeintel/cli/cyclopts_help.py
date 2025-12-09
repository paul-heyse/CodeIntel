"""Harden Cyclopts help rendering to tolerate missing default metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import attrs
import cyclopts.help.help as help_mod
from cyclopts.argument import Argument, ArgumentCollection
from cyclopts.core import App
from rich.console import Console

if TYPE_CHECKING:
    from cyclopts._result_action import ResultAction
    from cyclopts.group import Group
    from cyclopts.help.help import HelpPanel


class _AppCallKwargs(TypedDict, total=False):
    console: Console | None
    error_console: Console | None
    print_error: bool | None
    exit_on_error: bool | None
    help_on_error: bool | None
    verbose: bool | None
    end_of_options_delimiter: str | None
    backend: Literal["asyncio", "trio"] | None
    result_action: ResultAction | None

_ORIGINAL_CREATE_PARAMETER_HELP_PANEL = help_mod.create_parameter_help_panel


def _safe_default(argument: Argument) -> object:
    """Return a safe default object for help rendering.

    Returns
    -------
    object
        Either the original default (if it exposes ``.name``) or a sentinel with a
        human-friendly ``name`` for help output.
    """
    default = argument.field_info.default
    if hasattr(default, "name"):
        return default

    name = _format_default_value(default, argument_name=str(argument.name))
    return SimpleNamespace(name=name)


def _format_default_value(default: object, *, argument_name: str) -> str:
    """Produce a human-friendly default representation for help output.

    Returns
    -------
    str
        Render-friendly string for the provided default value.
    """
    if default is None:
        return "(none)"
    if isinstance(default, bool):
        return "true" if default else "false"
    if isinstance(default, Enum):
        return str(default.value)
    if isinstance(default, (int, float, str)):
        return str(default)
    return argument_name


def _patched_argument(argument: Argument) -> Argument:
    """Return a copy of the argument with a safe default.

    Returns
    -------
    Argument
        An argument whose default is safe for help rendering.
    """
    safe_default = _safe_default(argument)
    if safe_default is argument.field_info.default:
        return argument
    new_field_info = attrs.evolve(argument.field_info, default=safe_default)
    return attrs.evolve(argument, field_info=new_field_info)


def _patched_collection(argument_collection: Iterable[Argument]) -> ArgumentCollection:
    """Return an ArgumentCollection-like sequence of arguments with normalized defaults.

    Returns
    -------
    ArgumentCollection
        A fresh collection of patched arguments safe for help rendering.
    """
    patched_arguments = [_patched_argument(argument) for argument in argument_collection]
    collection_type = (
        argument_collection.__class__
        if isinstance(argument_collection, ArgumentCollection)
        else None
    )
    try:
        if collection_type is not None:
            return collection_type(patched_arguments)
        return ArgumentCollection(patched_arguments)
    except (TypeError, ValueError):
        return ArgumentCollection(patched_arguments)


def create_parameter_help_panel(
    group: Group,
    argument_collection: ArgumentCollection,
    help_format: str,
) -> HelpPanel:
    """Normalize argument defaults before delegating to the original renderer.

    Returns
    -------
    HelpPanel
        The panel produced by the original Cyclopts renderer.
    """
    patched_collection = _patched_collection(argument_collection)
    return _ORIGINAL_CREATE_PARAMETER_HELP_PANEL(group, patched_collection, help_format)


def _iter_patch_targets() -> Iterator[tuple[object, str]]:
    yield help_mod, "create_parameter_help_panel"
    for module_name in ("cyclopts.core", "cyclopts.help"):
        try:
            module = __import__(module_name, fromlist=["create_parameter_help_panel"])
        except ModuleNotFoundError:
            continue
        if hasattr(module, "create_parameter_help_panel"):
            yield module, "create_parameter_help_panel"


@contextmanager
def _patched_help_renderer() -> Iterator[None]:
    originals: list[tuple[object, str, object]] = []
    for module, attr in _iter_patch_targets():
        original = getattr(module, attr)
        originals.append((module, attr, original))
        setattr(module, attr, create_parameter_help_panel)
    try:
        yield
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)


def apply_help_patch() -> None:
    """Install the hardened help renderer globally for Cyclopts."""
    help_mod.create_parameter_help_panel = create_parameter_help_panel


def build_patched_app(make_app: Callable[[], App]) -> App:
    """Construct an App with help rendering patched only for that instance.

    Returns
    -------
    App
        Application instance whose help rendering is wrapped safely.
    """
    base_app = make_app()

    class PatchedAppProxy:
        """Proxy that wraps help rendering with the patched renderer."""

        def __init__(self, inner: App) -> None:
            self._inner = inner

        def __call__(
            self,
            tokens: str | Iterable[str] | None = None,
            **call_kwargs: Unpack[_AppCallKwargs],
        ) -> object:
            with _patched_help_renderer():
                return self._inner(tokens, **call_kwargs)

        def help_print(
            self,
            tokens: str | Iterable[str] | None = None,
            *,
            console: Console | None = None,
        ) -> object:
            with _patched_help_renderer():
                return self._inner.help_print(tokens, console=console)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    return PatchedAppProxy(base_app)  # type: ignore[return-value]


__all__ = ["apply_help_patch", "build_patched_app"]
