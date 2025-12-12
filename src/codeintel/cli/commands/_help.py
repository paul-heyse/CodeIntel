"""Harden Cyclopts help rendering to tolerate missing default metadata."""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

import attrs
import cyclopts.help.help as help_mod
from cyclopts.argument import ArgumentCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from cyclopts._result_action import ResultAction
    from cyclopts.argument import Argument
    from cyclopts.core import App
    from cyclopts.group import Group
    from cyclopts.help.help import HelpPanel
    from rich.console import Console


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
_GROUP_INDEX = 0
_ARGUMENT_COLLECTION_INDEX = 1
_FORMAT_INDEX = 2


class _DisplayDefault:
    """A sentinel object that displays a human-readable default in help.

    This class is used to replace None and other defaults with objects that:
    1. Have a ``.name`` attribute for Cyclopts Enum handling
    2. Have ``__repr__`` returning just the display name (not ``namespace(...)``)
    3. Are falsy like None for boolean contexts
    """

    __slots__ = ("name",)
    __hash__: ClassVar[None] = None

    def __init__(self, name: str) -> None:
        """Initialize with the display name.

        Parameters
        ----------
        name
            Human-readable representation for help output.
        """
        self.name = name

    def __repr__(self) -> str:
        """Return clean display name for help rendering.

        Returns
        -------
        str
            The display name without wrapper text.
        """
        return self.name

    def __str__(self) -> str:
        """Return clean display name for string conversion.

        Returns
        -------
        str
            The display name.
        """
        return self.name

    def __bool__(self) -> bool:
        """Return False to be falsy like None.

        Returns
        -------
        bool
            Always False.
        """
        return False

    def __eq__(self, other: object) -> bool:
        """Compare to other DisplayDefault instances.

        Note: We intentionally do NOT equal None, because Cyclopts checks
        ``default not in (None, empty)`` to decide whether to show defaults.
        If we equaled None, this check would fail and defaults wouldn't show.

        Parameters
        ----------
        other
            Object to compare with.

        Returns
        -------
        bool
            True if other is a matching DisplayDefault.
        """
        if isinstance(other, _DisplayDefault):
            return self.name == other.name
        return False


def _safe_default(argument: Argument) -> object:
    """Return a safe default object for help rendering.

    Returns
    -------
    object
        Either the original default (if it exposes ``.name``) or a sentinel with a
        human-friendly ``name`` for help output.
    """
    default = argument.field_info.default
    if default is not None and hasattr(default, "name"):
        return default

    name = _format_default_value(default, argument_name=str(argument.name))
    return _DisplayDefault(name)


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


def _patched_create_parameter_help_panel(
    group: Group,
    argument_collection: ArgumentCollection,
    help_format: str,
    /,
) -> HelpPanel:
    """Normalize argument defaults before delegating to the original renderer.

    Returns
    -------
    HelpPanel
        The panel produced by the original Cyclopts renderer.
    """
    patched_collection = _patched_collection(argument_collection)
    return _ORIGINAL_CREATE_PARAMETER_HELP_PANEL(group, patched_collection, help_format)


@wraps(_ORIGINAL_CREATE_PARAMETER_HELP_PANEL)
def create_parameter_help_panel(*args: object, **kwargs: object) -> HelpPanel:
    """Normalize defaults before delegating to the Cyclopts renderer.

    Returns
    -------
    HelpPanel
        The rendered help panel.
    """
    group = cast("Group", args[_GROUP_INDEX] if args else kwargs["group"])
    argument_collection = cast(
        "ArgumentCollection",
        args[_ARGUMENT_COLLECTION_INDEX]
        if len(args) > _ARGUMENT_COLLECTION_INDEX
        else kwargs["argument_collection"],
    )
    help_format = cast(
        "str",
        args[_FORMAT_INDEX] if len(args) > _FORMAT_INDEX else kwargs["format"],
    )
    return _patched_create_parameter_help_panel(group, argument_collection, help_format)


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
    """Install the hardened help renderer globally for Cyclopts.

    This patches ALL locations where ``create_parameter_help_panel`` is imported
    to ensure the patch is effective regardless of how Cyclopts accesses it.
    Without patching all locations, Python's import aliasing means some code
    paths will still use the unpatched original function.
    """
    for module, attr in _iter_patch_targets():
        setattr(module, attr, create_parameter_help_panel)


_HELP_STATE = {"applied": False}


def build_patched_app(make_app: Callable[[], App]) -> App:
    """Construct an App with help rendering hardened globally.

    This function applies the help rendering patch globally once (on first call)
    rather than wrapping each invocation with a proxy. This is simpler and avoids
    the complexity of the PatchedAppProxy pattern.

    Returns
    -------
    App
        Application instance with global help rendering hardening applied.
    """
    if not _HELP_STATE["applied"]:
        apply_help_patch()
        _HELP_STATE["applied"] = True

    return make_app()


__all__ = ["_AppCallKwargs", "_DisplayDefault", "apply_help_patch", "build_patched_app"]
