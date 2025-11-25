"""Parser registry for analytics modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from codeintel.analytics.parsing.models import ParsedFunction
from codeintel.config.parser_types import FunctionParserKind

ParseModuleFn = Callable[[Path, bytes], Iterable[ParsedFunction]]
DEFAULT_PARSERS: dict[FunctionParserKind, ParseModuleFn] = {}


class FunctionParserRegistry:
    """Registry for function parsers keyed by FunctionParserKind."""

    def __init__(self) -> None:
        self._by_kind: dict[FunctionParserKind, ParseModuleFn] = dict(DEFAULT_PARSERS)

    def register(self, kind: FunctionParserKind, fn: ParseModuleFn) -> None:
        """
        Register a parser implementation for the given kind.

        Parameters
        ----------
        kind :
            Parser identifier.
        fn :
            Parser implementation that returns parsed functions for a module.

        Raises
        ------
        ValueError
            If a parser is already registered for the requested kind.
        """
        if kind in self._by_kind:
            message = f"Parser already registered for kind {kind}"
            raise ValueError(message)
        self._by_kind[kind] = fn

    def get(self, kind: FunctionParserKind | None) -> ParseModuleFn:
        """
        Return the parser for the given kind (defaults to PYTHON).

        Parameters
        ----------
        kind :
            Requested parser identifier; defaults to `FunctionParserKind.PYTHON`.

        Returns
        -------
        ParseModuleFn
            Parser callable registered for the requested kind.

        Raises
        ------
        KeyError
            When the requested parser kind is not registered.
        """
        target = kind or FunctionParserKind.PYTHON
        try:
            return self._by_kind[target]
        except KeyError as exc:
            message = f"No parser registered for kind {target}"
            raise KeyError(message) from exc


_registry = FunctionParserRegistry()


def register_parser(kind: FunctionParserKind, fn: ParseModuleFn) -> None:
    """
    Register a parser globally.

    Raises
    ------
    ValueError
        If a parser is already registered for the given kind.
    """
    if kind in DEFAULT_PARSERS:
        message = f"Parser already registered for kind {kind}"
        raise ValueError(message)
    DEFAULT_PARSERS[kind] = fn
    _registry.register(kind, fn)


def get_parser(kind: FunctionParserKind | None = None) -> ParseModuleFn:
    """
    Return the globally registered parser for the requested kind.

    Returns
    -------
    ParseModuleFn
        Parser callable registered for the requested kind.
    """
    return _registry.get(kind)
