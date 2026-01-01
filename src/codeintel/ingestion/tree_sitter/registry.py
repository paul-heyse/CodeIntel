"""Tree-sitter language registry and query pack loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from tree_sitter import LANGUAGE_VERSION, MIN_COMPATIBLE_LANGUAGE_VERSION, Language, Parser
from tree_sitter_language_pack import get_language

if TYPE_CHECKING:
    from tree_sitter_language_pack import SupportedLanguage


@dataclass(frozen=True)
class LanguageSpec:
    """Tree-sitter language specification."""

    name: SupportedLanguage
    extensions: tuple[str, ...]
    pack_name: str


@dataclass(frozen=True)
class QueryPack:
    """Tree-sitter query pack definition."""

    name: str
    query_text: str


_PACKS_ROOT = Path(__file__).resolve().parent / "packs"

_LANGUAGE_SPECS: tuple[LanguageSpec, ...] = (
    LanguageSpec(name="python", extensions=(".py", ".pyi"), pack_name="python"),
)

_EXTENSION_TO_LANGUAGE: dict[str, SupportedLanguage] = {
    ext: spec.name for spec in _LANGUAGE_SPECS for ext in spec.extensions
}


def language_for_path(path: Path) -> SupportedLanguage | None:
    """Return the tree-sitter language name for a file path.

    Returns
    -------
    SupportedLanguage | None
        Language name when the file suffix is supported.
    """
    return _EXTENSION_TO_LANGUAGE.get(path.suffix.lower())


def supported_languages() -> tuple[SupportedLanguage, ...]:
    """Return the list of supported tree-sitter language names.

    Returns
    -------
    tuple[SupportedLanguage, ...]
        Supported language names.
    """
    return tuple(spec.name for spec in _LANGUAGE_SPECS)


def _spec_for_language(language: SupportedLanguage) -> LanguageSpec | None:
    for spec in _LANGUAGE_SPECS:
        if spec.name == language:
            return spec
    return None


def _assert_language_abi(language: Language) -> None:
    if not (MIN_COMPATIBLE_LANGUAGE_VERSION <= language.abi_version <= LANGUAGE_VERSION):
        msg = (
            "Tree-sitter language ABI not supported: "
            f"{language.abi_version} "
            f"(expected {MIN_COMPATIBLE_LANGUAGE_VERSION}-{LANGUAGE_VERSION})"
        )
        raise RuntimeError(msg)


@cache
def load_language(language: SupportedLanguage) -> Language:
    """Load a tree-sitter Language with ABI checks.

    Returns
    -------
    Language
        Loaded tree-sitter language binding.

    Raises
    ------
    ValueError
        If the language is not supported.
    """
    spec = _spec_for_language(language)
    if spec is None:
        msg = f"Unsupported tree-sitter language: {language}"
        raise ValueError(msg)
    ts_language = get_language(spec.name)
    _assert_language_abi(ts_language)
    return ts_language


@cache
def load_parser(language: SupportedLanguage) -> Parser:
    """Return a cached tree-sitter Parser for the given language.

    Returns
    -------
    Parser
        Cached parser instance.
    """
    return Parser(load_language(language))


@cache
def load_query_packs(language: SupportedLanguage) -> tuple[QueryPack, ...]:
    """Load query packs for a tree-sitter language.

    Returns
    -------
    tuple[QueryPack, ...]
        Query packs available for the language.
    """
    spec = _spec_for_language(language)
    if spec is None:
        return ()
    pack_dir = _PACKS_ROOT / spec.pack_name
    if not pack_dir.is_dir():
        return ()
    return tuple(
        QueryPack(name=path.stem, query_text=path.read_text(encoding="utf-8"))
        for path in sorted(pack_dir.glob("*.scm"))
    )


__all__ = [
    "LanguageSpec",
    "QueryPack",
    "language_for_path",
    "load_language",
    "load_parser",
    "load_query_packs",
    "supported_languages",
]
