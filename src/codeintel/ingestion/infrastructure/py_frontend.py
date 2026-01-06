"""Shared Python frontend caches for ingestion steps."""

from __future__ import annotations

import ast
import io
import threading
import tokenize
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource

if TYPE_CHECKING:
    from types import CodeType

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord


@dataclass(frozen=True)
class PyFrontendOptions:
    """Configuration options for shared Python frontend caching."""

    max_entries: int = 4096
    cache_bytes: bool = True
    cache_text: bool = True
    cache_line_index: bool = True
    cache_ast: bool = True
    cache_code: bool = True
    decode_errors: str = "replace"


@dataclass(frozen=True)
class PySourceBundle:
    """Source bundle for a Python module."""

    source_text: str
    source_bytes: bytes
    source_index: LineIndexedSource


class _LruCache:
    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._data: OrderedDict[Hashable, object] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> object | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: Hashable, value: object) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._max_entries:
                self._data.popitem(last=False)


class PyFrontend:
    """Shared frontend for reading and parsing Python source once per run."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: PyFrontendOptions | None = None,
    ) -> None:
        self._discovery = discovery
        self._options = options or PyFrontendOptions()
        max_entries = self._options.max_entries
        self._bytes_cache = _LruCache(max_entries) if self._options.cache_bytes else None
        self._text_cache = _LruCache(max_entries) if self._options.cache_text else None
        self._line_index_cache = _LruCache(max_entries) if self._options.cache_line_index else None
        self._ast_cache = _LruCache(max_entries) if self._options.cache_ast else None
        self._code_cache = _LruCache(max_entries) if self._options.cache_code else None

    @property
    def discovery(self) -> ModuleDiscoveryPort:
        """Return the discovery port used by the frontend."""
        return self._discovery

    def get_source_bytes(self, module: ModuleRecord) -> bytes | None:
        """Return module source bytes, reading and caching as needed.

        Parameters
        ----------
        module
            Module record to read.

        Returns
        -------
        bytes | None
            Source bytes for the module, or None when unavailable.
        """
        if self._bytes_cache is not None:
            cached = self._bytes_cache.get(module.rel_path)
            if isinstance(cached, bytes):
                return cached
        source_bytes = self._discovery.read_module_bytes(module)
        if source_bytes is None:
            source_text = self.get_source_text(module)
            if source_text is None:
                return None
            source_bytes = source_text.encode("utf-8", errors=self._options.decode_errors)
        if self._bytes_cache is not None:
            self._bytes_cache.set(module.rel_path, source_bytes)
        return source_bytes

    def get_source_text(self, module: ModuleRecord) -> str | None:
        """Return module source text, reading and caching as needed.

        Parameters
        ----------
        module
            Module record to read.

        Returns
        -------
        str | None
            Source text for the module, or None when unavailable.
        """
        if self._text_cache is not None:
            cached = self._text_cache.get(module.rel_path)
            if isinstance(cached, str):
                return cached
        source_text = self._discovery.read_module_source(module)
        if source_text is None:
            source_bytes = self.get_source_bytes(module)
            if source_bytes is None:
                return None
            source_text, _encoding = _decode_source_bytes(
                source_bytes, errors=self._options.decode_errors
            )
        if self._text_cache is not None:
            self._text_cache.set(module.rel_path, source_text)
        return source_text

    def get_source_bundle(self, module: ModuleRecord) -> PySourceBundle | None:
        """Return a cached source bundle for a module.

        Parameters
        ----------
        module
            Module record to read.

        Returns
        -------
        PySourceBundle | None
            Source bundle with bytes, text, and line index, or None if unavailable.
        """
        if self._line_index_cache is not None:
            cached = self._line_index_cache.get(module.rel_path)
            if isinstance(cached, PySourceBundle):
                return cached
        source_bytes = self.get_source_bytes(module)
        if source_bytes is None:
            return None
        source_text, encoding = _decode_source_bytes(
            source_bytes, errors=self._options.decode_errors
        )
        source_index = LineIndexedSource(source_text, source_bytes, encoding=encoding)
        bundle = PySourceBundle(
            source_text=source_text,
            source_bytes=source_bytes,
            source_index=source_index,
        )
        if self._line_index_cache is not None:
            self._line_index_cache.set(module.rel_path, bundle)
        return bundle

    def get_ast(self, module: ModuleRecord) -> ast.Module | None:
        """Return a parsed AST for the module when possible.

        Parameters
        ----------
        module
            Module record to parse.

        Returns
        -------
        ast.AST | None
            Parsed AST, or None if parsing fails or source is missing.
        """
        if self._ast_cache is not None:
            cached = self._ast_cache.get(module.rel_path)
            if isinstance(cached, ast.Module):
                return cached
        source_text = self.get_source_text(module)
        if source_text is None:
            return None
        try:
            tree = ast.parse(
                source_text,
                filename=str(module.file_path),
                type_comments=True,
            )
        except (SyntaxError, ValueError, TypeError):
            return None
        if self._ast_cache is not None:
            self._ast_cache.set(module.rel_path, tree)
        return tree

    def get_code(
        self,
        module: ModuleRecord,
        *,
        dont_inherit: bool,
        optimize: int,
        flags: int,
    ) -> CodeType | None:
        """Return compiled bytecode for the module when possible.

        Parameters
        ----------
        module
            Module record to compile.
        dont_inherit
            Whether to inherit compiler flags from the calling scope.
        optimize
            Optimization level for the compiler.
        flags
            Compiler flags to apply during compilation.

        Returns
        -------
        CodeType | None
            Compiled code object, or None if compilation fails.
        """
        cache_key = (module.rel_path, dont_inherit, optimize, flags)
        if self._code_cache is not None:
            cached = self._code_cache.get(cache_key)
            if cached is not None:
                return cached  # type: ignore[return-value]
        tree = self.get_ast(module)
        source_text = None
        try:
            if tree is not None:
                code = compile(
                    tree,
                    str(module.file_path),
                    "exec",
                    dont_inherit=dont_inherit,
                    optimize=optimize,
                    flags=flags,
                )
            else:
                source_text = self.get_source_text(module)
                if source_text is None:
                    return None
                code = compile(
                    source_text,
                    str(module.file_path),
                    "exec",
                    dont_inherit=dont_inherit,
                    optimize=optimize,
                    flags=flags,
                )
        except (SyntaxError, ValueError, TypeError):
            return None
        if self._code_cache is not None:
            self._code_cache.set(cache_key, code)
        return code


def _decode_source_bytes(source_bytes: bytes, *, errors: str) -> tuple[str, str]:
    encoding = "utf-8"
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
    except (LookupError, SyntaxError, UnicodeDecodeError):
        encoding = "utf-8"
    return source_bytes.decode(encoding, errors=errors), encoding


__all__ = ["PyFrontend", "PyFrontendOptions", "PySourceBundle"]
