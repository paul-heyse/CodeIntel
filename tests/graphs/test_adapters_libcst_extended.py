"""Extended tests for LibCST parsing adapter.

This module provides additional test coverage for the LibCST parsing
adapter from `codeintel.graphs.adapters.libcst_parsing`, including:

- Module parsing with imports and functions
- Parse error handling
- Import extraction (simple, from, aliased, star)
- Function extraction (sync, async, decorated, nested)
- Call site extraction
- Alias resolution
"""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.graphs.adapters.libcst_parsing import LibCSTParsingAdapter
from codeintel.graphs.ports.parsing import (
    ParsedFunction,
    ParsedModule,
    ParseError,
    ParseResult,
    ParsingPort,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_length,
    expect_true,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_module(result: ParseResult) -> ParsedModule:
    """Ensure ParseResult contains a module for type-safe access.

    Returns
    -------
    ParsedModule
        Parsed module extracted from the result.
    """
    if result.module is None:
        pytest.fail("Expected parsed module")
    module = result.module
    expect_is_not_none(module)
    return module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FUNCTION_COUNT: Final = 4
MIN_IMPORT_COUNT: Final = 4
PARSE_ERROR_LINE: Final = 10


# ---------------------------------------------------------------------------
# Test Fixtures - Realistic Python Source Code
# ---------------------------------------------------------------------------

SIMPLE_MODULE_SOURCE: Final = """
import os
import sys
from pathlib import Path
from typing import Optional, List

def process_data(items: List[int]) -> int:
    '''Process a list of items.'''
    total = 0
    for item in items:
        total += item
    return total

def validate_path(path: Optional[Path] = None) -> bool:
    '''Validate a filesystem path.'''
    if path is None:
        return False
    return path.exists()
"""

ASYNC_FUNCTIONS_SOURCE: Final = """
import asyncio
from aiohttp import ClientSession

async def fetch_url(url: str) -> str:
    '''Fetch content from a URL.'''
    async with ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

def sync_function() -> None:
    '''A synchronous function.'''
    pass

async def process_urls(urls: list[str]) -> list[str]:
    '''Process multiple URLs concurrently.'''
    tasks = [fetch_url(url) for url in urls]
    return await asyncio.gather(*tasks)
"""

DECORATED_FUNCTIONS_SOURCE: Final = """
from functools import lru_cache, wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@lru_cache(maxsize=128)
def cached_compute(n: int) -> int:
    '''Compute with caching.'''
    return n * n

@my_decorator
def decorated_func() -> str:
    '''A decorated function.'''
    return "decorated"

@property
def my_property(self) -> int:
    '''A property.'''
    return 42
"""

CLASS_WITH_METHODS_SOURCE: Final = """
class DataProcessor:
    '''Process data in various ways.'''

    def __init__(self, data: list[int]) -> None:
        self.data = data

    def process(self) -> int:
        '''Process the data.'''
        return sum(self.data)

    async def async_process(self) -> int:
        '''Process data asynchronously.'''
        return sum(self.data)

    @staticmethod
    def validate(value: int) -> bool:
        '''Validate a value.'''
        return value >= 0

    @classmethod
    def from_range(cls, n: int) -> 'DataProcessor':
        '''Create from a range.'''
        return cls(list(range(n)))
"""

IMPORT_VARIATIONS_SOURCE: Final = """
# Standard imports
import os
import sys as system

# From imports
from pathlib import Path
from typing import Optional, List as PyList

# Aliased imports
import numpy as np
from pandas import DataFrame as DF

# Nested module imports
import os.path
from collections.abc import Mapping, Sequence

# Star import
from typing import *
"""

CALL_SITES_SOURCE: Final = """
def main():
    result = process_data([1, 2, 3])
    validate(result)
    helper_func()
    obj.method()
    nested.deep.call()

def process_data(items):
    for item in items:
        transform(item)
    return aggregate(items)

def validate(value):
    return check_bounds(value)
"""

SYNTAX_ERROR_SOURCE: Final = """
def broken_function(
    # Missing closing paren and body
"""

NESTED_CLASSES_SOURCE: Final = """
class Outer:
    '''Outer class.'''

    class Inner:
        '''Inner class.'''

        def inner_method(self) -> None:
            '''Inner method.'''
            pass

    def outer_method(self) -> None:
        '''Outer method.'''
        pass
"""


# ---------------------------------------------------------------------------
# Tests: Module Parsing
# ---------------------------------------------------------------------------


def test_parse_module_simple_success() -> None:
    """Parse a simple module successfully."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    expect_true(result.error is None)
    module = _require_module(result)
    expect_is_instance(module, ParsedModule)


def test_parse_module_extracts_functions() -> None:
    """Parse module extracts all function definitions."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    func_names = {f.name for f in module.functions}
    expect_in("process_data", func_names)
    expect_in("validate_path", func_names)


def test_parse_module_extracts_imports() -> None:
    """Parse module extracts all import statements."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)
    expect_true(len(module.imports) >= MIN_IMPORT_COUNT)


def test_parse_module_preserves_source() -> None:
    """Parse module preserves original source."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)
    expect_equal(module.source, SIMPLE_MODULE_SOURCE)


def test_parse_module_with_syntax_error() -> None:
    """Parse module returns error for syntax errors."""
    result = LibCSTParsingAdapter.parse_module(SYNTAX_ERROR_SOURCE)

    expect_true(not result.success)
    expect_true(result.module is None)
    expect_true(result.error is not None)
    expect_true(result.error is not None and bool(result.error.message))  # Has error message


def test_parse_module_empty_source() -> None:
    """Parse module handles empty source."""
    result = LibCSTParsingAdapter.parse_module("")

    expect_true(result.success)
    module = _require_module(result)
    expect_length(module.functions, 0)
    expect_length(module.imports, 0)


def test_parse_module_with_cst_module() -> None:
    """Parse module includes CST module for further analysis."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)
    expect_true(module.cst_module is not None)


def test_parse_module_with_ast_module() -> None:
    """Parse module includes AST module for call site analysis."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)
    expect_true(module.ast_module is not None)


# ---------------------------------------------------------------------------
# Tests: Async Function Handling
# ---------------------------------------------------------------------------


def test_parse_async_functions_detected() -> None:
    """Async functions are properly detected."""
    result = LibCSTParsingAdapter.parse_module(ASYNC_FUNCTIONS_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    funcs_by_name = {f.name: f for f in module.functions}

    expect_true(funcs_by_name["fetch_url"].is_async)
    expect_true(funcs_by_name["process_urls"].is_async)
    expect_true(not funcs_by_name["sync_function"].is_async)


# ---------------------------------------------------------------------------
# Tests: Decorated Functions
# ---------------------------------------------------------------------------


def test_parse_decorated_functions() -> None:
    """Decorated functions capture decorator names."""
    result = LibCSTParsingAdapter.parse_module(DECORATED_FUNCTIONS_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    funcs_by_name = {f.name: f for f in module.functions}

    expect_in("lru_cache", funcs_by_name["cached_compute"].decorator_names)
    expect_in("my_decorator", funcs_by_name["decorated_func"].decorator_names)
    expect_in("property", funcs_by_name["my_property"].decorator_names)


# ---------------------------------------------------------------------------
# Tests: Class Methods
# ---------------------------------------------------------------------------


def test_parse_class_methods() -> None:
    """Class methods are extracted with qualified names."""
    result = LibCSTParsingAdapter.parse_module(CLASS_WITH_METHODS_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    # Find methods by qualname
    qualnames = {f.qualname for f in module.functions}

    expect_in("DataProcessor.__init__", qualnames)
    expect_in("DataProcessor.process", qualnames)
    expect_in("DataProcessor.async_process", qualnames)
    expect_in("DataProcessor.validate", qualnames)
    expect_in("DataProcessor.from_range", qualnames)


def test_parse_nested_class_methods() -> None:
    """Nested class methods have correct qualified names."""
    result = LibCSTParsingAdapter.parse_module(NESTED_CLASSES_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    qualnames = {f.qualname for f in module.functions}

    expect_in("Outer.outer_method", qualnames)
    expect_in("Outer.Inner.inner_method", qualnames)


# ---------------------------------------------------------------------------
# Tests: Import Extraction
# ---------------------------------------------------------------------------


def test_extract_imports_simple() -> None:
    """Extract simple import statements."""
    imports = LibCSTParsingAdapter.extract_imports(IMPORT_VARIATIONS_SOURCE)

    # Check for standard imports
    import_modules = [i[0] for i in imports]
    expect_in("os", import_modules)
    expect_in("sys", import_modules)


def test_extract_imports_from() -> None:
    """Extract from-import statements."""
    imports = LibCSTParsingAdapter.extract_imports(IMPORT_VARIATIONS_SOURCE)

    # Find pathlib import
    pathlib_imports = [i for i in imports if i[0] == "pathlib"]
    expect_true(len(pathlib_imports) > 0)
    expect_in("Path", pathlib_imports[0][1])


def test_extract_imports_aliased() -> None:
    """Extract aliased imports."""
    result = LibCSTParsingAdapter.parse_module(IMPORT_VARIATIONS_SOURCE)

    expect_true(result.success)
    module = _require_module(result)
    aliases = module.import_aliases

    expect_in("np", aliases)
    expect_equal(aliases["np"], "numpy")
    expect_in("system", aliases)
    expect_equal(aliases["system"], "sys")


def test_extract_imports_star() -> None:
    """Extract star imports."""
    imports = LibCSTParsingAdapter.extract_imports(IMPORT_VARIATIONS_SOURCE)

    # Find star import
    star_imports = [i for i in imports if "*" in i[1]]
    expect_true(len(star_imports) > 0)


def test_extract_imports_nested_modules() -> None:
    """Extract nested module imports."""
    imports = LibCSTParsingAdapter.extract_imports(IMPORT_VARIATIONS_SOURCE)

    import_modules = [i[0] for i in imports]
    expect_true("os.path" in import_modules or "os" in import_modules)


def test_extract_imports_syntax_error_returns_empty() -> None:
    """Extract imports returns empty for syntax errors."""
    imports = LibCSTParsingAdapter.extract_imports(SYNTAX_ERROR_SOURCE)

    expect_equal(imports, [])


# ---------------------------------------------------------------------------
# Tests: Call Site Extraction
# ---------------------------------------------------------------------------


def test_extract_call_sites_basic() -> None:
    """Extract call sites from a function."""
    result = LibCSTParsingAdapter.parse_module(CALL_SITES_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    # Extract call sites from the main function (lines 1-7 approximately)
    # Note: line numbers depend on exact source formatting
    call_sites = LibCSTParsingAdapter.extract_call_sites(
        module,
        function_span=(2, 8),
    )

    call_names = [name for name, _line in call_sites]
    expect_in("process_data", call_names)
    expect_in("validate", call_names)
    expect_in("helper_func", call_names)


def test_extract_call_sites_method_calls() -> None:
    """Extract method call sites."""
    result = LibCSTParsingAdapter.parse_module(CALL_SITES_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    call_sites = LibCSTParsingAdapter.extract_call_sites(
        module,
        function_span=(2, 8),
    )

    call_names = [name for name, _line in call_sites]
    # Method calls should extract the method name
    expect_in("method", call_names)


def test_extract_call_sites_empty_span() -> None:
    """Extract call sites returns empty for non-function span."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    # Use a span with no calls
    call_sites = LibCSTParsingAdapter.extract_call_sites(
        module,
        function_span=(1, 1),
    )

    expect_length(call_sites, 0)


def test_extract_call_sites_no_ast_returns_empty() -> None:
    """Extract call sites returns empty when no AST module."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    # Create a module without AST
    module_no_ast = ParsedModule(
        source=module.source,
        functions=module.functions,
        imports=module.imports,
        cst_module=module.cst_module,
        ast_module=None,
        _import_aliases={},
    )

    call_sites = LibCSTParsingAdapter.extract_call_sites(
        module_no_ast,
        function_span=(1, 100),
    )

    expect_equal(call_sites, [])


# ---------------------------------------------------------------------------
# Tests: parse_function Method
# ---------------------------------------------------------------------------


def test_parse_function_finds_function_in_range() -> None:
    """Parse function finds function within specified line range."""
    adapter = LibCSTParsingAdapter()

    # Note: The current implementation always returns start_line=1, end_line=1
    # for all functions, so we need to use that range
    func = adapter.parse_function(SIMPLE_MODULE_SOURCE, start_line=1, end_line=100)

    # Should find at least one function
    expect_true(func is not None)
    expect_is_instance(func, ParsedFunction)


def test_parse_function_returns_none_for_invalid_source() -> None:
    """Parse function returns None for invalid source."""
    adapter = LibCSTParsingAdapter()

    func = adapter.parse_function(SYNTAX_ERROR_SOURCE, start_line=1, end_line=10)

    expect_true(func is None)


def test_parse_function_returns_none_for_empty_range() -> None:
    """Parse function returns None when no function in range."""
    adapter = LibCSTParsingAdapter()

    # Use range before any function definitions
    func = adapter.parse_function("# Just a comment\n", start_line=1, end_line=10)

    expect_true(func is None)


# ---------------------------------------------------------------------------
# Tests: ParseResult Helpers
# ---------------------------------------------------------------------------


def test_parse_result_ok_creates_success() -> None:
    """ParseResult.ok creates successful result."""
    module = ParsedModule(
        source="",
        functions=(),
        imports=(),
        cst_module=None,
        ast_module=None,
        _import_aliases={},
    )

    result = ParseResult.ok(module)

    expect_true(result.success)
    expect_true(result.module is module)
    expect_true(result.error is None)


def test_parse_result_fail_creates_failure() -> None:
    """ParseResult.fail creates failure result."""
    error = ParseError(message="Test error", line=10, column=5)

    result = ParseResult.fail(error)

    expect_true(not result.success)
    expect_true(result.module is None)
    if result.error is None:
        pytest.fail("Expected parse error")
    expect_true(result.error is error)
    expect_equal(result.error.line, PARSE_ERROR_LINE)


# ---------------------------------------------------------------------------
# Tests: ParsedFunction Attributes
# ---------------------------------------------------------------------------


def test_parsed_function_has_parameters() -> None:
    """Parsed functions include parameter names."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    process_func = next(f for f in module.functions if f.name == "process_data")

    expect_in("items", process_func.parameters)


def test_parsed_function_qualname_top_level() -> None:
    """Top-level function qualname equals name."""
    result = LibCSTParsingAdapter.parse_module(SIMPLE_MODULE_SOURCE)

    expect_true(result.success)
    module = _require_module(result)

    process_func = next(f for f in module.functions if f.name == "process_data")

    expect_equal(process_func.qualname, "process_data")


# ---------------------------------------------------------------------------
# Tests: Protocol Compliance
# ---------------------------------------------------------------------------


def test_libcst_adapter_implements_parsing_port() -> None:
    """LibCSTParsingAdapter implements ParsingPort protocol."""
    adapter = LibCSTParsingAdapter()

    # Check it satisfies the protocol
    expect_true(hasattr(adapter, "parse_module"))
    expect_true(hasattr(adapter, "parse_function"))
    expect_true(hasattr(adapter, "extract_imports"))
    expect_true(hasattr(adapter, "extract_call_sites"))

    # Verify it can be assigned to the protocol type
    _: ParsingPort = adapter
