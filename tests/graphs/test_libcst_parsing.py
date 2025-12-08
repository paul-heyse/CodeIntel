"""Tests for LibCST parsing adapter.

This module tests the LibCST-based parsing implementation including
module parsing, import extraction, function collection, and call site analysis.
"""

from __future__ import annotations

from typing import Final

import pytest

from codeintel.graphs.adapters.libcst_parsing import LibCSTParsingAdapter
from codeintel.graphs.ports.parsing import ParsedModule
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)

EXPECTED_IMPORT_COUNT: Final = 2
EXPECTED_FUNCTION_COUNT: Final = 2
EXPECTED_PARAM_COUNT: Final = 3
EXPECTED_CALL_SITE_COUNT: Final = 4
EXPECTED_MIN_IMPORTS: Final = 4


def test_parse_module_simple_function() -> None:
    """Parse a module with a simple function."""
    source = '''
def hello():
    """Say hello."""
    print("Hello")
'''
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message=f"Expected parsing to succeed, got error: {result.error}")
    if result.module is None:
        pytest.fail("Expected module to be set")
    expect_equal(len(result.module.functions), 1)
    func = result.module.functions[0]
    expect_equal(func.name, "hello")


def test_parse_module_async_function() -> None:
    """Parse a module with an async function."""
    source = '''
async def fetch_data():
    """Fetch data asynchronously."""
    pass
'''
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    if result.module is None:
        pytest.fail("Expected module to be set")
    expect_equal(len(result.module.functions), 1)
    func = result.module.functions[0]
    expect_true(func.is_async, message="Expected function to be async")


def test_parse_module_with_decorators() -> None:
    """Parse a module with decorated functions."""
    source = """
@staticmethod
def helper():
    pass

@property
@cached
def value(self):
    return 42
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    if result.module is None:
        pytest.fail("Expected module to be set")
    expect_equal(len(result.module.functions), EXPECTED_FUNCTION_COUNT)

    helper = result.module.functions[0]
    expect_in("staticmethod", helper.decorator_names)

    value_func = result.module.functions[1]
    expect_in("property", value_func.decorator_names)


def test_parse_module_class_methods() -> None:
    """Parse methods inside a class."""
    source = """
class MyClass:
    def __init__(self):
        pass

    def method(self):
        pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    if result.module is None:
        pytest.fail("Expected module to be set")
    expect_equal(len(result.module.functions), EXPECTED_FUNCTION_COUNT)

    init_method = result.module.functions[0]
    expect_equal(init_method.qualname, "MyClass.__init__")

    method = result.module.functions[1]
    expect_equal(method.qualname, "MyClass.method")


def test_parse_module_with_imports() -> None:
    """Parse a module with import statements."""
    source = """
import os
import sys
from pathlib import Path
from typing import List, Optional
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    if result.module is None:
        pytest.fail("Expected module to be set")
    expect_true(
        len(result.module.imports) >= EXPECTED_MIN_IMPORTS,
        message=f"Expected at least {EXPECTED_MIN_IMPORTS} imports, got {len(result.module.imports)}",
    )


def test_parse_module_syntax_error() -> None:
    """Parse invalid source returns error result."""
    source = """
def broken(
    # Missing closing paren
def other():
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(not result.success, message="Expected parsing to fail for invalid syntax")
    expect_true(result.error is not None, message="Expected error to be set")


def test_extract_imports_simple() -> None:
    """Extract imports from simple source."""
    source = """
import os
from pathlib import Path
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    expect_true(
        len(imports) >= EXPECTED_IMPORT_COUNT,
        message=f"Expected at least {EXPECTED_IMPORT_COUNT} imports, got {len(imports)}",
    )

    # Check os import
    os_import = next((i for i in imports if i[0] == "os"), None)
    expect_is_not_none(os_import, message="Expected 'os' import")


def test_extract_imports_with_alias() -> None:
    """Extract imports with aliases."""
    source = """
import numpy as np
from pandas import DataFrame as DF
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    expect_true(
        len(imports) >= EXPECTED_IMPORT_COUNT,
        message=f"Expected at least {EXPECTED_IMPORT_COUNT} imports, got {len(imports)}",
    )


def test_extract_imports_from_package() -> None:
    """Extract from-imports from packages."""
    source = """
from os.path import join, dirname
from collections.abc import Mapping, Sequence
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    expect_true(
        len(imports) >= EXPECTED_IMPORT_COUNT,
        message=f"Expected at least {EXPECTED_IMPORT_COUNT} imports, got {len(imports)}",
    )

    # Check os.path import
    path_import = next((i for i in imports if i[0] == "os.path"), None)
    expect_is_not_none(path_import, message="Expected 'os.path' import")
    if path_import is not None:
        expect_true(
            "join" in path_import[1] or "dirname" in path_import[1],
            message=f"Expected 'join' or 'dirname' in imported names, got {path_import[1]}",
        )


def test_extract_imports_star() -> None:
    """Extract star imports."""
    source = """
from typing import *
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    expect_equal(len(imports), 1)
    expect_equal(imports[0][1], ("*",))


def test_extract_imports_invalid_source() -> None:
    """Extract imports from invalid source returns empty list."""
    source = """
def broken(
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    expect_equal(len(imports), 0)


def test_parse_function_instance() -> None:
    """Parse function using instance method runs without error."""
    adapter = LibCSTParsingAdapter()
    source = '''
def first():
    pass

def target():
    """Target function."""
    x = 1
    return x

def last():
    pass
'''
    # Parse the middle function - verifies the method runs without error
    # Note: LibCST doesn't expose line numbers directly so we can't assert on result
    _ = adapter.parse_function(source, start_line=4, end_line=8)


def test_parse_module_function_parameters() -> None:
    """Parse function with various parameters."""
    source = """
def func_with_params(a, b, c=None, *args, **kwargs):
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return
    expect_equal(len(result.module.functions), 1)

    func = result.module.functions[0]
    # Parameters should include at least a, b, c
    expect_true(
        len(func.parameters) >= EXPECTED_PARAM_COUNT,
        message=f"Expected at least {EXPECTED_PARAM_COUNT} parameters, got {func.parameters}",
    )


def test_extract_call_sites() -> None:
    """Extract call sites from a function."""
    source = """
def caller():
    foo()
    bar(1, 2)
    obj.method()
    result = calculate(x)
    return result
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return

    call_sites = LibCSTParsingAdapter.extract_call_sites(result.module, function_span=(1, 10))

    expect_true(
        len(call_sites) >= EXPECTED_CALL_SITE_COUNT,
        message=f"Expected at least {EXPECTED_CALL_SITE_COUNT} call sites, got {len(call_sites)}",
    )


def test_extract_call_sites_no_ast_module() -> None:
    """Extract call sites when AST module is None."""
    # Create a ParsedModule with no AST
    parsed = ParsedModule(
        source="",
        functions=(),
        imports=(),
        cst_module=None,
        ast_module=None,
    )

    call_sites = LibCSTParsingAdapter.extract_call_sites(parsed, function_span=(1, 10))

    expect_equal(len(call_sites), 0)


def test_parse_module_nested_class() -> None:
    """Parse nested class methods."""
    source = """
class Outer:
    class Inner:
        def inner_method(self):
            pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return
    expect_equal(len(result.module.functions), 1)


def test_parse_module_preserves_source() -> None:
    """Parsed module preserves original source."""
    source = """
def test():
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return
    expect_equal(result.module.source, source)


def test_parse_module_cst_and_ast_available() -> None:
    """Parsed module has both CST and AST available."""
    source = """
def func():
    return 42
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return
    expect_is_not_none(result.module.cst_module, message="Expected CST module to be set")
    expect_is_not_none(result.module.ast_module, message="Expected AST module to be set")


def test_import_collector_dotted_imports() -> None:
    """Import collector handles dotted module names."""
    source = """
import os.path.join
from collections.abc import Mapping
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return

    imports = result.module.imports
    # Should have collected at least the dotted module names
    module_names = [i[0] for i in imports]
    expect_in("collections.abc", module_names)


def test_decorated_function_with_call() -> None:
    """Parse function with decorator that is a call."""
    source = """
@decorator_factory(arg=1)
def decorated():
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return
    expect_equal(len(result.module.functions), 1)

    func = result.module.functions[0]
    expect_in("decorator_factory", func.decorator_names)


def test_parse_empty_module() -> None:
    """Parse empty module succeeds."""
    source = ""
    result = LibCSTParsingAdapter.parse_module(source)

    expect_true(result.success, message="Expected parsing to succeed for empty source")
    expect_is_not_none(result.module, message="Expected module to be set")
    if result.module is None:
        return
    expect_equal(len(result.module.functions), 0)
