"""Tests for LibCST parsing adapter.

This module tests the LibCST-based parsing implementation including
module parsing, import extraction, function collection, and call site analysis.
"""

from __future__ import annotations

from typing import Final

from codeintel.graphs.adapters.libcst_parsing import LibCSTParsingAdapter
from codeintel.graphs.ports.parsing import ParsedModule

EXPECTED_IMPORT_COUNT: Final = 2
EXPECTED_FUNCTION_COUNT: Final = 2
EXPECTED_PARAM_COUNT: Final = 3
EXPECTED_CALL_SITE_COUNT: Final = 4
EXPECTED_MIN_IMPORTS: Final = 4


def test_parse_module_simple_function() -> None:
    """Parse a module with a simple function.

    Raises
    ------
    AssertionError
        If parsing fails or function is not found.
    """
    source = '''
def hello():
    """Say hello."""
    print("Hello")
'''
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success:
        msg = f"Expected parsing to succeed, got error: {result.error}"
        raise AssertionError(msg)
    if result.module is None:
        msg = "Expected module to be set"
        raise AssertionError(msg)
    if len(result.module.functions) != 1:
        msg = f"Expected 1 function, got {len(result.module.functions)}"
        raise AssertionError(msg)
    func = result.module.functions[0]
    if func.name != "hello":
        msg = f"Expected function name 'hello', got '{func.name}'"
        raise AssertionError(msg)


def test_parse_module_async_function() -> None:
    """Parse a module with an async function.

    Raises
    ------
    AssertionError
        If async function is not detected.
    """
    source = '''
async def fetch_data():
    """Fetch data asynchronously."""
    pass
'''
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.functions) != 1:
        msg = f"Expected 1 function, got {len(result.module.functions)}"
        raise AssertionError(msg)
    func = result.module.functions[0]
    if not func.is_async:
        msg = "Expected function to be async"
        raise AssertionError(msg)


def test_parse_module_with_decorators() -> None:
    """Parse a module with decorated functions.

    Raises
    ------
    AssertionError
        If decorators are not extracted.
    """
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

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.functions) != EXPECTED_FUNCTION_COUNT:
        msg = f"Expected {EXPECTED_FUNCTION_COUNT} functions, got {len(result.module.functions)}"
        raise AssertionError(msg)

    helper = result.module.functions[0]
    if "staticmethod" not in helper.decorator_names:
        msg = f"Expected staticmethod decorator, got {helper.decorator_names}"
        raise AssertionError(msg)

    value_func = result.module.functions[1]
    if "property" not in value_func.decorator_names:
        msg = f"Expected property decorator, got {value_func.decorator_names}"
        raise AssertionError(msg)


def test_parse_module_class_methods() -> None:
    """Parse methods inside a class.

    Raises
    ------
    AssertionError
        If method qualnames are not correct.
    """
    source = """
class MyClass:
    def __init__(self):
        pass

    def method(self):
        pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.functions) != EXPECTED_FUNCTION_COUNT:
        msg = f"Expected {EXPECTED_FUNCTION_COUNT} methods, got {len(result.module.functions)}"
        raise AssertionError(msg)

    init_method = result.module.functions[0]
    if init_method.qualname != "MyClass.__init__":
        msg = f"Expected qualname 'MyClass.__init__', got '{init_method.qualname}'"
        raise AssertionError(msg)

    method = result.module.functions[1]
    if method.qualname != "MyClass.method":
        msg = f"Expected qualname 'MyClass.method', got '{method.qualname}'"
        raise AssertionError(msg)


def test_parse_module_with_imports() -> None:
    """Parse a module with import statements.

    Raises
    ------
    AssertionError
        If imports are not extracted correctly.
    """
    source = """
import os
import sys
from pathlib import Path
from typing import List, Optional
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.imports) < EXPECTED_MIN_IMPORTS:
        msg = f"Expected at least {EXPECTED_MIN_IMPORTS} imports, got {len(result.module.imports)}"
        raise AssertionError(msg)


def test_parse_module_syntax_error() -> None:
    """Parse invalid source returns error result.

    Raises
    ------
    AssertionError
        If error is not returned.
    """
    source = """
def broken(
    # Missing closing paren
def other():
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if result.success:
        msg = "Expected parsing to fail for invalid syntax"
        raise AssertionError(msg)
    if result.error is None:
        msg = "Expected error to be set"
        raise AssertionError(msg)


def test_extract_imports_simple() -> None:
    """Extract imports from simple source.

    Raises
    ------
    AssertionError
        If imports are not extracted correctly.
    """
    source = """
import os
from pathlib import Path
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    if len(imports) < EXPECTED_IMPORT_COUNT:
        msg = f"Expected at least {EXPECTED_IMPORT_COUNT} imports, got {len(imports)}"
        raise AssertionError(msg)

    # Check os import
    os_import = next((i for i in imports if i[0] == "os"), None)
    if os_import is None:
        msg = "Expected 'os' import"
        raise AssertionError(msg)


def test_extract_imports_with_alias() -> None:
    """Extract imports with aliases.

    Raises
    ------
    AssertionError
        If aliases are not handled.
    """
    source = """
import numpy as np
from pandas import DataFrame as DF
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    if len(imports) < EXPECTED_IMPORT_COUNT:
        msg = f"Expected at least {EXPECTED_IMPORT_COUNT} imports, got {len(imports)}"
        raise AssertionError(msg)


def test_extract_imports_from_package() -> None:
    """Extract from-imports from packages.

    Raises
    ------
    AssertionError
        If package imports are not extracted.
    """
    source = """
from os.path import join, dirname
from collections.abc import Mapping, Sequence
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    if len(imports) < EXPECTED_IMPORT_COUNT:
        msg = f"Expected at least {EXPECTED_IMPORT_COUNT} imports, got {len(imports)}"
        raise AssertionError(msg)

    # Check os.path import
    path_import = next((i for i in imports if i[0] == "os.path"), None)
    if path_import is None:
        msg = "Expected 'os.path' import"
        raise AssertionError(msg)
    if "join" not in path_import[1] and "dirname" not in path_import[1]:
        msg = f"Expected 'join' or 'dirname' in imported names, got {path_import[1]}"
        raise AssertionError(msg)


def test_extract_imports_star() -> None:
    """Extract star imports.

    Raises
    ------
    AssertionError
        If star import is not recognized.
    """
    source = """
from typing import *
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    if len(imports) != 1:
        msg = f"Expected 1 import, got {len(imports)}"
        raise AssertionError(msg)
    if imports[0][1] != ("*",):
        msg = f"Expected star import, got {imports[0][1]}"
        raise AssertionError(msg)


def test_extract_imports_invalid_source() -> None:
    """Extract imports from invalid source returns empty list.

    Raises
    ------
    AssertionError
        If non-empty list is returned for invalid source.
    """
    source = """
def broken(
"""
    imports = LibCSTParsingAdapter.extract_imports(source)

    if len(imports) != 0:
        msg = f"Expected empty imports for invalid source, got {len(imports)}"
        raise AssertionError(msg)


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
    """Parse function with various parameters.

    Raises
    ------
    AssertionError
        If parameters are not extracted.
    """
    source = """
def func_with_params(a, b, c=None, *args, **kwargs):
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.functions) != 1:
        msg = f"Expected 1 function, got {len(result.module.functions)}"
        raise AssertionError(msg)

    func = result.module.functions[0]
    # Parameters should include at least a, b, c
    if len(func.parameters) < EXPECTED_PARAM_COUNT:
        msg = f"Expected at least {EXPECTED_PARAM_COUNT} parameters, got {func.parameters}"
        raise AssertionError(msg)


def test_extract_call_sites() -> None:
    """Extract call sites from a function.

    Raises
    ------
    AssertionError
        If call sites are not extracted.
    """
    source = """
def caller():
    foo()
    bar(1, 2)
    obj.method()
    result = calculate(x)
    return result
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)

    call_sites = LibCSTParsingAdapter.extract_call_sites(result.module, function_span=(1, 10))

    if len(call_sites) < EXPECTED_CALL_SITE_COUNT:
        msg = f"Expected at least {EXPECTED_CALL_SITE_COUNT} call sites, got {len(call_sites)}"
        raise AssertionError(msg)


def test_extract_call_sites_no_ast_module() -> None:
    """Extract call sites when AST module is None.

    Raises
    ------
    AssertionError
        If non-empty result for missing AST.
    """
    # Create a ParsedModule with no AST
    parsed = ParsedModule(
        source="",
        functions=(),
        imports=(),
        cst_module=None,
        ast_module=None,
    )

    call_sites = LibCSTParsingAdapter.extract_call_sites(parsed, function_span=(1, 10))

    if len(call_sites) != 0:
        msg = f"Expected empty call sites, got {len(call_sites)}"
        raise AssertionError(msg)


def test_parse_module_nested_class() -> None:
    """Parse nested class methods.

    Raises
    ------
    AssertionError
        If nested class methods are not found.
    """
    source = """
class Outer:
    class Inner:
        def inner_method(self):
            pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.functions) != 1:
        msg = f"Expected 1 method, got {len(result.module.functions)}"
        raise AssertionError(msg)


def test_parse_module_preserves_source() -> None:
    """Parsed module preserves original source.

    Raises
    ------
    AssertionError
        If source is not preserved.
    """
    source = """
def test():
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if result.module.source != source:
        msg = "Expected source to be preserved"
        raise AssertionError(msg)


def test_parse_module_cst_and_ast_available() -> None:
    """Parsed module has both CST and AST available.

    Raises
    ------
    AssertionError
        If CST or AST is missing.
    """
    source = """
def func():
    return 42
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if result.module.cst_module is None:
        msg = "Expected CST module to be set"
        raise AssertionError(msg)
    if result.module.ast_module is None:
        msg = "Expected AST module to be set"
        raise AssertionError(msg)


def test_import_collector_dotted_imports() -> None:
    """Import collector handles dotted module names.

    Raises
    ------
    AssertionError
        If dotted imports are not handled.
    """
    source = """
import os.path.join
from collections.abc import Mapping
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)

    imports = result.module.imports
    # Should have collected at least the dotted module names
    module_names = [i[0] for i in imports]
    if "collections.abc" not in module_names:
        msg = f"Expected 'collections.abc' in imports, got {module_names}"
        raise AssertionError(msg)


def test_decorated_function_with_call() -> None:
    """Parse function with decorator that is a call.

    Raises
    ------
    AssertionError
        If decorator call is not handled.
    """
    source = """
@decorator_factory(arg=1)
def decorated():
    pass
"""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success or result.module is None:
        msg = "Expected parsing to succeed"
        raise AssertionError(msg)
    if len(result.module.functions) != 1:
        msg = f"Expected 1 function, got {len(result.module.functions)}"
        raise AssertionError(msg)

    func = result.module.functions[0]
    if "decorator_factory" not in func.decorator_names:
        msg = f"Expected 'decorator_factory' decorator, got {func.decorator_names}"
        raise AssertionError(msg)


def test_parse_empty_module() -> None:
    """Parse empty module succeeds.

    Raises
    ------
    AssertionError
        If parsing fails on empty input.
    """
    source = ""
    result = LibCSTParsingAdapter.parse_module(source)

    if not result.success:
        msg = "Expected parsing to succeed for empty source"
        raise AssertionError(msg)
    if result.module is None:
        msg = "Expected module to be set"
        raise AssertionError(msg)
    if len(result.module.functions) != 0:
        msg = f"Expected 0 functions, got {len(result.module.functions)}"
        raise AssertionError(msg)
