"""Test function signature extraction from AST.

Test the pure computation functions for extracting structured
signature information from Python function AST nodes.
"""

from __future__ import annotations

import ast

from codeintel.analytics.compute.functions.signatures import (
    FunctionSignature,
    ParameterInfo,
    extract_signature,
)
from tests._helpers import assert_frozen

# =============================================================================
# Constants
# =============================================================================

EXPECTED_PARAMS_1 = 1
EXPECTED_PARAMS_2 = 2
EXPECTED_PARAMS_3 = 3
EXPECTED_DECORATORS_3 = 3

# =============================================================================
# Test Data: Realistic Function Source Code
# =============================================================================

SIMPLE_FUNCTION = """
def greet(name):
    '''Say hello.'''
    return f"Hello, {name}!"
"""

TYPED_FUNCTION = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
"""

ASYNC_FUNCTION = """
async def fetch_data(url: str, timeout: float = 30.0) -> bytes:
    '''Fetch data from URL.'''
    pass
"""

METHOD_WITH_SELF = """
def process(self, data: dict) -> None:
    '''Process data.'''
    pass
"""

CLASSMETHOD_EXAMPLE = """
@classmethod
def from_dict(cls, data: dict) -> 'MyClass':
    '''Create from dictionary.'''
    pass
"""

STATICMETHOD_EXAMPLE = """
@staticmethod
def validate(value: str) -> bool:
    '''Validate value.'''
    pass
"""

PROPERTY_EXAMPLE = """
@property
def count(self) -> int:
    '''Return count.'''
    return self._count
"""

DECORATED_FUNCTION = """
@decorator1
@decorator2.option('value')
@module.decorator3
def decorated_func(x: int) -> int:
    '''Decorated function.'''
    return x
"""

COMPLEX_SIGNATURE = """
def complex_func(
    pos_only: int,
    /,
    regular: str,
    with_default: float = 1.0,
    *args: int,
    keyword_only: bool,
    keyword_with_default: str = "default",
    **kwargs: object,
) -> tuple[int, str]:
    '''Complex function with all parameter kinds.'''
    pass
"""

VAR_POSITIONAL_ONLY = """
def varargs_only(*args: str) -> list[str]:
    '''Accept only varargs.'''
    return list(args)
"""

VAR_KEYWORD_ONLY = """
def kwargs_only(**kwargs: object) -> dict[str, object]:
    '''Accept only kwargs.'''
    return kwargs
"""

NO_PARAMS = """
def no_params() -> None:
    '''Function with no parameters.'''
    pass
"""

NO_DOCSTRING = """
def no_docstring(x: int) -> int:
    return x * 2
"""

MULTIPLE_DECORATORS = """
@cache
@logged
@retry(times=3)
def resilient_fetch(url: str) -> bytes:
    '''Fetch with retries.'''
    pass
"""


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_function(source: str) -> ast.AST:
    """
    Parse source code and return the first function definition.

    Parameters
    ----------
    source
        Python source code containing a function definition.

    Returns
    -------
    ast.AST
        The first AST node from the parsed source.
    """
    tree = ast.parse(source.strip())
    return tree.body[0]


# =============================================================================
# ParameterInfo Tests
# =============================================================================


def test_parameter_create_with_all_fields() -> None:
    """Create parameter info with all fields."""
    param = ParameterInfo(
        name="value",
        annotation="int",
        has_default=True,
        kind="positional_or_keyword",
    )
    assert param.name == "value"
    assert param.annotation == "int"
    assert param.has_default
    assert param.kind == "positional_or_keyword"


def test_parameter_no_annotation() -> None:
    """Create parameter without annotation."""
    param = ParameterInfo(
        name="x",
        annotation=None,
        has_default=False,
        kind="positional_or_keyword",
    )
    assert param.annotation is None


def test_parameter_is_frozen() -> None:
    """Parameter info is immutable."""
    param = ParameterInfo(
        name="x",
        annotation="int",
        has_default=False,
        kind="positional_or_keyword",
    )
    assert_frozen(param, "name", "y")


# =============================================================================
# FunctionSignature Tests
# =============================================================================


def test_signature_create_with_all_fields() -> None:
    """Create function signature with all fields."""
    sig = FunctionSignature(
        name="test_func",
        qualname="module.test_func",
        is_async=True,
        is_method=False,
        is_classmethod=False,
        is_staticmethod=False,
        is_property=False,
        parameters=(ParameterInfo("x", "int", has_default=False, kind="positional_or_keyword"),),
        return_annotation="str",
        decorators=("decorator",),
        docstring="Test function.",
    )
    assert sig.name == "test_func"
    assert sig.is_async


def test_signature_is_frozen() -> None:
    """Function signature is immutable."""
    sig = FunctionSignature(
        name="func",
        qualname="func",
        is_async=False,
        is_method=False,
        is_classmethod=False,
        is_staticmethod=False,
        is_property=False,
        parameters=(),
        return_annotation=None,
        decorators=(),
        docstring=None,
    )
    assert_frozen(sig, "name", "other")


# =============================================================================
# extract_signature Basic Tests
# =============================================================================


def test_extract_simple_function() -> None:
    """Extract signature from simple function."""
    node = _parse_function(SIMPLE_FUNCTION)
    sig = extract_signature(node)
    assert sig.name == "greet"
    assert sig.docstring == "Say hello."
    assert len(sig.parameters) == EXPECTED_PARAMS_1
    assert sig.parameters[0].name == "name"


def test_extract_typed_function() -> None:
    """Extract signature with type annotations."""
    node = _parse_function(TYPED_FUNCTION)
    sig = extract_signature(node)
    assert sig.parameters[0].annotation == "int"
    assert sig.return_annotation == "int"


def test_extract_async_function() -> None:
    """Extract async function signature."""
    node = _parse_function(ASYNC_FUNCTION)
    sig = extract_signature(node)
    assert sig.is_async
    assert sig.name == "fetch_data"


def test_extract_no_params() -> None:
    """Extract function with no parameters."""
    node = _parse_function(NO_PARAMS)
    sig = extract_signature(node)
    assert not sig.parameters
    assert sig.return_annotation == "None"


def test_extract_no_docstring() -> None:
    """Extract function without docstring."""
    node = _parse_function(NO_DOCSTRING)
    sig = extract_signature(node)
    assert sig.docstring is None


def test_extract_non_function_node() -> None:
    """Non-function node returns empty signature."""
    node = ast.parse("x = 1").body[0]
    sig = extract_signature(node)
    assert not sig.name
    assert not sig.parameters


# =============================================================================
# extract_signature Method Detection Tests
# =============================================================================


def test_extract_method_with_self() -> None:
    """Detect method by self parameter."""
    node = _parse_function(METHOD_WITH_SELF)
    sig = extract_signature(node)
    assert sig.is_method
    assert not sig.is_classmethod
    assert not sig.is_staticmethod


def test_extract_classmethod() -> None:
    """Detect classmethod decorator."""
    node = _parse_function(CLASSMETHOD_EXAMPLE)
    sig = extract_signature(node)
    assert sig.is_classmethod
    assert sig.is_method


def test_extract_staticmethod() -> None:
    """Detect staticmethod decorator."""
    node = _parse_function(STATICMETHOD_EXAMPLE)
    sig = extract_signature(node)
    assert sig.is_staticmethod
    assert not sig.is_method


def test_extract_property() -> None:
    """Detect property decorator."""
    node = _parse_function(PROPERTY_EXAMPLE)
    sig = extract_signature(node)
    assert sig.is_property


# =============================================================================
# extract_signature Decorator Tests
# =============================================================================


def test_extract_simple_decorator() -> None:
    """Extract simple decorator name."""
    source = "@mydecorator\ndef func(): pass"
    node = _parse_function(source)
    sig = extract_signature(node)
    assert "mydecorator" in sig.decorators


def test_extract_multiple_decorators() -> None:
    """Extract multiple decorators."""
    node = _parse_function(MULTIPLE_DECORATORS)
    sig = extract_signature(node)
    assert len(sig.decorators) == EXPECTED_DECORATORS_3
    assert "cache" in sig.decorators


def test_extract_attribute_decorator() -> None:
    """Extract attribute-style decorator."""
    node = _parse_function(DECORATED_FUNCTION)
    sig = extract_signature(node)
    # Should include module.decorator3
    has_module_dec = any("decorator3" in d for d in sig.decorators)
    assert has_module_dec


def test_extract_call_decorator() -> None:
    """Extract decorator with arguments."""
    node = _parse_function(DECORATED_FUNCTION)
    sig = extract_signature(node)
    # decorator2.option('value') should be present
    has_option = any("option" in d or "decorator2" in d for d in sig.decorators)
    assert has_option


# =============================================================================
# extract_signature Parameter Kind Tests
# =============================================================================


def test_extract_positional_or_keyword_param() -> None:
    """Detect positional_or_keyword parameters."""
    node = _parse_function(TYPED_FUNCTION)
    sig = extract_signature(node)
    assert sig.parameters[0].kind == "positional_or_keyword"


def test_extract_positional_only_param() -> None:
    """Detect positional-only parameters."""
    node = _parse_function(COMPLEX_SIGNATURE)
    sig = extract_signature(node)
    pos_only = [p for p in sig.parameters if p.kind == "positional_only"]
    assert pos_only
    assert pos_only[0].name == "pos_only"


def test_extract_var_positional_param() -> None:
    """Detect *args parameter."""
    node = _parse_function(VAR_POSITIONAL_ONLY)
    sig = extract_signature(node)
    var_pos = [p for p in sig.parameters if p.kind == "var_positional"]
    assert len(var_pos) == EXPECTED_PARAMS_1
    assert var_pos[0].name == "args"


def test_extract_var_keyword_param() -> None:
    """Detect **kwargs parameter."""
    node = _parse_function(VAR_KEYWORD_ONLY)
    sig = extract_signature(node)
    var_kw = [p for p in sig.parameters if p.kind == "var_keyword"]
    assert len(var_kw) == EXPECTED_PARAMS_1
    assert var_kw[0].name == "kwargs"


def test_extract_keyword_only_param() -> None:
    """Detect keyword-only parameters."""
    node = _parse_function(COMPLEX_SIGNATURE)
    sig = extract_signature(node)
    kw_only = [p for p in sig.parameters if p.kind == "keyword_only"]
    assert kw_only
    names = {p.name for p in kw_only}
    assert "keyword_only" in names


def test_extract_complex_signature_all_kinds() -> None:
    """Complex signature has all parameter kinds."""
    node = _parse_function(COMPLEX_SIGNATURE)
    sig = extract_signature(node)
    kinds = {p.kind for p in sig.parameters}
    expected = {
        "positional_only",
        "positional_or_keyword",
        "var_positional",
        "keyword_only",
        "var_keyword",
    }
    assert kinds == expected


# =============================================================================
# extract_signature Default Value Tests
# =============================================================================


def test_extract_no_defaults() -> None:
    """Parameters without defaults."""
    node = _parse_function(TYPED_FUNCTION)
    sig = extract_signature(node)
    for param in sig.parameters:
        assert not param.has_default


def test_extract_with_default() -> None:
    """Parameter with default value."""
    node = _parse_function(ASYNC_FUNCTION)
    sig = extract_signature(node)
    timeout = next((p for p in sig.parameters if p.name == "timeout"), None)
    assert timeout is not None
    assert timeout.has_default


def test_extract_mixed_defaults() -> None:
    """Mix of parameters with and without defaults."""
    node = _parse_function(COMPLEX_SIGNATURE)
    sig = extract_signature(node)
    # regular should not have default
    regular = next((p for p in sig.parameters if p.name == "regular"), None)
    assert regular is not None
    assert not regular.has_default
    # with_default should have default
    with_def = next((p for p in sig.parameters if p.name == "with_default"), None)
    assert with_def is not None
    assert with_def.has_default


def test_extract_keyword_only_default() -> None:
    """Keyword-only parameter with default."""
    node = _parse_function(COMPLEX_SIGNATURE)
    sig = extract_signature(node)
    kw_def = next(
        (p for p in sig.parameters if p.name == "keyword_with_default"),
        None,
    )
    assert kw_def is not None
    assert kw_def.has_default


# =============================================================================
# extract_signature Qualname Tests
# =============================================================================


def test_extract_default_qualname() -> None:
    """Default qualname equals function name."""
    node = _parse_function(SIMPLE_FUNCTION)
    sig = extract_signature(node)
    assert sig.qualname == sig.name


def test_extract_custom_qualname() -> None:
    """Custom qualname overrides default."""
    node = _parse_function(SIMPLE_FUNCTION)
    sig = extract_signature(node, qualname="module.submodule.greet")
    assert sig.qualname == "module.submodule.greet"


# =============================================================================
# extract_signature Annotation Tests
# =============================================================================


def test_extract_simple_annotation() -> None:
    """Extract simple type annotation."""
    node = _parse_function(TYPED_FUNCTION)
    sig = extract_signature(node)
    assert sig.parameters[0].annotation == "int"


def test_extract_complex_annotation() -> None:
    """Extract complex type annotation."""
    node = _parse_function(COMPLEX_SIGNATURE)
    sig = extract_signature(node)
    assert sig.return_annotation == "tuple[int, str]"


def test_extract_no_annotation() -> None:
    """Handle missing annotation."""
    node = _parse_function(SIMPLE_FUNCTION)
    sig = extract_signature(node)
    assert sig.parameters[0].annotation is None
    assert sig.return_annotation is None


# =============================================================================
# Integration Tests
# =============================================================================


def test_realistic_api_handler() -> None:
    """Extract signature from realistic API handler."""
    source = '''
async def create_user(
    request: Request,
    user_data: UserCreate,
    db: Database = Depends(get_db),
) -> UserResponse:
    """Create a new user.

    Parameters
    ----------
    request
        The incoming request.
    user_data
        User creation data.
    db
        Database connection.

    Returns
    -------
    UserResponse
        The created user.
    """
    pass
'''
    node = _parse_function(source)
    sig = extract_signature(node)
    assert sig.is_async
    assert len(sig.parameters) == EXPECTED_PARAMS_3
    assert sig.return_annotation == "UserResponse"


def test_realistic_test_function() -> None:
    """Extract signature from realistic test function."""
    source = '''
@pytest.fixture
def database_connection(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> Generator[Connection, None, None]:
    """Create test database connection.

    Yields
    ------
    Connection
        Database connection for testing.
    """
    pass
'''
    node = _parse_function(source)
    sig = extract_signature(node)
    assert "pytest.fixture" in sig.decorators[0]
    assert len(sig.parameters) == EXPECTED_PARAMS_2


def test_realistic_cli_command() -> None:
    """Extract signature from realistic CLI command."""
    source = '''
@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.argument('input_file', type=click.Path(exists=True))
def process(
    input_file: str,
    verbose: bool = False,
) -> None:
    """Process the input file."""
    pass
'''
    node = _parse_function(source)
    sig = extract_signature(node)
    assert len(sig.decorators) >= EXPECTED_PARAMS_2
    assert sig.return_annotation == "None"
