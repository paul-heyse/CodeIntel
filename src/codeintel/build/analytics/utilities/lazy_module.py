"""Lazy module loading utilities for analytics packages.

This module provides helpers for implementing lazy attribute loading in
package __init__.py files, reducing import-time overhead while maintaining
clean public APIs.

Example
-------
```python
from codeintel.build.analytics.utilities.lazy_module import (
    LazyAttrMap,
    lazy_callable,
    make_lazy_getattr,
)


_LAZY_ATTRS: LazyAttrMap = {
    "compute_metrics": ("mypackage.metrics", "compute_metrics"),
}


__getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__, cache_in_globals=globals())

# Optional: create explicit callable wrappers for IDE support
compute_metrics = lazy_callable(_LAZY_ATTRS, "compute_metrics")
```
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

LazyAttrMap = dict[str, tuple[str, str]]
"""Mapping of attribute names to (module_path, attr_name) tuples."""


def make_lazy_getattr(
    lazy_attrs: LazyAttrMap,
    module_name: str,
    *,
    cache_in_globals: dict[str, object] | None = None,
) -> Callable[[str], object]:
    """Create a __getattr__ function for lazy module attribute loading.

    Parameters
    ----------
    lazy_attrs
        Mapping of attribute names to (module_path, attr_name) tuples.
    module_name
        Name of the module (for error messages).
    cache_in_globals
        Optional globals() dict to cache loaded attributes in.

    Returns
    -------
    Callable[[str], object]
        A __getattr__ function suitable for module-level use.

    Example
    -------
    ```python
    _LAZY_ATTRS = {
        "compute_metrics": ("mypackage.metrics", "compute_metrics"),
    }


    __getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__, cache_in_globals=globals())
    ```
    """

    def _getattr_impl(name: str) -> object:
        if name not in lazy_attrs:
            message = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(message)
        module_path, attr_name = lazy_attrs[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        if cache_in_globals is not None:
            cache_in_globals[name] = attr
        return attr

    return _getattr_impl


def lazy_callable(
    lazy_attrs: LazyAttrMap,
    name: str,
) -> Callable[..., object]:
    """Create a lazy-loading callable wrapper for a module attribute.

    Parameters
    ----------
    lazy_attrs
        Mapping of attribute names to (module_path, attr_name) tuples.
    name
        Name of the attribute to wrap.

    Returns
    -------
    Callable[..., object]
        A wrapper that lazily loads and calls the target function.

    Example
    -------
    ```python
    _LAZY_ATTRS = {
        "compute_metrics": ("mypackage.metrics", "compute_metrics"),
    }


    compute_metrics = lazy_callable(_LAZY_ATTRS, "compute_metrics")
    ```
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        module_path, attr_name = lazy_attrs[name]
        module = importlib.import_module(module_path)
        func = getattr(module, attr_name)
        return func(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


__all__ = [
    "LazyAttrMap",
    "lazy_callable",
    "make_lazy_getattr",
]
