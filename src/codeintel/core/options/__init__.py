"""Unified options and configuration protocol for CodeIntel.

This module provides a consistent pattern for options/config classes
across all modules, enabling validation, composition, and serialization.

Examples
--------
>>> from codeintel.core.options import OptionsProtocol, BaseOptions
>>> @dataclass(frozen=True)
... class MyOptions(BaseOptions):
...     timeout_ms: int = 5000
...     retry_count: int = 3
>>> opts = MyOptions()
>>> opts.validate().ok
True
"""

from __future__ import annotations

from codeintel.core.options.base import BaseOptions
from codeintel.core.options.protocol import OptionsProtocol, ValidationResult

__all__ = [
    "BaseOptions",
    "OptionsProtocol",
    "ValidationResult",
]
