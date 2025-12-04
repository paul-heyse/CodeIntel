"""Core infrastructure shared between graphs and analytics subsystems.

This package contains unified protocols, types, and utilities that are
used by both the graphs and analytics subsystems, eliminating duplication
and ensuring consistency.

Modules
-------
- plugins: Unified plugin protocol, result types, and registry
- recipes: Unified recipe DSL and executor
- resources: Unified resource provider protocol and registry
- config_registry: Type-safe configuration registry
- singleton: Thread-safe singleton holder pattern
"""

from __future__ import annotations

from codeintel.core.config_registry import (
    ConfigNotFoundError,
    ConfigRegistry,
    ConfigTypeError,
    ConfigValidationError,
)
from codeintel.core.singleton import SingletonHolder

__all__ = [
    "ConfigNotFoundError",
    "ConfigRegistry",
    "ConfigTypeError",
    "ConfigValidationError",
    "SingletonHolder",
]
