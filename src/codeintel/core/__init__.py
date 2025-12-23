"""Core infrastructure shared between graphs and analytics subsystems.

This package contains unified protocols, types, and utilities that are
used by both the graphs and analytics subsystems, eliminating duplication
and ensuring consistency.

Subpackages
-----------
- config: Runtime settings dataclasses
- execution: Runtime execution infrastructure (telemetry, retry, timing)
- plugins: Unified plugin protocol, result types, and execution context
- resources: Unified resource provider protocol and registry

Modules
-------
- singleton: Thread-safe singleton holder pattern
"""

from __future__ import annotations

from codeintel.core.singleton import SingletonHolder

__all__ = [
    "SingletonHolder",
]
