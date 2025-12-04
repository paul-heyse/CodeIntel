"""Core infrastructure shared between graphs and analytics subsystems.

This package contains unified protocols, types, and utilities that are
used by both the graphs and analytics subsystems, eliminating duplication
and ensuring consistency.

Modules
-------
- plugins: Unified plugin protocol, result types, and registry
- recipes: Unified recipe DSL and executor
- resources: Unified resource provider protocol and registry
"""

from __future__ import annotations

__all__: list[str] = []
