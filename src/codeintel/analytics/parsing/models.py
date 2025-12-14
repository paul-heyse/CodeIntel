"""Shared parsing dataclasses for analytics subsystems.

Note
----
As of v5.0.0, SourceSpan, ParsedFunction, and ParsedModule are defined
in codeintel.core.parsing and re-exported here for backward compatibility.
New code should import from codeintel.core.parsing directly.
"""

from __future__ import annotations

# Re-export from core for backward compatibility
from codeintel.core.parsing import ParsedFunction, ParsedModule, SourceSpan

__all__ = [
    "ParsedFunction",
    "ParsedModule",
    "SourceSpan",
]
