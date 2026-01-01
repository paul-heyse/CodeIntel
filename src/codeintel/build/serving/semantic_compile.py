"""Deprecated semantic registry compiler shim.

Use `codeintel.serving.semantic_compile` instead.
"""

from __future__ import annotations

msg = "codeintel.build.serving.semantic_compile moved to codeintel.serving.semantic_compile"
raise ImportError(msg)
