"""Pure computation layer for semantic role classification.

This module provides pure functions for classifying functions and modules
by their semantic role (api_handler, cli_command, repository, service, etc.).
All functions are side-effect-free and operate on in-memory data structures.
"""

from __future__ import annotations

from codeintel.analytics.compute.semantic_roles.classification import (
    FunctionContext,
    ModuleRecord,
    RoleAccumulator,
    RoleArtifacts,
    classify_function_role,
    classify_modules,
)

__all__ = [
    "FunctionContext",
    "ModuleRecord",
    "RoleAccumulator",
    "RoleArtifacts",
    "classify_function_role",
    "classify_modules",
]
