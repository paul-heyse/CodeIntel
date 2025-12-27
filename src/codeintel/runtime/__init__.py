"""Runtime composition and bundle interfaces."""

from __future__ import annotations

from codeintel.runtime.compose import RuntimeComposition, compose_runtime, set_execution_active
from codeintel.runtime.inputs import ExecutionInputs
from codeintel.runtime.module_resolver import resolve_module_paths, resolve_modules
from codeintel.runtime.registry import RuntimeRegistry
from codeintel.runtime.runtime_bundle import RuntimeBundle, RuntimeKey

__all__ = [
    "ExecutionInputs",
    "RuntimeBundle",
    "RuntimeComposition",
    "RuntimeKey",
    "RuntimeRegistry",
    "compose_runtime",
    "resolve_module_paths",
    "resolve_modules",
    "set_execution_active",
]
