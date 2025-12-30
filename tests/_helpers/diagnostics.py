"""Debug helpers for inspecting runtime composition in tests."""

from __future__ import annotations

import logging
import os

from codeintel.runtime.runtime_bundle import RuntimeBundle

log = logging.getLogger(__name__)


def maybe_log_runtime_modules(runtime: RuntimeBundle, *, label: str = "runtime") -> None:
    """Optionally log resolved module imports for runtime composition."""
    if not _debug_modules_enabled():
        return
    entries = sorted(runtime.module_provenance.values(), key=lambda item: item.module_import)
    module_paths = [entry.module_import for entry in entries]
    log.warning(
        "debug.runtime.modules label=%s count=%d modules=%s",
        label,
        len(module_paths),
        module_paths,
    )


def _debug_modules_enabled() -> bool:
    value = os.getenv("CODEINTEL_DEBUG_MODULES", "")
    return value.lower() in {"1", "true", "yes", "on"}


__all__ = ["maybe_log_runtime_modules"]
