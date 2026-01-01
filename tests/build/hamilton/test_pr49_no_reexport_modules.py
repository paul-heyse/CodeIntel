"""PR49: compatibility re-export modules are removed."""

from __future__ import annotations

import importlib.util

import pytest


def test_pr49_compat_reexport_modules_removed() -> None:
    """Verify legacy compatibility modules are not importable."""
    removed_modules = (
        "codeintel.build.analytics.resources.registry",
        "codeintel.build.graphs.catalog",
        "codeintel.build.graphs.ports.catalog",
        "codeintel.build.graphs.ports.storage",
    )
    still_present = [name for name in removed_modules if importlib.util.find_spec(name) is not None]
    if still_present:
        message = "Compatibility modules still importable:\n" + "\n".join(still_present)
        pytest.fail(message)
