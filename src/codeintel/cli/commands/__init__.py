"""CLI commands package (deprecated - use handler modules directly).

This package previously contained Typer-based command groups. The CLI has been
migrated to Cyclopts; handlers are now located in the parent ``cli`` package:

- ``build_handlers.py`` - Build system handlers
- ``docs_handlers.py`` - Document export handlers
- ``datasets_handlers.py`` - Dataset management handlers
- ``graphs_handlers.py`` - Graph plugin handlers
- ``history_handlers.py`` - History timeseries handlers
- ``ide_handlers.py`` - IDE integration handlers
- ``subsystem_handlers.py`` - Subsystem exploration handlers
- ``storage_handlers.py`` - Storage validation handlers
- ``ops_handlers.py`` - Operation and serve handlers
- ``common_handlers.py`` - Shared utilities
"""

from __future__ import annotations

__all__: list[str] = []
