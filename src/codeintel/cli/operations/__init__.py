"""Operation definitions package.

Operations are now registered via the @cli_command decorator in command modules.
This package is retained for potential future use but no longer triggers
side-effect registrations.

Note: The LEGACY operation registration files have been removed in Phase 6.
Operations are now registered by the @cli_command decorator in commands/*.py.
"""

from __future__ import annotations

__all__: list[str] = []
