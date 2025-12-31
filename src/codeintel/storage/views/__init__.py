"""View discovery helpers.

View discovery now relies on Hamilton tag metadata rather than precompiled SQL
view registries. The public surface remains minimal to avoid import cycles
during storage bootstrap.
"""

from __future__ import annotations

__all__: list[str] = []
