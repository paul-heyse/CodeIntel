"""Runtime typing exports for modules compiled by Hamilton.

Hamilton evaluates function annotations at runtime via ``typing.get_type_hints``.
When annotations reference names like ``Mapping``/``Sequence``, they must be
present in module globals at runtime (not only under ``TYPE_CHECKING``).

Import these re-exports in Hamilton node modules to keep annotations resolvable
without scattering ``collections.abc`` imports everywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence

__all__ = ["Callable", "Collection", "Iterable", "Iterator", "Mapping", "Sequence"]
