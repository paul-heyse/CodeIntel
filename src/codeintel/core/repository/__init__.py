"""Unified repository infrastructure.

This module provides the core repository patterns for the codebase,
including protocols, pagination, and filtering utilities.

Examples
--------
Using the repository protocol:

>>> from codeintel.core.repository import RepositoryProtocol, PagedResult, Pagination
>>>
>>> class UserRepository:
...     def get(self, id: int | str) -> User | None:
...         return self._db.get(id)
...
...     def list(
...         self,
...         *,
...         filters: Mapping[str, object] | None = None,
...         pagination: Pagination | None = None,
...     ) -> PagedResult[User]:
...         # Implementation here
...         pass

Using pagination:

>>> pagination = Pagination(limit=10, offset=20)
>>> pagination = Pagination.for_page(2, page_size=10)

Using filters:

>>> from codeintel.core.repository import FilterBuilder
>>>
>>> filters = FilterBuilder().eq("status", "active").gte("age", 18).build()
"""

from codeintel.core.repository.filtering import (
    FilterBuilder,
    FilterCondition,
    FilterOperator,
    parse_filters,
)
from codeintel.core.repository.pagination import (
    PagedResult,
    Pagination,
)
from codeintel.core.repository.protocol import (
    BulkRepositoryProtocol,
    RepositoryProtocol,
    WriteableRepositoryProtocol,
)

__all__ = [
    "BulkRepositoryProtocol",
    "FilterBuilder",
    "FilterCondition",
    "FilterOperator",
    "PagedResult",
    "Pagination",
    "RepositoryProtocol",
    "WriteableRepositoryProtocol",
    "parse_filters",
]
