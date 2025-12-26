"""Attribute normalizer budget tests."""

from __future__ import annotations

from collections.abc import Sequence

from hypothesis import given
from hypothesis import strategies as st

from codeintel.observability.attribute_schema import build_attribute_normalizer
from codeintel.observability.policy import AttributeBudget, ObservabilityPolicy
from codeintel.observability.semconv_keys import (
    CLI_ARG_NAMES,
    DB_QUERY_PARAMETER_PREFIX,
    HTTP_ROUTE,
    MCP_TOOL_NAME,
)

_MAX_LIST_LEN = 2
_MAX_STR_LEN = 5


def _budgeted_policy() -> ObservabilityPolicy:
    return ObservabilityPolicy(
        budget=AttributeBudget(max_list_len=_MAX_LIST_LEN, max_str_len=_MAX_STR_LEN)
    )


@given(
    st.dictionaries(
        keys=st.sampled_from([HTTP_ROUTE, MCP_TOOL_NAME, CLI_ARG_NAMES]),
        values=st.one_of(
            st.text(min_size=0, max_size=20),
            st.lists(st.text(min_size=0, max_size=20), min_size=0, max_size=5),
            st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=5),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=1,
        max_size=3,
    )
)
def test_normalizer_enforces_budget_limits(attrs: dict[str, object]) -> None:
    """Normalized attributes should respect string and list budgets."""
    normalizer = build_attribute_normalizer(_budgeted_policy())
    normalized = normalizer.normalize(attrs)
    for value in normalized.values():
        if isinstance(value, str):
            assert len(value) <= _MAX_STR_LEN
        elif isinstance(value, Sequence):
            assert len(value) <= _MAX_LIST_LEN
            for item in value:
                if isinstance(item, str):
                    assert len(item) <= _MAX_STR_LEN


def test_normalizer_allows_registered_prefix_keys() -> None:
    """Dynamic prefix keys should be accepted by the registry."""
    normalizer = build_attribute_normalizer(_budgeted_policy())
    attrs = {f"{DB_QUERY_PARAMETER_PREFIX}user_id": "abc123"}
    normalized = normalizer.normalize(attrs)
    assert f"{DB_QUERY_PARAMETER_PREFIX}user_id" in normalized
