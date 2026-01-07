"""Tests for codeintel.build.analytics.compute.graphs.conversions module.

Testing Charter Compliance:
- Pure function tests with realistic edge cases
- No monkeypatching or test-only code paths
- Tests actual production code paths for ID normalization
"""

from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from codeintel.build.analytics.compute.graphs.conversions import (
    log_empty_graph,
    log_projection_skipped,
    safe_float,
)
from codeintel.build.graphs.rx.store import RxGraphStore
from codeintel.core.data_models.ids import normalize_decimal_id
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_none,
    expect_not_in,
)


class TestNormalizeDecimalId:
    """Tests for normalize_decimal_id function."""

    @staticmethod
    def test_returns_none_for_none() -> None:
        """Verify None input returns None."""
        expect_is_none(normalize_decimal_id(None))

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (123, 123),
            (0, 0),
            (-1, -1),
        ],
    )
    def test_passes_through_integers(input_val: int, expected: int) -> None:
        """Verify integer values pass through unchanged."""
        result = normalize_decimal_id(input_val)
        expect_equal(result, expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (Decimal("123"), 123),
            (Decimal("0"), 0),
            (Decimal("-456"), -456),
            (Decimal("999999999999999999"), 999999999999999999),
        ],
    )
    def test_converts_decimal_to_int(input_val: Decimal, expected: int) -> None:
        """Verify Decimal values are converted to integers."""
        result = normalize_decimal_id(input_val)
        expect_equal(result, expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (b"123", 123),
            (b"456", 456),
            (b"0", 0),
        ],
    )
    def test_decodes_bytes(input_val: bytes, expected: int) -> None:
        """Verify bytes values are decoded and converted."""
        result = normalize_decimal_id(input_val)
        expect_equal(result, expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (bytearray(b"123"), 123),
            (bytearray(b"789"), 789),
        ],
    )
    def test_decodes_bytearray(input_val: bytearray, expected: int) -> None:
        """Verify bytearray values are decoded and converted."""
        result = normalize_decimal_id(input_val)
        expect_equal(result, expected)

    @staticmethod
    def test_returns_none_for_invalid_bytes() -> None:
        """Verify invalid UTF-8 bytes return None."""
        invalid_bytes = b"\xff\xfe"
        result = normalize_decimal_id(invalid_bytes)
        expect_is_none(result)

    @staticmethod
    def test_returns_none_for_non_numeric_bytes() -> None:
        """Verify non-numeric bytes return None."""
        result = normalize_decimal_id(b"not_a_number")
        expect_is_none(result)

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            ("123", 123),
            ("456", 456),
        ],
    )
    def test_converts_string(input_val: str, expected: int) -> None:
        """Verify string values are converted."""
        result = normalize_decimal_id(input_val)
        expect_equal(result, expected)

    @staticmethod
    def test_converts_float_via_string() -> None:
        """Verify float values are converted via string representation."""
        result = normalize_decimal_id(123.0)
        expect_is_none(result)

    @staticmethod
    def test_returns_none_for_unconvertible() -> None:
        """Verify unconvertible values return None."""
        result = normalize_decimal_id("not_a_number")
        expect_is_none(result)


class TestSafeFloat:
    """Tests for safe_float function."""

    @staticmethod
    def test_returns_none_for_none() -> None:
        """Verify None input returns None."""
        expect_is_none(safe_float(None))

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (1.5, 1.5),
            (0.0, 0.0),
            (-3.14, -3.14),
        ],
    )
    def test_passes_through_floats(input_val: float, expected: float) -> None:
        """Verify float values pass through unchanged."""
        result = safe_float(input_val)
        expect_equal(result, expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (Decimal("1.5"), 1.5),
            (Decimal("0.0"), 0.0),
            (Decimal("-3.14"), -3.14),
        ],
    )
    def test_converts_decimal(input_val: Decimal, expected: float) -> None:
        """Verify Decimal values are converted to float."""
        result = safe_float(input_val)
        expect_equal(result, pytest.approx(expected))

    @staticmethod
    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            ("1.5", 1.5),
            ("0.0", 0.0),
            ("-3.14", -3.14),
        ],
    )
    def test_converts_string(input_val: str, expected: float) -> None:
        """Verify string values are converted to float."""
        result = safe_float(input_val)
        expect_equal(result, pytest.approx(expected))

    @staticmethod
    def test_returns_none_for_invalid() -> None:
        """Verify invalid values return None."""
        result = safe_float("not_a_float")
        expect_is_none(result)


class TestLogEmptyGraph:
    """Tests for log_empty_graph function."""

    @staticmethod
    def test_logs_for_empty_graph(caplog: pytest.LogCaptureFixture) -> None:
        """Verify debug log is emitted for empty graph."""
        empty_graph = RxGraphStore.undirected()

        with caplog.at_level(logging.DEBUG):
            log_empty_graph("test_graph", empty_graph)

        expect_in("test_graph is empty", caplog.text)

    @staticmethod
    def test_no_log_for_nonempty_graph(caplog: pytest.LogCaptureFixture) -> None:
        """Verify no log is emitted for non-empty graph."""
        graph = RxGraphStore.undirected()
        graph.ensure_node(1)

        with caplog.at_level(logging.DEBUG):
            log_empty_graph("test_graph", graph)

        expect_not_in("empty", caplog.text)

    @staticmethod
    def test_works_with_digraph(caplog: pytest.LogCaptureFixture) -> None:
        """Verify function works with DiGraph."""
        empty_digraph = RxGraphStore.directed()

        with caplog.at_level(logging.DEBUG):
            log_empty_graph("call_graph", empty_digraph)

        expect_in("call_graph is empty", caplog.text)


class TestLogProjectionSkipped:
    """Tests for log_projection_skipped function."""

    @staticmethod
    def test_logs_info(caplog: pytest.LogCaptureFixture) -> None:
        """Verify info log is emitted with correct details."""
        with caplog.at_level(logging.INFO):
            log_projection_skipped(
                "test_projection",
                "insufficient nodes",
                nodes=5,
                graph_nodes=10,
            )

        expect_in("test_projection", caplog.text)
        expect_in("insufficient nodes", caplog.text)
        expect_in("partition_size=5", caplog.text)
        expect_in("graph_nodes=10", caplog.text)

    @staticmethod
    def test_logs_with_different_reasons(caplog: pytest.LogCaptureFixture) -> None:
        """Verify various skip reasons are logged correctly."""
        reasons = [
            "no bipartite structure",
            "empty partition",
            "single node partition",
        ]

        for reason in reasons:
            with caplog.at_level(logging.INFO):
                log_projection_skipped("proj", reason, nodes=1, graph_nodes=2)
                expect_in(reason, caplog.text)
            caplog.clear()


class TestIntegrationScenarios:
    """Integration tests combining multiple conversion functions."""

    @staticmethod
    def test_roundtrip_decimal_normalization() -> None:
        """Verify Decimal IDs normalize to ints."""
        original = Decimal("123456789012345678901234567890")
        normalized = normalize_decimal_id(original)
        expect_equal(normalized, int(original))

    @staticmethod
    def test_graph_node_id_normalization() -> None:
        """Verify numeric graph nodes normalize to ints."""
        graph = RxGraphStore.directed()
        graph.ensure_node(Decimal("123"))
        graph.ensure_node(456)
        graph.ensure_node("789")
        graph.ensure_node("func_name")

        normalized = {normalize_decimal_id(n) for n in graph.node_ids()}
        expected = {123, 456, 789, None}
        expect_equal(normalized, expected)

    @staticmethod
    def test_safe_float_for_metrics() -> None:
        """Verify safe_float handles metric values correctly."""
        metrics = [
            "0.75",
            "0.5",
            0.25,
            None,
            "invalid",
        ]
        results = [safe_float(m) for m in metrics]
        expected = [0.75, 0.5, 0.25, None, None]

        for result, exp in zip(results, expected, strict=True):
            if exp is None:
                expect_is_none(result)
            else:
                expect_equal(result, pytest.approx(exp))
