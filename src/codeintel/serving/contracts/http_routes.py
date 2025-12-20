"""Canonical registry of serving HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteSpec:
    """HTTP route specification for serving endpoints."""

    method: str
    path: str
    name: str


SERVING_HTTP_ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(method="GET", path="/v1/semantic/views", name="semantic.list_views"),
    RouteSpec(method="GET", path="/v1/semantic/views/{view_id}", name="semantic.describe_view"),
    RouteSpec(method="POST", path="/v1/semantic/explain", name="semantic.explain_view"),
    RouteSpec(method="POST", path="/v1/semantic/query", name="semantic.query_view"),
    RouteSpec(method="POST", path="/v1/export/semantic/{view_id}", name="export.export_view"),
    RouteSpec(method="POST", path="/v1/search", name="search.search_view"),
)


__all__ = ["SERVING_HTTP_ROUTES", "RouteSpec"]
