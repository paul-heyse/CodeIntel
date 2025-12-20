"""Canonical CodeIntel resource URIs and templates for serving."""

from __future__ import annotations

RESOURCE_URI_SCHEME = "codeintel://"

META_SERVING_URI = "codeintel://meta/serving"
META_RESOURCES_URI = "codeintel://meta/resources"
META_ENVIRONMENT_URI = "codeintel://meta/environment"
META_VIEWS_SQL_URI = "codeintel://meta/views_sql"
META_VIEWS_SQL_DIFF_URI = "codeintel://meta/views_sql_diff"

SEMANTIC_VIEWS_URI = "codeintel://semantic/views"
SEMANTIC_VIEW_URI_TEMPLATE = "codeintel://semantic/views/{view_id}"

EXPORT_RESOURCE_PREFIX = "codeintel://exports/"
EXPORT_URI_TEMPLATE = "codeintel://exports/{export_id}"
EXPORT_META_URI_TEMPLATE = "codeintel://exports/{export_id}/meta"
EXPORT_PREVIEW_URI_TEMPLATE = "codeintel://exports/{export_id}/preview"
EXPORT_SQL_URI_TEMPLATE = "codeintel://exports/{export_id}/sql"
EXPORT_LINES_URI_TEMPLATE = "codeintel://exports/{export_id}/lines{?offset,limit}"
EXPORT_BYTES_URI_TEMPLATE = "codeintel://exports/{export_id}/bytes{?offset,limit}"


def semantic_view_uri(view_id: str) -> str:
    """Return the URI for a semantic view description.

    Returns
    -------
    str
        Fully qualified semantic view URI.
    """
    return f"{SEMANTIC_VIEWS_URI}/{view_id}"


def export_uri(export_id: str) -> str:
    """Return the URI for an export payload.

    Returns
    -------
    str
        Fully qualified export payload URI.
    """
    return f"{EXPORT_RESOURCE_PREFIX}{export_id}"


def export_meta_uri(export_id: str) -> str:
    """Return the URI for export metadata.

    Returns
    -------
    str
        Fully qualified export metadata URI.
    """
    return f"{EXPORT_RESOURCE_PREFIX}{export_id}/meta"


def export_preview_uri(export_id: str) -> str:
    """Return the URI for an export preview.

    Returns
    -------
    str
        Fully qualified export preview URI.
    """
    return f"{EXPORT_RESOURCE_PREFIX}{export_id}/preview"


def export_sql_uri(export_id: str) -> str:
    """Return the URI for compiled export SQL.

    Returns
    -------
    str
        Fully qualified export SQL URI.
    """
    return f"{EXPORT_RESOURCE_PREFIX}{export_id}/sql"


def export_lines_uri_template(export_id: str) -> str:
    """Return the lines template URI for an export.

    Returns
    -------
    str
        Lines template URI with query expansion parameters.
    """
    return f"{EXPORT_RESOURCE_PREFIX}{export_id}/lines{{?offset,limit}}"


def export_bytes_uri_template(export_id: str) -> str:
    """Return the bytes template URI for an export.

    Returns
    -------
    str
        Bytes template URI with query expansion parameters.
    """
    return f"{EXPORT_RESOURCE_PREFIX}{export_id}/bytes{{?offset,limit}}"


__all__ = [
    "EXPORT_BYTES_URI_TEMPLATE",
    "EXPORT_LINES_URI_TEMPLATE",
    "EXPORT_META_URI_TEMPLATE",
    "EXPORT_PREVIEW_URI_TEMPLATE",
    "EXPORT_RESOURCE_PREFIX",
    "EXPORT_SQL_URI_TEMPLATE",
    "EXPORT_URI_TEMPLATE",
    "META_ENVIRONMENT_URI",
    "META_RESOURCES_URI",
    "META_SERVING_URI",
    "META_VIEWS_SQL_DIFF_URI",
    "META_VIEWS_SQL_URI",
    "RESOURCE_URI_SCHEME",
    "SEMANTIC_VIEWS_URI",
    "SEMANTIC_VIEW_URI_TEMPLATE",
    "export_bytes_uri_template",
    "export_lines_uri_template",
    "export_meta_uri",
    "export_preview_uri",
    "export_sql_uri",
    "export_uri",
    "semantic_view_uri",
]
