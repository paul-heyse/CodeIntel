"""Shared pattern bundles for IO, concurrency, and framework detection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_IO_SPEC: dict[str, dict[str, list[str]]] = {
    "network": {
        "libs": ["requests", "httpx", "urllib3", "aiohttp", "socket", "boto3", "paramiko"],
        "funcs": ["get", "post", "put", "delete", "request", "send"],
    },
    "db": {
        "libs": ["sqlalchemy", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis"],
        "funcs": ["execute", "session", "commit", "query"],
    },
    "filesystem": {
        "libs": ["pathlib", "os", "shutil"],
        "funcs": ["open", "remove", "unlink", "copy", "move", "rmdir", "mkdir"],
    },
    "subprocess": {
        "libs": ["subprocess"],
        "funcs": ["run", "Popen", "call", "check_call", "check_output"],
    },
}

CONCURRENCY_LIBS: set[str] = {
    "asyncio",
    "anyio",
    "trio",
    "threading",
    "concurrent",
    "multiprocessing",
}

HTTP_CLIENT_LIBS: set[str] = {"requests", "httpx", "aiohttp"}
HTTP_SERVER_LIBS: set[str] = {"fastapi", "flask", "django", "starlette"}
DB_LIBS: set[str] = {"sqlalchemy", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis"}
MESSAGE_LIBS: set[str] = {"kafka", "pika", "celery", "kombu"}


@dataclass(frozen=True)
class AstFeaturePatterns:
    """
    Bundle of patterns used to derive FunctionAstFeatures.

    Allows easier testing and later customization (e.g., project-specific patterns).
    """

    io_spec: Mapping[str, dict[str, list[str]]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_IO_SPEC)
    )
    concurrency_libs: set[str] = field(default_factory=lambda: set(CONCURRENCY_LIBS))
    http_client_libs: set[str] = field(default_factory=lambda: set(HTTP_CLIENT_LIBS))
    http_server_libs: set[str] = field(default_factory=lambda: set(HTTP_SERVER_LIBS))
    db_libs: set[str] = field(default_factory=lambda: set(DB_LIBS))
    message_libs: set[str] = field(default_factory=lambda: set(MESSAGE_LIBS))


DEFAULT_PATTERNS = AstFeaturePatterns()

__all__ = [
    "CONCURRENCY_LIBS",
    "DB_LIBS",
    "DEFAULT_IO_SPEC",
    "DEFAULT_PATTERNS",
    "HTTP_CLIENT_LIBS",
    "HTTP_SERVER_LIBS",
    "MESSAGE_LIBS",
    "AstFeaturePatterns",
]
