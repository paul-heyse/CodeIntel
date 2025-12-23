"""Data models and config data flow seed pack.

This module provides DataModelsPack which seeds data for testing data model
analytics, config data flow, and model heuristics. It includes modules with
data classes, ORM models, and config-consuming code.

The pack is designed for tests like test_model_config_heuristics.py that need
realistic data model source code and type annotations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.assertions import ModulesAssertions
from tests._helpers.builders import (
    CallGraphNodeRow,
    FunctionTypesRow,
    GoidRow,
    ModuleRow,
    RepoMapRow,
    insert_rows,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree
from tests._helpers.seeds.core import CORE_PACK

if TYPE_CHECKING:
    from pathlib import Path

    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Data Models Constants
# =============================================================================

# Module paths for data model tests
MOD_MODELS_PATH = "pkg/models.py"
MOD_DB_PATH = "pkg/db.py"
MOD_API_HANDLERS_PATH = "pkg/api_handlers.py"
MOD_CONFIG_PATH = "pkg/config_loader.py"

MOD_MODELS_FQN = "pkg.models"
MOD_DB_FQN = "pkg.db"
MOD_API_HANDLERS_FQN = "pkg.api_handlers"
MOD_CONFIG_FQN = "pkg.config_loader"

# GOID hashes for data model test functions/classes
GOID_USER_CLASS = 2001
GOID_POST_CLASS = 2002
GOID_USER_PAYLOAD_CLASS = 2003
GOID_CREATE_USER = 2004
GOID_FETCH_USER = 2005
GOID_SERIALIZE_POST = 2006
GOID_SERIALIZE_PAYLOAD = 2007
GOID_CONFIG_CHECKS = 2008

# Sample source code for data model tests
MODELS_SOURCE = '''\
"""Data models for testing."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """User data model."""

    id: int
    name: str
    email: Optional[str] = None


@dataclass
class Post:
    """Post data model."""

    id: int
    title: str
    author_id: int
    content: str


class UserPayload:
    """User payload for API."""

    def __init__(self, user_id: int, action: str) -> None:
        self.user_id = user_id
        self.action = action
'''

DB_SOURCE = '''\
"""Database operations for testing."""

from typing import Optional
from sqlalchemy.orm import Session
from pkg.models import User


def create_user(session: Session, name: str) -> User:
    """Create a new user.

    Parameters
    ----------
    session
        Database session.
    name
        User name.

    Returns
    -------
    User
        Created user instance.
    """
    user = User(id=1, name=name)
    return user


def fetch_user(session: Session) -> Optional[User]:
    """Fetch a user.

    Parameters
    ----------
    session
        Database session.

    Returns
    -------
    Optional[User]
        User if found, None otherwise.
    """
    return None
'''

API_HANDLERS_SOURCE = '''\
"""API handlers for testing."""

from pkg.models import Post, UserPayload


def serialize_post(post: Post) -> dict[str, object]:
    """Serialize a post.

    Parameters
    ----------
    post
        Post to serialize.

    Returns
    -------
    dict[str, object]
        Serialized post data.
    """
    return {"id": post.id, "title": post.title}


def serialize_payload(payload: UserPayload) -> dict[str, object]:
    """Serialize a payload.

    Parameters
    ----------
    payload
        Payload to serialize.

    Returns
    -------
    dict[str, object]
        Serialized payload data.
    """
    return {"user_id": payload.user_id, "action": payload.action}
'''

CONFIG_SOURCE = '''\
"""Config loader for testing."""


def config_checks(settings: dict[str, object]) -> bool:
    """Check config settings.

    Parameters
    ----------
    settings
        Settings dictionary.

    Returns
    -------
    bool
        True if valid.
    """
    return "enabled" in settings
'''


# =============================================================================
# Data Models Pack Implementation
# =============================================================================


@dataclass
class DataModelsPack:
    """Seed pack for data model analytics testing.

    Seeds modules, GOIDs, call graph nodes, and function types for testing
    data model detection, config data flow, and type heuristics.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    write_source_files : bool
        Whether to write sample source files to repo root.
    include_function_types : bool
        Whether to seed function types.
    include_call_graph_nodes : bool
        Whether to seed call graph nodes.
    """

    name: str = "data_models"
    write_source_files: bool = True
    include_function_types: bool = True
    include_call_graph_nodes: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for base data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply data models seeds to the test context.

        Seeds modules, GOIDs, call graph nodes, and function types.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        # Write source files if requested
        if self.write_source_files:
            self._write_source_files(ctx.repo_root)

        # Seed database tables
        module_map = self._resolve_module_map(ctx)
        self._seed_repo_map(ctx, module_map)
        self._seed_modules(ctx, module_map)
        ModulesAssertions(ctx.gateway, ctx.snapshot).inventory_consistent()
        self._seed_goids(ctx, now)

        if self.include_call_graph_nodes:
            self._seed_call_graph_nodes(ctx)

        if self.include_function_types:
            self._seed_function_types(ctx, now)

    @staticmethod
    def _write_source_files(repo_root: Path) -> None:
        """Write sample source files to repository.

        Parameters
        ----------
        repo_root
            Repository root path.
        """
        pkg_dir = repo_root / "pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        # Ensure __init__.py exists
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Test package."""\n', encoding="utf-8")

        # Write source files
        (pkg_dir / "models.py").write_text(MODELS_SOURCE, encoding="utf-8")
        (pkg_dir / "db.py").write_text(DB_SOURCE, encoding="utf-8")
        (pkg_dir / "api_handlers.py").write_text(API_HANDLERS_SOURCE, encoding="utf-8")
        (pkg_dir / "config_loader.py").write_text(CONFIG_SOURCE, encoding="utf-8")

    @staticmethod
    def _seed_repo_map(ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed the core.repo_map table."""
        rows = [
            RepoMapRow(
                repo=ctx.repo,
                commit=ctx.commit,
                modules=module_map,
                overlays={},
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_modules(ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed modules table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        module_map
            Module map keyed by module name to repo-relative paths.
        """
        rows = [
            ModuleRow(module=module, path=path, repo=ctx.repo, commit=ctx.commit)
            for module, path in sorted(module_map.items())
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _resolve_module_map(ctx: TestContext) -> dict[str, str]:
        path_map = modules_expected_from_repo_tree(ctx.repo_root)
        module_map = {module: path for path, module in path_map.items()}
        if not module_map:
            module_map = {
                MOD_MODELS_FQN: MOD_MODELS_PATH,
                MOD_DB_FQN: MOD_DB_PATH,
                MOD_API_HANDLERS_FQN: MOD_API_HANDLERS_PATH,
                MOD_CONFIG_FQN: MOD_CONFIG_PATH,
            }
        return module_map

    @staticmethod
    def _seed_goids(ctx: TestContext, now: datetime) -> None:
        """Seed GOIDs table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows = [
            # Classes
            GoidRow(
                goid_h128=GOID_USER_CLASS,
                urn=f"goid:{ctx.repo}/{MOD_MODELS_PATH}#{MOD_MODELS_FQN}.User",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_MODELS_PATH,
                kind="class",
                qualname=f"{MOD_MODELS_FQN}.User",
                start_line=7,
                end_line=14,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_POST_CLASS,
                urn=f"goid:{ctx.repo}/{MOD_MODELS_PATH}#{MOD_MODELS_FQN}.Post",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_MODELS_PATH,
                kind="class",
                qualname=f"{MOD_MODELS_FQN}.Post",
                start_line=17,
                end_line=25,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_USER_PAYLOAD_CLASS,
                urn=f"goid:{ctx.repo}/{MOD_MODELS_PATH}#{MOD_MODELS_FQN}.UserPayload",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_MODELS_PATH,
                kind="class",
                qualname=f"{MOD_MODELS_FQN}.UserPayload",
                start_line=28,
                end_line=35,
                language="python",
                created_at=now,
            ),
            # Functions
            GoidRow(
                goid_h128=GOID_CREATE_USER,
                urn=f"goid:{ctx.repo}/{MOD_DB_PATH}#{MOD_DB_FQN}.create_user",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_DB_PATH,
                kind="function",
                qualname=f"{MOD_DB_FQN}.create_user",
                start_line=8,
                end_line=25,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_FETCH_USER,
                urn=f"goid:{ctx.repo}/{MOD_DB_PATH}#{MOD_DB_FQN}.fetch_user",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_DB_PATH,
                kind="function",
                qualname=f"{MOD_DB_FQN}.fetch_user",
                start_line=28,
                end_line=44,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_SERIALIZE_POST,
                urn=(
                    f"goid:{ctx.repo}/{MOD_API_HANDLERS_PATH}"
                    f"#{MOD_API_HANDLERS_FQN}.serialize_post"
                ),
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_API_HANDLERS_PATH,
                kind="function",
                qualname=f"{MOD_API_HANDLERS_FQN}.serialize_post",
                start_line=6,
                end_line=20,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_SERIALIZE_PAYLOAD,
                urn=f"goid:{ctx.repo}/{MOD_API_HANDLERS_PATH}#"
                f"{MOD_API_HANDLERS_FQN}.serialize_payload",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_API_HANDLERS_PATH,
                kind="function",
                qualname=f"{MOD_API_HANDLERS_FQN}.serialize_payload",
                start_line=23,
                end_line=37,
                language="python",
                created_at=now,
            ),
            GoidRow(
                goid_h128=GOID_CONFIG_CHECKS,
                urn=f"goid:{ctx.repo}/{MOD_CONFIG_PATH}#{MOD_CONFIG_FQN}.config_checks",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=MOD_CONFIG_PATH,
                kind="function",
                qualname=f"{MOD_CONFIG_FQN}.config_checks",
                start_line=4,
                end_line=18,
                language="python",
                created_at=now,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_call_graph_nodes(ctx: TestContext) -> None:
        """Seed call graph nodes for all GOIDs.

        Parameters
        ----------
        ctx
            Test context with gateway.
        """
        goid_configs = [
            (GOID_USER_CLASS, "class", MOD_MODELS_PATH),
            (GOID_POST_CLASS, "class", MOD_MODELS_PATH),
            (GOID_USER_PAYLOAD_CLASS, "class", MOD_MODELS_PATH),
            (GOID_CREATE_USER, "function", MOD_DB_PATH),
            (GOID_FETCH_USER, "function", MOD_DB_PATH),
            (GOID_SERIALIZE_POST, "function", MOD_API_HANDLERS_PATH),
            (GOID_SERIALIZE_PAYLOAD, "function", MOD_API_HANDLERS_PATH),
            (GOID_CONFIG_CHECKS, "function", MOD_CONFIG_PATH),
        ]

        rows = [
            CallGraphNodeRow(
                goid_h128=goid,
                language="python",
                kind=kind,
                arity=0,
                is_public=True,
                rel_path=rel_path,
            )
            for goid, kind, rel_path in goid_configs
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_function_types(ctx: TestContext, now: datetime) -> None:
        """Seed function types for type heuristics testing.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        # Type information for each function
        type_configs = [
            (
                GOID_CREATE_USER,
                MOD_DB_FQN,
                MOD_DB_PATH,
                "create_user",
                {"session": "Session", "name": "str"},
                "User",
            ),
            (
                GOID_FETCH_USER,
                MOD_DB_FQN,
                MOD_DB_PATH,
                "fetch_user",
                {"session": "Session"},
                "User | None",
            ),
            (
                GOID_SERIALIZE_POST,
                MOD_API_HANDLERS_FQN,
                MOD_API_HANDLERS_PATH,
                "serialize_post",
                {"post": "Post"},
                "dict[str, object]",
            ),
            (
                GOID_SERIALIZE_PAYLOAD,
                MOD_API_HANDLERS_FQN,
                MOD_API_HANDLERS_PATH,
                "serialize_payload",
                {"payload": "UserPayload"},
                "dict[str, object]",
            ),
            (
                GOID_CONFIG_CHECKS,
                MOD_CONFIG_FQN,
                MOD_CONFIG_PATH,
                "config_checks",
                {"settings": "dict[str, object]"},
                "bool",
            ),
        ]

        rows = []
        for goid, module_fqn, rel_path, func_name, params, return_type in type_configs:
            qualname = f"{module_fqn}.{func_name}"
            total_params = len(params)
            rows.append(
                FunctionTypesRow(
                    function_goid_h128=goid,
                    urn=f"goid:{ctx.repo}/{rel_path}#{qualname}",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=rel_path,
                    language="python",
                    kind="function",
                    qualname=qualname,
                    start_line=1,
                    end_line=10,
                    total_params=total_params,
                    annotated_params=total_params,
                    unannotated_params=0,
                    param_typed_ratio=1.0,
                    has_return_annotation=True,
                    return_type=return_type,
                    return_type_source="annotation",
                    type_comment=None,
                    param_types_json=json.dumps(params),
                    fully_typed=True,
                    partial_typed=False,
                    untyped=False,
                    typedness_bucket="fully_typed",
                    typedness_source="annotation",
                    created_at=now,
                )
            )
        insert_rows(ctx.gateway, rows)


# Default instance for common usage
DATA_MODELS_PACK = DataModelsPack()


__all__ = [
    "API_HANDLERS_SOURCE",
    "CONFIG_SOURCE",
    "DATA_MODELS_PACK",
    "DB_SOURCE",
    "GOID_CONFIG_CHECKS",
    "GOID_CREATE_USER",
    "GOID_FETCH_USER",
    "GOID_POST_CLASS",
    "GOID_SERIALIZE_PAYLOAD",
    "GOID_SERIALIZE_POST",
    "GOID_USER_CLASS",
    "GOID_USER_PAYLOAD_CLASS",
    "MODELS_SOURCE",
    "MOD_API_HANDLERS_FQN",
    "MOD_API_HANDLERS_PATH",
    "MOD_CONFIG_FQN",
    "MOD_CONFIG_PATH",
    "MOD_DB_FQN",
    "MOD_DB_PATH",
    "MOD_MODELS_FQN",
    "MOD_MODELS_PATH",
    "DataModelsPack",
]
