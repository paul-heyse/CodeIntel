"""Global catalog hash computation for canonical registries."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeintel.build.config import CONFIG_FILE_NAME
from codeintel.core.hashing.content import content_hash, file_hash
from codeintel.core.schemas.hashing import schema_hash

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.hamilton.contracts.schemas.schema import DatasetSchema


@dataclass(frozen=True, slots=True)
class CatalogHashInputs:
    """Inputs used to compute the canonical catalog hash."""

    hamilton_digest: str
    schema_registry_hash: str
    build_config_hash: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable mapping of hash inputs.

        Returns
        -------
        dict[str, str | None]
            Mapping of input names to their hash values.
        """
        return {
            "hamilton_digest": self.hamilton_digest,
            "schema_registry_hash": self.schema_registry_hash,
            "build_config_hash": self.build_config_hash,
        }


def _iter_source_files(root: Path, *, subpaths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for subpath in subpaths:
        base = root / subpath
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(files)


def compute_hamilton_module_digest() -> str:
    """Compute a deterministic digest for Hamilton module sources.

    Returns
    -------
    str
        Deterministic digest representing Hamilton source files.
    """
    codeintel_root = Path(__file__).resolve().parents[2]
    source_files = _iter_source_files(
        codeintel_root,
        subpaths=(
            Path("build/hamilton/native"),
            Path("build/hamilton/templates"),
        ),
    )
    parts: list[str] = []
    for path in source_files:
        rel = path.relative_to(codeintel_root).as_posix()
        parts.append(f"{rel}:{file_hash(path)}")
    joined = "|".join(parts)
    return content_hash(joined)


def _schema_registry() -> Mapping[str, DatasetSchema]:
    module = importlib.import_module("codeintel.build.hamilton.contracts.schemas")
    registry = module.SCHEMA_REGISTRY
    return cast("Mapping[str, DatasetSchema]", registry)


def _schema_entry_hash(schema: DatasetSchema) -> str:
    if schema.ddl_schema is not None:
        return schema_hash(schema.ddl_schema)
    json_schema = schema.json_schema()
    payload = json.dumps(json_schema, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_schema_registry_hash() -> str:
    """Compute a deterministic hash over the canonical schema registry.

    Returns
    -------
    str
        Hash representing the schema registry contents.
    """
    entries: list[str] = []
    for table_key, schema in sorted(_schema_registry().items(), key=lambda item: item[0]):
        entries.append(f"{table_key}:{_schema_entry_hash(schema)}")
    joined = "|".join(entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_build_config_hash(root: Path | None = None) -> str | None:
    """Return the hash of the build config file when present.

    Returns
    -------
    str | None
        Hash of the build config file, or None when no config exists.
    """
    base = root or Path.cwd()
    config_path = base / CONFIG_FILE_NAME
    if not config_path.exists():
        return None
    return file_hash(config_path)


def compute_catalog_hash(inputs: CatalogHashInputs) -> str:
    """Compute the catalog hash for the given inputs.

    Returns
    -------
    str
        Digest of the catalog inputs.
    """
    payload = json.dumps(inputs.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_global_catalog_hash(root: Path | None = None) -> tuple[str, CatalogHashInputs]:
    """Compute the global catalog hash and return inputs.

    Returns
    -------
    tuple[str, CatalogHashInputs]
        Catalog hash and the inputs used to compute it.
    """
    inputs = CatalogHashInputs(
        hamilton_digest=compute_hamilton_module_digest(),
        schema_registry_hash=compute_schema_registry_hash(),
        build_config_hash=compute_build_config_hash(root),
    )
    return compute_catalog_hash(inputs), inputs


__all__ = [
    "CatalogHashInputs",
    "compute_build_config_hash",
    "compute_catalog_hash",
    "compute_global_catalog_hash",
    "compute_hamilton_module_digest",
    "compute_schema_registry_hash",
]
