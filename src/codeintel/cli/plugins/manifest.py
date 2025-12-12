"""Plugin manifest schema and validation.

Define the structure for plugin manifests with semantic
versioning and capability declarations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


SEMVER_PATTERN = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$",
)


PLUGIN_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


CLI_API_VERSION = "1.0.0"


class PluginCapability(Enum):
    """Capabilities a plugin can request.

    Values
    ------
    REGISTER_OPERATIONS
        Register new operations.
    READ_CONFIG
        Read CLI configuration.
    WRITE_CONFIG
        Modify CLI configuration.
    READ_STORAGE
        Read from storage.
    WRITE_STORAGE
        Write to storage.
    EXECUTE_EXTERNAL
        Run external commands.
    NETWORK_ACCESS
        Make network requests.
    FILE_READ
        Read arbitrary files.
    FILE_WRITE
        Write arbitrary files.
    """

    REGISTER_OPERATIONS = "register_operations"
    READ_CONFIG = "read_config"
    WRITE_CONFIG = "write_config"
    READ_STORAGE = "read_storage"
    WRITE_STORAGE = "write_storage"
    EXECUTE_EXTERNAL = "execute_external"
    NETWORK_ACCESS = "network_access"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"


@dataclass(frozen=True)
class SemanticVersion:
    """Semantic version representation.

    Parameters
    ----------
    major
        Major version.
    minor
        Minor version.
    patch
        Patch version.
    prerelease
        Prerelease identifier.
    build
        Build metadata.
    """

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    @classmethod
    def parse(cls, version: str) -> SemanticVersion:
        """Parse version string.

        Parameters
        ----------
        version
            Version string.

        Returns
        -------
        SemanticVersion
            Parsed version.

        Raises
        ------
        ValueError
            If version is invalid.
        """
        match = SEMVER_PATTERN.match(version)
        if not match:
            msg = f"Invalid semantic version: {version}"
            raise ValueError(msg)

        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease"),
            build=match.group("buildmetadata"),
        )

    def __str__(self) -> str:
        """Convert to string.

        Returns
        -------
        str
            Version string.
        """
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result += f"-{self.prerelease}"
        if self.build:
            result += f"+{self.build}"
        return result

    def is_compatible_with(self, required: SemanticVersion) -> bool:
        """Check compatibility with required version.

        Use semver compatibility rules:
        - Major must match
        - Minor must be >= required
        - Patch can be any value

        Parameters
        ----------
        required
            Required version.

        Returns
        -------
        bool
            True if compatible.
        """
        if self.major != required.major:
            return False
        return self.minor >= required.minor


@dataclass
class PluginDependency:
    """Plugin dependency declaration.

    Parameters
    ----------
    name
        Dependency plugin name.
    version_requirement
        Version requirement string.
    optional
        Whether dependency is optional.
    """

    name: str
    version_requirement: str
    optional: bool = False


@dataclass
class PluginManifest:
    """Plugin manifest with metadata and capabilities.

    Parameters
    ----------
    name
        Plugin name.
    version
        Plugin version.
    api_version
        Required CLI API version.
    description
        Plugin description.
    author
        Plugin author.
    capabilities
        Requested capabilities.
    dependencies
        Plugin dependencies.
    entry_point
        Module entry point.
    """

    name: str
    version: str
    api_version: str
    description: str = ""
    author: str = ""
    capabilities: list[PluginCapability] = field(default_factory=list)
    dependencies: list[PluginDependency] = field(default_factory=list)
    entry_point: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        """Create manifest from dictionary.

        Parameters
        ----------
        data
            Manifest data.

        Returns
        -------
        PluginManifest
            Parsed manifest.
        """
        capabilities = [PluginCapability(cap) for cap in data.get("capabilities", [])]
        dependencies = [
            PluginDependency(
                name=dep["name"],
                version_requirement=dep.get("version", "*"),
                optional=dep.get("optional", False),
            )
            for dep in data.get("dependencies", [])
        ]

        return cls(
            name=data["name"],
            version=data["version"],
            api_version=data.get("api_version", CLI_API_VERSION),
            description=data.get("description", ""),
            author=data.get("author", ""),
            capabilities=capabilities,
            dependencies=dependencies,
            entry_point=data.get("entry_point", ""),
        )

    @classmethod
    def load(cls, path: Path) -> PluginManifest:
        """Load manifest from file.

        Parameters
        ----------
        path
            Path to manifest file.

        Returns
        -------
        PluginManifest
            Loaded manifest.
        """
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, Any]
            Manifest data.
        """
        return {
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "description": self.description,
            "author": self.author,
            "capabilities": [cap.value for cap in self.capabilities],
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version_requirement,
                    "optional": dep.optional,
                }
                for dep in self.dependencies
            ],
            "entry_point": self.entry_point,
        }

    def save(self, path: Path) -> None:
        """Save manifest to file.

        Parameters
        ----------
        path
            Path to save manifest.
        """
        path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    def validate(self) -> list[str]:
        """Validate manifest.

        Returns
        -------
        list[str]
            Validation errors.
        """
        errors: list[str] = []

        try:
            SemanticVersion.parse(self.version)
        except ValueError as e:
            errors.append(f"Invalid version: {e}")

        try:
            plugin_api = SemanticVersion.parse(self.api_version)
            cli_api = SemanticVersion.parse(CLI_API_VERSION)
            if not cli_api.is_compatible_with(plugin_api):
                errors.append(
                    f"Incompatible API version: plugin requires {self.api_version}, "
                    f"CLI provides {CLI_API_VERSION}",
                )
        except ValueError as e:
            errors.append(f"Invalid API version: {e}")

        if not PLUGIN_NAME_PATTERN.match(self.name):
            errors.append(
                "Invalid name: must be lowercase alphanumeric with hyphens/underscores",
            )

        if not self.entry_point:
            errors.append("Missing entry_point")

        return errors


__all__ = [
    "CLI_API_VERSION",
    "PLUGIN_NAME_PATTERN",
    "SEMVER_PATTERN",
    "PluginCapability",
    "PluginDependency",
    "PluginManifest",
    "SemanticVersion",
]
