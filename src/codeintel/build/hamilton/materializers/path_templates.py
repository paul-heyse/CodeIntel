"""Artifact path template helpers for Hamilton materializers."""

from __future__ import annotations

from string import Formatter

_ALLOWED_KEYS: frozenset[str] = frozenset({"build_dir", "export_dir", "repo_root", "scip_dir"})
_FORMATTER = Formatter()


class PathTemplateError(ValueError):
    """Raised when an artifact path template uses unsupported placeholders."""

    def __init__(self, *, placeholder: str, allowed: frozenset[str]) -> None:
        message = f"Unsupported artifact path_template placeholder {placeholder!r} (allowed={sorted(allowed)})"
        super().__init__(message)
        self.placeholder = placeholder
        self.allowed = allowed


def validate_path_template(template: str) -> None:
    """Validate that a path template uses only allowed placeholders.

    Parameters
    ----------
    template
        Template string containing placeholder fields.

    Raises
    ------
    PathTemplateError
        If the template includes a placeholder outside the allowed set.
    """
    for _, field_name, _, _ in _FORMATTER.parse(template):
        if field_name is None:
            continue
        if field_name not in _ALLOWED_KEYS:
            raise PathTemplateError(
                placeholder=field_name,
                allowed=_ALLOWED_KEYS,
            )


def format_path_template(template: str, *, formatter: dict[str, str]) -> str:
    """Validate and format a path template.

    Parameters
    ----------
    template
        Template string containing placeholder fields.
    formatter
        Mapping used to format placeholders.

    Returns
    -------
    str
        Formatted path string.

    """
    validate_path_template(template)
    return template.format(**formatter)


def default_formatter(
    *,
    build_dir: str,
    scip_dir: str,
    export_dir: str,
    repo_root: str,
) -> dict[str, str]:
    """Return the default formatter mapping for artifact paths.

    Returns
    -------
    dict[str, str]
        Mapping of placeholder names to formatted path values.
    """
    return {
        "build_dir": build_dir,
        "scip_dir": scip_dir,
        "export_dir": export_dir,
        "repo_root": repo_root,
    }


__all__ = [
    "PathTemplateError",
    "default_formatter",
    "format_path_template",
    "validate_path_template",
]
