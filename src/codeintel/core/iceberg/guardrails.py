"""Guardrails for Iceberg enforcement policies."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.core.config.settings import IcebergSettings


@dataclass(frozen=True, slots=True)
class IcebergGuardrailError(RuntimeError):
    """Raised when Iceberg enforcement policies are violated."""

    table_key: str
    message: str

    def __str__(self) -> str:
        """Return a formatted guardrail error string.

        Returns
        -------
        str
            Formatted guardrail message.
        """
        return f"{self.table_key}: {self.message}"


def iceberg_enforced_table(*, settings: IcebergSettings, table_key: str) -> bool:
    """Return True when the table key must use Iceberg.

    Returns
    -------
    bool
        True when the table key matches enforcement prefixes.
    """
    prefixes = _normalized_prefixes(settings.enforced_table_prefixes)
    if not prefixes:
        return False
    return any(table_key.startswith(prefix) for prefix in prefixes)


def require_iceberg_write(*, settings: IcebergSettings, table_key: str) -> None:
    """Raise when an enforced table is written without Iceberg enabled.

    Raises
    ------
    IcebergGuardrailError
        When writes are required but disabled in settings.
    """
    if not iceberg_enforced_table(settings=settings, table_key=table_key):
        return
    if settings.write_enabled:
        return
    raise IcebergGuardrailError(
        table_key=table_key,
        message="Iceberg writes are required but write_enabled is false.",
    )


def require_iceberg_read(*, settings: IcebergSettings, table_key: str) -> None:
    """Raise when an enforced table is read without Iceberg enabled.

    Raises
    ------
    IcebergGuardrailError
        When reads are required but disabled in settings.
    """
    if not iceberg_enforced_table(settings=settings, table_key=table_key):
        return
    if settings.read_enabled:
        return
    raise IcebergGuardrailError(
        table_key=table_key,
        message="Iceberg reads are required but read_enabled is false.",
    )


def _normalized_prefixes(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = [prefix.strip() for prefix in prefixes if prefix.strip()]
    return tuple(cleaned)


__all__ = [
    "IcebergGuardrailError",
    "iceberg_enforced_table",
    "require_iceberg_read",
    "require_iceberg_write",
]
