"""Core constants for CodeIntel.

This package provides named constants for common values used throughout
the codebase, eliminating magic numbers (PLR2004).

Submodules
----------
- crypto: Cryptographic constants (SHA-256 digest lengths, etc.)
"""

from __future__ import annotations

from codeintel.core.constants.crypto import (
    SHA256_DIGEST_BYTES,
    SHA256_HEX_DIGEST_LENGTH,
)

__all__ = [
    "SHA256_DIGEST_BYTES",
    "SHA256_HEX_DIGEST_LENGTH",
]
