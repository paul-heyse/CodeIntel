"""Cryptographic constants used throughout the codebase.

This module provides named constants for cryptographic values,
eliminating magic numbers (PLR2004) and documenting domain knowledge.

Examples
--------
>>> from codeintel.core.constants.crypto import SHA256_HEX_DIGEST_LENGTH
>>> digest = "a" * 64  # Valid SHA-256 hex digest
>>> len(digest) == SHA256_HEX_DIGEST_LENGTH
True
"""

from __future__ import annotations

# SHA-256 produces 32 bytes = 256 bits
SHA256_DIGEST_BYTES: int = 32

# SHA-256 hex digest is 64 characters (32 bytes * 2 hex chars per byte)
SHA256_HEX_DIGEST_LENGTH: int = SHA256_DIGEST_BYTES * 2  # 64

__all__ = [
    "SHA256_DIGEST_BYTES",
    "SHA256_HEX_DIGEST_LENGTH",
]
