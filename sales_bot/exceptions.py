from __future__ import annotations


class SalesBotError(Exception):
    """Base exception for bot domain failures."""


class ConfigurationError(SalesBotError):
    """Raised when required environment variables are missing."""


class AlreadyExistsError(SalesBotError):
    """Raised when a record already exists."""


class NotFoundError(SalesBotError):
    """Raised when a record cannot be found."""


class PermissionDeniedError(SalesBotError):
    """Raised when an action is not allowed."""


class ExternalServiceError(SalesBotError):
    """Raised when an upstream integration fails."""
