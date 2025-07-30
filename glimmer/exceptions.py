"""Exceptions for the GLIMMER package.

This module defines custom exceptions used throughout the GLIMMER package
for error handling and validation.
"""

class GlimmerError(Exception):
    """Base exception for GLIMMER-related errors."""
    pass

class GlimmerValidationError(GlimmerError):
    """Raised when a GLIMMER pattern fails validation."""
    pass

class GlimmerProcessingError(GlimmerError):
    """Raised when there's an error processing a GLIMMER pattern."""
    pass

class GlimmerFormatError(GlimmerError):
    """Raised when there's an error in the format of a GLIMMER pattern."""
    pass
