"""
GLIMMER: Generative Language Interface for Meta-Modeling and Evolutionary Responses

This module provides tools for working with GLIMMER patterns, including validation,
processing, and transformation for fine-tuning language models with the STARWEAVE framework.
"""

__version__ = "0.1.0"

from .core import GlimmerPattern, validate_pattern, load_pattern, save_pattern
from .processor import GlimmerProcessor
from .exceptions import GlimmerValidationError

__all__ = [
    'GlimmerPattern',
    'validate_pattern',
    'load_pattern',
    'save_pattern',
    'GlimmerProcessor',
    'GlimmerValidationError',
]
