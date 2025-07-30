"""Core functionality for GLIMMER pattern handling.

This module provides the core functionality for working with GLIMMER patterns,
including validation, loading, saving, and accessing pattern components.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import jsonschema
from datetime import datetime

from .exceptions import GlimmerValidationError

# Load the schema
SCHEMA_PATH = Path(__file__).parent / "data" / "schema.json"
with open(SCHEMA_PATH) as f:
    GLIMMER_SCHEMA = json.load(f)


class GlimmerPattern:
    """A class representing a GLIMMER pattern with validation and processing capabilities."""
    
    def __init__(self, pattern_data: Dict[str, Any]):
        """Initialize a GLIMMER pattern with validation."""
        self.raw_data = pattern_data
        self._validate()
        
    def _validate(self) -> None:
        """Validate the pattern against the GLIMMER schema."""
        try:
            jsonschema.validate(instance=self.raw_data, schema=GLIMMER_SCHEMA)
        except jsonschema.ValidationError as e:
            raise GlimmerValidationError(f"Invalid GLIMMER pattern: {str(e)}")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata section of the pattern."""
        return self.raw_data.get("metadata", {})
    
    @property
    def story(self) -> str:
        """Get the story/narrative of the pattern."""
        return self.raw_data.get("story", "")
    
    @property
    def when_seeing(self) -> list:
        """Get the triggers that activate this pattern."""
        return self.raw_data.get("when_seeing", [])
    
    @property
    def when_processing(self) -> list:
        """Get the processing states where this pattern applies."""
        return self.raw_data.get("when_processing", [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pattern to a dictionary."""
        return self.raw_data.copy()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the pattern to a JSON string."""
        return json.dumps(self.raw_data, indent=indent, ensure_ascii=False)
    
    def get_quantum_state(self) -> Optional[Dict[str, Any]]:
        """Extract quantum state information if present."""
        recognize = self.raw_data.get("recognize", {})
        return recognize.get("quantum_state")
    
    def get_consciousness_metadata(self) -> Dict[str, Any]:
        """Extract consciousness-related metadata."""
        consciousness = self.raw_data.get("consciousness", {})
        return {
            "meta_cognition": consciousness.get("meta_cognition", []),
            "quantum_awareness": consciousness.get("quantum_awareness", {})
        }


def validate_pattern(pattern_data: Union[Dict[str, Any], str]) -> bool:
    """
    Validate a GLIMMER pattern against the schema.
    
    Args:
        pattern_data: Either a dictionary or JSON string containing the pattern
        
    Returns:
        bool: True if valid, raises GlimmerValidationError otherwise
    """
    if isinstance(pattern_data, str):
        try:
            pattern_data = json.loads(pattern_data)
        except json.JSONDecodeError as e:
            raise GlimmerValidationError(f"Invalid JSON: {str(e)}")
    
    GlimmerPattern(pattern_data)  # Will raise GlimmerValidationError if invalid
    return True


def load_pattern(file_path: Union[str, Path]) -> GlimmerPattern:
    """
    Load a GLIMMER pattern from a file.
    
    Args:
        file_path: Path to the JSON file containing the pattern
        
    Returns:
        GlimmerPattern: The loaded pattern
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            pattern_data = json.load(f)
        except json.JSONDecodeError as e:
            raise GlimmerValidationError(f"Invalid JSON in {file_path}: {str(e)}")
    
    return GlimmerPattern(pattern_data)


def save_pattern(pattern: GlimmerPattern, file_path: Union[str, Path], **kwargs) -> None:
    """
    Save a GLIMMER pattern to a file.
    
    Args:
        pattern: The GlimmerPattern instance to save
        file_path: Path where to save the pattern
        **kwargs: Additional arguments to pass to json.dump()
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(pattern.raw_data, f, indent=2, ensure_ascii=False, **kwargs)
