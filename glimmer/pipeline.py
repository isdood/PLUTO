"""GLIMMER data pipeline for preparing training data for LLM fine-tuning.

This module provides functionality to process GLIMMER patterns into formats
suitable for training language models.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, asdict
import logging

from .core import GlimmerPattern
from .processor import GlimmerProcessor
from .exceptions import GlimmerValidationError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    """A single training example for LLM fine-tuning."""
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the training example to a dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }

class GlimmerDataset:
    """Manages the conversion of GLIMMER patterns into training data."""
    
    def __init__(self, patterns_dir: Optional[Union[str, Path]] = None):
        """Initialize the dataset with GLIMMER patterns.
        
        Args:
            patterns_dir: Optional directory containing GLIMMER pattern files
        """
        self.processor = GlimmerProcessor()
        if patterns_dir:
            self.load_patterns(patterns_dir)
    
    def load_patterns(self, directory: Union[str, Path]) -> None:
        """Load patterns from a directory.
        
        Args:
            directory: Path to directory containing GLIMMER pattern files
            
        Raises:
            ValueError: If the directory does not exist or is not a directory
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Patterns directory not found: {directory}")
            
        logger.info(f"Loading patterns from {dir_path}")
        
        # First try to load .we files, fall back to .json if none found
        we_files = list(dir_path.glob("*.we"))
        json_files = list(dir_path.glob("*.json")) if not we_files else []
        
        if not we_files and not json_files:
            logger.warning(f"No pattern files found in {dir_path}")
            return
            
        for file_path in we_files + json_files:
            try:
                if file_path.suffix == '.we':
                    pattern = self._load_we_file(file_path)
                else:
                    pattern = GlimmerPattern.load(file_path)
                self.processor.patterns.append(pattern)
                logger.debug(f"Loaded pattern from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {str(e)}")
                continue
                
        logger.info(f"Successfully loaded {len(self.processor.patterns)} patterns")
    
    def _load_we_file(self, file_path: Path) -> GlimmerPattern:
        """Load and parse a .we pattern file.
        
        Args:
            file_path: Path to the .we file
            
        Returns:
            Parsed GlimmerPattern
            
        Raises:
            ValueError: If the file is not a valid .we pattern file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Define the pattern marker (without escape sequences)
            pattern_marker = '@pattern_meta@'
            
            # Simple parsing of .we format (this is a simplified version)
            if pattern_marker not in content:
                raise ValueError(f"Invalid .we file: missing {pattern_marker} section")
            
            # Find the JSON content between pattern markers
            parts = content.split(pattern_marker)
            if len(parts) < 3:  # Should have content before, between, and after markers
                raise ValueError(f"Invalid .we file: missing content between {pattern_marker} markers")
                
            # The JSON is between the first and second pattern marker
            json_content = parts[1].strip()
            
            # Try to parse the JSON content directly first
            try:
                pattern_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                # If direct parsing fails, try to extract the first valid JSON object
                try:
                    json_start = json_content.find('{')
                    json_end = json_content.rfind('}') + 1
                    if json_start == -1 or json_end == 0:
                        raise ValueError("Could not find JSON data in .we file") from e
                    json_str = json_content[json_start:json_end]
                    pattern_data = json.loads(json_str)
                except Exception as inner_e:
                    raise ValueError(f"Invalid JSON in .we file: {str(inner_e)}") from inner_e
            
            # Create and validate the pattern
            pattern = GlimmerPattern(pattern_data)
            return pattern
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path.name}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing {file_path.name}: {str(e)}")
