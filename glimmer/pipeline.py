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
            raise ValueError(f"Error loading {file_path.name}: {str(e)}")
            
    def to_training_examples(self) -> List[TrainingExample]:
        """Convert loaded patterns into training examples.
        
        Returns:
            List of TrainingExample objects ready for LLM fine-tuning
        """
        examples = []
        for pattern in self.processor.patterns:
            try:
                # Create a training example from each pattern
                example = TrainingExample(
                    text=pattern.story,  # Use the story field from the pattern
                    metadata={
                        "author": pattern.metadata.get("author", "Unknown"),
                        "pattern_type": pattern.metadata.get("type", "unknown"),
                        "pattern_version": pattern.metadata.get("pattern_version", "unknown"),
                        "timestamp": pattern.metadata.get("timestamp", ""),
                        "source_file": getattr(pattern, "source_file", "unknown")
                    }
                )
                examples.append(example)
            except Exception as e:
                logger.error(f"Error creating training example from pattern: {str(e)}")
                continue
                
        return examples
        
    def to_jsonl(self, output_file: Union[str, Path], **kwargs) -> None:
        """Save the dataset as a JSONL file for LLM fine-tuning.
        
        Args:
            output_file: Path to the output JSONL file
            **kwargs: Additional keyword arguments to pass to json.dumps()
        """
        output_path = Path(output_file)
        examples = self.to_training_examples()
        
        if not examples:
            logger.warning("No training examples to save. Did you load any patterns?")
            return
            
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write examples to JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                # Convert to dict and add any additional fields
                example_dict = example.to_dict()
                
                # Add system prompt if provided
                system_prompt = kwargs.pop('system_prompt', None)
                if system_prompt:
                    example_dict['system_prompt'] = system_prompt
                    
                # Add instruction if provided
                instruction = kwargs.pop('instruction', None)
                if instruction:
                    example_dict['instruction'] = instruction
                
                # Write the JSON line
                json_line = json.dumps(example_dict, ensure_ascii=False, **kwargs)
                f.write(json_line + '\n')
                
        logger.info(f"Saved {len(examples)} training examples to {output_path}")
