"""GLIMMER data pipeline for preparing training data for LLM fine-tuning.

This module provides functionality to process GLIMMER patterns into formats
suitable for training language models.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, asdict
import logging

from .core import GlimmerPattern, load_pattern
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
        
        # Load all JSON pattern files
        json_files = list(dir_path.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON pattern files found in {dir_path}")
            return
            
        for file_path in json_files:
            try:
                pattern = load_pattern(file_path)
                self.processor.patterns.append(pattern)
                logger.debug(f"Loaded pattern from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {str(e)}")
                continue
                
        logger.info(f"Successfully loaded {len(self.processor.patterns)} patterns")
    

            
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
