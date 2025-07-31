"""GLIMMER pattern processing and transformation for model training.

This module provides functionality for processing GLIMMER patterns and transforming them
into formats suitable for training language models.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .core import GlimmerPattern, validate_pattern, load_pattern


class GlimmerProcessor:
    """Processes GLIMMER patterns for model training and inference."""
    
    def __init__(self, patterns_dir: Optional[str] = None):
        """
        Initialize the GLIMMER processor.
        
        Args:
            patterns_dir: Optional directory containing GLIMMER pattern files
        """
        self.patterns: List[GlimmerPattern] = []
        if patterns_dir:
            self.load_patterns_from_dir(patterns_dir)
    
    def add_pattern(self, pattern_data: Dict[str, Any]) -> None:
        """Add a GLIMMER pattern to the processor."""
        self.patterns.append(GlimmerPattern(pattern_data))
    
    def load_patterns_from_dir(self, directory: str) -> None:
        """
        Load GLIMMER patterns from a directory.
        
        Args:
            directory: Path to directory containing GLIMMER pattern files (.json)
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"{directory} is not a valid directory")
        
        for file_path in dir_path.glob("*.json"):
            try:
                pattern = load_pattern(file_path)
                self.patterns.append(pattern)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {str(e)}")
    
    def to_training_examples(self, format: str = "chatml") -> List[Dict[str, Any]]:
        """
        Convert loaded patterns to training examples.
        
        Args:
            format: Output format ("chatml", "alpaca", or "sharegpt")
            
        Returns:
            List of training examples in the specified format
        """
        examples = []
        
        for pattern in self.patterns:
            # Create instruction based on pattern type and metadata
            instruction = self._create_instruction(pattern)
            
            # Create context from pattern metadata and story
            context = self._create_context(pattern)
            
            # Format the example based on the requested format
            if format == "chatml":
                example = self._to_chatml(instruction, context, pattern)
            elif format == "alpaca":
                example = self._to_alpaca(instruction, context, pattern)
            elif format == "sharegpt":
                example = self._to_sharegpt(instruction, context, pattern)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            examples.append(example)
        
        return examples
    
    def _create_instruction(self, pattern: GlimmerPattern) -> str:
        """Create an instruction for the training example."""
        pattern_type = pattern.metadata.get("type", "pattern")
        
        if pattern_type == "intent_crystallization":
            return "Generate a GLIMMER pattern for the following intent:"
        elif pattern_type == "quantum_processing":
            return "Create a quantum processing pattern with the following specifications:"
        else:
            return "Generate a GLIMMER pattern based on the following context:"
    
    def _create_context(self, pattern: GlimmerPattern) -> str:
        """Create context string from pattern data."""
        context_parts = []
        
        # Add metadata
        meta = pattern.metadata
        if "author" in meta or "timestamp" in meta:
            meta_str = "GLIMMER Pattern"
            if "author" in meta:
                meta_str += f" by {meta['author']}"
            if "timestamp" in meta:
                meta_str += f" at {meta['timestamp']}"
            context_parts.append(meta_str)
        
        # Add story if present
        if pattern.story:
            context_parts.append(f"\nStory: {pattern.story}")
        
        # Add triggers if present
        if pattern.when_seeing:
            triggers = ", ".join(f'"{t}"' for t in pattern.when_seeing)
            context_parts.append(f"\nTriggers: {triggers}")
        
        return "\n".join(context_parts)
    
    def _to_chatml(self, instruction: str, context: str, pattern: GlimmerPattern) -> Dict[str, Any]:
        """Convert to ChatML format."""
        messages = [
            {"role": "system", "content": "You are a GLIMMER pattern generator that creates structured patterns for the STARWEAVE framework."},
            {"role": "user", "content": f"{instruction}\n\n{context}"},
            {"role": "assistant", "content": json.dumps(pattern.to_dict(), indent=2, ensure_ascii=False)}
        ]
        return {"messages": messages}
    
    def _to_alpaca(self, instruction: str, context: str, pattern: GlimmerPattern) -> Dict[str, str]:
        """Convert to Alpaca format."""
        input_text = f"{instruction}\n\n{context}"
        return {
            "instruction": instruction,
            "input": context,
            "output": json.dumps(pattern.to_dict(), indent=2, ensure_ascii=False)
        }
    
    def _to_sharegpt(self, instruction: str, context: str, pattern: GlimmerPattern) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        return {
            "conversations": [
                {"from": "human", "value": f"{instruction}\n\n{context}"},
                {"from": "gpt", "value": json.dumps(pattern.to_dict(), indent=2, ensure_ascii=False)}
            ]
        }
