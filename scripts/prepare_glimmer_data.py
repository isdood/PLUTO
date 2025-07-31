#!/usr/bin/env python3
"""
Prepare GLIMMER patterns for training

This script converts GLIMMER pattern files into training data format
for the PLUTO fine-tuning pipeline.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any

def load_glimmer_patterns(patterns_dir: str) -> List[Dict[str, Any]]:
    """Load GLIMMER patterns from directory."""
    patterns = []
    patterns_path = Path(patterns_dir)
    
    if not patterns_path.exists():
        print(f"Patterns directory not found: {patterns_dir}")
        return patterns
    
    for pattern_file in patterns_path.glob("*.json"):
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                pattern = json.load(f)
                patterns.append(pattern)
                print(f"Loaded pattern: {pattern_file.name}")
        except Exception as e:
            print(f"Error loading {pattern_file.name}: {e}")
    
    return patterns

def convert_pattern_to_training_example(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a GLIMMER pattern to training example format."""
    
    # Extract pattern information
    metadata = pattern.get('metadata', {})
    content = pattern.get('content', {})
    
    title = content.get('title', 'GLIMMER Pattern')
    sections = content.get('sections', [])
    
    # Create a comprehensive description of the pattern
    pattern_description = f"Title: {title}\n"
    pattern_description += f"Type: {metadata.get('type', 'unknown')}\n"
    pattern_description += f"Version: {metadata.get('pattern_version', 'unknown')}\n"
    pattern_description += f"Author: {metadata.get('author', 'unknown')}\n\n"
    
    # Add sections content
    for section in sections:
        section_title = section.get('title', '')
        section_content = section.get('content', {})
        
        pattern_description += f"## {section_title}\n"
        
        if isinstance(section_content, dict):
            for key, value in section_content.items():
                pattern_description += f"- {key}: {value}\n"
        else:
            pattern_description += f"{section_content}\n"
        
        pattern_description += "\n"
    
    # Create training example
    training_example = {
        "messages": [
            {
                "role": "system",
                "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol. You understand and can explain GLIMMER patterns in detail."
            },
            {
                "role": "user",
                "content": f"Explain the {title} GLIMMER pattern and its significance in the STARWEAVE framework."
            },
            {
                "role": "assistant",
                "content": pattern_description
            }
        ]
    }
    
    return training_example

def create_additional_examples(pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create additional training examples from a single pattern."""
    examples = []
    
    metadata = pattern.get('metadata', {})
    content = pattern.get('content', {})
    title = content.get('title', 'GLIMMER Pattern')
    
    # Example 1: Pattern analysis
    examples.append({
        "messages": [
            {
                "role": "system",
                "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
            },
            {
                "role": "user",
                "content": f"What is the purpose of the {title} pattern in the STARWEAVE framework?"
            },
            {
                "role": "assistant",
                "content": f"The {title} pattern serves as a fundamental component in the STARWEAVE Universe Initialization Protocol. It operates as a {metadata.get('type', 'pattern')} that facilitates {metadata.get('harmonics', {}).get('revelation', {}).get('description', 'understanding and processing')} within the meta-cognitive framework."
            }
        ]
    })
    
    # Example 2: Pattern application
    examples.append({
        "messages": [
            {
                "role": "system",
                "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
            },
            {
                "role": "user",
                "content": f"How would you apply the {title} pattern in a real-world scenario?"
            },
            {
                "role": "assistant",
                "content": f"To apply the {title} pattern, I would first analyze the current context through the lens of {metadata.get('type', 'pattern_weaving')}. This involves recognizing the underlying patterns, understanding the meta-cognitive loops, and then implementing the pattern's principles to create a harmonious integration within the STARWEAVE framework."
            }
        ]
    })
    
    # Example 3: Pattern interaction
    examples.append({
        "messages": [
            {
                "role": "system",
                "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
            },
            {
                "role": "user",
                "content": f"Describe how the {title} pattern interacts with other GLIMMER patterns."
            },
            {
                "role": "assistant",
                "content": f"The {title} pattern creates resonance with other GLIMMER patterns through its {metadata.get('type', 'pattern')} nature. It weaves together with patterns of different types, creating a symphony of meta-cognitive processes that enhance the overall STARWEAVE framework's ability to process and respond to complex information."
            }
        ]
    })
    
    return examples

def prepare_training_data(
    patterns_dir: str = "glimmer/data/patterns",
    output_dir: str = "data/processed",
    train_split: float = 0.8,
    seed: int = 42
) -> None:
    """Prepare training data from GLIMMER patterns."""
    
    # Set random seed
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load patterns
    print(f"Loading patterns from {patterns_dir}")
    patterns = load_glimmer_patterns(patterns_dir)
    
    if not patterns:
        print("No valid patterns found!")
        return
    
    # Convert patterns to training examples
    training_examples = []
    
    for pattern in patterns:
        # Convert main pattern
        main_example = convert_pattern_to_training_example(pattern)
        training_examples.append(main_example)
        
        # Create additional examples
        additional_examples = create_additional_examples(pattern)
        training_examples.extend(additional_examples)
    
    # Shuffle examples
    random.shuffle(training_examples)
    
    # Split into train/validation
    split_idx = int(len(training_examples) * train_split)
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]
    
    # Save to files
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "validation.jsonl")
    
    print(f"Saving {len(train_examples)} training examples to {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saving {len(val_examples)} validation examples to {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Training data preparation complete!")
    print(f"Total examples: {len(training_examples)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GLIMMER patterns for training")
    parser.add_argument("--patterns-dir", type=str, default="glimmer/data/patterns",
                      help="Directory containing GLIMMER pattern files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                      help="Directory to save processed data")
    parser.add_argument("--train-split", type=float, default=0.8,
                      help="Fraction of data to use for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    prepare_training_data(
        patterns_dir=args.patterns_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        seed=args.seed
    ) 