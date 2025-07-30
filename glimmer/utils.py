"""Utility functions for working with GLIMMER patterns.

This module provides various utility functions for working with GLIMMER patterns,
including file operations, pattern analysis, and format conversion.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def find_glow_file(directory: str, glob_func=None) -> List[str]:
    """
    Find all .we files in the specified directory.
    
    Args:
        directory: Path to search for .we files
        glob_func: Optional function to use for globbing (for testing)
        
    Returns:
        List of paths to .we files
    """
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")
        
    # Use provided glob function or default to path.glob
    glob_func = glob_func or path.glob
    return [str(p) for p in glob_func("**/*.we") if p.is_file()]


def convert_we_to_json(we_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convert a .we file to a JSON file.
    
    Args:
        we_path: Path to the .we file
        output_dir: Optional directory to save the JSON file
        
    Returns:
        Path to the generated JSON file
    """
    path = Path(we_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {we_path}")
    
    # Read the .we file content
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # This is a simplified parser - in a real implementation, you'd want to
    # properly parse the .we format based on its specification
    try:
        # Extract JSON-like content between @pattern_meta@ tags
        if "@pattern_meta@" in content:
            parts = content.split("@pattern_meta@")
            if len(parts) >= 3:
                json_str = parts[1].strip()
                pattern_data = json.loads(json_str)
            else:
                raise ValueError("Invalid .we file format: Could not find pattern metadata")
        else:
            # Try to parse the entire file as JSON
            pattern_data = json.loads(content)
        
        # Determine output path
        if output_dir is None:
            output_dir = path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{path.stem}.json"
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pattern_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse .we file: {str(e)}")


def analyze_patterns(patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a collection of GLIMMER patterns and return statistics.
    
    Args:
        patterns: List of GLIMMER pattern dictionaries
        
    Returns:
        Dictionary containing analysis results
    """
    stats = {
        "total_patterns": len(patterns),
        "pattern_types": {},
        "metadata_fields": {},
        "quantum_usage": 0,
        "has_consciousness": 0,
        "has_processing": 0,
    }
    
    for pattern in patterns:
        # Count pattern types
        pattern_type = pattern.get("metadata", {}).get("type", "unknown")
        stats["pattern_types"][pattern_type] = stats["pattern_types"].get(pattern_type, 0) + 1
        
        # Count metadata fields
        for field in pattern.get("metadata", {}).keys():
            stats["metadata_fields"][field] = stats["metadata_fields"].get(field, 0) + 1
        
        # Check for quantum features
        if "quantum_state" in pattern.get("recognize", {}):
            stats["quantum_usage"] += 1
        
        # Check for consciousness section
        if "consciousness" in pattern:
            stats["has_consciousness"] += 1
        
        # Check for processing section
        if "processing" in pattern:
            stats["has_processing"] += 1
    
    return stats
