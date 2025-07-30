""
Command-line interface for the GLIMMER package.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .core import GlimmerPattern, validate_pattern, load_pattern, save_pattern
from .utils import find_glow_file, convert_we_to_json
from .exceptions import GlimmerValidationError


def validate_cli():
    """CLI command to validate GLIMMER patterns."""
    parser = argparse.ArgumentParser(description="Validate GLIMMER patterns")
    parser.add_argument("path", help="Path to a GLIMMER pattern file or directory")
    parser.add_argument(
        "--recursive", "-r", 
        action="store_true", 
        help="Recursively process directories"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Show detailed validation errors"
    )
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    
    if path.is_file():
        files = [path]
    else:
        if args.recursive:
            files = [Path(f) for f in path.rglob("*.json")]
        else:
            files = [f for f in path.glob("*.json") if f.is_file()]
    
    valid_count = 0
    invalid_count = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)
            
            validate_pattern(pattern_data)
            valid_count += 1
            if args.verbose:
                print(f"✅ Valid: {file_path}")
                
        except (json.JSONDecodeError, GlimmerValidationError) as e:
            invalid_count += 1
            print(f"❌ Invalid: {file_path}", file=sys.stderr)
            if args.verbose:
                print(f"   Error: {str(e)}", file=sys.stderr)
    
    print(f"\nValidation complete. {valid_count} valid, {invalid_count} invalid.")
    sys.exit(1 if invalid_count > 0 else 0)


def convert_cli():
    """CLI command to convert .we files to JSON."""
    parser = argparse.ArgumentParser(description="Convert .we files to JSON")
    parser.add_argument(
        "input", 
        help="Input .we file or directory containing .we files"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Output directory for JSON files (default: same as input)"
    )
    parser.add_argument(
        "--recursive", "-r", 
        action="store_true", 
        help="Recursively process directories"
    )
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    if input_path.is_file():
        files = [input_path]
    else:
        if args.recursive:
            files = [f for f in input_path.rglob("*.we") if f.is_file()]
        else:
            files = [f for f in input_path.glob("*.we") if f.is_file()]
    
    if not files:
        print("No .we files found.", file=sys.stderr)
        sys.exit(1)
    
    success_count = 0
    error_count = 0
    
    for we_file in files:
        try:
            output_path = convert_we_to_json(we_file, output_dir)
            print(f"Converted: {we_file} -> {output_path}")
            success_count += 1
        except Exception as e:
            print(f"Error converting {we_file}: {str(e)}", file=sys.stderr)
            error_count += 1
    
    print(f"\nConversion complete. {success_count} succeeded, {error_count} failed.")
    sys.exit(1 if error_count > 0 else 0)


def main():
    """Main entry point for the GLIMMER CLI."""
    parser = argparse.ArgumentParser(description="GLIMMER: Generative Language Interface for Meta-Modeling and Evolutionary Responses")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", 
        help="Validate GLIMMER patterns"
    )
    validate_parser.add_argument("path", help="Path to a GLIMMER pattern file or directory")
    validate_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively process directories")
    validate_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed validation errors")
    validate_parser.set_defaults(func=validate_cli)
    
    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", 
        help="Convert .we files to JSON"
    )
    convert_parser.add_argument("input", help="Input .we file or directory containing .we files")
    convert_parser.add_argument("--output", "-o", help="Output directory for JSON files (default: same as input)")
    convert_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively process directories")
    convert_parser.set_defaults(func=convert_cli)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        # Remove the command from args to avoid duplicate arguments
        cmd = args.command
        delattr(args, 'command')
        delattr(args, 'func')
        
        # Call the appropriate function with remaining arguments
        if cmd == 'validate':
            validate_cli()
        elif cmd == 'convert':
            convert_cli()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
