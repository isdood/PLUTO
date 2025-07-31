"""Tests for the GLIMMER pattern module."""
import unittest
import tempfile
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from glimmer.core import GlimmerPattern, load_pattern, validate_pattern

class TestGlimmerPattern(unittest.TestCase):
    """Test cases for the GlimmerPattern class and related functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.valid_pattern_data = {
            "metadata": {
                "timestamp": "2023-01-01T00:00:00Z",
                "author": "Test",
                "pattern_version": "1.0.0",
                "type": "pattern_weaving",
                "title": "Test Pattern",
                "color": "#4B0082"
            },
            "content": {
                "title": "Test Pattern Title",
                "sections": [
                    {
                        "title": "Test Section",
                        "content": {
                            "key1": "value1"
                        }
                    }
                ]
            }
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_validate_valid_pattern(self):
        """Test that a valid pattern passes validation."""
        self.assertTrue(validate_pattern(self.valid_pattern_data))
    
    def test_validate_invalid_pattern(self):
        """Test that an invalid pattern fails validation."""
        from glimmer.exceptions import GlimmerValidationError
        
        invalid_data = self.valid_pattern_data.copy()
        del invalid_data["content"]
        
        with self.assertRaises(GlimmerValidationError) as cm:
            # This should raise a validation error because 'content' is required
            GlimmerPattern(invalid_data)
        
        # Check that the error message contains information about the missing 'content' field
        self.assertIn("'content' is a required property", str(cm.exception))
    
    def test_load_pattern_from_dict(self):
        """Test loading a pattern from a dictionary."""
        pattern = GlimmerPattern(self.valid_pattern_data)
        self.assertEqual(pattern.metadata["title"], "Test Pattern")
        self.assertEqual(len(pattern.raw_data["content"]["sections"]), 1)
    
    def test_load_pattern_from_file(self):
        """Test loading a pattern from a file."""
        # Create a test file
        test_file = self.test_dir / "test_pattern.json"
        with open(test_file, 'w') as f:
            json.dump(self.valid_pattern_data, f)
        
        # Test loading the file
        pattern = load_pattern(test_file)
        self.assertEqual(pattern.metadata["title"], "Test Pattern")
        self.assertEqual(pattern.metadata["author"], "Test")
    
    def test_save_pattern(self):
        """Test saving a pattern to a file."""
        pattern = GlimmerPattern(self.valid_pattern_data)
        output_file = self.test_dir / "output_pattern.json"
        
        # Save the pattern
        with open(output_file, 'w') as f:
            json.dump(pattern.raw_data, f, indent=2)
        
        # Verify the file was created and contains valid JSON
        self.assertTrue(output_file.exists())
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data["metadata"]["title"], "Test Pattern")


if __name__ == "__main__":
    unittest.main()
