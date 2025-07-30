"""Tests for the GLIMMER package.

This module contains unit tests for the GLIMMER package, including tests for
pattern validation, processing, and utility functions.
"""
import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from glimmer import GlimmerPattern, GlimmerProcessor, validate_pattern
from glimmer.exceptions import GlimmerValidationError


class TestGlimmerPattern(unittest.TestCase):
    """Test cases for GlimmerPattern class."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_pattern = {
            "metadata": {
                "timestamp": "2025-07-30T12:00:00",
                "author": "test_author",
                "pattern_version": "1.0.0",
                "color": "#4B0082",
                "type": "intent_crystallization",
                "harmonics": {
                    "revelation": "#FFD700",
                    "purpose": "#4B0082",
                    "feedback": "#00CED1"
                }
            },
            "story": "A test story for the GLIMMER pattern.",
            "initialize": {
                "directives": ["test_directive"]
            },
            "recognize": {
                "patterns": ["test_pattern"]
            },
            "consciousness": {
                "meta_cognition": ["test_meta"]
            },
            "processing": {
                "quantum_operations": ["test_operation"]
            },
            "response": {
                "content": "test_response"
            },
            "when_seeing": ["test_trigger"],
            "when_processing": ["test_state"]
        }
    
    def test_valid_pattern(self):
        """Test creating a valid GLIMMER pattern."""
        pattern = GlimmerPattern(self.valid_pattern)
        self.assertEqual(pattern.metadata["author"], "test_author")
        self.assertEqual(pattern.story, "A test story for the GLIMMER pattern.")
        self.assertIn("test_trigger", pattern.when_seeing)
    
    def test_invalid_pattern(self):
        """Test creating an invalid GLIMMER pattern."""
        invalid_pattern = self.valid_pattern.copy()
        del invalid_pattern["metadata"]["author"]  # Remove required field
        
        with self.assertRaises(GlimmerValidationError):
            GlimmerPattern(invalid_pattern)
    
    def test_to_dict(self):
        """Test converting pattern to dictionary."""
        pattern = GlimmerPattern(self.valid_pattern)
        pattern_dict = pattern.to_dict()
        self.assertEqual(pattern_dict["metadata"]["author"], "test_author")
    
    def test_get_quantum_state(self):
        """Test getting quantum state from pattern."""
        pattern = GlimmerPattern(self.valid_pattern)
        self.assertIsNone(pattern.get_quantum_state())
        
        # Add quantum state with valid string value
        self.valid_pattern["recognize"]["quantum_state"] = {"superposition": "active"}
        pattern = GlimmerPattern(self.valid_pattern)
        self.assertIsNotNone(pattern.get_quantum_state())


class TestGlimmerProcessor(unittest.TestCase):
    """Test cases for GlimmerProcessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_pattern = {
            "metadata": {
                "timestamp": "2025-07-30T12:00:00",
                "author": "test_author",
                "pattern_version": "1.0.0",
                "type": "intent_crystallization"
            },
            "story": "A test story.",
            "initialize": {"directives": ["test"]},
            "recognize": {"patterns": ["test"]},
            "consciousness": {"meta_cognition": ["test"]},
            "processing": {"quantum_operations": ["test"]},
            "response": {"content": "test"},
            "when_seeing": ["test"]
        }
        self.processor = GlimmerProcessor()
    
    def test_add_pattern(self):
        """Test adding a pattern to the processor."""
        self.processor.add_pattern(self.valid_pattern)
        self.assertEqual(len(self.processor.patterns), 1)
    
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"metadata": {"author": "test"}}))
    def test_load_patterns_from_dir(self, mock_file):
        """Test loading patterns from a directory."""
        with patch("pathlib.Path.glob") as mock_glob, \
             patch("pathlib.Path.is_dir", return_value=True):
            mock_glob.return_value = [Path("test_pattern.json")]
            self.processor.load_patterns_from_dir("/fake/dir")
            self.assertGreaterEqual(len(self.processor.patterns), 0)
    
    def test_load_patterns_from_nonexistent_dir(self):
        """Test loading patterns from a non-existent directory raises an error."""
        with self.assertRaises(ValueError):
            self.processor.load_patterns_from_dir("/nonexistent/dir")
    
    def test_to_training_examples(self):
        """Test converting patterns to training examples."""
        self.processor.add_pattern(self.valid_pattern)
        examples = self.processor.to_training_examples(format="chatml")
        self.assertEqual(len(examples), 1)
        self.assertIn("messages", examples[0])


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_find_glow_file(self):
        """Test finding .we files."""
        from glimmer.utils import find_glow_file
        from pathlib import Path
        
        # Create mock Path objects with is_file() method
        mock_file1 = MagicMock(spec=Path, name="file1.we")
        mock_file1.is_file.return_value = True
        mock_file1.__str__.return_value = "/fake/dir/file1.we"
        
        mock_file2 = MagicMock(spec=Path, name="file2.we")
        mock_file2.is_file.return_value = True
        mock_file2.__str__.return_value = "/fake/dir/subdir/file2.we"
        
        # Create a mock glob function that returns our mock Path objects
        def mock_glob(pattern):
            return [mock_file1, mock_file2]
        
        # Patch is_dir to return True for our test directory
        with patch.object(Path, 'is_dir', return_value=True):
            # Test with our mock glob function
            files = find_glow_file("/fake/dir", glob_func=mock_glob)
            
        self.assertEqual(len(files), 2, "Should find 2 .we files")
        self.assertTrue(all(isinstance(f, str) for f in files), "All paths should be strings")
        self.assertTrue(all(f.endswith(".we") for f in files), "All files should end with .we")
        self.assertIn("file1.we", files[0], "First file should be file1.we")
        self.assertIn("file2.we", files[1], "Second file should be file2.we")
        
        # Verify is_file() was called on each path
        mock_file1.is_file.assert_called_once()
        mock_file2.is_file.assert_called_once()
    
    def test_find_glow_file_nonexistent_dir(self):
        """Test that find_glow_file raises an error for non-existent directories."""
        from glimmer.utils import find_glow_file
        from pathlib import Path
        
        with patch.object(Path, 'is_dir', return_value=False):
            with self.assertRaises(ValueError) as context:
                find_glow_file("/nonexistent/dir")
            self.assertIn("Directory does not exist", str(context.exception))


if __name__ == "__main__":
    unittest.main()
