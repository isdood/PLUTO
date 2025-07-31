"""Tests for the GLIMMER dataset module."""
import unittest
import tempfile
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from glimmer.pipeline import GlimmerDataset, TrainingExample

class TestGlimmerDataset(unittest.TestCase):
    """Test cases for the GlimmerDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.valid_pattern = {
            "metadata": {
                "timestamp": "2023-01-01T00:00:00Z",
                "author": "Test",
                "pattern_version": "1.0.0",
                "type": "pattern_weaving",
                "title": "Test Pattern"
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
    
    def test_initialization(self):
        """Test that the dataset initializes correctly."""
        dataset = GlimmerDataset()
        self.assertEqual(len(dataset.processor.patterns), 0)
    
    def test_load_patterns_nonexistent_dir(self):
        """Test loading patterns from a non-existent directory."""
        dataset = GlimmerDataset()
        with self.assertRaises(ValueError):
            dataset.load_patterns("/nonexistent/directory")
    
    def test_load_patterns(self):
        """Test loading patterns from a directory."""
        # Create a test pattern file
        pattern_file = self.test_dir / "test_pattern.json"
        with open(pattern_file, 'w') as f:
            json.dump(self.valid_pattern, f)
        
        # Create dataset and load patterns
        dataset = GlimmerDataset()
        dataset.load_patterns(self.test_dir)
        
        # Verify results
        self.assertEqual(len(dataset.processor.patterns), 1)
        self.assertEqual(dataset.processor.patterns[0].raw_data, self.valid_pattern)
    
    @patch('glimmer.pipeline.TrainingExample')
    def test_to_training_examples(self, mock_training_example):
        """Test converting patterns to training examples."""
        # Create a test pattern
        from glimmer.core import GlimmerPattern
        pattern = GlimmerPattern(self.valid_pattern)
        
        # Create dataset with test pattern
        dataset = GlimmerDataset()
        dataset.processor.patterns = [pattern]
        
        # Set up mock return value for TrainingExample
        mock_example = MagicMock()
        mock_example.metadata = {"author": "Test"}
        mock_example.text = "Test Pattern Title: This is a test pattern"
        mock_training_example.return_value = mock_example
        
        # Convert to training examples
        examples = dataset.to_training_examples()
        
        # Verify results
        self.assertEqual(len(examples), 1)
        self.assertIs(examples[0], mock_example)
        mock_training_example.assert_called_once()


if __name__ == "__main__":
    unittest.main()
