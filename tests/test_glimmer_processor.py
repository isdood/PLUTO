"""Tests for the GLIMMER processor module."""
import unittest
from unittest.mock import MagicMock

from glimmer.processor import GlimmerProcessor

class TestGlimmerProcessor(unittest.TestCase):
    """Test cases for the GlimmerProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = GlimmerProcessor()
        self.test_pattern_data = {
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
        
        # Store the pattern data for testing
        self.test_pattern = self.test_pattern_data
    
    def test_initialization(self):
        """Test that the processor initializes correctly."""
        self.assertEqual(len(self.processor.patterns), 0)
    
    def test_add_pattern(self):
        """Test adding a pattern to the processor."""
        from glimmer.core import GlimmerPattern
        self.processor.add_pattern(self.test_pattern)
        self.assertEqual(len(self.processor.patterns), 1)
        self.assertIsInstance(self.processor.patterns[0], GlimmerPattern)
    
    def test_process_pattern(self):
        """Test processing a pattern."""
        # Add the pattern to the processor and get training examples
        self.processor.add_pattern(self.test_pattern)
        examples = self.processor.to_training_examples()
        
        # Should get at least one example
        self.assertGreater(len(examples), 0)
        result = examples[0]
        
        # Verify the result contains expected content
        self.assertIsNotNone(result)
        self.assertIn(self.test_pattern["metadata"]["title"], str(result))
    
    def test_process_all_patterns(self):
        """Test processing all patterns in the processor."""
        # Create a second test pattern
        pattern2_data = {
            "metadata": {**self.test_pattern_data["metadata"], "title": "Pattern 2"},
            "content": {
                "title": "Second Test Pattern",
                "sections": [
                    {
                        "title": "Second Section",
                        "content": {"key2": "value2"}
                    }
                ]
            }
        }
        
        # Add patterns to processor and get training examples
        self.processor.add_pattern(self.test_pattern)
        self.processor.add_pattern(pattern2_data)
        
        # Get training examples for all patterns
        examples = self.processor.to_training_examples()
        
        # Should get at least two examples (one per pattern)
        self.assertGreaterEqual(len(examples), 2)
        
        # Convert examples to strings for easier searching
        example_texts = [str(example) for example in examples]
        
        # Check that we have content from both patterns
        self.assertTrue(any(self.test_pattern["metadata"]["title"] in text for text in example_texts))
        self.assertTrue(any("Pattern 2" in text for text in example_texts))


if __name__ == "__main__":
    unittest.main()
