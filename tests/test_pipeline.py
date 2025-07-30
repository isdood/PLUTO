"""Tests for the GLIMMER data pipeline."""
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import json
import tempfile
import shutil

from glimmer.pipeline import GlimmerDataset, TrainingExample
from glimmer.core import GlimmerPattern

class TestGlimmerDataset(unittest.TestCase):
    """Test cases for the GlimmerDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        # Using raw string to avoid escape sequence warnings
        self.valid_we_content = r"""@pattern_meta@
        {"metadata": {"title": "Test Pattern", "author": "Test"}}
        @pattern_meta@
        Some content here
        """
        
    def tearDown(self):
        """Clean up test fixtures."""
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
    
    def test_load_we_file(self):
        """Test loading a .we pattern file."""
        # Create a test .we file
        we_file = self.test_dir / "test.we"
        we_file.write_text(self.valid_we_content)
        print(f"Created .we file at {we_file}")
        print(f"File content: {we_file.read_text()}")
        
        # Create a test .json file for fallback
        json_file = self.test_dir / "test.json"
        json_content = '{"metadata": {"title": "JSON Pattern"}}'
        json_file.write_text(json_content)
        print(f"Created .json file at {json_file}")
        print(f"JSON content: {json_content}")
        
        # Test loading with patched file operations to see what's happening
        with patch('glimmer.pipeline.GlimmerPattern') as mock_glimmer_pattern, \
             patch('glimmer.pipeline.GlimmerProcessor') as mock_processor, \
             patch('glimmer.pipeline.logger') as mock_logger, \
             patch('glimmer.pipeline.GlimmerPattern.load') as mock_load:
            
            # Set up the mock processor
            mock_processor.return_value = MagicMock()
            mock_processor.return_value.patterns = []
            
            # Set up the mock pattern
            mock_pattern = MagicMock()
            mock_pattern.metadata = {"title": "Test Pattern", "author": "Test"}
            mock_glimmer_pattern.return_value = mock_pattern
            
            # Create dataset and load patterns
            dataset = GlimmerDataset()
            print(f"Dataset processor before load: {dataset.processor}")
            
            # Patch the file reading to see what's being read
            def mock_open_file(file_path, *args, **kwargs):
                print(f"Attempting to read file: {file_path}")
                if str(file_path).endswith('.we'):
                    return mock_open(read_data=self.valid_we_content)(file_path, *args, **kwargs)
                elif str(file_path).endswith('.json'):
                    return mock_open(read_data=json_content)(file_path, *args, **kwargs)
                return open(file_path, *args, **kwargs)
            
            with patch('builtins.open', mock_open_file):
                # Also patch Path.glob to return our test files
                with patch('pathlib.Path.glob') as mock_glob:
                    # Make sure glob returns our test files
                    mock_glob.return_value = [we_file, json_file]
                    
                    # Now load the patterns
                    dataset.load_patterns(self.test_dir)
                    print(f"Loaded {len(dataset.processor.patterns)} patterns")
                    
                    # Check that the processor was called with the patterns
                    print(f"Processor patterns: {dataset.processor.patterns}")
                    
                    # Verify the files were processed
                    print(f"Mock GlimmerPattern calls: {mock_glimmer_pattern.mock_calls}")
                    
                    # Check that GlimmerPattern was called with the .we file content
                    mock_glimmer_pattern.assert_called_once_with({"metadata": {"title": "Test Pattern", "author": "Test"}})
                    
                    # Check that load() was called with the JSON file path
                    mock_load.assert_called_once_with(json_file)
                    
                    # Check that we have the right number of patterns
                    self.assertEqual(len(dataset.processor.patterns), 2)
        
        # Verify results
        self.assertEqual(len(dataset.processor.patterns), 2)  # Both files should be loaded
        self.assertIsInstance(dataset.processor.patterns[0], MagicMock)
        self.assertEqual(dataset.processor.patterns[0].metadata['title'], "Test Pattern")
        
    def test_load_invalid_we_file(self):
        """Test loading an invalid .we pattern file."""
        # Create an invalid .we file (missing @pattern_meta@)
        we_file = self.test_dir / "invalid.we"
        we_file.write_text('{"invalid": "json"}')
        
        dataset = GlimmerDataset()
        with self.assertLogs('glimmer.pipeline', level='ERROR') as cm:
            dataset.load_patterns(self.test_dir)
            
        # Should log an error but not raise
        self.assertIn('Error loading invalid.we', '\n'.join(cm.output))
        self.assertEqual(len(dataset.processor.patterns), 0)  # No valid patterns loaded


class TestTrainingExample(unittest.TestCase):
    """Test cases for the TrainingExample class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        example = TrainingExample(
            text="Test text",
            metadata={"key": "value"}
        )
        result = example.to_dict()
        self.assertEqual(result, {
            "text": "Test text",
            "metadata": {"key": "value"}
        })


if __name__ == "__main__":
    unittest.main()
