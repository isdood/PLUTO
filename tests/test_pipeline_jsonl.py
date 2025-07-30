"""Tests for the JSONL conversion functionality in the GLIMMER data pipeline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, PropertyMock

from glimmer.pipeline import GlimmerDataset, TrainingExample
from glimmer.core import GlimmerPattern

class TestJSONLConversion(unittest.TestCase):
    """Test the JSONL conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.valid_we_content = """@pattern_meta@
        {"metadata": {"title": "Test Pattern", "author": "Test"}}
        @pattern_meta@
        Some content here
        """
        
        # Create a test .we file
        self.we_file = self.test_dir / "test.we"
        self.we_file.write_text(self.valid_we_content)
        
        # Create a test .json file
        self.json_file = self.test_dir / "test.json"
        self.json_content = '{"metadata": {"title": "JSON Pattern"}}'
        self.json_file.write_text(self.json_content)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
        
    @patch('glimmer.pipeline.GlimmerProcessor')
    def test_to_training_examples(self, mock_processor):
        """Test converting patterns to training examples."""
        # Create actual GlimmerPattern instances with test data that matches the schema
        pattern_data1 = {
            "metadata": {
                "timestamp": "2023-01-01T00:00:00Z",
                "author": "Test",
                "pattern_version": "1.0.0",
                "type": "intent_crystallization"
            },
            "story": "Test pattern content",
            "initialize": {
                "directives": ["test_directive"]
            },
            "recognize": {
                "patterns": ["test_pattern"],
                "quantum_state": {
                    "uncertainty": "low",
                    "entanglement": "none",
                    "superposition": "none"
                }
            },
            "consciousness": {
                "meta_cognition": ["test_meta_cognition"]
            },
            "processing": {
                "quantum_operations": ["test_operation"]
            },
            "response": {
                "content": "Test response"
            }
        }
        
        pattern_data2 = {
            "metadata": {
                "timestamp": "2023-01-02T00:00:00Z",
                "author": "System",
                "pattern_version": "1.0.0",
                "type": "quantum_processing"
            },
            "story": "JSON pattern content",
            "initialize": {
                "directives": ["json_directive"]
            },
            "recognize": {
                "patterns": ["json_pattern"],
                "quantum_state": {
                    "uncertainty": "medium",
                    "entanglement": "partial",
                    "superposition": "partial"
                }
            },
            "consciousness": {
                "meta_cognition": ["json_meta_cognition"]
            },
            "processing": {
                "quantum_operations": ["json_operation"]
            },
            "response": {
                "content": "JSON response"
            }
        }
        
        # Create mock processor with patterns
        mock_processor.return_value.patterns = [
            GlimmerPattern(pattern_data1),
            GlimmerPattern(pattern_data2)
        ]
        
        # Add source_file attribute to patterns for testing
        patterns = mock_processor.return_value.patterns
        patterns[0].source_file = str(self.we_file)
        patterns[1].source_file = str(self.json_file)
        
        # Create dataset and convert to training examples
        dataset = GlimmerDataset()
        dataset.processor = mock_processor.return_value
        
        examples = dataset.to_training_examples()
        
        # Verify results
        self.assertEqual(len(examples), 2)
        self.assertIsInstance(examples[0], TrainingExample)
        
        # Check the examples
        self.assertEqual(len(examples), 2)
        
        # Check that both patterns were converted correctly
        self.assertEqual(examples[0].text, "Test pattern content")
        self.assertEqual(examples[0].metadata["author"], "Test")
        self.assertEqual(examples[0].metadata["pattern_type"], "intent_crystallization")
        self.assertEqual(examples[0].metadata["pattern_version"], "1.0.0")
        
        self.assertEqual(examples[1].text, "JSON pattern content")
        self.assertEqual(examples[1].metadata["author"], "System")
        self.assertEqual(examples[1].metadata["pattern_type"], "quantum_processing")
        self.assertEqual(examples[1].metadata["pattern_version"], "1.0.0")
        
    @patch('glimmer.pipeline.GlimmerProcessor')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_to_jsonl(self, mock_mkdir, mock_file, mock_processor):
        """Test saving dataset to JSONL format."""
        # Create a test pattern that matches the schema
        pattern_data = {
            "metadata": {
                "timestamp": "2023-01-01T00:00:00Z",
                "author": "Test",
                "pattern_version": "1.0.0",
                "type": "intent_crystallization"
            },
            "story": "Test pattern content",
            "initialize": {
                "directives": ["test_directive"]
            },
            "recognize": {
                "patterns": ["test_pattern"],
                "quantum_state": {
                    "uncertainty": "low",
                    "entanglement": "none",
                    "superposition": "none"
                }
            },
            "consciousness": {
                "meta_cognition": ["test_meta_cognition"]
            },
            "processing": {
                "quantum_operations": ["test_operation"]
            },
            "response": {
                "content": "Test response"
            }
        }
        pattern = GlimmerPattern(pattern_data)
        pattern.source_file = str(self.we_file)
        
        # Set up mock processor with the test pattern
        mock_processor.return_value.patterns = [pattern]
        
        # Create dataset and save to JSONL
        dataset = GlimmerDataset()
        dataset.processor = mock_processor.return_value
        
        output_file = self.test_dir / "output.jsonl"
        dataset.to_jsonl(output_file, system_prompt="You are a helpful assistant.", indent=2)
        
        # Verify file was written
        mock_file.assert_called_once_with(output_file, 'w', encoding='utf-8')
        
        # Check the content written to the file
        file_handle = mock_file()
        
        # Get the first argument of the first call to write
        write_calls = file_handle.write.call_args_list
        self.assertGreater(len(write_calls), 0, "No data was written to the file")
        
        # Get the first line of JSON data
        first_write = write_calls[0][0][0]
        
        # Parse the JSON data
        try:
            data = json.loads(first_write)
            
            # Check the content
            self.assertEqual(data["text"], "Test pattern content")
            self.assertEqual(data["metadata"]["author"], "Test")
            self.assertEqual(data["metadata"]["pattern_type"], "intent_crystallization")
            self.assertEqual(data["system_prompt"], "You are a helpful assistant.")
        except json.JSONDecodeError as e:
            self.fail(f"Failed to parse JSON: {e}\nContent: {first_write}")

if __name__ == "__main__":
    unittest.main()
