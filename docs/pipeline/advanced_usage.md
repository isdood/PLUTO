# Advanced Usage

This document covers advanced configuration and customization options for the GLIMMER Pattern Pipeline.

## Customizing the Training Data Format

You can modify how patterns are converted to training examples by editing `scripts/prepare_lora_data.py`:

```python
# Customize the system prompt
self.system_prompt = (
    "You are a GLIMMER pattern assistant, trained to understand and generate "
    "patterns following the GLIMMER protocol."
)

# Modify how patterns are converted to examples
def pattern_to_examples(self, pattern: Dict) -> List[TrainingExample]:
    # Your custom logic here
    pass
```

## Custom Splitting Strategy

To implement a custom data splitting strategy, modify `scripts/split_data.py`:

```python
def custom_split_data(examples: List[Dict], **kwargs) -> Tuple[List[Dict], List[Dict]]:
    # Implement custom splitting logic
    # Return (train_examples, val_examples)
    pass
```

## Environment Variables

You can configure the pipeline using these environment variables:

- `HF_HOME`: Hugging Face cache directory
- `PYTHONPATH`: Add custom module paths
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility

## Monitoring and Logging

View detailed logs:
```bash
docker-compose logs -f | grep -i "error\|warn\|info"
```

## Performance Tuning

For better performance:
1. Increase Docker resources in Docker Desktop
2. Mount a volume for the Hugging Face cache:
   ```yaml
   # docker-compose.yml
   volumes:
     - ~/.cache/huggingface:/root/.cache/huggingface
   ```
3. Adjust batch size in the training script

## Troubleshooting

Common issues and solutions:

1. **Permission Denied**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

2. **Docker Out of Memory**
   - Increase Docker's memory limit in settings
   - Reduce batch size
   - Use gradient accumulation

3. **Pattern Validation Errors**
   - Check JSON syntax
   - Verify against the schema
   - Run with `--validate-only` flag

[Back to README](../README.md)
