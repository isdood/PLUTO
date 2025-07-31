# Adding New Patterns

This guide explains how to add new GLIMMER patterns to the pipeline.

## Pattern File Format

Each pattern should be a JSON file with the following structure:

```json
{
  "metadata": {
    "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
    "author": "Your Name or Identifier",
    "pattern_version": "1.0.0",
    "color": "#RRGGBB",
    "type": "pattern_type",
    "harmonics": {
      "key1": {
        "color": "#RRGGBB",
        "description": "Description of this harmonic"
      },
      "key2": {
        "color": "#RRGGBB",
        "description": "Description of this harmonic"
      }
    }
  },
  "content": {
    "title": "Pattern Title",
    "sections": [
      {
        "title": "Section Title",
        "content": {
          "key1": "value1",
          "key2": "value2"
        }
      }
    ]
  }
}
```

## Naming Conventions

- **Filename**: `{number:04d}-{descriptive_name}.json`
  - Example: `0010-pattern_name.json`
- **Numbering**: Use the next available number in sequence
- **Descriptive Name**: Use lowercase with underscores for spaces

## Step-by-Step Guide

1. **Create a New Pattern File**
   ```bash
   # Navigate to the patterns directory
   cd glimmer/data/patterns
   
   # Find the highest number currently in use
   ls -1 *.json | sort -n | tail -1
   
   # Create a new file with the next number
   touch 0010-descriptive_name.json
   ```

2. **Edit the Pattern File**
   - Use the template above
   - Ensure all required fields are present
   - Follow the existing pattern structure

3. **Validate the Pattern**
   ```bash
   # Run the data preparation script in validation mode
   docker-compose exec -T pluto python3 scripts/prepare_lora_data.py \
       --patterns-dir glimmer/data/patterns \
       --output-dir /tmp/validation \
       --validate-only
   ```

4. **Process the New Pattern**
   ```bash
   # Run the full pipeline
   docker-compose exec -T pluto python3 scripts/prepare_lora_data.py \
       --patterns-dir glimmer/data/patterns \
       --output-dir data/processed
   
   docker-compose exec -T pluto python3 scripts/split_data.py \
       --input-file data/processed/training_data.jsonl \
       --output-dir data/processed \
       --train-ratio 0.8 \
       --seed 42
   ```

## Best Practices

1. **Content Guidelines**
   - Keep section titles clear and consistent
   - Use markdown formatting in content fields
   - Include examples where helpful

2. **Versioning**
   - Update the `pattern_version` when making changes
   - Document changes in the commit message

3. **Testing**
   - Always validate new patterns
   - Check the output format
   - Verify the training/validation split

## Example Pattern

See [Example Pattern](example_pattern.md) for a complete example with explanations.

[Next: Docker Setup →](docker_setup.md)
