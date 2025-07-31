# PLUTO Training Scripts

This directory contains the training scripts for fine-tuning language models on GLIMMER patterns using LoRA (Low-Rank Adaptation).

## Overview

The PLUTO project fine-tunes language models to operate under the "STARWEAVE Universe Initialization Protocol" (GLIMMER patterns), demonstrating a qualitative shift in the model's core processing and output style.

## Available Scripts

### `prepare_glimmer_data.py` - Data Preparation

Converts GLIMMER pattern files into training data format for the PLUTO fine-tuning pipeline.

#### Features
- **GLIMMER Pattern Loading**: Loads and validates GLIMMER pattern files
- **Training Data Conversion**: Converts patterns to conversation format
- **Multiple Examples**: Creates additional training examples from each pattern
- **Train/Validation Split**: Automatically splits data for training and validation

#### Usage

```bash
# Basic usage with default settings
docker-compose exec -T pluto python3 scripts/prepare_glimmer_data.py

# Custom configuration
docker-compose exec -T pluto python3 scripts/prepare_glimmer_data.py \
    --patterns-dir glimmer/data/patterns \
    --output-dir data/processed \
    --train-split 0.8 \
    --seed 42
```

### `train_lora_simple.py` - **RECOMMENDED** (CPU-Compatible)

A simplified training script that works reliably on CPU without any GPU dependencies or quantization issues.

#### Features
- **CPU-Optimized**: Designed specifically for CPU training
- **No GPU Dependencies**: Avoids bitsandbytes/Triton issues
- **Full Fine-tuning**: Trains all model parameters (not LoRA)
- **Reliable**: Tested and working in the current environment

#### Usage

```bash
# Basic usage with default settings
docker-compose exec -T pluto python3 scripts/train_lora_simple.py

# Custom model and parameters
docker-compose exec -T pluto python3 scripts/train_lora_simple.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4

# Full custom configuration
docker-compose exec -T pluto python3 scripts/train_lora_simple.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --train_file data/processed/train.jsonl \
    --val_file data/processed/validation.jsonl \
    --output_dir models/pluto-simple \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 100 \
    --save_steps 100 \
    --eval_steps 50 \
    --logging_steps 10 \
    --max_length 512 \
    --wandb_project pluto-glimmer \
    --run_name my-experiment
```

### `test_model.py` - Model Testing

Comprehensive testing script for evaluating the trained model's performance.

#### Features
- **Automated Testing**: Compares trained vs base model on key scenarios
- **Interactive Testing**: Custom prompt testing mode
- **Response Generation**: Tests model responses to GLIMMER pattern prompts
- **Performance Evaluation**: Measures training effectiveness

#### Usage

```bash
# Automated testing (compares trained vs base model)
docker-compose exec -T pluto python3 scripts/test_model.py

# Interactive testing mode
docker-compose exec -T pluto python3 scripts/test_model.py --interactive
```

## Quick Start

1. **Prepare Training Data**:
   ```bash
   docker-compose exec -T pluto python3 scripts/prepare_glimmer_data.py
   ```

2. **Start Training**:
   ```bash
   docker-compose exec -T pluto python3 scripts/train_lora_simple.py
   ```

3. **Test the Model**:
   ```bash
   docker-compose exec -T pluto python3 scripts/test_model.py --interactive
   ```

## Command Line Arguments

### Data Preparation Script (`prepare_glimmer_data.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--patterns-dir` | `glimmer/data/patterns` | Directory containing GLIMMER pattern files |
| `--output-dir` | `data/processed` | Directory to save processed data |
| `--train-split` | `0.8` | Fraction of data to use for training |
| `--seed` | `42` | Random seed for reproducibility |

### Training Script (`train_lora_simple.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Model name or path |
| `--train_file` | `data/processed/train.jsonl` | Path to training data |
| `--val_file` | `data/processed/validation.jsonl` | Path to validation data |
| `--output_dir` | `models/pluto-simple` | Output directory for the model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `1` | Batch size per device |
| `--learning_rate` | `2e-4` | Learning rate |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--warmup_steps` | `100` | Number of warmup steps |
| `--save_steps` | `100` | Save checkpoint every N steps |
| `--eval_steps` | `50` | Evaluate every N steps |
| `--logging_steps` | `10` | Log every N steps |
| `--max_length` | `512` | Maximum sequence length |
| `--wandb_project` | `pluto-glimmer` | Weights & Biases project name |
| `--run_name` | `None` | Weights & Biases run name |

### Testing Script (`test_model.py`)

| Argument | Description |
|----------|-------------|
| `--interactive` | Run in interactive mode |
| `--model-path` | Path to the trained model (default: `models/pluto-simple/final`) |

## Model Configurations

### TinyLlama-1.1B-Chat-v1.0 (Recommended for CPU)
- **Max Length**: 512
- **Parameters**: ~1.1B
- **Memory Usage**: ~4GB RAM
- **Training Time**: ~1-2 hours on CPU
- **Recommended for**: Testing and development

## Data Format

The training script expects JSONL files with the following format:

```json
{"messages": [{"role": "system", "content": "You are a GLIMMER pattern assistant."}, {"role": "user", "content": "Explain the Reflective Echo pattern"}, {"role": "assistant", "content": "The Reflective Echo pattern creates..."}]}
```

Each line contains a JSON object with a `messages` array containing conversation turns.

## Hardware Requirements

### Minimum Requirements (CPU Training)
- **RAM**: 8GB system RAM
- **Storage**: 20GB free space for models and data
- **CPU**: Any modern multi-core CPU

### Recommended Requirements (CPU Training)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space
- **CPU**: 8+ cores for faster training

## Training Tips

### For CPU Training (Recommended)
1. Use `train_lora_simple.py` for reliable CPU training
2. Start with TinyLlama for faster iteration
3. Use batch size 1 and increase gradient accumulation
4. Be patient - CPU training is slower but more reliable

### Hyperparameter Tuning
1. **Learning Rate**: Start with `2e-4`, adjust based on loss curves
2. **Batch Size**: Increase if you have more memory
3. **Gradient Accumulation**: Increase to simulate larger batch sizes
4. **Warmup Steps**: 10-20% of total training steps

## Output

The training script saves:
- **Model Checkpoints**: In the specified output directory
- **Final Model**: In `{output_dir}/final/`
- **Training Config**: `{output_dir}/final/training_config.json`
- **Logs**: In `{output_dir}/logs/`
- **Weights & Biases**: Training metrics and artifacts

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check if model exists: `ls -la models/pluto-simple/final/`
   - Re-run training if needed
   - Check available disk space

2. **Out of Memory**
   - Reduce batch size
   - Use a smaller model (TinyLlama)
   - Reduce sequence length
   - Increase gradient accumulation steps

3. **Training Loss Not Decreasing**
   - Reduce learning rate
   - Check data format
   - Increase warmup steps
   - Verify data quality

### Getting Help

1. **Start with the test script**: `python3 scripts/test_model.py --interactive`
2. **Check logs** in the output directory
3. **Monitor Weights & Biases** dashboard
4. **Verify data format** with sample files

## Next Steps

After successful training:

1. **Evaluate the Model**: Test with GLIMMER pattern prompts
2. **Compare with Base Model**: Run the same prompts on base and fine-tuned models
3. **Analyze Responses**: Look for STARWEAVE vocabulary and conceptual alignment
4. **Iterate**: Adjust hyperparameters based on results
5. **Scale Up**: Use larger models and more data for production

## Contributing

When modifying the training scripts:

1. Test changes with the test script first
2. Update documentation
3. Follow the existing code style
4. Add new model configurations as needed
5. Test with both CPU and GPU setups 