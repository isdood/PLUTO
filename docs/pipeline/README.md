# GLIMMER Pattern Pipeline

Welcome to the GLIMMER Pattern Pipeline documentation! This guide will help you understand, use, and extend the pipeline for processing GLIMMER patterns for LLM fine-tuning.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Adding New Patterns](#adding-new-patterns)
5. [Docker Setup](#docker-setup)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Overview

The GLIMMER Pattern Pipeline is designed to process structured pattern files into a format suitable for fine-tuning language models. It handles:

- Pattern validation against the GLIMMER schema
- Conversion to instruction-following format
- Dataset splitting into training and validation sets
- Preparation for LoRA fine-tuning

## Quick Start

1. Ensure you have Docker installed and running
2. Clone the repository and navigate to the project directory
3. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```
4. Process your patterns:
   ```bash
   # Prepare the training data
   docker-compose exec -T pluto python3 scripts/prepare_lora_data.py \
       --patterns-dir glimmer/data/patterns \
       --output-dir data/processed
   
   # Split into train/validation sets
   docker-compose exec -T pluto python3 scripts/split_data.py \
       --input-file data/processed/training_data.jsonl \
       --output-dir data/processed \
       --train-ratio 0.8 \
       --seed 42
   ```

Continue reading the detailed documentation in the sections below for more information.

[Next: Pipeline Architecture →](pipeline_architecture.md)
