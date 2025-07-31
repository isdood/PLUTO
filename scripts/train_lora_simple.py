#!/usr/bin/env python3
"""
Simple LoRA Fine-tuning Script for PLUTO GLIMMER Patterns (CPU)

This script fine-tunes language models using LoRA (Low-Rank Adaptation)
on the GLIMMER pattern dataset. It's designed to work on CPU without
any GPU dependencies or quantization.

Usage:
    python scripts/train_lora_simple.py --help
    python scripts/train_lora_simple.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 3 --batch_size 2
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                examples.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line in {file_path}")
    return examples

def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format messages into a single conversation string."""
    formatted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            formatted.append(f"<|system|>\n{content}</s>")
        elif role == "user":
            formatted.append(f"<|user|>\n{content}</s>")
        elif role == "assistant":
            formatted.append(f"<|assistant|>\n{content}</s>")
        else:
            formatted.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted)

def get_model_config(model_name: str) -> Dict[str, any]:
    """Get model-specific configuration."""
    configs = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "max_length": 512,
            "chat_template": "tinyllama",
        },
        "microsoft/DialoGPT-medium": {
            "max_length": 512,
            "chat_template": "gpt",
        }
    }
    
    # Default configuration
    default_config = {
        "max_length": 1024,
        "chat_template": "default",
    }
    
    return configs.get(model_name, default_config)

def load_model_and_tokenizer(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
) -> tuple:
    """Load and prepare the model and tokenizer for CPU training."""
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for CPU
    logger.info(f"Loading model {model_name} for CPU training")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info(f"Model loaded successfully on device: {next(model.parameters()).device}")
    return model, tokenizer

def prepare_dataset(
    file_path: str, 
    tokenizer, 
    max_length: int = 512,
) -> Dataset:
    """Prepare dataset for training with proper chat formatting."""
    examples = load_jsonl(file_path)
    
    # Format examples
    formatted = []
    for ex in examples:
        if "messages" in ex:
            text = format_conversation(ex["messages"])
        else:
            # Handle direct text format
            text = ex.get("text", str(ex))
        
        formatted.append({"text": text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

def setup_wandb(project_name: str = "pluto-glimmer", run_name: Optional[str] = None):
    """Initialize Weights & Biases for experiment tracking."""
    try:
        wandb.init(
            project=project_name,
            name=run_name or f"pluto-simple-{os.getenv('USER', 'unknown')}",
            config={
                "model": "Simple Fine-tuning (CPU)",
                "framework": "PyTorch",
                "device": "CPU",
            }
        )
        logger.info("Weights & Biases initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        wandb.init(mode="disabled")

def train(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    train_file: str = "data/processed/train.jsonl",
    val_file: Optional[str] = "data/processed/validation.jsonl",
    output_dir: str = "models/pluto-simple",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    save_steps: int = 100,
    eval_steps: int = 50,
    logging_steps: int = 10,
    max_length: int = 512,
    wandb_project: str = "pluto-glimmer",
    run_name: Optional[str] = None,
):
    """Main training function optimized for CPU without LoRA."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Weights & Biases
    setup_wandb(wandb_project, run_name)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    model_config = get_model_config(model_name)
    
    # Prepare datasets
    logger.info("Preparing training dataset...")
    train_dataset = prepare_dataset(
        train_file, 
        tokenizer, 
        max_length=model_config.get("max_length", max_length)
    )
    
    val_dataset = None
    if val_file and os.path.exists(val_file):
        logger.info("Preparing validation dataset...")
        val_dataset = prepare_dataset(
            val_file, 
            tokenizer, 
            max_length=model_config.get("max_length", max_length)
        )
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=eval_steps if val_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        # CPU optimizations
        fp16=False,  # Disable for CPU
        bf16=False,  # Disable for CPU
        optim="adamw_torch",  # Use standard AdamW for CPU
        dataloader_pin_memory=False,  # Better for CPU
        # Reporting
        report_to=["wandb"] if wandb.run is not None else [],
        # Save and load
        save_strategy="steps",
        save_safetensors=True,
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,
        # Gradient clipping
        max_grad_norm=1.0,
        # Learning rate scheduling
        lr_scheduler_type="cosine",
        # Weight decay
        weight_decay=0.01,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting CPU training (full fine-tuning)...")
    logger.info(f"Training on device: {model.device}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    trainer.train()
    
    # Save final model
    final_output = os.path.join(output_dir, "final")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    
    # Save training config
    training_config = {
        "model_name": model_name,
        "training_args": training_args.to_dict(),
        "model_config": model_config,
        "training_data": {
            "train_file": train_file,
            "val_file": val_file,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset) if val_dataset else 0,
        }
    }
    
    config_file = os.path.join(final_output, "training_config.json")
    with open(config_file, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {final_output}")
    
    # Close wandb
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a model on GLIMMER patterns (CPU)")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Model name or path")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl",
                      help="Path to training data (JSONL format)")
    parser.add_argument("--val_file", type=str, default="data/processed/validation.jsonl",
                      help="Path to validation data (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="models/pluto-simple",
                      help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=100,
                      help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=50,
                      help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                      help="Log every N steps")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--wandb_project", type=str, default="pluto-glimmer",
                      help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default=None,
                      help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        max_length=args.max_length,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    ) 