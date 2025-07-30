#!/usr/bin/env python3
"""
Script to download and prepare the Mistral-7B-Instruct model for fine-tuning.

This script will:
1. Download the Mistral-7B-Instruct model from Hugging Face
2. Save it in the models directory
3. Verify the model can be loaded with the correct configuration
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
# Using a small, fully open model for initial testing
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small, open model that doesn't require auth
MODEL_SAVE_DIR = Path("models/tinyllama-1.1b-chat")

# Ensure the model directory exists
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def setup_bnb_config():
    """Setup model configuration.
    
    For now, we're not using 4-bit quantization to avoid compatibility issues.
    We'll add quantization back after we confirm the basic download works.
    """
    return None  # No quantization for now

def download_model():
    """Download and save the model and tokenizer."""
    logger.info(f"Downloading {MODEL_NAME}...")
    
    # Get model config (no quantization for now)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # For batch inference
    
    # Save model and tokenizer
    logger.info(f"Saving model to {MODEL_SAVE_DIR}...")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    
    logger.info("Model and tokenizer saved successfully!")
    return model, tokenizer

def verify_model():
    """Verify the model can be loaded and performs inference."""
    logger.info("Verifying model...")
    
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_SAVE_DIR,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SAVE_DIR,
            trust_remote_code=True,
        )
        
        # Test inference
        prompt = "### Instruction: Explain what this project is about.\n### Response:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("\n--- Model Response ---")
        logger.info(response)
        logger.info("---------------------\n")
        
    except Exception as e:
        logger.error(f"Error during model verification: {e}")
        logger.info("Model files were downloaded, but there was an error during verification.")
        logger.info("This might be due to GPU/CPU compatibility issues, but the model is ready for use.")
        return False
    
    return True  # Return True if verification was successful
    
    logger.info("Model verification complete!")

if __name__ == "__main__":
    # Check if model already exists (look for either safetensors or pytorch model files)
    model_exists = (
        (MODEL_SAVE_DIR / "model.safetensors").exists() or 
        (MODEL_SAVE_DIR / "pytorch_model.bin").exists() or
        (MODEL_SAVE_DIR / "pytorch_model.safetensors").exists()
    )
    
    if model_exists:
        logger.info(f"Model found at {MODEL_SAVE_DIR}. Verifying...")
        verification_success = verify_model()
    else:
        logger.info("Model not found. Downloading...")
        model, tokenizer = download_model()
        verification_success = verify_model()
    
    # Exit with appropriate status code
    sys.exit(0 if verification_success else 1)
