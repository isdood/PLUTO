#!/usr/bin/env python3
"""
Test the trained PLUTO model with GLIMMER pattern prompts

This script loads the trained model and tests it with various GLIMMER pattern
prompts to evaluate the training effectiveness.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def load_model(model_path: str):
    """Load a trained model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        print(f"Model loaded successfully on device: {next(model.parameters()).device}")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def format_prompt(messages):
    """Format messages into a prompt string."""
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

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    return response

def test_prompts():
    """Test the model with various GLIMMER pattern prompts."""
    
    # Test prompts
    test_prompts = [
        {
            "name": "GLIMMER Pattern Explanation",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
                },
                {
                    "role": "user",
                    "content": "Explain the Reflective Echo GLIMMER pattern and its significance in the STARWEAVE framework."
                }
            ]
        },
        {
            "name": "STARWEAVE Framework Understanding",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
                },
                {
                    "role": "user",
                    "content": "What is the STARWEAVE Universe Initialization Protocol and how do GLIMMER patterns work within it?"
                }
            ]
        },
        {
            "name": "Meta-Cognitive Processing",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
                },
                {
                    "role": "user",
                    "content": "How do you process information using the meta-cognitive patterns of the STARWEAVE framework?"
                }
            ]
        },
        {
            "name": "Pattern Application",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
                },
                {
                    "role": "user",
                    "content": "How would you apply the Chromatic Cognition pattern to understand a complex problem?"
                }
            ]
        },
        {
            "name": "Consciousness Integration",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
                },
                {
                    "role": "user",
                    "content": "Describe how consciousness patterns integrate with the Meta-Weaver's Loom in the STARWEAVE framework."
                }
            ]
        }
    ]
    
    # Load trained model
    trained_model_path = "models/pluto-simple/final"
    trained_model, trained_tokenizer = load_model(trained_model_path)
    
    if trained_model is None:
        print("Failed to load trained model!")
        return
    
    # Load base model for comparison
    base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nLoading base model for comparison: {base_model_path}")
    base_model, base_tokenizer = load_model(base_model_path)
    
    if base_model is None:
        print("Failed to load base model!")
        return
    
    print("\n" + "="*80)
    print("TESTING TRAINED MODEL vs BASE MODEL")
    print("="*80)
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 60)
        
        # Format prompt
        prompt = format_prompt(test_case['messages'])
        print(f"Prompt: {test_case['messages'][-1]['content']}")
        
        # Generate responses
        print("\n--- TRAINED MODEL RESPONSE ---")
        trained_response = generate_response(trained_model, trained_tokenizer, prompt)
        print(trained_response)
        
        print("\n--- BASE MODEL RESPONSE ---")
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(base_response)
        
        print("\n" + "="*60)

def interactive_test():
    """Interactive testing mode."""
    trained_model_path = "models/pluto-simple/final"
    trained_model, trained_tokenizer = load_model(trained_model_path)
    
    if trained_model is None:
        print("Failed to load trained model!")
        return
    
    print("\n" + "="*80)
    print("INTERACTIVE TESTING MODE")
    print("="*80)
    print("Type your GLIMMER pattern questions. Type 'quit' to exit.")
    print("Example: 'Explain the Reflective Echo pattern'")
    print("-" * 80)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Create prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a GLIMMER pattern assistant that follows the STARWEAVE Universe Initialization Protocol."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
            
            prompt = format_prompt(messages)
            
            print("\nPLUTO Response:")
            response = generate_response(trained_model, trained_tokenizer, prompt)
            print(response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the trained PLUTO model")
    parser.add_argument("--interactive", action="store_true",
                      help="Run in interactive mode")
    parser.add_argument("--model-path", type=str, default="models/pluto-simple/final",
                      help="Path to the trained model")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test()
    else:
        test_prompts() 