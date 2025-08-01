# PLUTO: STARWEAVE AI Fine-Tuning Project

This repository contains the environment setup, data, and code for fine-tuning an open-source Large Language Model (LLM) to operate under the "STARWEAVE Universe Initialization Protocol" (GLIMMER patterns).

**Current Status**: ✅ **WORKING** - CPU-optimized pipeline with TinyLlama-1.1B-Chat model successfully trained and tested.

## 🎯 Project Overview

PLUTO fine-tunes language models to operate under the "STARWEAVE Universe Initialization Protocol" (GLIMMER patterns), demonstrating a qualitative shift in the model's core processing and output style.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended for CPU operation
- At least 6GB disk space for models and data

### 1. Build and Start Container

```bash
# Clone the repository
git clone https://github.com/yourusername/PLUTO.git
cd PLUTO

# Build and start the container
docker-compose build
docker-compose up -d
```

### 2. Prepare Training Data

```bash
# Inside container
docker-compose exec -T pluto python3 scripts/prepare_glimmer_data.py
```

This converts GLIMMER patterns from `glimmer/data/patterns/` into training data in `data/processed/`.

### 3. Train the Model

```bash
# Basic training with default settings
docker-compose exec -T pluto python3 scripts/train_lora_simple.py

# Custom training parameters
docker-compose exec -T pluto python3 scripts/train_lora_simple.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4
```

### 4. Test the Model

```bash
# Automated testing (compares trained vs base model)
docker-compose exec -T pluto python3 scripts/test_model.py

# Interactive testing mode
docker-compose exec -T pluto python3 scripts/test_model.py --interactive
```

## 📁 Project Structure

```
PLUTO/
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies (consolidated)
├── data/
│   └── processed/             # Training data
│       ├── train.jsonl        # Training examples (29 lines)
│       ├── validation.jsonl   # Validation examples (9 lines)
│       └── training_data.jsonl # Original data (10 lines)
├── models/
│   ├── pluto-simple/          # Trained PLUTO model
│   │   ├── final/            # Final trained model (4.1GB)
│   │   └── checkpoint-21/    # Training checkpoint
│   └── tinyllama-1.1b-chat/  # Base model for comparison (2.0GB)
├── scripts/                   # Working pipeline scripts
│   ├── prepare_glimmer_data.py # Data preparation
│   ├── train_lora_simple.py   # CPU-optimized training
│   ├── test_model.py          # Model testing
│   ├── README.md              # Scripts documentation
│   ├── MODEL_TESTING_GUIDE.md # Testing guide
│   └── PIPELINE_SUMMARY.md    # Pipeline overview
├── glimmer/
│   └── data/patterns/         # GLIMMER pattern files
└── docs/                      # Project documentation
```

## 🎯 Current Implementation

### ✅ **Working Pipeline**

- **Model**: TinyLlama-1.1B-Chat (CPU-optimized)
- **Training**: Full fine-tuning (CPU-compatible)
- **Hardware**: CPU-only operation (tested and working)
- **Data**: GLIMMER patterns converted to conversation format
- **Testing**: Automated and interactive testing available

### 📊 **Training Results**

- **Loss Reduction**: Significant decrease in training loss
- **Model Size**: 4.1GB trained model
- **Training Time**: ~5 mins on CPU (9950X @ 6.0GHz)
- **Memory Usage**: ~4GB RAM during training

## 🔧 Available Scripts

### Data Preparation
```bash
# Prepare GLIMMER patterns for training
docker-compose exec -T pluto python3 scripts/prepare_glimmer_data.py
```

### Training
```bash
# CPU-optimized training
docker-compose exec -T pluto python3 scripts/train_lora_simple.py
```

### Testing
```bash
# Automated testing
docker-compose exec -T pluto python3 scripts/test_model.py

# Interactive testing
docker-compose exec -T pluto python3 scripts/test_model.py --interactive
```

## 📖 Documentation

- **`scripts/README.md`** - Complete script documentation
- **`scripts/MODEL_TESTING_GUIDE.md`** - Comprehensive testing guide
- **`scripts/PIPELINE_SUMMARY.md`** - Pipeline overview
- **`docs/`** - Project documentation

## 🎯 Key Features

### ✅ **Working Components**
- **Data Preparation**: Converts GLIMMER patterns to training format
- **CPU Training**: Reliable training without GPU dependencies
- **Model Testing**: Automated comparison and interactive testing
- **Documentation**: Complete guides and troubleshooting

### 🚀 **Pipeline Benefits**
- **No GPU Dependencies**: Works on any CPU
- **Reliable**: Tested and working
- **Fast Iteration**: Quick training cycles
- **Easy Testing**: Automated and interactive modes

## 🔍 Testing the Model

The trained model demonstrates qualitative differences from the base model:

- **STARWEAVE Vocabulary**: Uses GLIMMER pattern terminology
- **Conceptual Alignment**: Responds with pattern-specific content
- **Self-Awareness**: Shows meta-cognitive processing
- **Structured Responses**: Follows GLIMMER pattern formats

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size: `--batch_size 1`
   - Use gradient accumulation: `--gradient_accumulation_steps 4`

2. **Model Loading Errors**
   - Check model exists: `ls -la models/pluto-simple/final/`
   - Re-run training if needed

3. **Training Loss Not Decreasing**
   - Reduce learning rate: `--learning_rate 1e-4`
   - Increase warmup steps: `--warmup_steps 200`

## 🎯 Next Steps

### Current Status: ✅ **PRODUCTION READY**

The pipeline is working and ready for:
- **Model Training**: Successfully trains on GLIMMER patterns
- **Model Testing**: Comprehensive testing capabilities
- **Documentation**: Complete guides and troubleshooting

### Future Enhancements
- [ ] Add GPU support for larger models
- [ ] Implement LoRA for memory efficiency
- [ ] Scale to larger models (Mistral-7B, etc.)
- [ ] Add advanced training techniques

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎉 The PLUTO pipeline is now clean, focused, and ready for production use!**