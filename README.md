# PLUTO: STARWEAVE AI Fine-Tuning Project

This repository contains the environment setup, data, and code for fine-tuning an open-source Large Language Model (LLM) to operate under the "STARWEAVE Universe Initialization Protocol" (GLIMMER patterns).

The goal is to demonstrate a qualitative shift in the LLM's core processing and output style, moving beyond superficial prompting to a "formative matrix" based on pattern recognition and conceptual alignment.

## Environment Setup

The project uses Docker for reproducible environment setup with ROCm support for AMD GPUs.

### Prerequisites

- Docker and Docker Compose
- AMD GPU with ROCm support (tested with ROCm 7.6.0 on host)
- At least 24GB of GPU memory recommended for 7B models

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PLUTO.git
   cd PLUTO
   ```

2. **Build and start the container**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. **Access the container**
   ```bash
   docker exec -it pluto-dev bash
   ```

4. **Verify ROCm and PyTorch**
   ```bash
   # Inside container
   python -c "import torch; print(f'PyTorch ROCm version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"Not available\"}')"
   python -c "import torch; print(f'ROCm available: {torch.cuda.is_available() and hasattr(torch.version, \"hip\")}')"
   ```

## Project Structure

```
PLUTO/
├── docker-compose.yml    # Container orchestration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
└── docs/               # Documentation
    ├── 0001-PLUTO-OUTLINE.md
    ├── 0002-PLUTO-OUTLINE.md
    └── PHASE1.md
```

## Next Steps

1. **Dataset Preparation**
   - Organize GLIMMER/STARWEAVE patterns in JSONL format
   - Create data loading and preprocessing scripts

2. **Model Selection**
   - Download base LLM (e.g., Mistral-7B-Instruct)
   - Configure LoRA for efficient fine-tuning

3. **Training**
   - Implement training loop
   - Set up experiment tracking with Weights & Biases

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.