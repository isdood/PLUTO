Excellent\! "PLUTO" is a perfect designation for this initial, foundational step – a small but crucial piece of the larger STARWEAVE vision.

Given you're on Arch Linux with excellent hardware, we can set you up efficiently. And yes, absolutely, creating a repository is highly recommended. It provides version control, makes it easy to share or replicate your work, and keeps your project organized.

Here's a detailed, step-by-step guide for Phase 1: **PLUTO: Environment Setup & Model Selection.**

-----

## Phase 1: PLUTO - Environment Setup & Model Selection (for Arch Linux)

### **Step 0: Create the Project Repository (PLUTO)**

1.  **Initialize Git Repository:**
    ```bash
    mkdir PLUTO_STARWEAVE_AI
    cd PLUTO_STARWEAVE_AI
    git init
    ```
2.  **Create a `README.md`:**
    ```bash
    touch README.md
    ```
    Add some initial content to `README.md` (you can expand this later):
    ```markdown
    # PLUTO: STARWEAVE AI Fine-Tuning Project

    This repository contains the environment setup, data, and code for fine-tuning an open-source Large Language Model (LLM) to operate under the "STARWEAVE Universe Initialization Protocol" (GLIMMER patterns).

    The goal is to demonstrate a qualitative shift in the LLM's core processing and output style, moving beyond superficial prompting to a "formative matrix" based on pattern recognition and conceptual alignment.

    ## Phase 1: Environment Setup & Model Selection
    ```
3.  **Initial Git Commit:**
    ```bash
    git add .
    git commit -m "Initial project setup for PLUTO: STARWEAVE AI"
    ```
    (Optional: You can now create a remote repository on GitHub/GitLab and push your initial commit if you wish).

### **Step 1: System Updates & Essential Tools**

It's always a good idea to ensure your Arch Linux system is up-to-date.

1.  **Update System:**
    ```bash
    sudo pacman -Syu
    ```
2.  **Install Essential Build Tools (if not already present):** These are often needed for compiling Python packages.
    ```bash
    sudo pacman -S --needed base-devel git python-pip
    ```

### **Step 2: Python Environment (Conda Recommended)**

While `venv` works, `conda` (Miniconda/Anaconda) is generally preferred in the ML community for its robust dependency management, especially with CUDA/ROCm versions.

1.  **Install Miniconda:**
      * Download the latest Miniconda installer for Linux from the official Miniconda website: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
      * Example (check for latest version):
        ```bash
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        chmod +x Miniconda3-latest-Linux-x86_64.sh
        ./Miniconda3-latest-Linux-x86_64.sh
        ```
      * Follow the prompts. Press `ENTER` to accept the license terms. Type `yes` when prompted to initialize Miniconda3.
      * After installation, close and reopen your terminal, or source your `.bashrc`/`.zshrc`:
        ```bash
        source ~/.bashrc # or ~/.zshrc if you use zsh
        ```
      * Verify installation:
        ```bash
        conda --version
        ```
2.  **Create a New Conda Environment for PLUTO:**
    ```bash
    conda create -n pluto_starweave python=3.10
    ```
    (Using Python 3.10 as it's generally well-supported by ML libraries)
3.  **Activate the Environment:**
    ```bash
    conda activate pluto_starweave
    ```
    Your terminal prompt should change to `(pluto_starweave) user@hostname:~/PLUTO_STARWEAVE_AI$`

### **Step 3: PyTorch with ROCm Support for AMD GPU (Crucial\!)**

This is the most specific part for your AMD GPU. We need to ensure PyTorch is built with ROCm support.

1.  **Check ROCm Driver Installation:** Ensure your AMDGPU drivers and ROCm are correctly installed and configured on Arch. This is often handled by `amdgpu` kernel module and `rocm-hip-sdk` or similar packages from AUR or official repos.

      * You should ideally be able to run `rocminfo` and `rocm-smi` successfully.
      * If you encounter issues here, you might need to consult the Arch Wiki or AMD's documentation for your specific GPU/driver version.

2.  **Install PyTorch with ROCm (5.6 is a common target for 7900XTX, but verify AMD's latest recommendation):**

    ```bash
    # (Ensure you are in the 'pluto_starweave' conda environment)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # This is for CUDA, not ROCm directly from conda-forge
    ```

    **Correction for AMD ROCm:** Conda-forge generally lags for ROCm. The official PyTorch website provides `pip` wheels.

      * Go to: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
      * Select: `Stable`, `Linux`, `Pip`, `ROCm`.
      * It will provide a command like:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 # (or 5.6, 6.0, etc.)
        ```
        **Crucially, check which ROCm version is best supported by your 7900XTX and your installed ROCm drivers on Arch. As of mid-2024, ROCm 5.6 or 5.7 are common. ROCm 6.0+ is newer.**

3.  **Verify PyTorch & GPU Recognition:**

    ```bash
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.current_device())"
    ```

      * Expected Output: `True`, `AMD Radeon RX 7900 XTX` (or similar), `0`. If `False` or no device name, ROCm/PyTorch setup is incorrect.

### **Step 4: Install HuggingFace Libraries & Others**

These are the core libraries for LLM handling and fine-tuning.

1.  **Install Core Libraries:**
    ```bash
    # (Ensure you are in the 'pluto_starweave' conda environment)
    pip install transformers peft accelerate trl safetensors xformers bitsandbytes
    ```
      * `bitsandbytes`: Crucial for QLoRA. AMD ROCm support for `bitsandbytes` can sometimes be tricky. If `pip install bitsandbytes` fails or doesn't provide ROCm-enabled binaries, you might need to install it from source (more complex) or use a custom build. We can troubleshoot this if it becomes an issue.
      * `xformers`: Provides optimized attention mechanisms (like Flash Attention 2). Check for AMD/ROCm compatibility; if `pip install xformers` fails, you might skip it for now.

### **Step 5: Model Selection & Initial Download (Optional, but good for verification)**

You can download your chosen model to ensure you have access and it loads correctly.

1.  **Choose Your Model:** Let's go with **Mistral-7B-Instruct-v0.2** as a solid starting point.
2.  **Python Script to Download/Load:**
    Create a new file `test_model_load.py` in your `PLUTO_STARWEAVE_AI` directory:
    ```python
    # test_model_load.py
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    print(f"Attempting to load tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded successfully!")

    print(f"Attempting to load model {model_name}...")
    # Using float16 or bfloat16 for efficiency and to fit on GPU
    # If VRAM is an issue, try load_in_4bit=True with bitsandbytes
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto" # This automatically maps layers to available devices (GPU/CPU)
        )
        print("Model loaded successfully!")
        print(f"Model on device: {model.device}")
        print(f"Model Dtype: {model.dtype}")

        # Basic test inference
        prompt = "Hello, I am a language model."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Running basic inference on {model_name}...")
        outputs = model.generate(**inputs, max_new_tokens=20)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Inference result: {decoded_output}")

    except Exception as e:
        print(f"Error loading or inferring with model: {e}")
        print("This might be due to VRAM issues or incorrect bitsandbytes/ROCm setup. Try `load_in_4bit=True` if you're hitting VRAM limits.")

    print("--- Model Load Test Complete ---")
    ```
3.  **Run the Test Script:**
    ```bash
    python test_model_load.py
    ```
    This script will download the model (takes time) and attempt to load it onto your GPU. If it completes successfully, you've got your basic environment working\! If it errors, particularly with `bitsandbytes` or `CUDA/ROCm` memory issues, we'll need to troubleshoot.

### **Step 6: Update `README.md` and Commit**

1.  **Add Environment Details to `README.md`:**
    ```markdown
    ## Phase 1: Environment Setup & Model Selection

    **Hardware:**
    - CPU: AMD Ryzen 9950X (32-thread)
    - GPU: AMD Radeon RX 7900 XTX (24GB VRAM)
    - RAM: 64GB 6400MHz DDR5

    **Operating System:** Arch Linux

    **Python Environment:** Conda environment `pluto_starweave` (Python 3.10)

    **PyTorch:** Installed with ROCm support (version TBD based on `pip install` output)

    **Key Libraries:** `transformers`, `peft`, `accelerate`, `trl`, `safetensors`, `xformers` (if compatible), `bitsandbytes` (if ROCm compatible).

    **Selected Base Model:** `mistralai/Mistral-7B-Instruct-v0.2`

    ---
    ```
2.  **Commit Your Changes:**
    ```bash
    git add .
    git commit -m "Phase 1: Completed environment setup and confirmed base model loading."
    ```

-----

**Reporting Back:**

Go through these steps. Take your time, especially with the PyTorch/ROCm installation. If you encounter *any* errors, paste the full error output here, and we'll troubleshoot it together. When you successfully load `Mistral-7B-Instruct-v0.2` and get an inference output, you'll have completed Phase 1\!

Good luck, PLUTO\!
