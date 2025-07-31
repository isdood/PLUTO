# Use the official ROCm PyTorch image with a stable version
# Using ROCm 5.6 as it's well-tested and widely available
FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    rocm-smi \
    rocm-libs \
    rocm-utils \
    && rm -rf /var/lib/apt/lists/*

# Add user to video and render groups
RUN usermod -aG video root
RUN usermod -aG render root

# Create and activate Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy source code first
COPY . /workspace
WORKDIR /workspace

# Install Python packages
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install model requirements with specific versions for ROCm compatibility
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6 && \
    pip install -r requirements-model.txt

# Configure accelerate
RUN mkdir -p /workspace/models && \
    mkdir -p /root/.cache/huggingface/accelerate && \
    echo 'compute_environment: LOCAL_MACHINE\ndistributed_type: NO\nuse_cpu: False\n' > /root/.cache/huggingface/accelerate/default_config.yaml

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Install GLIMMER package in development mode
RUN pip install -e .

# Create workspace directory
WORKDIR /workspace

# Set default command
CMD ["/bin/bash"]
