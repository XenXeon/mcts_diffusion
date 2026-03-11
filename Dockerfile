# Start from the official NVIDIA CUDA image with Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Prevent timezone prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies and native Python 3.10
RUN apt-get update && apt-get install -y \
    git wget curl \
    python3.10 python3.10-dev python3.10-distutils \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Download MuJoCo C++ Engine (Required for the older mujoco_py wrapper)
RUN mkdir -p /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xzf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

# 3. Set MuJoCo environment variables globally
ENV MUJOCO_PATH=/root/.mujoco/mujoco210
ENV MUJOCO_PLUGIN_PATH=/root/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# 4. Install pip and set Python 3.10 as the default system Python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 5. Install PyTorch and torchvision with CUDA 12.8 for RTX 50-Series (sm_120) support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /workspace
CMD ["/bin/bash"]