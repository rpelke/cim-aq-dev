##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Use CUDA base image
FROM docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.11
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    git \
    curl \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies with cleanup
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /workspace/requirements.txt \
    && pip3 cache purge \
    && find /usr/local/lib/python3*/dist-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3*/dist-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3*/dist-packages -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3*/dist-packages -name "test" -type d -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /tmp/* /var/tmp/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Set working directory
WORKDIR /workspace

# Copy only the necessary files and folders
COPY lib/ /workspace/lib/
COPY models/ /workspace/models/
COPY run/ /workspace/run/
COPY utils/ /workspace/utils/
COPY finetune.py /workspace/finetune.py
COPY pretrain.py /workspace/pretrain.py
COPY rl_quantize.py /workspace/rl_quantize.py

# Create necessary directories
RUN mkdir -p data /workspace/pretrained/imagenet /workspace/checkpoints /workspace/save

# Create a home directory for cache and make workspace writable when running as different users
RUN mkdir -p /workspace/home && chmod -R 777 /workspace
ENV HOME=/workspace/home

# Set the default command
CMD ["bash"]