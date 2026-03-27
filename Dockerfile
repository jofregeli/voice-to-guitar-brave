# Dockerfile for Voice-to-Guitar TFG
# Uses PyTorch with CUDA 12.1 for RTX 5080 compatibility
# Usage: docker-compose up --build

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL description="Voice-to-Guitar BRAVE training environment"
LABEL author="TFG Student"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first (layer caching — avoids reinstall on code changes)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install BRAVE
# NOTE: update this URL if you fork BRAVE
RUN git clone https://github.com/fcaspe/BRAVE /opt/BRAVE && \
    pip install -e /opt/BRAVE

# Copy project code (not data — mount data as a volume)
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories expected at runtime
RUN mkdir -p data/raw data/processed models

# Default command: show help
CMD ["python", "--version"]
