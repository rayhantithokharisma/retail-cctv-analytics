FROM python:3.11-slim

# System dependencies for OpenCV and multimedia processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Optional CUDA runtime notes:
# For NVIDIA GPU support, use nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base image
# and install torch with --index-url https://download.pytorch.org/whl/cu121

# Default command runs the entire pipeline
CMD ["python", "-m", "src.cli", "run-all"]
