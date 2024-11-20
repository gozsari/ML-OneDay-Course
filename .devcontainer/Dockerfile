# Start from the base Python image
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m venv /workspace/.venv \
    && /workspace/.venv/bin/pip install --upgrade pip \
    && /workspace/.venv/bin/pip install -r requirements.txt

# Set PATH to include the virtual environment
ENV PATH="/workspace/.venv/bin:$PATH"
