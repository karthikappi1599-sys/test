# Use a clean Python base
FROM python:3.10-slim

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for soundfile + datasets
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /app

# Copy your script into the container
COPY temp.py /app/temp.py

# Install Python dependencies
RUN pip install --no-cache-dir \
    datasets \
    huggingface_hub \
    soundfile \
    fsspec

# HuggingFace token passed at runtime
ENV HF_TOKEN=""

# Run the script
CMD ["python3", "temp.py"]
