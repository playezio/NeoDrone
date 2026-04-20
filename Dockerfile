FROM python:3.10-slim

LABEL maintainer="NEODrone Project Contributors"
LABEL description="NEODrone Dataset Processing Pipeline - Complete data extraction, cleaning, annotation, and validation pipeline for drone imagery datasets"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements_extraction.txt .
COPY requirements_annotation.txt .
COPY requirements_metadata.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_extraction.txt && \
    pip install --no-cache-dir -r requirements_annotation.txt && \
    pip install --no-cache-dir -r requirements_metadata.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY README.md .
COPY LICENSE .

# Create necessary directories
RUN mkdir -p /app/data/raw_videos \
    /app/data/frames \
    /app/data/synced \
    /app/data/cleaned \
    /app/data/annotations \
    /app/data/validation \
    /app/data/qa_audit \
    /app/weights

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Create a non-root user
RUN useradd -m -u 1000 neodrone && \
    chown -R neodrone:neodrone /app

USER neodrone

# Expose port for potential web interface
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.extraction.extract_frames", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1