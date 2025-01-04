# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install build dependencies and create virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Install runtime TBB dependencies
RUN apt-get update && apt-get install -y \
    libtbb-dev \
    libtbbmalloc2 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Create entrypoint script to run FastAPI server
RUN echo '#!/bin/sh\n\
    uvicorn server:app --host 0.0.0.0 --port 8000' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Run the application
CMD ["/entrypoint.sh"]