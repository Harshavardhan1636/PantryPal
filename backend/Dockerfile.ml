# ============================================================================
# PantryPal ML Model Server - BentoML Dockerfile
# Optimized for inference with GPU support option
# ============================================================================

# Stage 1: Build stage
FROM bentoml/bento-server:1.2.0-python3.11 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy ML dependencies
COPY backend/ml/requirements.txt ./

# Install ML libraries
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Stage 2: Runtime stage
FROM bentoml/bento-server:1.2.0-python3.11

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BENTOML_HOME=/app/bentoml \
    BENTOML_PORT=3000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy ML code and models
COPY --chown=mluser:mluser backend/ml /app/ml
COPY --chown=mluser:mluser backend/ml_deployment /app/ml_deployment

# Create BentoML directories
RUN mkdir -p /app/bentoml /app/models /app/logs && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start BentoML server
CMD ["bentoml", "serve", "backend.ml_deployment.model_deployment:WastePredictor", \
     "--host", "0.0.0.0", \
     "--port", "3000", \
     "--workers", "2", \
     "--production"]
