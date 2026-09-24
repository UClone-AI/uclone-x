# Multi-stage Docker build for UClone-X Framework & Embedded Dashboard

# Stage 1: Build React 19 Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Runtime Environment
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install build dependencies for tree-sitter C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package requirements and sources
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Copy built frontend static assets from stage 1
COPY --from=frontend-builder /app/src/uclone_x/ui_static/ ./src/uclone_x/ui_static/

# Install UClone-X and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all]"

# Expose developer GUI and API port
EXPOSE 5180

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5180/api/health || exit 1

# Default runtime command
ENTRYPOINT ["ucx"]
CMD ["ui", "--host", "0.0.0.0", "--port", "5180"]
