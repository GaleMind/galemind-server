# Build stage
FROM rust:latest-slim-bullseye AS builder

# Build arguments for metadata
ARG BUILDTIME
ARG VERSION
ARG REVISION

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies only (this layer will be cached)
RUN cargo build --release && rm src/main.rs

# Copy source code
COPY src/ src/

# Build the actual application
RUN cargo build --release

# Strip the binary to reduce size
RUN strip target/release/galemind

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false galemind

WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/galemind /app/galemind

# Create directory for models
RUN mkdir -p /app/models && chown galemind:galemind /app/models

# Switch to non-root user
USER galemind

# Set environment variables
ENV MODELS_DIR=/app/models

# Expose ports
EXPOSE 8080 50051

# Add metadata labels
ARG BUILDTIME
ARG VERSION
ARG REVISION
LABEL org.opencontainers.image.created="${BUILDTIME}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.title="Galemind Server" \
      org.opencontainers.image.description="AI-powered server for Galemind platform" \
      org.opencontainers.image.source="https://github.com/Galemind/galemind-server"

# Health check (install curl first)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./galemind", "start", "--rest-host", "0.0.0.0", "--grpc-host", "0.0.0.0"]