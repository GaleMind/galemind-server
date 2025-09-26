# Build stage
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Build dependencies in release mode
RUN cargo build --release

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

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["./galemind", "start", "--rest-host", "0.0.0.0", "--grpc-host", "0.0.0.0"]