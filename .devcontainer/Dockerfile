# Use the same base image as before (Rust on Debian Bookworm)
FROM mcr.microsoft.com/devcontainers/rust:1-1-bookworm

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install protoc (protobuf compiler) and other dev tools required for building gRPC/tonic/prost
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       protobuf-compiler \
       git \
       htop \
       make \
       libprotobuf-dev \
       pkg-config \
       build-essential \
       clang \
       cmake \
       git \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> /home/vscode/.bashrc

# Install protoc-gen-grpc-web if needed (optional) - commented out for now
# RUN curl -L -o /usr/local/bin/protoc-gen-grpc-web \ 
#     https://github.com/grpc/grpc-web/releases/download/1.4.0/protoc-gen-grpc-web-1.4.0-linux-x86_64 \
#     && chmod +x /usr/local/bin/protoc-gen-grpc-web

# Ensure the vscode user can use installed tools (paths are system-wide)

# Switch back to default non-root user set by the base image (if any)
USER vscode
