# syntax=docker/dockerfile:1

# ---- build -----------------------------------------------------------------
# Trixie, not bookworm: the ONNX Runtime binaries that ort-sys downloads are
# built against glibc 2.38 and a matching libstdc++, so they reference symbols
# (__isoc23_strtol, std::__cxx11::...::_M_replace_cold) that Debian 12's glibc
# 2.36 does not provide, and the final link fails on undefined symbols. The
# runtime stage tracks the same release so the two libraries agree.
FROM rust:1-trixie AS builder

# protoc compiles the service definition; libclang is needed to generate
# RocksDB's bindings. Neither is needed at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        libclang-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# ONNX Runtime and RocksDB link statically, so the runtime image needs nothing
# beyond a C and C++ standard library. Only the two binaries are built; the
# test-only crates and dev-dependencies are left out.
# The registry and the build directory are cache mounts, so a rebuild after a
# source change recompiles only what changed. Neither survives into the image,
# which is why the binaries are copied out before the mount goes away.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/build/target,sharing=locked \
    cargo build --release --locked -p galemind -p galemind-client \
    && mkdir -p /out \
    && cp target/release/galemind target/release/gmctl /out/ \
    && strip /out/galemind /out/gmctl

# ---- runtime ---------------------------------------------------------------
FROM debian:trixie-slim AS runtime

ARG BUILDTIME
ARG VERSION
ARG REVISION

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --shell /usr/sbin/nologin galemind

WORKDIR /app

COPY --from=builder /out/galemind /usr/local/bin/galemind
# The client ships too, so the container can check its own health and be
# poked at without another image.
COPY --from=builder /out/gmctl /usr/local/bin/gmctl

# The model repository and the durable store. Both are writable by the
# service account so a mounted volume works without extra setup.
RUN mkdir -p /app/models /app/data && chown -R galemind:galemind /app
COPY --chown=galemind:galemind models/ /app/models/

USER galemind

# MALLOC_ARENA_MAX bounds glibc's per-thread freed-memory arenas, of which it
# allows up to eight per core by default -- 96 on a twelve-core host.
# Inference frees in large blocks from many threads, so those arenas retain
# most of what a batch allocated and resident memory settles two to three
# times above what the work in flight needs. Measured on the ten-model
# repository: 196 MB unset against 69 MB with this, and flat rather than still
# climbing. It belongs in the environment because glibc reads it before main
# runs; see the memory section of the README.
ENV GALEMIND_MODEL_REPOSITORY=/app/models \
    GALEMIND_STORE_PATH=/app/data/models.db \
    GALEMIND_REST_ADDRESS=0.0.0.0:8000 \
    GALEMIND_GRPC_ADDRESS=0.0.0.0:8001 \
    GALEMIND_LOG=info \
    MALLOC_ARENA_MAX=2

# 8000 REST, 8001 gRPC. Metrics are served at /metrics on the REST port.
EXPOSE 8000 8001

# The server's own liveness endpoint, asked by its own client.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["gmctl", "--endpoint", "http://127.0.0.1:8000", "live"]

LABEL org.opencontainers.image.created="${BUILDTIME}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.title="GaleMind Server" \
      org.opencontainers.image.description="CPU inference server speaking the Predict Protocol v2" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/Galemind/galemind-server"

ENTRYPOINT ["galemind"]
CMD ["serve"]
