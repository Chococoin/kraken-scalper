# Stage 1: Build Rust binary
FROM rust:1.83-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY migrations ./migrations

# Build release binary
RUN cargo build --release --bin recorder

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub for dataset uploads
RUN pip3 install --break-system-packages huggingface_hub

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/recorder /app/

# Copy configuration and scripts
COPY config/default.toml /app/config/
COPY scripts/hf_upload.py /app/scripts/

# Create data directory for parquet files
RUN mkdir -p /app/data

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Run the recorder daemon
CMD ["/app/recorder"]
