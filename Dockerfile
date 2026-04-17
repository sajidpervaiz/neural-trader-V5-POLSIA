# Production-ready multi-stage Dockerfile for Neural Trader v4

# Stage 1: Rust build
FROM rust:1.85 AS rust-builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    clang \
    cmake \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust workspace and build
COPY rust/ ./rust/
COPY proto/ ./rust/proto/
COPY proto/ ./rust/gateway/proto/
WORKDIR /build/rust

# Build Rust components
RUN cargo build --workspace --release

# Stage 2: TypeScript build
FROM node:20-alpine AS ts-builder

WORKDIR /build

# Copy TypeScript source
COPY ts/ ./ts/
WORKDIR /build/ts/dex-layer

# Install dependencies and build
RUN npm ci && npm run build

# Stage 3: Python base
FROM python:3.12-slim AS python-base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TA-Lib
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf ta-lib/

# Stage 4: Python gateway service
FROM python-base AS python

WORKDIR /app

# Copy Python code
COPY core/ ./core/
COPY data_ingestion/ ./data_ingestion/
COPY engine/ ./engine/
COPY execution/ ./execution/
COPY interface/ ./interface/
COPY storage/ ./storage/
COPY main.py .
COPY config/ ./config/

# Copy Rust Python bindings
COPY --from=rust-builder /build/rust/target/release/libneural_trader_rust.so /usr/local/lib/
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf

# Create non-root user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

EXPOSE 8000

CMD ["python", "main.py"]

# Stage 5: Rust gateway service
FROM gcr.io/distroless/python3-debian12 AS gateway

WORKDIR /app

# Copy Rust gateway binary
COPY --from=rust-builder /build/rust/target/release/gateway /app/gateway

# Copy generated protobuf files (if any)
# COPY generated/rust/ ./proto/

# Create non-root user
USER 65534:65534

EXPOSE 50051 8001

CMD ["/app/gateway"]

# Stage 6: Full application
FROM python-base AS production

WORKDIR /app

# Copy all code
COPY core/ ./core/
COPY data_ingestion/ ./data_ingestion/
COPY engine/ ./engine/
COPY execution/ ./execution/
COPY interface/ ./interface/
COPY storage/ ./storage/
COPY research/ ./research/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY proto/ ./proto/
COPY main.py .
COPY config/ ./config/

# Copy built assets
COPY --from=rust-builder /build/rust/target/release/libneural_trader_rust.so /usr/local/lib/
COPY --from=ts-builder /build/ts/dex-layer/dist /app/ts/dex-layer/

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf

# Create non-root user
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

EXPOSE 8000 8001 50051

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["python", "main.py"]

# Stage 7: Development environment
FROM python-base AS dev

WORKDIR /app

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    black \
    flake8 \
    mypy \
    ipython \
    jupyterlab

# Copy all source code
COPY . .

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Node.js for TypeScript
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Create directories
RUN mkdir -p logs data models notebooks

USER 1000:1000

EXPOSE 8000 3001 50051

CMD ["/bin/bash"]
