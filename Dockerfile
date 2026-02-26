FROM rust:latest AS builder

WORKDIR /app
# Cache dependencies: copy manifests first
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY migrations/ migrations/
RUN cargo build --release

# Build web UI
FROM node:22-slim AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Runtime
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    libssl3 ca-certificates curl python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/quant /usr/local/bin/quant
COPY --from=web-builder /web/dist /app/web/dist
COPY config/default.toml /app/config/default.toml
COPY migrations/ /app/migrations/
COPY scripts/ /app/scripts/
COPY ml_models/ /app/ml_models/

# Install Python deps for ML/market data
RUN pip3 install --break-system-packages akshare lightgbm scikit-learn pandas numpy 2>/dev/null || true

WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -sf http://localhost:8080/api/health || exit 1

CMD ["quant", "serve", "--config", "config/default.toml"]
