FROM rust:1.82 AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
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
    libssl3 ca-certificates python3 python3-pip \
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

CMD ["quant", "serve", "--config", "config/default.toml"]
