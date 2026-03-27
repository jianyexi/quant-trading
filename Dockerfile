# ── Build Rust backend ─────────────────────────────────
FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY migrations/ migrations/
RUN cargo build --release

# ── Build web UI ──────────────────────────────────────
FROM node:22-slim AS web-builder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# ── Runtime ───────────────────────────────────────────
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3t64 ca-certificates curl python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python deps in a venv — fail loudly on missing packages
RUN python3 -m venv /opt/ml-venv
ENV PATH="/opt/ml-venv/bin:$PATH"
RUN pip install --no-cache-dir \
    tushare akshare yfinance lightgbm scikit-learn pandas numpy flask

COPY --from=builder /app/target/release/quant /usr/local/bin/quant
COPY --from=web-builder /web/dist /app/web/dist
COPY config/ /app/config/
COPY migrations/ /app/migrations/
COPY scripts/ /app/scripts/
COPY ml_models/ /app/ml_models/

# Non-root user for production security
RUN groupadd -r quant && useradd -r -g quant -d /app quant \
    && mkdir -p /app/logs /app/data \
    && chown -R quant:quant /app \
    && chmod +x /app/scripts/docker-entrypoint.sh

USER quant
WORKDIR /app
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -sf http://localhost:8080/api/health || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["quant", "--config", "config/default.toml", "serve"]
