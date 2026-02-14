FROM rust:1.82 AS builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/quant /usr/local/bin/quant
COPY config/default.toml /app/config/default.toml

WORKDIR /app
EXPOSE 8080

CMD ["quant", "serve", "--config", "config/default.toml"]
