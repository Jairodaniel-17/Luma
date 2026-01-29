# ================================
# Builder - MUSL static binary
# ================================
FROM clux/muslrust:stable AS builder

WORKDIR /app

# 1. Cache SOLO dependencias
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --target x86_64-unknown-linux-musl && \
    rm -rf src

# 2. Copiar proyecto REAL (incluye docs/ y ui/)
COPY . .

# 3. Build final (con assets reales)
RUN cargo build --release --target x86_64-unknown-linux-musl --bin luma

# ================================
# Runtime - scratch
# ================================
FROM scratch

COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/luma /luma

ENTRYPOINT ["/luma"]
CMD ["--port", "1234", "--bind", "0.0.0.0", "--DATA_DIR", "/data"]
