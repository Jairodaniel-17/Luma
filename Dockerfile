FROM clux/muslrust:stable AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY benches ./benches
COPY src ./src
COPY docs ./docs
COPY ui ./ui

# cache deps + build parcial
RUN cargo build --release --target x86_64-unknown-linux-musl

# ahora sí copia TODO
COPY . .

RUN cargo build --release --target x86_64-unknown-linux-musl --bin luma

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/luma /luma
ENTRYPOINT ["/luma"]
CMD ["--port", "1234", "--bind", "0.0.0.0", "--DATA_DIR", "/data"]
