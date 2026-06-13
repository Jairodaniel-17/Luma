FROM clux/muslrust:stable AS builder
WORKDIR /app

# Dependency layer — only rebuilds when Cargo.toml / Cargo.lock changes
COPY Cargo.toml Cargo.lock ./
# Create stub src so cargo can resolve the workspace
RUN mkdir -p src && echo 'fn main(){}' > src/main.rs
RUN cargo build --release --target x86_64-unknown-linux-musl 2>&1 | tail -5; \
    rm -f target/x86_64-unknown-linux-musl/release/deps/luma*

# Application layer
COPY src      ./src
COPY benches  ./benches
COPY docs     ./docs
COPY ui       ./ui
RUN cargo build --release --target x86_64-unknown-linux-musl --bin luma

# Minimal runtime image (no shell, no libc)
FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/luma /luma
EXPOSE 1234
ENTRYPOINT ["/luma"]
CMD ["serve"]
