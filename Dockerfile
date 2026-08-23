FROM clux/muslrust:stable AS builder
WORKDIR /app

# Dependency layer — only rebuilds when Cargo.toml / Cargo.lock changes
COPY Cargo.toml Cargo.lock ./
# Stubs for everything `Cargo.toml` declares a target for — not just `src`.
#
# `src/main.rs` alone was not enough and the manifest would not parse at all:
# four `[[bench]]` sections name files under `benches/`, which this layer copies
# later, and cargo refuses to read a manifest whose targets are missing. So the
# dependency layer failed with "can't find `vector_bench` bench", the image was
# never built, and the release stopped there.
#
# It went unnoticed because this is the only path that builds the image, and it
# runs only when a `v*` tag is pushed — the plan recorded the job as "nobody has
# seen this pass yet", and the first time anybody did, it did not.
RUN mkdir -p src benches \
    && echo 'fn main(){}' > src/main.rs \
    && for bench in vector_bench sqlite_bench vector_mmap_bench hybrid_hub_bench; do \
         echo 'fn main(){}' > "benches/$bench.rs"; \
       done
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
