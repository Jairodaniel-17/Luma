//! S3-compatible API over the blob store.
pub mod credentials;
pub mod routes;
pub mod sigv4;
pub mod xml;

/// The S3 router.
///
/// Separate from the `/v1` router because S3 owns the root of the path space:
/// `GET /` is ListBuckets and `GET /{bucket}/{key}` is any object. Mounted
/// together, one would shadow the other — `/v1/health` is a perfectly good S3
/// bucket named `v1` with a key named `health`.
pub fn router(state: crate::api::AppState) -> axum::Router {
    use axum::extract::DefaultBodyLimit;
    use axum::routing::{delete, get, head, post, put};

    // Without this the S3 API inherits axum's default of **2 MiB**, which made
    // it useless for anything a person would actually store: `PutObject` and
    // `UploadPart` above 2 MiB had their connection closed with no status and
    // nothing logged, so a client saw `ConnectionClosedError` and had no way to
    // learn why.
    //
    // It went unnoticed because every S3 test used hundred-byte parts —
    // `docs/integrar/S3.md` even said so under "cargas grandes de verdad", and
    // listed it as untested rather than broken. `tests/e2e/s3_scale.py` is what
    // found it, on the first 8 MiB part it tried to upload.
    //
    // The bound stays, and is the same knob the `/v1` router uses
    // (`max_body_bytes`, 100 MB by default, `MAX_BODY_MB` to change it): these
    // handlers take the body as `Bytes`, so it is buffered in full, and
    // `DefaultBodyLimit::disable()` would trade a broken API for one that any
    // client can use to exhaust the server's memory. Real S3 allows 5 GiB parts;
    // matching that would mean streaming the body to disk, which is a different
    // change and belongs with it, not smuggled into a limit constant.
    let max_body = state.config.max_body_bytes;

    axum::Router::new()
        .route("/", get(routes::list_buckets))
        .route(
            "/:bucket",
            get(routes::bucket_get)
                .put(routes::bucket_put)
                .delete(routes::bucket_delete),
        )
        // `*key` rather than `:key`: S3 keys contain slashes, and they are part
        // of the key rather than path structure. With `:key`, every object in a
        // folder-shaped prefix would 404.
        .route("/:bucket/*key", get(routes::object_get))
        .route("/:bucket/*key", put(routes::object_put))
        .route("/:bucket/*key", head(routes::object_head))
        .route("/:bucket/*key", delete(routes::object_delete))
        .route("/:bucket/*key", post(routes::object_post))
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state)
}

/// Start the S3 listener. Returns the bound port, or `None` when disabled.
///
/// Off unless `s3_port` is set, for the same reason the RESP port is: a server
/// that starts answering on a new port after an upgrade is a surprise, and on a
/// shared host it is a conflict.
pub async fn spawn(
    config: &crate::config::Config,
    state: crate::api::AppState,
    shutdown: tokio_util::sync::CancellationToken,
) -> std::io::Result<Option<u16>> {
    if config.s3_port == 0 {
        return Ok(None);
    }
    let addr = std::net::SocketAddr::new(config.bind_addr, config.s3_port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?.port();
    let max_body = config.max_body_bytes;

    let app = router(state);
    tokio::spawn(async move {
        let served = axum::serve(listener, app).with_graceful_shutdown(async move {
            shutdown.cancelled().await;
        });
        if let Err(e) = served.await {
            tracing::error!("S3 listener stopped: {e}");
        }
    });

    tracing::info!(port = bound, "S3-compatible listener started");
    // What this warns about has to be true, or it teaches operators to ignore
    // it. It used to say multipart ETags were not MD5; that divergence was
    // closed and the line was not updated, so the one warning the S3 API prints
    // was advertising a defect it no longer had.
    tracing::warn!(
        max_body_mb = max_body / (1024 * 1024),
        "the S3 API is experimental: several subresources are refused rather \
         than implemented, and a single object or part is capped at max_body_mb \
         where real S3 allows 5 GiB. See docs/integrar/S3.md."
    );
    Ok(Some(bound))
}
