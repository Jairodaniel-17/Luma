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
    use axum::routing::{delete, get, head, post, put};
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
    tracing::warn!(
        "the S3 API is experimental: multipart ETags are not MD5 and several \
         subresources are refused rather than implemented. See docs/integrar/S3.md."
    );
    Ok(Some(bound))
}
