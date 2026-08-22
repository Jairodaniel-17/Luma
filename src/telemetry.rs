//! Tracing setup, with optional OTLP export.
//!
//! W5.1 of `docs/PLAN-MAESTRO.md`. Metrics say *how much*; a trace says *where
//! the time went in this one request*, which is the question you cannot answer
//! from a histogram once several engines are involved in one call — a hub search
//! that pre-filters in SQL, embeds, then searches vectors is three subsystems and
//! one latency number.
//!
//! ## Opt-in, and off by default
//!
//! Export only happens when `otel_endpoint` is set. There is no fallback
//! endpoint: guessing `localhost:4317` would make every developer's process
//! spend its life retrying a connection to a collector that is not there, and
//! the retries would be the loudest thing in the log.
//!
//! ## A failure to export is not a failure to start
//!
//! If the exporter cannot be built, the server logs it and runs with local
//! logging only. Refusing to boot because a telemetry sidecar is missing turns an
//! observability problem into an outage, which is the wrong trade for something
//! whose whole purpose is watching.

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Held for the lifetime of the process; dropping it flushes pending spans.
///
/// Returned rather than leaked so the flush happens at a point the caller
/// chooses. A dropped-early guard means the last spans before shutdown — the
/// interesting ones, if it is shutting down badly — never leave the process.
pub struct Telemetry {
    provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl Telemetry {
    /// Flush and shut down the exporter.
    ///
    /// Explicit rather than only in `Drop`, because a flush can block and doing
    /// that inside a destructor during an abnormal exit is how a shutdown hangs.
    pub fn shutdown(&mut self) {
        if let Some(provider) = self.provider.take() {
            if let Err(e) = provider.shutdown() {
                tracing::warn!("OTLP shutdown did not complete cleanly: {e}");
            }
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Install the tracing subscriber, exporting to OTLP when configured.
///
/// The filter comes from `RUST_LOG` as before. Without it the default is `info`,
/// which is a change worth stating: previously a bare `luma serve` printed
/// nothing at all on a successful start, so an operator checking which port it
/// bound to had to know to set an environment variable first.
pub fn init(endpoint: Option<&str>, service_name: &str) -> Telemetry {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let Some(endpoint) = endpoint.filter(|e| !e.trim().is_empty()) else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
        return Telemetry { provider: None };
    };

    match build_provider(endpoint, service_name) {
        Ok(provider) => {
            let tracer = opentelemetry::trace::TracerProvider::tracer(&provider, "luma");
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .init();
            tracing::info!(endpoint, "OTLP trace export enabled");
            Telemetry {
                provider: Some(provider),
            }
        }
        Err(e) => {
            // Local logging still comes up. See the module note: a missing
            // collector must not be an outage.
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .init();
            tracing::error!(
                endpoint,
                "OTLP export could not be set up; continuing with local logging only: {e}"
            );
            Telemetry { provider: None }
        }
    }
}

fn build_provider(
    endpoint: &str,
    service_name: &str,
) -> anyhow::Result<opentelemetry_sdk::trace::SdkTracerProvider> {
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let resource = opentelemetry_sdk::Resource::builder()
        .with_attributes([
            KeyValue::new("service.name", service_name.to_string()),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        ])
        .build();

    Ok(opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build())
}

#[cfg(test)]
mod tests {

    #[test]
    fn an_absent_or_blank_endpoint_means_no_exporter() {
        // Blank as well as absent: an empty string in a config file or an unset
        // environment variable expanded by a shell both arrive as `Some("")`,
        // and treating that as an endpoint would make the process retry a
        // connection to nothing.
        //
        // `init` installs a global subscriber and can only run once per process,
        // so the branch is checked through the same predicate it uses rather than
        // by calling it twice.
        for candidate in [None, Some(""), Some("   ")] {
            assert!(
                candidate.filter(|e: &&str| !e.trim().is_empty()).is_none(),
                "{candidate:?} must not be treated as an endpoint"
            );
        }
        assert!(Some("http://collector:4317")
            .filter(|e: &&str| !e.trim().is_empty())
            .is_some());
    }
}
