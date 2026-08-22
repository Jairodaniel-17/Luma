//! The RESP port over TLS.
//!
//! F4.1 of `docs/PLAN-MAESTRO.md`, the part that was left pending. It became
//! more than hygiene once `AUTH` started carrying an organization's api key: in
//! the clear, anyone on the path has a credential that binds a connection to
//! that org's whole keyspace.
//!
//! What these check, in order of what would hurt most if it were wrong:
//!
//! 1. A TLS listener refuses a plaintext client rather than serving it.
//! 2. A TLS client gets the same RESP semantics as a plaintext one.
//! 3. Turning TLS on with no certificate **refuses to start**, instead of
//!    falling back to plaintext because a file was missing.
//! 4. TLS off stays plaintext, so enabling HTTPS elsewhere cannot silently
//!    change this port's protocol under a running client.

use luma::config::Config;
use luma::engine::Engine;
use luma::resp::listener::{spawn, RespMetrics};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// A self-signed certificate for `localhost`, written to a temp directory.
///
/// Generated rather than committed: a checked-in key is a key someone will
/// eventually reuse, and a checked-in certificate expires and fails the suite on
/// a date nobody chose.
fn self_signed(dir: &std::path::Path) -> (String, String) {
    let certified = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("certificate generation");
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    std::fs::write(&cert_path, certified.cert.pem()).unwrap();
    // `serialize_pem` emits PKCS#8, which is what the loader requires.
    std::fs::write(&key_path, certified.signing_key.serialize_pem()).unwrap();
    (
        cert_path.to_str().unwrap().to_string(),
        key_path.to_str().unwrap().to_string(),
    )
}

struct Server {
    port: u16,
    shutdown: CancellationToken,
    _dir: tempfile::TempDir,
}

/// Start a listener, returning the error instead of panicking when it refuses.
async fn try_start(
    mut tune: impl FnMut(&mut Config, &std::path::Path),
) -> Result<Server, std::io::Error> {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config {
        data_dir: Some(dir.path().to_str().unwrap().to_string()),
        resp_port: 0,
        api_key: String::new(),
        ..Config::default()
    };
    tune(&mut config, dir.path());
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    config.resp_port = probe.local_addr().unwrap().port();
    drop(probe);

    let shutdown = CancellationToken::new();
    let engine = Engine::new(config.clone(), shutdown.clone()).unwrap();
    let port = spawn(
        &config,
        engine,
        Arc::new(RespMetrics::default()),
        None,
        shutdown.clone(),
    )
    .await?
    .expect("listener must bind");

    Ok(Server {
        port,
        shutdown,
        _dir: dir,
    })
}

async fn start_tls() -> Server {
    try_start(|config, dir| {
        let (cert, key) = self_signed(dir);
        config.resp_tls_enabled = true;
        config.resp_tls_cert_path = Some(cert);
        config.resp_tls_key_path = Some(key);
    })
    .await
    .expect("a TLS listener with a valid certificate must start")
}

/// A rustls client that trusts anything, because the certificate is self-signed
/// and generated seconds ago. Verifying it would test rustls, not Luma.
fn permissive_client_config() -> rustls::ClientConfig {
    // The client side needs it too, and for the same reason.
    luma::install_crypto_provider();

    #[derive(Debug)]
    struct TrustAnything;

    impl rustls::client::danger::ServerCertVerifier for TrustAnything {
        fn verify_server_cert(
            &self,
            _end_entity: &rustls::pki_types::CertificateDer<'_>,
            _intermediates: &[rustls::pki_types::CertificateDer<'_>],
            _server_name: &rustls::pki_types::ServerName<'_>,
            _ocsp: &[u8],
            _now: rustls::pki_types::UnixTime,
        ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &rustls::pki_types::CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
            Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes()
        }
    }

    rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(TrustAnything))
        .with_no_client_auth()
}

/// Send one command over TLS and return the raw reply.
fn tls_call(port: u16, frame: &[u8]) -> Vec<u8> {
    let config = Arc::new(permissive_client_config());
    let name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
    let mut connection = rustls::ClientConnection::new(config, name).unwrap();
    let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    socket
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut tls = rustls::Stream::new(&mut connection, &mut socket);
    tls.write_all(frame).unwrap();
    tls.flush().unwrap();
    let mut reply = [0u8; 4096];
    let read = tls.read(&mut reply).unwrap();
    reply[..read].to_vec()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_tls_listener_serves_resp_over_tls() {
    let server = start_tls().await;
    let port = server.port;
    let reply = tokio::task::spawn_blocking(move || {
        tls_call(port, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$5\r\nvalue\r\n")
    })
    .await
    .unwrap();
    assert_eq!(String::from_utf8_lossy(&reply), "+OK\r\n");
    server.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_plaintext_client_gets_nothing_from_a_tls_listener() {
    // The property that matters: it must not answer in the clear. A server that
    // fell back would hand the credential to exactly the observer TLS is there
    // to stop.
    let server = start_tls().await;
    let port = server.port;
    let reply = tokio::task::spawn_blocking(move || {
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket.write_all(b"PING\r\n").unwrap();
        let mut buffer = [0u8; 128];
        match socket.read(&mut buffer) {
            Ok(0) => Vec::new(),
            Ok(n) => buffer[..n].to_vec(),
            // A reset or a timeout is a refusal too.
            Err(_) => Vec::new(),
        }
    })
    .await
    .unwrap();

    assert!(
        !String::from_utf8_lossy(&reply).contains("PONG"),
        "a TLS listener answered a plaintext client: {:?}",
        String::from_utf8_lossy(&reply)
    );
    server.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn tls_carries_the_same_semantics_as_plaintext() {
    // Encryption must not change what the protocol says. A subtly different
    // framing over TLS would only show up under a real client.
    let server = start_tls().await;
    let port = server.port;
    let replies = tokio::task::spawn_blocking(move || {
        (
            tls_call(
                port,
                b"*4\r\n$5\r\nRPUSH\r\n$1\r\nq\r\n$1\r\na\r\n$1\r\nb\r\n",
            ),
            tls_call(port, b"*2\r\n$3\r\nGET\r\n$7\r\nmissing\r\n"),
        )
    })
    .await
    .unwrap();
    assert_eq!(String::from_utf8_lossy(&replies.0), ":2\r\n");
    // A nil, not an empty bulk — the same distinction as in the clear.
    assert_eq!(String::from_utf8_lossy(&replies.1), "$-1\r\n");
    server.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn tls_without_a_certificate_refuses_to_start() {
    // Not a warning and not a fallback. Serving plaintext because a file was
    // missing is how credentials end up on the wire while the operator believes
    // the port is encrypted.
    let outcome = try_start(|config, _| {
        config.resp_tls_enabled = true;
    })
    .await;
    let error = outcome.err().expect("the listener must refuse to start");
    assert!(
        error.to_string().contains("no certificate resolves"),
        "the refusal must say what is missing: {error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tls_with_an_unreadable_certificate_refuses_to_start() {
    let outcome = try_start(|config, dir| {
        config.resp_tls_enabled = true;
        config.resp_tls_cert_path = Some(dir.join("absent.pem").to_str().unwrap().to_string());
        config.resp_tls_key_path = Some(dir.join("absent.key").to_str().unwrap().to_string());
    })
    .await;
    let error = outcome.err().expect("the listener must refuse to start");
    assert!(
        error.to_string().contains("cannot open"),
        "the refusal must name the file: {error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_http_certificate_is_used_when_no_resp_specific_one_is_set() {
    // The common deployment: one certificate for the host. A separate pair
    // exists only because the Redis port is often published under another name.
    let server = try_start(|config, dir| {
        let (cert, key) = self_signed(dir);
        config.resp_tls_enabled = true;
        config.tls_cert_path = Some(cert);
        config.tls_key_path = Some(key);
    })
    .await
    .expect("the shared certificate must be picked up");
    let port = server.port;
    let reply = tokio::task::spawn_blocking(move || tls_call(port, b"PING\r\n"))
        .await
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&reply), "+PONG\r\n");
    server.shutdown.cancel();
}

#[tokio::test(flavor = "multi_thread")]
async fn tls_stays_off_unless_asked_for() {
    // A certificate configured for HTTPS must not silently change this port's
    // protocol: every connected client would break at once, on an upgrade that
    // looked unrelated.
    let server = try_start(|config, dir| {
        let (cert, key) = self_signed(dir);
        config.tls_cert_path = Some(cert);
        config.tls_key_path = Some(key);
    })
    .await
    .expect("plaintext must still start");
    let port = server.port;
    let reply = tokio::task::spawn_blocking(move || {
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        socket.write_all(b"PING\r\n").unwrap();
        let mut buffer = [0u8; 128];
        let read = socket.read(&mut buffer).unwrap_or(0);
        buffer[..read].to_vec()
    })
    .await
    .unwrap();
    assert_eq!(String::from_utf8_lossy(&reply), "+PONG\r\n");
    server.shutdown.cancel();
}
